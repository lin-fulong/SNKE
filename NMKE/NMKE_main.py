import csv
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words
from .NMKE_hparams import NMKEHyperParams


CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}
P_loaded = False
cache_c_new = False
P_CACHE = None
P_CACHE_LAYERS = None
CACHE_C_CACHE = None


def _empty_projector_cache(num_layers: int, proj_dim: int) -> torch.Tensor:
    return torch.zeros((num_layers, proj_dim, proj_dim), device="cpu")


def _load_projector_cache(
    file_path: str, target_layers: List[int], proj_dim: int
) -> Tuple[torch.Tensor, List[int], bool]:
    target_layers = list(target_layers)
    projector = _empty_projector_cache(len(target_layers), proj_dim)
    missing_indices = list(range(len(target_layers)))

    if not os.path.exists(file_path):
        return projector, missing_indices, False

    loaded_obj = torch.load(file_path, map_location="cpu")
    loaded_layers = None
    loaded_projector = loaded_obj
    if isinstance(loaded_obj, dict) and "projector" in loaded_obj:
        loaded_layers = loaded_obj.get("layers")
        loaded_projector = loaded_obj["projector"]

    if not torch.is_tensor(loaded_projector):
        print(f"[NMKE] Ignoring malformed projector cache at {file_path}")
        return projector, missing_indices, False

    if loaded_projector.ndim != 3 or tuple(loaded_projector.shape[1:]) != (
        proj_dim,
        proj_dim,
    ):
        print(
            f"[NMKE] Projector cache shape mismatch at {file_path}: "
            f"cached={tuple(loaded_projector.shape)} expected=(*, {proj_dim}, {proj_dim})"
        )
        return projector, missing_indices, False

    if loaded_layers is not None:
        layer_to_idx = {
            int(layer): idx
            for idx, layer in enumerate(loaded_layers[: loaded_projector.shape[0]])
        }
        missing_indices = []
        for new_idx, layer in enumerate(target_layers):
            old_idx = layer_to_idx.get(int(layer))
            if old_idx is None:
                missing_indices.append(new_idx)
                continue
            projector[new_idx, :, :] = loaded_projector[old_idx, :, :]
    else:
        reusable = min(loaded_projector.shape[0], len(target_layers))
        if reusable > 0:
            projector[:reusable, :, :] = loaded_projector[:reusable, :, :]
        missing_indices = list(range(reusable, len(target_layers)))

    return projector, missing_indices, True


def _save_projector_cache(
    file_path: str, projector: torch.Tensor, layers: List[int]
) -> None:
    torch.save(
        {"layers": list(layers), "projector": projector.detach().cpu()},
        file_path,
    )


def apply_NMKE_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: NMKEHyperParams,
    cache_template: Optional[str] = None,
    cache_c=None,
    P=None,
    neuron_stats_path: Optional[str] = None,
    neuron_stats_context: Optional[Dict] = None,
    **kwargs,
) -> Dict[str, Tuple[torch.Tensor]]:
    global P_loaded, cache_c_new, P_CACHE, P_CACHE_LAYERS, CACHE_C_CACHE

    print(
        f"[NMKE] apply_NMKE_to_model start | requests={len(requests)} "
        f"| layers={hparams.layers}"
    )

    if P is None or cache_c is None:
        weight_name = f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight"
        w_out = nethook.get_parameter(model, weight_name)
        if "gpt2-xl" in hparams.model_name.lower():
            proj_dim = w_out.shape[0]
        else:
            proj_dim = w_out.shape[1]
        print(f"[NMKE] Projection dimension resolved from {weight_name}: {proj_dim}")
        del w_out

    if cache_c is None:
        expected_cache_shape = (len(hparams.layers), proj_dim, proj_dim)
        if (
            not cache_c_new
            or CACHE_C_CACHE is None
            or tuple(CACHE_C_CACHE.shape) != expected_cache_shape
        ):
            CACHE_C_CACHE = torch.zeros(expected_cache_shape, device="cpu")
            cache_c_new = True
        cache_c = CACHE_C_CACHE

    if P is None:
        if (
            not P_loaded
            or P_CACHE is None
            or P_CACHE_LAYERS != list(hparams.layers)
            or tuple(P_CACHE.shape) != (len(hparams.layers), proj_dim, proj_dim)
        ):
            file_path = f"nmke_null_space_project_{hparams.model_name.replace('/', '_')}.pt"
            P_CACHE, missing_indices, loaded = _load_projector_cache(
                file_path, hparams.layers, proj_dim
            )
            if loaded:
                print(f"[NMKE] Loaded compatible null-space projector entries from: {file_path}")
            else:
                print(f"[NMKE] Null-space projector missing. Computing and saving to: {file_path}")

            for idx in missing_indices:
                layer = hparams.layers[idx]
                print(f"[NMKE] Computing projector for layer {layer} ({idx + 1}/{len(hparams.layers)})")
                P_CACHE[idx, :, :] = get_project(model, tok, layer, hparams)

            if missing_indices or not loaded:
                _save_projector_cache(file_path, P_CACHE, hparams.layers)
                print(f"[NMKE] Saved null-space projector to: {file_path}")
            P_CACHE_LAYERS = list(hparams.layers)
            P_loaded = True
        P = P_CACHE

    deltas, cache_c = execute_NMKE(
        model,
        tok,
        requests,
        hparams,
        cache_template=cache_template,
        cache_c=cache_c,
        P=P,
        neuron_stats_path=neuron_stats_path,
        neuron_stats_context=neuron_stats_context,
    )

    with torch.no_grad():
        for w_name, upd_m in deltas.items():
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_m.to(w.device), w.shape)
            print(
                f"[NMKE] Applying delta to {w_name} | "
                f"weight_shape={tuple(w.shape)} | delta_norm={torch.linalg.norm(upd_matrix).item():.4f}"
            )
            w[...] += upd_matrix.float()

    print(f"[NMKE] New weights successfully inserted into {list(deltas.keys())}")
    return model, cache_c


def execute_NMKE(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: NMKEHyperParams,
    cache_template: Optional[str] = None,
    cache_c=None,
    P=None,
    neuron_stats_path: Optional[str] = None,
    neuron_stats_context: Optional[Dict] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    deltas: Dict[str, torch.Tensor] = {}
    requests = deepcopy(requests)

    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]

    for request in requests[:10]:
        print(
            f"[NMKE] request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if cache_fname is not None and cache_fname.exists():
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
                print(f"[NMKE] Loaded cached z for case_id={request['case_id']} from {cache_fname}")
            except Exception as e:
                print(f"[NMKE] Error reading cache file due to {e}. Recomputing...")

        if not data_loaded:
            cur_z = compute_z(model, tok, request, hparams, z_layer, context_templates)
            z_list.append(cur_z)
            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(cache_fname, **{"v_star": cur_z.detach().cpu().numpy()})
                print(f"[NMKE] Cached k/v pair at {cache_fname}")

    zs = torch.stack(z_list, dim=1)
    print(f"[NMKE] Stacked z tensor shape: {tuple(zs.shape)}")

    input_prompts = [
        context.format(request["prompt"].replace("{}", request["subject"]))
        for request in requests
        for context_list in context_templates
        for context in context_list
    ]
    input_tok = tok(input_prompts, return_tensors="pt", padding=True).to("cuda")
    tracked_layers = [
        hparams.rewrite_module_tmp.format(layer) for layer in hparams.layers
    ]
    with torch.no_grad():
        with nethook.TraceDict(
            model,
            layers=tracked_layers,
            retain_input=True,
            retain_output=False,
            detach=True,
        ) as tr:
            outputs = model(**input_tok)
    model_output = {
        "outputs": outputs,
        "mlp_traces": {
            layer: tr[hparams.rewrite_module_tmp.format(layer)].input
            for layer in hparams.layers
        },
        "attention_mask": input_tok["attention_mask"].detach().clone(),
    }
    del input_tok
    torch.cuda.empty_cache()

    importance_masks = {}
    for layer in hparams.layers:
        print(f"[NMKE] Computing importance mask for layer {layer}")
        if "llama" in hparams.model_name.lower():
            score_matrix = get_importance_scores(model_output, model, layer)
        else:
            score_matrix = get_gpt_importance_scores(model_output, model, layer)

        resonance_ratio, burst_ratio = entropy_adaptive_mask_ratio(score_matrix)
        final_mask, resonance_mask, burst_mask, resonance_counts, burst_score, _, _ = (
            compute_hybrid_resonant_mask(
                score_matrix,
                resonance_ratio=resonance_ratio,
                burst_ratio=burst_ratio,
                use_resonance=True,
                use_burst=True,
            )
        )
        importance_masks[layer] = final_mask
        _append_neuron_mask_stats(
            neuron_stats_path,
            neuron_stats_context,
            layer,
            final_mask,
            resonance_mask,
            burst_mask,
            resonance_ratio,
            burst_ratio,
        )
        print(
            f"[NMKE] Layer {layer} mask ready | "
            f"active_neurons={int(final_mask.sum().item())}"
        )
        del score_matrix, final_mask, resonance_mask, burst_mask
        del resonance_counts, burst_score
        torch.cuda.empty_cache()

    del model_output
    torch.cuda.empty_cache()

    for i, layer in enumerate(hparams.layers):
        print(f"\n\n[NMKE] ===== Layer {layer} ({i + 1}/{len(hparams.layers)}) =====\n")
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"[NMKE] Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        targets = zs - cur_zs
        print(f"[NMKE] Layer {layer} z error: {torch.linalg.norm(targets, dim=0).mean().item():.6f}")

        repeat_factor = layer_ks.size(1) // targets.size(1)
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(hparams.layers) - i)
        upd_matrix = torch.linalg.solve(
            P[i, :, :].cuda() @ (layer_ks @ layer_ks.T + cache_c[i, :, :].cuda())
            + hparams.L2 * torch.eye(layer_ks.shape[0], dtype=torch.float, device="cuda"),
            P[i, :, :].cuda() @ layer_ks @ resid.T,
        )

        neuron_mask = importance_masks[layer].to(upd_matrix.device)
        upd_matrix = neuron_mask[:, None] * upd_matrix

        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        print(f"[NMKE] {weight_name} orig norm: {torch.linalg.norm(weights[weight_name]).item():.6f}")
        print(f"[NMKE] {weight_name} upd norm: {torch.linalg.norm(upd_matrix).item():.6f}")

        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix.float()
            deltas[weight_name] = upd_matrix.detach().cpu()

        for x in [layer_ks, cur_zs, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i, :, :] += layer_ks.cpu() @ layer_ks.cpu().T
        print(f"[NMKE] Updated cache_c for layer {layer}")

    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"[NMKE] Deltas successfully computed for {list(weights.keys())}")
    return deltas, cache_c


def _append_neuron_mask_stats(
    stats_path: Optional[str],
    context: Optional[Dict],
    layer: int,
    final_mask: torch.Tensor,
    resonance_mask: torch.Tensor,
    burst_mask: torch.Tensor,
    resonance_ratio: float,
    burst_ratio: float,
) -> None:
    if stats_path is None:
        return

    path = Path(stats_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    final_cpu = final_mask.detach().float().cpu()
    resonance_cpu = resonance_mask.detach().float().cpu()
    burst_cpu = burst_mask.detach().float().cpu()
    total_neurons = int(final_cpu.numel())
    one_count = int(final_cpu.sum().item())
    resonance_one_count = int(resonance_cpu.sum().item())
    burst_one_count = int(burst_cpu.sum().item())
    row = {
        **(context or {}),
        "layer": layer,
        "total_neurons": total_neurons,
        "mask_1_count": one_count,
        "mask_0_count": total_neurons - one_count,
        "resonance_mask_1_count": resonance_one_count,
        "resonance_mask_0_count": total_neurons - resonance_one_count,
        "burst_mask_1_count": burst_one_count,
        "burst_mask_0_count": total_neurons - burst_one_count,
        "resonance_ratio": f"{resonance_ratio:.6f}",
        "burst_ratio": f"{burst_ratio:.6f}",
    }
    fieldnames = [
        "alg_name",
        "model_name",
        "ds_name",
        "edit_step",
        "edit_num",
        "num_edits",
        "request_count",
        "case_ids",
        "layer",
        "total_neurons",
        "mask_1_count",
        "mask_0_count",
        "resonance_mask_1_count",
        "resonance_mask_0_count",
        "burst_mask_1_count",
        "burst_mask_0_count",
        "resonance_ratio",
        "burst_ratio",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"[NMKE] Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if matrix.shape == shape:
        return matrix
    if matrix.T.shape == shape:
        return matrix.T
    raise ValueError(
        "Update matrix computed by NMKE does not match original weight shape. Check for bugs in the code?"
    )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        print("[NMKE] Context templates cache miss. Generating templates.")
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]
        ]
        print(f"[NMKE] Cached context templates {CONTEXT_TEMPLATES_CACHE}")
    return CONTEXT_TEMPLATES_CACHE


def get_project(model, tok, layer, hparams):
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
    ).cpu()
    u, s, _ = torch.linalg.svd(cov, full_matrices=False)
    small_singular_indices = (s < hparams.nullspace_threshold).nonzero(as_tuple=True)[0]
    print(
        f"[NMKE] get_project complete for layer {layer} | "
        f"threshold={hparams.nullspace_threshold} | nullspace_rank={len(small_singular_indices)}"
    )
    return u[:, small_singular_indices] @ u[:, small_singular_indices].T


def get_importance_scores(model_output, model, layer):
    coeff_tensor = model_output["mlp_traces"][layer]
    coeff_tensor = coeff_tensor[0] if isinstance(coeff_tensor, tuple) else coeff_tensor
    attention_mask = model_output["attention_mask"]
    last_token_idxs = attention_mask.sum(dim=1) - 1
    if coeff_tensor.shape[0] != attention_mask.shape[0]:
        coeff_tensor = coeff_tensor.transpose(0, 1)
    batch_indices = torch.arange(coeff_tensor.shape[0], device=coeff_tensor.device)
    token_activations = coeff_tensor[
        batch_indices, last_token_idxs.to(coeff_tensor.device), :
    ]
    return torch.nan_to_num(
        token_activations.abs().float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def get_gpt_importance_scores(model_output, model, layer):
    return get_importance_scores(model_output, model, layer)


def compute_hybrid_resonant_mask(
    score_matrix: torch.Tensor,
    resonance_ratio: float = 0.25,
    burst_ratio: float = 0.15,
    use_resonance: bool = True,
    use_burst: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized = (score_matrix - score_matrix.mean(dim=1, keepdim=True)) / (
        score_matrix.std(dim=1, keepdim=True) + 1e-6
    )
    resonance_counts = (normalized > 0.0).float().sum(dim=0)
    resonance_cut = torch.quantile(resonance_counts, 1 - resonance_ratio)
    resonance_mask = (resonance_counts >= resonance_cut).float()

    burst_score = score_matrix.max(dim=0).values
    burst_cut = torch.quantile(burst_score, 1 - burst_ratio)
    burst_mask = (burst_score >= burst_cut).float()

    if use_resonance and use_burst:
        final_mask = torch.clamp(resonance_mask + burst_mask, max=1.0)
    elif use_resonance:
        final_mask = resonance_mask
    elif use_burst:
        final_mask = burst_mask
    else:
        final_mask = torch.zeros_like(resonance_mask)

    return (
        final_mask,
        resonance_mask,
        burst_mask,
        resonance_counts,
        burst_score,
        resonance_cut,
        burst_cut,
    )


def entropy_adaptive_mask_ratio(
    score_matrix: torch.Tensor,
    resonance_bounds: Tuple[float, float] = (0.3, 0.4),
    burst_bounds: Tuple[float, float] = (0.3, 0.4),
    gamma_r: float = 3.0,
    gamma_b: float = 2.0,
    alpha: float = 30.0,
) -> Tuple[float, float]:
    dim = score_matrix.shape[1]
    log_dim = np.log(dim)

    shifted = score_matrix - score_matrix.max(dim=1, keepdim=True)[0]
    softmax_scores = torch.softmax(shifted * alpha, dim=1)
    entropy_prompt = -(softmax_scores * (softmax_scores + 1e-8).log()).sum(dim=1)
    entropy_r = entropy_prompt.mean() / log_dim
    resonance_ratio = resonance_bounds[0] + (
        resonance_bounds[1] - resonance_bounds[0]
    ) * entropy_r.clamp(0, 1).pow(gamma_r)

    max_acts = torch.clamp(score_matrix.max(dim=0).values, min=0.0)
    if max_acts.sum() < 1e-8:
        return float(resonance_ratio), burst_bounds[0]

    burst_probs = max_acts / (max_acts.sum() + 1e-8)
    entropy_burst = -(burst_probs * (burst_probs + 1e-8).log()).sum()
    entropy_b = entropy_burst / log_dim
    burst_ratio = burst_bounds[0] + (
        burst_bounds[1] - burst_bounds[0]
    ) * entropy_b.clamp(0, 1).pow(gamma_b)

    return float(resonance_ratio), float(burst_ratio)
