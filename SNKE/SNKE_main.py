import csv
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .gnn import GNN
from .build_graph import build_graph_from_triples
from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words
from .gnn_training import optimize_graph_delta
from .SNKE_hparams import SNKEHyperParams


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
    """
    Load a cached null-space projector and reuse any compatible layers.

    Supports both the legacy tensor-only format and the new dict format:
    {
        "layers": [...],
        "projector": tensor
    }
    """
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
        print(f"[SNKE] Ignoring malformed projector cache at {file_path}")
        return projector, missing_indices, False

    if loaded_projector.ndim != 3 or tuple(loaded_projector.shape[1:]) != (
        proj_dim,
        proj_dim,
    ):
        print(
            f"[SNKE] Projector cache shape mismatch at {file_path}: "
            f"cached={tuple(loaded_projector.shape)} expected=(*, {proj_dim}, {proj_dim})"
        )
        return projector, missing_indices, False

    if loaded_layers is not None:
        layer_to_idx = {
            int(layer): idx for idx, layer in enumerate(loaded_layers[: loaded_projector.shape[0]])
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
        if loaded_projector.shape[0] != len(target_layers):
            print(
                f"[SNKE] Legacy projector cache layer count mismatch at {file_path}: "
                f"cached={loaded_projector.shape[0]} expected={len(target_layers)}. "
                "Reusing aligned prefix layers and recomputing the remainder."
            )

    return projector, missing_indices, True


def _save_projector_cache(
    file_path: str, projector: torch.Tensor, layers: List[int]
) -> None:
    torch.save(
        {
            "layers": list(layers),
            "projector": projector.detach().cpu(),
        },
        file_path,
    )


def apply_SNKE_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: SNKEHyperParams,
    graph_input: list,
    gnn_model: Optional[GNN] = None,
    cache_template: Optional[str] = None,
    cache_c=None,
    P=None,
    neuron_stats_path: Optional[str] = None,
    neuron_stats_context: Optional[Dict] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Applies SNKE updates after first computing masked deltas, following NMKE_main's structure.
    """
    global P_loaded, cache_c_new, P_CACHE, P_CACHE_LAYERS, CACHE_C_CACHE

    print(
        f"[SNKE] apply_SNKE_to_model start | requests={len(requests)} "
        f"| graphs={len(graph_input) if graph_input is not None else 0} "
        f"| layers={hparams.layers}"
    )

    if gnn_model is None:
        n_embed = model.config.n_embd if hasattr(model.config, "n_embd") else model.config.hidden_size
        print(f"[SNKE] Initializing internal RGCN | hidden_dim={n_embed}")
        gnn_h_dim = int(n_embed * hparams.gnn_dim_factor)
        num_bases = 100 if gnn_h_dim % 100 == 0 else 64
        if gnn_h_dim % num_bases != 0:
            num_bases = 1
        print(f"[SNKE] RGCN block bases={num_bases}")
        gnn_model = GNN(
            num_rels=1000,
            num_nodes=1000,
            h_dim=gnn_h_dim,
            out_dim=n_embed,
            num_bases=num_bases,
            num_basis=num_bases,
            num_hidden_layers=2,
            dropout=hparams.gnn_feat_drop,
            self_loop=True,
            skip_connect=False,
            encoder_name="rgcn",
            opn="sub",
            use_cuda=True,
            analysis=False,
        )

    if P is None or cache_c is None:
        weight_name = f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight"
        w_out = nethook.get_parameter(model, weight_name)
        if hparams.model_name == "gpt2-xl":
            proj_dim = w_out.shape[0]
        else:
            proj_dim = w_out.shape[1]
        print(f"[SNKE] Projection dimension resolved from {weight_name}: {proj_dim}")
        del w_out

    if cache_c is None:
        expected_cache_shape = (len(hparams.layers), proj_dim, proj_dim)
        if (
            not cache_c_new
            or CACHE_C_CACHE is None
            or tuple(CACHE_C_CACHE.shape) != expected_cache_shape
        ):
            print(f"[SNKE] Initializing cache_c with shape=({len(hparams.layers)}, {proj_dim}, {proj_dim})")
            CACHE_C_CACHE = torch.zeros(
                (len(hparams.layers), proj_dim, proj_dim), device="cpu"
            )
            cache_c_new = True
        else:
            print("[SNKE] Reusing cached cache_c tensor")
        cache_c = CACHE_C_CACHE

    if P is None:
        if (
            not P_loaded
            or P_CACHE is None
            or P_CACHE_LAYERS != list(hparams.layers)
            or tuple(P_CACHE.shape) != (len(hparams.layers), proj_dim, proj_dim)
        ):
            file_path = f"null_space_project_{hparams.model_name}.pt"
            P_CACHE, missing_indices, loaded = _load_projector_cache(
                file_path, hparams.layers, proj_dim
            )
            if loaded:
                print(f"[SNKE] Loaded compatible null-space projector entries from: {file_path}")
            else:
                print(f"[SNKE] Null-space projector missing. Computing and saving to: {file_path}")

            for idx in missing_indices:
                layer = hparams.layers[idx]
                print(f"[SNKE] Computing projector for layer {layer} ({idx + 1}/{len(hparams.layers)})")
                P_CACHE[idx, :, :] = get_project(model, tok, layer, hparams)

            if missing_indices or not loaded:
                _save_projector_cache(file_path, P_CACHE, hparams.layers)
                print(f"[SNKE] Saved null-space projector to: {file_path}")
            P_CACHE_LAYERS = list(hparams.layers)
            P_loaded = True
        else:
            print("[SNKE] Reusing cached null-space projector")
        P = P_CACHE

    deltas, cache_c = execute_SNKE(
        model,
        tok,
        requests,
        hparams,
        graph_input,
        gnn_model,
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
                f"[SNKE] Applying delta to {w_name} | "
                f"weight_shape={tuple(w.shape)} | delta_norm={torch.linalg.norm(upd_matrix).item():.4f}"
            )
            w[...] += upd_matrix.float()

    print(f"[SNKE] New weights successfully inserted into {list(deltas.keys())}")
    return model, cache_c


def execute_SNKE(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: SNKEHyperParams,
    graph_input: list,
    gnn_model: GNN,
    cache_template: Optional[str] = None,
    cache_c=None,
    P=None,
    neuron_stats_path: Optional[str] = None,
    neuron_stats_context: Optional[Dict] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Executes the SNKE update algorithm while restoring the original model afterward.
    """

    deltas: Dict[str, torch.Tensor] = {}
    requests = deepcopy(requests)
    graph_info = []

    print(
        f"[SNKE] execute_SNKE start | requests={len(requests)} "
        f"| graph_batches={len(graph_input) if graph_input is not None else 0}"
    )

    for i, (request, graph) in enumerate(zip(requests, graph_input)):
        if request["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]

        print(
            f"[SNKE] Building graph for case_id={request['case_id']} "
            f"| subject={request['subject']} | triples={len(graph)}"
        )
        g, node_indices, init_rel_emb = build_graph_from_triples(graph, model, tok, hparams)
        graph_info.append(
            {
                "graph": g,
                "node_indices": node_indices,
                "init_rel_emb": init_rel_emb,
            }
        )
        print(
            f"[SNKE] Graph ready for case_id={request['case_id']} | "
            f"nodes={g.num_nodes()} | edges={g.num_edges()} | rel_emb={tuple(init_rel_emb.shape)}"
        )

    for request in requests[:10]:
        print(
            f"[SNKE] request sample: "
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
    print(f"[SNKE] Context template groups: {[len(group) for group in context_templates]}")

    for request, graph_dict in zip(requests, graph_info):
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
                print(f"[SNKE] Loaded cached z for case_id={request['case_id']} from {cache_fname}")
            except Exception as e:
                print(f"[SNKE] Error reading cache file due to {e}. Recomputing...")

        if not data_loaded:
            print(f"[SNKE] Computing z for case_id={request['case_id']} at layer {z_layer}")
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
                gnn_model=gnn_model,
                graph_input=graph_dict["graph"],
                node_indices=graph_dict["node_indices"],
                init_rel_emb=graph_dict["init_rel_emb"],
            )
            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(cache_fname, **{"v_star": cur_z.detach().cpu().numpy()})
                print(f"[SNKE] Cached k/v pair at {cache_fname}")

    zs = torch.stack(z_list, dim=1)
    print(f"[SNKE] Stacked z tensor shape: {tuple(zs.shape)}")

    importance_masks = {}
    print("[SNKE] Computing structural guidance")
    structural_guidance = compute_structural_guidance(
        requests=requests,
        hparams=hparams,
        gnn_model=gnn_model,
        graph_info=graph_info,
        z_targets=z_list,
    )
    guidance_cur_zs = get_module_input_output_at_words(
        model,
        tok,
        z_layer,
        context_templates=[request["prompt"] for request in requests],
        words=[request["subject"] for request in requests],
        module_template=hparams.layer_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )[1].T
    guidance_target_residuals = (zs - guidance_cur_zs).T.detach()
    print(
        f"[SNKE] Target-residual guidance ready | "
        f"shape={tuple(guidance_target_residuals.shape)} "
        f"| mean_norm={torch.linalg.norm(guidance_target_residuals, dim=1).mean().item():.6f}"
    )
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
    print(f"[SNKE] Forward pass for attribution complete | prompt_batch={len(input_prompts)}")

    del input_tok
    torch.cuda.empty_cache()

    use_resonance = True
    use_burst = True
    for layer in hparams.layers:
        print(f"[SNKE] Computing importance mask for layer {layer}")
        if "llama" in hparams.model_name.lower():
            score_matrix = get_importance_scores(model_output, model, layer)
        else:
            score_matrix = get_gpt_importance_scores(model_output, model, layer)

        score_matrix = apply_target_residual_alignment(
            score_matrix=score_matrix,
            target_residuals=guidance_target_residuals,
            model=model,
            hparams=hparams,
            layer=layer,
            context_templates=context_templates,
        )

        # Structure-guided attribution reweighting: I_tilde = Imp * (1 + beta * z(alpha))
        score_matrix = apply_structural_guidance(
            score_matrix=score_matrix,
            structural_guidance=structural_guidance,
            model=model,
            hparams=hparams,
            layer=layer,
            requests=requests,
            context_templates=context_templates,
        )

        resonance_ratio, burst_ratio = entropy_adaptive_mask_ratio(score_matrix)
        final_mask, resonance_mask, burst_mask, resonance_counts, burst_score, _, _ = (
            compute_hybrid_resonant_mask(
                score_matrix,
                resonance_ratio=resonance_ratio,
                burst_ratio=burst_ratio,
                use_resonance=use_resonance,
                use_burst=use_burst,
            )
        )
        importance_masks[layer] = final_mask
        active_neurons = int(final_mask.sum().item())
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
            f"[SNKE] Layer {layer} mask ready | active_neurons={active_neurons} "
            f"| resonance_ratio={resonance_ratio:.4f} | burst_ratio={burst_ratio:.4f}"
        )

        del score_matrix, final_mask, resonance_mask, burst_mask
        del resonance_counts, burst_score
        torch.cuda.empty_cache()

    del model_output
    torch.cuda.empty_cache()

    for i, layer in enumerate(hparams.layers):
        print(f"\n\n[SNKE] ===== Layer {layer} ({i + 1}/{len(hparams.layers)}) =====\n")

        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"[SNKE] Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

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
        print(f"[SNKE] Layer {layer} z error: {torch.linalg.norm(targets, dim=0).mean().item():.6f}")

        repeat_factor = layer_ks.size(1) // targets.size(1)
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(hparams.layers) - i)
        upd_matrix = torch.linalg.solve(
            P[i, :, :].cuda() @ (layer_ks @ layer_ks.T + cache_c[i, :, :].cuda())
            + hparams.L2 * torch.eye(layer_ks.shape[0], dtype=torch.float, device="cuda"),
            P[i, :, :].cuda() @ layer_ks @ resid.T,
        )

        neuron_mask = importance_masks[layer].to(upd_matrix.device)
        raw_upd_norm = torch.linalg.norm(upd_matrix).item()
        soft_gate_strength = 0.2
        gate = 1.0 + soft_gate_strength * (neuron_mask - neuron_mask.mean())
        upd_matrix = gate[:, None] * upd_matrix
        print(
            f"[SNKE] Layer {layer} soft gate | "
            f"active={int(neuron_mask.sum().item())}/{neuron_mask.numel()} "
            f"| gate_min={gate.min().item():.4f} | gate_max={gate.max().item():.4f} "
            f"| raw_upd_norm={raw_upd_norm:.6f} "
            f"| gated_upd_norm={torch.linalg.norm(upd_matrix).item():.6f}"
        )

        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        print(f"[SNKE] {weight_name} orig norm: {torch.linalg.norm(weights[weight_name]).item():.6f}")
        print(f"[SNKE] {weight_name} upd norm: {torch.linalg.norm(upd_matrix).item():.6f}")

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
        print(f"[SNKE] Updated cache_c for layer {layer}")

    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"[SNKE] Deltas successfully computed for {list(weights.keys())}")
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

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
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
        "Update matrix computed by SNKE does not match original weight shape. Check for bugs in the code?"
    )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        print("[SNKE] Context templates cache miss. Generating templates.")
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
        print(f"[SNKE] Cached context templates {CONTEXT_TEMPLATES_CACHE}")
    else:
        print("[SNKE] Reusing cached context templates")

    return CONTEXT_TEMPLATES_CACHE


def compute_structural_guidance(
    requests: List[Dict],
    hparams: SNKEHyperParams,
    gnn_model: GNN,
    graph_info: List[Dict],
    z_targets: List[torch.Tensor],
) -> torch.Tensor:
    if gnn_model is None or len(graph_info) == 0:
        print("[SNKE] Structural guidance skipped because graph model/info is unavailable")
        return None

    structural_vectors = []
    for request, graph_dict, z_target in zip(requests, graph_info, z_targets):
        desired_z = z_target.detach()
        print(
            f"[SNKE] Training structural guidance for case_id={request['case_id']} "
            f"| desired_z_norm={desired_z.norm().item():.4f}"
        )

        def graph_alignment_loss(graph_delta_feature):
            mse_loss = F.mse_loss(graph_delta_feature, desired_z)
            zero = torch.zeros((), device=desired_z.device)
            return (
                mse_loss,
                mse_loss.detach().unsqueeze(0),
                mse_loss,
                zero,
                zero,
                desired_z,
            )

        _, graph_delta = optimize_graph_delta(
            gnn_model=gnn_model,
            hparams=hparams,
            graph_input=graph_dict["graph"],
            node_indices=graph_dict["node_indices"],
            init_rel_emb=graph_dict["init_rel_emb"],
            request=request,
            loss_fn=graph_alignment_loss,
        )
        structural_vectors.append(F.normalize(graph_delta.detach(), p=2, dim=-1))
        print(
            f"[SNKE] Structural guidance ready for case_id={request['case_id']} "
            f"| graph_delta_norm={graph_delta.norm().item():.4f}"
        )

    if len(structural_vectors) == 0:
        print("[SNKE] Structural guidance produced no vectors")
        return None

    stacked = torch.stack(structural_vectors, dim=0)
    print(f"[SNKE] Structural guidance stack shape: {tuple(stacked.shape)}")
    return stacked


def apply_structural_guidance(
    score_matrix: torch.Tensor,
    structural_guidance: Optional[torch.Tensor],
    model: AutoModelForCausalLM,
    hparams: SNKEHyperParams,
    layer: int,
    requests: List[Dict],
    context_templates: List[List[str]],
) -> torch.Tensor:
    if structural_guidance is None:
        print(f"[SNKE] Layer {layer} structural guidance skipped")
        return score_matrix

    num_neurons = score_matrix.shape[1]
    neuron_vectors = get_rewrite_neuron_vectors(model, hparams, layer, num_neurons)
    if neuron_vectors is None:
        print(f"[SNKE] Layer {layer} structural guidance skipped due to incompatible rewrite weight shape")
        return score_matrix

    neuron_vectors = neuron_vectors.to(score_matrix.device)
    z_edit = structural_guidance.to(score_matrix.device)
    alpha = torch.sigmoid(z_edit @ neuron_vectors.T)

    prompts_per_request = sum(len(context_group) for context_group in context_templates)
    alpha = alpha.repeat_interleave(prompts_per_request, dim=0)

    if alpha.shape[0] != score_matrix.shape[0]:
        print(f"[SNKE] Layer {layer} structural guidance skipped due to alpha/score shape mismatch")
        return score_matrix

    alpha = (alpha - alpha.mean(dim=1, keepdim=True)) / (
        alpha.std(dim=1, keepdim=True) + 1e-6
    )
    guidance_strength = 0.1
    gate = torch.clamp(1.0 + guidance_strength * alpha, min=0.5, max=1.5)
    print(
        f"[SNKE] Layer {layer} applying structural guidance | "
        f"score_shape={tuple(score_matrix.shape)} | alpha_shape={tuple(alpha.shape)} "
        f"| gate_mean={gate.mean().item():.4f} | gate_std={gate.std().item():.4f}"
    )
    return score_matrix * gate


def get_rewrite_neuron_vectors(
    model: AutoModelForCausalLM,
    hparams: SNKEHyperParams,
    layer: int,
    num_neurons: int,
) -> Optional[torch.Tensor]:
    rewrite_weight = nethook.get_parameter(
        model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
    ).detach()

    if rewrite_weight.shape[0] == num_neurons:
        return rewrite_weight
    if rewrite_weight.shape[1] == num_neurons:
        return rewrite_weight.T
    return None


def apply_target_residual_alignment(
    score_matrix: torch.Tensor,
    target_residuals: torch.Tensor,
    model: AutoModelForCausalLM,
    hparams: SNKEHyperParams,
    layer: int,
    context_templates: List[List[str]],
) -> torch.Tensor:
    num_neurons = score_matrix.shape[1]
    neuron_vectors = get_rewrite_neuron_vectors(model, hparams, layer, num_neurons)
    if neuron_vectors is None:
        print(f"[SNKE] Layer {layer} residual alignment skipped due to incompatible rewrite weight shape")
        return score_matrix

    prompts_per_request = sum(len(context_group) for context_group in context_templates)
    residuals = target_residuals.to(score_matrix.device).repeat_interleave(
        prompts_per_request, dim=0
    )
    if residuals.shape[0] != score_matrix.shape[0]:
        print(f"[SNKE] Layer {layer} residual alignment skipped due to residual/score shape mismatch")
        return score_matrix

    neuron_vectors = neuron_vectors.to(score_matrix.device)
    residuals = F.normalize(residuals.float(), p=2, dim=1)
    neuron_vectors = F.normalize(neuron_vectors.float(), p=2, dim=1)
    alignment = torch.abs(residuals @ neuron_vectors.T)
    aligned_scores = score_matrix * alignment
    print(
        f"[SNKE] Layer {layer} target-residual alignment | "
        f"alignment_mean={alignment.mean().item():.4f} "
        f"| alignment_std={alignment.std().item():.4f}"
    )
    return torch.nan_to_num(aligned_scores, nan=0.0, posinf=0.0, neginf=0.0)




def get_project(model, tok, layer, hparams):
    force_recompute = False
    print(f"[SNKE] get_project start for layer {layer}")
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
    threshold = hparams.nullspace_threshold
    small_singular_indices = (s < threshold).nonzero(as_tuple=True)[0]
    print(
        f"[SNKE] get_project complete for layer {layer} | "
        f"threshold={threshold} | nullspace_rank={len(small_singular_indices)}"
    )
    return u[:, small_singular_indices] @ u[:, small_singular_indices].T


def get_importance_scores(model_output, model, layer):
    if isinstance(model_output, dict):
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

    def get_prob(vector):
        return torch.nn.Softmax(dim=-1)(vector)

    def get_bsvalues(vector, model, final_var):
        vector = vector * torch.rsqrt(final_var + 1e-6)
        vector = vector.to(model.model.norm.weight.device)
        vector_rmsn = vector * model.model.norm.weight
        return model.lm_head(vector_rmsn)

    predicted_list = [model_output[0][i][-1] for i in range(model_output[0].shape[0])]
    predicted_top10_list = [
        torch.argsort(predicted, descending=True)[:10] for predicted in predicted_list
    ]
    predicted_indices_list = [
        predicted_indices[0].item() for predicted_indices in predicted_top10_list
    ]

    residual_outputs = model_output[-1][layer][1].tolist()
    coefficient_scores_all = model_output[-1][layer][0].tolist()
    final_var_list = model_output[-1][-1][2].tolist()

    def safe_log_prob(vector, eps=1e-8):
        return torch.log(torch.clamp(vector, min=eps, max=1 - eps))

    amplify_factor = 30
    importance_scores_all = []
    for batch_idx in range(len(residual_outputs)):
        final_var_tensor = torch.tensor(final_var_list[batch_idx][-1], dtype=torch.float32).cuda()
        final_var = final_var_tensor.pow(2).mean(-1, keepdim=True)
        coefficient_scores = torch.tensor(coefficient_scores_all[batch_idx][-1], dtype=torch.float).cuda()
        fc2_vectors = model.model.layers[layer].mlp.down_proj.weight.data.cuda()
        ffn_subvalues = (coefficient_scores * fc2_vectors).T
        cur_residual = torch.tensor(residual_outputs[batch_idx][-1], dtype=torch.float).cuda()
        with torch.no_grad():
            origin_prob = get_prob(get_bsvalues(cur_residual, model, final_var))[
                predicted_indices_list[batch_idx]
            ]
            cur_ffn_subvalues_plus = ffn_subvalues * amplify_factor + cur_residual
            cur_ffn_subvalues_probs = get_prob(
                get_bsvalues(cur_ffn_subvalues_plus, model, final_var)
            )[:, predicted_indices_list[batch_idx]]
            origin_prob = origin_prob.clamp(min=1e-8, max=1.0)
            cur_ffn_subvalues_probs = cur_ffn_subvalues_probs.clamp(min=1e-8, max=1.0)
            importance_scores = safe_log_prob(cur_ffn_subvalues_probs) - safe_log_prob(origin_prob)
        importance_scores_all.append(importance_scores)

    importance_scores_all = torch.stack(importance_scores_all).to("cuda")
    return torch.nan_to_num(importance_scores_all, nan=0.0, posinf=0.0, neginf=0.0)


def get_gpt_importance_scores(model_output, model, layer):
    if isinstance(model_output, dict):
        coeff_tensor = model_output["mlp_traces"][layer]
        coeff_tensor = coeff_tensor[0] if isinstance(coeff_tensor, tuple) else coeff_tensor
        attention_mask = model_output["attention_mask"]
        last_token_idxs = attention_mask.sum(dim=1) - 1

        if coeff_tensor.shape[0] != attention_mask.shape[0]:
            coeff_tensor = coeff_tensor.transpose(0, 1)

        batch_indices = torch.arange(coeff_tensor.shape[0], device=coeff_tensor.device)
        token_activations = coeff_tensor[batch_indices, last_token_idxs.to(coeff_tensor.device), :]
        return torch.nan_to_num(
            token_activations.abs().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def get_prob(vector):
        return torch.nn.Softmax(dim=-1)(vector)

    def get_bsvalues(vector, model, final_var):
        mean = torch.mean(vector, -1)
        vector_ln = (vector - mean.unsqueeze(-1)) / final_var * model.transformer.ln_f.weight.data
        return model.lm_head(vector_ln).data

    predicted_list = [model_output[0][i][-1] for i in range(model_output[0].shape[0])]
    predicted_top10_list = [
        torch.argsort(predicted, descending=True)[:10] for predicted in predicted_list
    ]
    predicted_indices_list = [
        predicted_indices[0].item() for predicted_indices in predicted_top10_list
    ]

    residual_outputs = model_output[-1][layer][1].tolist()
    coefficient_scores_all = model_output[-1][layer][0].tolist()
    final_var_list = model_output[-1][-1][2].tolist()

    def safe_log_prob(vector, eps=1e-8):
        return torch.log(torch.clamp(vector, min=eps, max=1 - eps))

    amplify_factor = 30
    importance_scores_all = []
    for batch_idx in range(len(residual_outputs)):
        final_var_tensor = torch.tensor(final_var_list[batch_idx][-1], dtype=torch.float32).cuda()
        final_var = torch.var(final_var_tensor, dim=-1, unbiased=False, keepdim=True).sqrt() + 1e-5
        coefficient_scores = torch.tensor(coefficient_scores_all[batch_idx][-1], dtype=torch.float).cuda()
        fc2_vectors = model.transformer.h[layer].mlp.c_proj.weight.data.T.cuda()
        ffn_subvalues = (coefficient_scores * fc2_vectors).T
        cur_residual = torch.tensor(residual_outputs[batch_idx][-1], dtype=torch.float).cuda()
        with torch.no_grad():
            origin_prob = get_prob(get_bsvalues(cur_residual, model, final_var))[
                predicted_indices_list[batch_idx]
            ]
            cur_ffn_subvalues_plus = ffn_subvalues * amplify_factor + cur_residual
            cur_ffn_subvalues_probs = get_prob(
                get_bsvalues(cur_ffn_subvalues_plus, model, final_var)
            )[:, predicted_indices_list[batch_idx]]
            origin_prob = origin_prob.clamp(min=1e-8, max=1.0)
            cur_ffn_subvalues_probs = cur_ffn_subvalues_probs.clamp(min=1e-8, max=1.0)
            importance_scores = safe_log_prob(cur_ffn_subvalues_probs) - safe_log_prob(origin_prob)
        importance_scores_all.append(importance_scores)

    importance_scores_all = torch.stack(importance_scores_all).to("cuda")
    return torch.nan_to_num(importance_scores_all, nan=0.0, posinf=0.0, neginf=0.0)


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
    scaled = shifted * alpha
    softmax_scores = torch.softmax(scaled, dim=1)

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
