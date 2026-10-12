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

from .gnn import GNN
from .build_graph import build_graph_from_triples
from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .KGMET_hparams import KGMETHyperParams
# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

def apply_KGMET_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: KGMETHyperParams,
    graph_input: Optional[list] = None,
    gnn_model: Optional[GNN] = None,
    cache_template: Optional[str] = None,
    cache_c=None,
    P=None,
    return_orig_weights_device: Optional[str] = None,
    **kwargs,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the KGMET update algorithm.

    This signature is intentionally compatible with the arguments passed by
    experiments.evaluate_sequential when the algorithm is registered later.
    """
    if graph_input is None:
        raise ValueError(
            "KGMET requires graph_input. When wiring it into an experiment "
            "entrypoint, load KGDataset and pass graph_input just like SNKE."
        )
    if len(graph_input) != len(requests):
        raise ValueError(
            f"KGMET expected one graph per request, got "
            f"{len(graph_input)} graph(s) for {len(requests)} request(s)."
        )

    if gnn_model is None:
        n_embed = model.config.n_embd if hasattr(model.config, "n_embd") else model.config.hidden_size
        print(f"[KGMET] Initializing internal RGCN | hidden_dim={n_embed}")
        gnn_h_dim = int(n_embed * hparams.gnn_dim_factor)
        num_bases = 100 if gnn_h_dim % 100 == 0 else 64
        if gnn_h_dim % num_bases != 0:
            num_bases = 1
        gnn_model = GNN(
            num_nodes=1000,
            h_dim=gnn_h_dim,
            out_dim=n_embed,
            num_rels=1000,
            num_bases=num_bases,
            num_basis=num_bases,
            num_hidden_layers=2,
            dropout=hparams.gnn_feat_drop,
            self_loop=True,
            skip_connect=False,
            encoder_name="uvrgcn",
            opn="sub",
            use_cuda=True,
            analysis=False,
        )

    if P is None or cache_c is None:
        weight_name = f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight"
        w_out = nethook.get_parameter(model, weight_name)
        proj_dim = w_out.shape[0] if hparams.model_name == "gpt2-xl" else w_out.shape[1]
        print(f"[KGMET] Projection dimension resolved from {weight_name}: {proj_dim}")
        del w_out

    if cache_c is None:
        cache_c = torch.zeros((len(hparams.layers), proj_dim, proj_dim), device="cpu")
        print(f"[KGMET] Initialized cache_c with shape={tuple(cache_c.shape)}")

    if P is None:
        P = torch.eye(proj_dim, device="cpu").repeat(len(hparams.layers), 1, 1)
        print(f"[KGMET] Initialized identity projector P with shape={tuple(P.shape)}")

    # Update target and print info
    requests = deepcopy(requests)
    graph_info = []
    for i, (request, graph) in enumerate(zip(requests, graph_input)):
        g, node_indices, init_rel_emb = build_graph_from_triples(
            graph, model, tok, hparams)
        graph_dict = {"graph":g, "node_indices": node_indices, "init_rel_emb": init_rel_emb}
        graph_info.append(graph_dict)
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"KGMET request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request, graph_dict in zip(requests, graph_info):
        # Retrieve k/v pair if already stored in cache
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
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z_mlp = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
                gnn_model = gnn_model,
                graph_input = graph_dict["graph"],
                node_indices = graph_dict["node_indices"],
                init_rel_emb = graph_dict["init_rel_emb"])

            z_list.append(cur_z_mlp)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z_mlp.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)

    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")
       # cov = get_cov(
       #     model,
       #     tok,
       #     hparams.rewrite_module_tmp.format(layer),
       #     hparams.mom2_dataset,
       #     hparams.mom2_n_samples,
       #     hparams.mom2_dtype,
       #     force_recompute=False,
       # )
        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")
        # Compute residual error
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
        print("z error", torch.linalg.norm(targets, dim=0).mean())
        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        
       # adj_k = torch.linalg.solve(
       #     hparams.mom2_update_weight * cov + layer_ks @ layer_ks.T,
       #     layer_ks,
       # )
        resid = targets / np.sqrt(len(hparams.layers) - i)  # Distribute residual across layers
        #upd_matrix = resid @ adj_k.T
        upd_matrix = torch.linalg.solve(
                P[i,:,:].cuda() @ (layer_ks @ layer_ks.T + cache_c[i,:,:].cuda()) + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda"), P[i,:,:].cuda() @ layer_ks @ resid.T
        )
        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix
        # Clear GPU memory
        #del U,S,cov
        for x in [layer_ks, cur_zs, targets, upd_matrix]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return model, cache_c


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
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

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

    return (
        torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by KGMET does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
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
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
