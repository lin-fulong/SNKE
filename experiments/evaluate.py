import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import csv
import json
import shlex
import shutil
import sys
from itertools import islice
from time import time
from typing import Tuple, Union
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer,
    KnownsDataset,
    KGDataset,
    MQuAKEDataset,
)
from experiments.logging_utils import install_run_logger
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre

from memit import MEMITHyperParams
from memit.compute_z import get_module_input_output_at_words, compute_z
from memit.memit_main import apply_memit_to_model, get_context_templates
from memit.memit_seq_main import apply_memit_seq_to_model
from memit.memit_rect_main import apply_memit_rect_to_model
from SNKE import SNKEHyperParams, GNN
from SNKE.SNKE_main import apply_SNKE_to_model
from KGMET import KGMETHyperParams, apply_KGMET_to_model
from NMKE import NMKEHyperParams, apply_NMKE_to_model
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.globals import *
from nse import NSEHyperParams
from nse.nse_main import apply_nse_to_model
from glue_eval.glue_eval import GLUEEval
ALG_DICT = {
    "SNKE": (SNKEHyperParams, apply_SNKE_to_model),
    "KGMET": (KGMETHyperParams, apply_KGMET_to_model),
    "NMKE": (NMKEHyperParams, apply_NMKE_to_model),
    "MEMIT_seq": (MEMITHyperParams, apply_memit_seq_to_model),
    "MEMIT_prune": (MEMITHyperParams, apply_memit_to_model),
    "MEMIT_rect": (MEMITHyperParams, apply_memit_rect_to_model),
    "NSE": (NSEHyperParams, apply_nse_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
    "mquake": (MQuAKEDataset, compute_rewrite_quality_counterfact)
}


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]
    print(
        f"[experiments.evaluate] start | alg={alg_name} | ds={ds_name} "
        f"| model={model_name} | num_edits={num_edits} | size_limit={dataset_size_limit}"
    )

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    install_run_logger(run_dir, "experiments.evaluate")
    print(f"[experiments.evaluate] Results will be stored at {run_dir}")
    runtime_metrics_path = run_dir / "runtime_metrics.csv"
    neuron_stats_path = run_dir / "neuron_mask_counts.csv"
    launch_command_path = run_dir / "launch_command.txt"
    launch_command = "python -m experiments.evaluate"
    if len(sys.argv) > 1:
        launch_command += " " + " ".join(shlex.quote(arg) for arg in sys.argv[1:])
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        launch_command = f"CUDA_VISIBLE_DEVICES={shlex.quote(cuda_visible_devices)} {launch_command}"
    with open(launch_command_path, "w", encoding="utf-8") as f:
        f.write(f"cwd: {os.getcwd()}\n")
        f.write(f"command: {launch_command}\n")
    print(f"[experiments.evaluate] Launch command saved to {launch_command_path}")
    if "MEMIT" in alg_name:
    # Get run hyperparameters
        params_path = (
            run_dir / "params.json"
            if continue_from_run is not None
            else HPARAMS_DIR / "MEMIT" / hparams_fname
        )
    else:
        params_path = (
            run_dir / "params.json"
            if continue_from_run is not None
            else HPARAMS_DIR / alg_name / hparams_fname
        )
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"[experiments.evaluate] Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    if type(model_name) is str:
        resolved_model_name = resolve_model_path(model_name)
        print(
            f"[experiments.evaluate] Instantiating model: {model_name}"
            f" | source={resolved_model_name}"
        )
        model = AutoModelForCausalLM.from_pretrained(resolved_model_name).cuda()
        tok = AutoTokenizer.from_pretrained(resolved_model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path
    
    gnn_model = None
    
    # Load data
    print("[experiments.evaluate] Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)
    eval_ds = KnownsDataset(DATA_DIR)
    graphds = KGDataset(DATA_DIR, size=dataset_size_limit,
                        tok=tok, ds_name=ds_name)
    print(
        f"[experiments.evaluate] Dataset ready | ds_len={len(ds)} "
        f"| eval_ds_len={len(eval_ds)} | graph_ds_len={len(graphds)}"
    )
    # Get cache templates
    cache_template = None
    if use_cache:
        if any(alg in alg_name for alg in ["MEMIT","SNKE", "KGMET", "NMKE", "MEMIT_seq", "MEMIT_prune", "MEMIT_rect"]):
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_MEMIT"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
            )
        else:
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
            )
        print(f"[experiments.evaluate] Will load cache from {cache_template}")
    if alg_name == "NSE":
        cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        for record in ds:
            # Retrieve k/v pair if already stored in cache
            cache_fname = (
                Path(
                    str(cache_template).format(
                        hparams.layers[-1], hparams.clamp_norm_factor, record["case_id"]
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
                continue
            # Compute k/v pair if not loaded from cache
            if not data_loaded:
                context_templates = get_context_templates(model, tok)
                cur_z = compute_z(
                    model,
                    tok,
                    {"case_id": record["case_id"], **record["requested_rewrite"]},
                    hparams,
                    hparams.layers[-1],
                    context_templates,
                )
                if cache_fname is not None:
                    cache_fname.parent.mkdir(exist_ok=True, parents=True)
                    np.savez(
                        cache_fname,
                        **{
                            "v_star": cur_z.detach().cpu().numpy(),
                        },
                    )
                    print(f"[experiments.evaluate] Cached k/v pair at {cache_fname}")
    if any(alg in alg_name for alg in ["MEMIT_seq", "MEMIT_prune", "NSE"]):
        # Iterate through dataset
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        if hparams.model_name == "gpt2-xl":
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
        elif hparams.model_name in ["EleutherAI_gpt-j-6B","Llama3-8B"]:
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
        del W_out
    # hs = get_module_input_output_at_words(
    #         model,
    #         tok,
    #         hparams.layers[-1],
    #         context_templates=[request["template"] for request in eval_ds],
    #         words=[request["subject"] for request in eval_ds],
    #         module_template=hparams.layer_module_tmp,
    #         fact_token_strategy=hparams.fact_token,
    #     )[1].T
    # torch.save(hs, "pre_edit_hs.pt")
    # del hs
    glue_save_location = str(run_dir) + '/' + 'glue_eval/'
    os.makedirs(glue_save_location, exist_ok=True)
    print(f"[experiments.evaluate] GLUE eval artifacts will be stored at {glue_save_location}")
    cnt = 0
    total_edit_time = 0.0
    executed_edit_chunks = 0
    peak_cuda_memory_allocated = 0
    peak_cuda_memory_reserved = 0
    #zipped_lists = list(zip(ds, graphds))
    for record_chunks in chunks(ds, num_edits):

        case_result_template = str(run_dir / "{}_edits-case_{}.json")
        print(f"[experiments.evaluate] ===== chunk {cnt + 1} =====")
        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            print(f"[experiments.evaluate] Chunk {cnt + 1} skipped because all case files already exist")
            continue
        
        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        request_count = (
            sum(len(record["requested_rewrite"]) for record in record_chunks)
            if ds_name == "mquake"
            else len(record_chunks)
        )
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT","SNKE", "KGMET", "NMKE", "MEMIT_seq", "MEMIT_prune", "NSE"]) else dict()
        seq_args = dict(cache_c=cache_c) if any(alg in alg_name for alg in ["MEMIT_seq", "NSE"]) else dict()
        nc_args = (
            {
                "neuron_stats_path": str(neuron_stats_path),
                "neuron_stats_context": {
                    "alg_name": alg_name,
                    "model_name": model_name,
                    "ds_name": ds_name,
                    "edit_step": cnt + 1,
                    "edit_num": (cnt + 1) * num_edits,
                    "num_edits": num_edits,
                    "request_count": request_count,
                    "case_ids": json.dumps(case_ids, ensure_ascii=False),
                },
            }
            if alg_name in ["SNKE", "NMKE"]
            else {}
        )

        if cnt == 0 and args.downstream_eval_steps > 0:#do initial GLUE EVAL WITH ORIGINAL MODEL
           glue_results = {'edit_num': -1}

           out_file = glue_save_location + "base.json"
            
           glue_eval = GLUEEval(model, tok, number_of_tests = 100)
           glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)

           #store the individual overall result file
           output_filename = out_file.replace('.json', '_glue.json')
           with open(output_filename, "w") as f:
               json.dump(glue_results, f, indent=4)

        if ds_name == "mquake":
            graph_input = [
                triples
                for item in graphds
                if item["case_id"] in case_ids
                for triples in item["triples"]
            ]
        else:
            graph_input = [item["triples"] for item in graphds if item["case_id"] in case_ids]
        print(
            f"[experiments.evaluate] Prepared chunk | case_ids={case_ids} "
            f"| graph_batches={len(graph_input)}"
        )
            
        glame_args = dict(graph_input=graph_input) if any(
            alg in alg_name for alg in ["SNKE", "KGMET"]) else dict()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time()
        print(
            f"[experiments.evaluate] Invoking {alg_name} | "
            f"seq_args={list(seq_args.keys())} | extra_args={list(glame_args.keys())}"
        )
        if any(alg in alg_name for alg in ["SNKE", "KGMET", "NMKE", "MEMIT_seq", "NSE"]):
            if ds_name == "mquake":
                if alg_name == "NMKE":
                    edited_model, cache_c = apply_algo(
                        model,
                        tok,
                        [
                            {"case_id": record["case_id"], **sub_request}
                            for record in record_chunks
                            for sub_request in record["requested_rewrite"]
                        ],
                        hparams,
                        **args_conserve_memory,
                        **etc_args,
                        **seq_args,
                        **nc_args,
                    )
                else:
                    edited_model, weights_copy = apply_algo(
                        model,
                        tok,
                        [
                            {"case_id": record["case_id"], **sub_request}
                            for record in record_chunks
                            for sub_request in record["requested_rewrite"]
                        ],
                        hparams,
                        **glame_args,
                        **args_conserve_memory,
                        **etc_args,
                        **seq_args,
                        **nc_args,
                        )
            else:
                edited_model, cache_c = apply_algo(
                    model,
                    tok,
                    [
                        {"case_id": record["case_id"], **record["requested_rewrite"]}
                        for record in record_chunks
                    ],
                    hparams,
                    **glame_args,
                    **args_conserve_memory,
                    **etc_args,
                    **seq_args,
                    **nc_args,
                )
        elif alg_name == "MEMIT_prune":
            if cnt == 0:
                edited_model, weights_copy = apply_algo(
                    model,
                    tok,
                    [
                        {"case_id": record["case_id"], **record["requested_rewrite"]}
                        for record in record_chunks
                    ],
                    hparams,
                    return_orig_weights=True,
                    **args_conserve_memory,
                    **etc_args,
                )
                # Initialize the upd_matrix dictionary
                upd_matrix = {}
            else:
                edited_model, _ = apply_algo(
                    model,
                    tok,
                    [
                        {"case_id": record["case_id"], **record["requested_rewrite"]}
                        for record in record_chunks
                    ],
                    hparams,
                    return_orig_weights=False,
                    **args_conserve_memory,
                    **etc_args,
                )
            if cnt == (dataset_size_limit/num_edits) - 1:
            # Calculate the weight update matrix
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        current_weight = nethook.get_parameter(model, k)
                        upd_matrix[k] = current_weight - v.to("cuda")
                        # Calculate max singular value of the original weight
                        _, S_orig, _ = torch.svd(v)
                        max_sigma = S_orig.max().item()

                        # Adjust the upd_matrix singular values
                        U_upd, S_upd, V_upd = torch.svd(upd_matrix[k])
                        adjusted_S = torch.where(
                            S_upd > max_sigma,
                            torch.log(S_upd) - torch.log(torch.tensor(max_sigma, device='cuda')) + max_sigma,
                            S_upd
                        )
                        upd_matrix[k] = torch.matmul(U_upd, torch.matmul(torch.diag(adjusted_S), V_upd.t()))

                # Apply the adjusted updates to the model
                with torch.no_grad():
                    for k in upd_matrix:
                        original_weight = nethook.get_parameter(model, k)
                        adjusted_weight = original_weight + upd_matrix[k]
                        original_weight.copy_(adjusted_weight)
        else:
            edited_model, _ = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **record["requested_rewrite"]}
                    for record in record_chunks
                ],
                hparams,
                return_orig_weights=False,
                **args_conserve_memory,
                **etc_args,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_cuda_memory_allocated = max(
                peak_cuda_memory_allocated,
                torch.cuda.max_memory_allocated(),
            )
            peak_cuda_memory_reserved = max(
                peak_cuda_memory_reserved,
                torch.cuda.max_memory_reserved(),
            )
        exec_time = time() - start
        total_edit_time += exec_time
        executed_edit_chunks += 1
        cnt+=1
        print(f"[experiments.evaluate] Execution took {exec_time:.4f}s")
        
        # Evaluate new model(Regular GLUE evaluation after editing)
        if args.downstream_eval_steps > 0 and cnt % args.downstream_eval_steps == 0:
           glue_results = {
                       'edit_num': cnt*num_edits,
                       'case_id': case_ids
                       }

           out_file = glue_save_location + "case_{}.json".format(record["case_id"])#stores the last case ID of the batch

           glue_eval = GLUEEval(model, tok, number_of_tests = 100)
           glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)
                    
            # store the individual overall result file
           output_filename = out_file.replace('.json', '_glue.json')
           with open(output_filename, "w") as f:
               json.dump(glue_results, f, indent=4)

    avg_edit_time = total_edit_time / executed_edit_chunks if executed_edit_chunks else 0.0
    avg_single_edit_time = total_edit_time / (executed_edit_chunks * num_edits) if executed_edit_chunks and num_edits else 0.0
    with open(runtime_metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "alg_name",
                "model_name",
                "ds_name",
                "dataset_size_limit",
                "num_edits",
                "executed_edit_chunks",
                "total_edit_time_sec",
                "avg_edit_chunk_time_sec",
                "avg_single_edit_time_sec",
                "peak_cuda_memory_allocated_bytes",
                "peak_cuda_memory_allocated_mib",
                "peak_cuda_memory_reserved_bytes",
                "peak_cuda_memory_reserved_mib",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "alg_name": alg_name,
                "model_name": model_name,
                "ds_name": ds_name,
                "dataset_size_limit": dataset_size_limit,
                "num_edits": num_edits,
                "executed_edit_chunks": executed_edit_chunks,
                "total_edit_time_sec": f"{total_edit_time:.6f}",
                "avg_edit_chunk_time_sec": f"{avg_edit_time:.6f}",
                "avg_single_edit_time_sec": f"{avg_single_edit_time:.6f}",
                "peak_cuda_memory_allocated_bytes": peak_cuda_memory_allocated,
                "peak_cuda_memory_allocated_mib": f"{peak_cuda_memory_allocated / 1024 ** 2:.2f}",
                "peak_cuda_memory_reserved_bytes": peak_cuda_memory_reserved,
                "peak_cuda_memory_reserved_mib": f"{peak_cuda_memory_reserved / 1024 ** 2:.2f}",
            }
        )
    print(f"[experiments.evaluate] Runtime metrics saved to {runtime_metrics_path}")

    hs = get_module_input_output_at_words(
            edited_model,
            tok,
            hparams.layers[-1],
            context_templates=[request["template"] for request in eval_ds],
            words=[request["subject"] for request in eval_ds],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
    torch.save(hs, "post_edit_hs_memit.pt")
    start = time()
    gen_test_vars = [snips, vec]
    for record_idx, record in enumerate(ds):
        out_file = Path(case_result_template.format(num_edits, record["case_id"]))
        if out_file.exists():
            print(f"[experiments.evaluate] Skipping {out_file}; already exists")
            continue
        metrics = {
            "case_id": record["case_id"],
            "grouped_case_ids": case_ids,
            "num_edits": num_edits,
            "requested_rewrite": record["requested_rewrite"],
            "time": exec_time,
            "post": ds_eval_method(
                edited_model,
                tok,
                record,
                *(
                    gen_test_vars
                    if generation_test_interval > 0 and (record_idx + 1) % generation_test_interval == 0
                    else [None, None]
                ),  # Only test generation every generation_test_interval records
            ),
        }
        # Dump metrics in .json
        with open(out_file, "w") as f:
            json.dump(metrics, f, indent=1)
        print(f"[experiments.evaluate] Wrote metrics for case_id={record['case_id']} to {out_file}")

        # Restore original weights
        # with torch.no_grad():
        #     for k, v in weights_copy.items():
        #         nethook.get_parameter(model, k)[...] = v.to("cuda")

        print(f"[experiments.evaluate] Evaluation took {time() - start:.4f}s")
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
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(len(small_singular_indices))
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T
def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["SNKE", "KGMET", "NMKE", "MEMIT_rect", "MEMIT_seq","MEMIT_prune", "MEMIT", "ROME", "FT", "MEND","NSE"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre", "mquake"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=2000,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=100,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--downstream_eval_steps",
        type=int,
        default=0,
        help="If we want to do sequential editing or not",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
    )
