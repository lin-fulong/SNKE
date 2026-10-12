# SNKE

Official code accompanying **“Structure-Guided Neural Mask Knowledge Editing for Large Language Models via Graph-Aware Attribution”** by Ruinan Liu, Fulong Lin, Qingyu Wang, Xin Wang, and Zhihua Fu. The manuscript is available as [SNKE.pdf](SNKE.pdf).

SNKE edits factual knowledge in large language models by combining target-direction-aware neuron attribution with local knowledge graph structure. It is designed to improve edit efficacy and paraphrase generality while supporting multi-hop and sequential editing. The paper also discusses trade-offs in locality, fluency, and GPU memory use.

## Method

Given an edit request and its associated knowledge subgraph, SNKE:

1. Optimizes a target representation for the new fact and derives a target residual direction.
2. Encodes an edit-centered knowledge subgraph with an RGCN to obtain structural guidance.
3. Scores rewrite-module neurons using activation strength, alignment with the target residual, and graph guidance. Consistency and peak-response masks capture complementary neuron groups.
4. Computes a multi-layer, null-space-constrained weight update and applies a soft neuron gate to the update.

The implementation is in [`SNKE/`](SNKE/). [`KGMET/`](KGMET/) and the other editing methods are comparison implementations; `--alg_name=SNKE` selects the method proposed in this paper.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`SNKE/`](SNKE/) | SNKE editor, graph encoder, attribution, masks, and hyperparameters class |
| [`hparams/SNKE/`](hparams/SNKE/) | Released SNKE configurations for GPT-2-XL, GPT-J-6B, and LLaMA 3 |
| [`experiments/`](experiments/) | Standard, sequential, and MQuAKE evaluation entry points; result summaries |
| [`dsets/`](dsets/) | Editing datasets and knowledge graph dataset loader |
| [`wikidata_tools/`](wikidata_tools/) | Utilities used to prepare graph data |
| [`glue_eval/`](glue_eval/) | Optional downstream evaluation code and its bundled subsets |
| [`KGMET/`](KGMET/), [`NMKE/`](NMKE/), [`rome/`](rome/), [`memit/`](memit/), [`nse/`](nse/), [`baselines/`](baselines/) | Comparison methods |

## Environment

The paper's experiments use an NVIDIA A800 GPU. The code calls CUDA directly, so an NVIDIA GPU and a compatible CUDA software stack are required for the editing runs.

The released experiments were conducted in a Conda environment with Linux, CUDA 11.8, PyTorch `2.1.2+cu118`, and `tqdm` `4.64.1`. The DGL wheel specified in `requirements.txt` also targets CUDA 11.8.

Install the dependencies from the provided `requirements.txt` after creating the Conda environment.

Create and activate the Conda environment:

```bash
conda create -n snke python=3.10 -y
conda activate snke
pip install -r requirements.txt
```

On Windows, use the activation command for your shell. Run the commands below from the repository root because [`globals.yml`](globals.yml) and several evaluation assets use relative paths. A model can be specified by a Hugging Face identifier or an accessible local directory. Set `MODEL_DIR` in `globals.yml` if you want model identifiers to resolve under a local model root. Access to gated models, including LLaMA 3, must be arranged separately.

## Prepare datasets and graphs

Set the data location in [`globals.yml`](globals.yml); the default is `data/`. SNKE needs **both** the editing records and a corresponding graph file. The graph loader does not download graph files automatically.

| `--ds_name` | Editing records in `data/` | Required graph in `data/` |
| --- | --- | --- |
| `mcf` | `multi_counterfact.json` | `multi_counterfact_graph.json` |
| `zsre` | `filtered_zsre.json` or `zsre_mend_eval.json` | `zsre_graph.json` |
| `mquake` | `mquake.json` | `mquake_graph.json` |
| `cf` | `counterfact.json` | `counterfact_graph.json` |

The loader can download the standard CounterFact, MultiCounterFact, and ZsRE editing records from the URL configured in `globals.yml` when they are absent. `mquake.json` and all graph files must be supplied separately. The evaluation entry points also load `known_1000.json`, which is downloaded automatically if absent. Generation-based evaluation may additionally download attribute snippets and TF-IDF files.

Each graph file must be a JSON list of records with a `case_id` and a `triples` field. For `mcf`, `zsre`, and `cf`, `triples` contains objects with `subject`, `relation`, and `target` strings:

```json
[
  {
    "case_id": 0,
    "triples": [
      {"subject": "Example subject", "relation": "example relation", "target": "Example target"}
    ]
  }
]
```

The graph records must match the editing records by `case_id` and order. MQuAKE graph records use one list of triples per requested rewrite. [`wikidata_tools/generate_multi_counterfact_graph.py`](wikidata_tools/generate_multi_counterfact_graph.py) can generate `multi_counterfact_graph.json` after its prerequisite CounterFact and MQuAKE graph files are available; it may query Wikidata. The bundled [`wikidata_tools/zsre_graph.json`](wikidata_tools/zsre_graph.json) is a source graph artifact, not a file that the evaluation loader reads in place. Prepare a graph aligned with the chosen ZsRE editing records and put it in `data/zsre_graph.json`.

## Run SNKE

The following example uses the released GPT-2-XL configuration and a small MultiCounterFact slice. Supply the matching model and graph data before running it:

```bash
python -m experiments.evaluate \
  --alg_name=SNKE \
  --model_name=gpt2-xl \
  --hparams_fname=gpt2-xl.json \
  --ds_name=mcf \
  --dataset_size_limit=10 \
  --num_edits=1 \
  --skip_generation_tests \
  --generation_test_interval=-1
```

For the paper-style batch setting, increase `--dataset_size_limit` and `--num_edits` to the desired values; for example, 2,000 and 100 for MultiCounterFact. The first run can take much longer because SNKE computes covariance statistics and null-space projectors. Results are written under `results/SNKE/run_XXX/`. `--skip_generation_tests` disables the slower generation-based metrics; omit it when reporting fluency and generation consistency.

Sequential editing uses a separate entry point and writes to `results/SNKE_sequential/run_XXX/`:

```bash
python -m experiments.evaluate_sequential \
  --alg_name=SNKE \
  --model_name=gpt2-xl \
  --hparams_fname=gpt2-xl.json \
  --ds_name=zsre \
  --dataset_size_limit=100 \
  --num_edits=100 \
  --skip_generation_tests \
  --generation_test_interval=-1
```

The MQuAKE entry point is `python -m experiments.evaluate_mquake` with `--alg_name=SNKE --ds_name=mquake`; it requires the matching `mquake.json` and `mquake_graph.json` files. The paper evaluates 2-hop and 3-hop portability with 2,000 cases and 50 edits per step. Its graph layout differs from the other datasets, so validate record alignment before a full run.

Do not enable `--conserve_memory` for SNKE with the current entry points: they pass an argument that `apply_SNKE_to_model` does not accept. The optional `--use_cache` stores computed key/value targets under `KV_DIR` in `globals.yml`; the directory and `.npz` files are created during execution.

## Summarize results

Replace `run_000` with the directory created by your run:

```bash
python -m experiments.summarize --dir_name=SNKE --runs=run_000
python -m experiments.summarize_sequential --dir_name=SNKE --runs=run_000
python -m experiments.summarize_mquake --dir_name=SNKE --runs=run_000
```

Use the summarizer corresponding to the experiment entry point; the sequential summarizer looks under `results/SNKE_sequential/` when given `--dir_name=SNKE`.

## Results reported in the paper

The manuscript evaluates GPT-2-XL, GPT-J-6B, and LLaMA 3 on MultiCounterFact, ZsRE, and MQuAKE. It reports strong edit efficacy and generality for SNKE and improved multi-hop portability. For example, on the paper's 2,000-case MQuAKE evaluation, SNKE's average scores are 42.32 (GPT-2-XL), 48.87 (GPT-J-6B), and 46.63 (LLaMA 3). These are **paper results**, not measurements from a fresh run of this checkout. See [the manuscript](SNKE.pdf) for the full tables, evaluation protocol, ablations, and limitations.

## Citation and acknowledgments

If you use SNKE, please cite the accompanying manuscript:

```bibtex
@misc{liu2026snke,
  title  = {Structure-Guided Neural Mask Knowledge Editing for Large Language Models via Graph-Aware Attribution},
  author = {Ruinan Liu and Fulong Lin and Qingyu Wang and Xin Wang and Zhihua Fu},
  year   = {2026}
}
```

This repository includes and adapts code for comparison methods. SNKE builds on research and implementations including [AlphaEdit](https://github.com/jianghoucheng/AlphaEdit), [GLAME](https://github.com/Acruxos/GLAME), [KGMET](https://github.com/FredJDean/KGMET), and NMKE. Please also cite the original methods when using their implementations or reproducing comparisons.
