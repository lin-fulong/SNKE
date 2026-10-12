import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from util.globals import *

REMOTE_URL = f"{REMOTE_ROOT_URL}/data/dsets/zsre_mend_eval.json"


class MENDQADataset:
    """
    Dataset of factual knowledge based on zsRE.
    Specifically selected from the QA validation slice from Mitchell et al.
    Project page: http://nlp.cs.washington.edu/zeroshot/
    """

    def __init__(self, data_dir: str, tok: AutoTokenizer, size=None, *args, **kwargs):
        data_dir = Path(data_dir)
        zsre_loc = data_dir / "filtered_zsre.json"
        fallback_loc = data_dir / "zsre_mend_eval.json"
        print(f"[zsre] Checking primary dataset file: {zsre_loc}")
        if not zsre_loc.exists() and fallback_loc.exists():
            print(f"[zsre] Primary file missing. Falling back to local file: {fallback_loc}")
            zsre_loc = fallback_loc
        if not zsre_loc.exists():
            print(f"[zsre] File missing. Downloading from {REMOTE_URL}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(REMOTE_URL, zsre_loc)

        print(f"[zsre] Loading JSON from {zsre_loc}")
        with open(zsre_loc, "r", encoding="utf-8") as f:
            raw = json.load(f)
        print(f"[zsre] Loaded {len(raw)} raw records")

        data = []
        for i, record in enumerate(raw):
            assert (
                "nq question: " in record["loc"]
            ), f"Neighborhood prompt missing `nq question:`. Check for errors?"
            ans_toks = tok(" " + record["loc_ans"])["input_ids"]
            data.append(
                {
                    "case_id": i,
                    "requested_rewrite": {
                        "prompt": record["src"].replace(record["subject"], "{}"),
                        "subject": record["subject"],
                        "target_new": {"str": record["answers"][0]},
                        "target_true": {"str": "<|endoftext|>"},
                    },
                    "paraphrase_prompts": [record["rephrase"]],
                    "neighborhood_prompts": [
                        {
                            "prompt": record["loc"] + "?" + tok.decode(ans_toks[:i]),
                            "target": tok.decode(ans_toks[i]),
                        }
                        for i in range(len(ans_toks))
                    ],
                    "attribute_prompts": [],
                    "generation_prompts": [],
                }
            )

        self._data = data[:size]
        if size is not None:
            print(f"[zsre] Truncated dataset to first {len(self._data)} records")
        first_case = self._data[0]["case_id"] if self._data else "NA"
        print(f"[zsre] Ready with {len(self)} elements; first_case_id={first_case}")

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
