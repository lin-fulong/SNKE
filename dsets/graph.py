import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset

from util.globals import *


class KGDataset(Dataset):
    def __init__(
        self, data_dir: str, ds_name: str, size: typing.Optional[int] = None, *args, **kwargs
    ):
        data_dir = Path(data_dir)
        if ds_name == "cf":
            data_loc = data_dir / "counterfact_graph.json"
        elif ds_name == "cf-one-hop":
            data_loc = data_dir / "cf_plus_graph.json"
        elif ds_name == "mquake":
            data_loc = data_dir / "mquake_graph.json"
        elif ds_name == "mcf":
            data_loc = data_dir / "multi_counterfact_graph.json"
        elif ds_name == "zsre":
            data_loc = data_dir / "zsre_graph.json"
        else:
            raise NotImplementedError(
                "can not found corresponding KG in /data/!")

        print(f"[graph:{ds_name}] Resolving graph dataset file: {data_loc}")
        print(f"[graph:{ds_name}] Loading JSON from {data_loc}")
        with open(data_loc, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        print(f"[graph:{ds_name}] Loaded {len(self.data)} raw graph records")
        if size is not None:
            self.data = self.data[:size]
            print(f"[graph:{ds_name}] Truncated graph dataset to first {len(self.data)} records")

        first_case = self.data[0].get("case_id", "NA") if self.data else "NA"
        print(f"[graph:{ds_name}] Ready with {len(self)} elements; first_case_id={first_case}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
