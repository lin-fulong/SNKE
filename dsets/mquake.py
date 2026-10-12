import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset

from util.globals import *

REMOTE_ROOT = f"{REMOTE_ROOT_URL}/data/dsets"


class MQuAKEDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        multi: bool = False,
        size: typing.Optional[int] = None,
        *args,
        **kwargs,
    ):
        data_dir = Path(data_dir)
        cf_loc = data_dir / (
            "mquake.json"
        )
        print(f"[mquake] Resolving dataset file: {cf_loc}")

        print(f"[mquake] Loading JSON from {cf_loc}")
        with open(cf_loc, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        print(f"[mquake] Loaded {len(self.data)} raw records")
        if size is not None:
            self.data = self.data[:size]
            print(f"[mquake] Truncated dataset to first {len(self.data)} records")

        first_case = self.data[0].get("case_id", "NA") if self.data else "NA"
        print(f"[mquake] Ready with {len(self)} elements; first_case_id={first_case}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
