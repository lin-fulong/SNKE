import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset

from util.globals import *

REMOTE_ROOT = f"{REMOTE_ROOT_URL}/data/dsets"


class CounterFactDataset(Dataset):
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
            "counterfact.json" if not multi else "multi_counterfact.json"
        )
        dataset_name = "multi_counterfact" if multi else "counterfact"
        print(f"[{dataset_name}] Resolving dataset file: {cf_loc}")
        if not cf_loc.exists():
            remote_url = f"{REMOTE_ROOT}/{'multi_' if multi else ''}counterfact.json"
            print(f"[{dataset_name}] File missing. Downloading from {remote_url}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(remote_url, cf_loc)

        print(f"[{dataset_name}] Loading JSON from {cf_loc}")
        with open(cf_loc, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        print(f"[{dataset_name}] Loaded {len(self.data)} raw records")
        if size is not None:
            self.data = self.data[:size]
            print(f"[{dataset_name}] Truncated dataset to first {len(self.data)} records")

        first_case = self.data[0].get("case_id", "NA") if self.data else "NA"
        print(f"[{dataset_name}] Ready with {len(self)} elements; first_case_id={first_case}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class MultiCounterFactDataset(CounterFactDataset):
    def __init__(
        self, data_dir: str, size: typing.Optional[int] = None, *args, **kwargs
    ):
        super().__init__(data_dir, *args, multi=True, size=size, **kwargs)
