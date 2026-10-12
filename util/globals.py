from pathlib import Path

import yaml

with open("globals.yml", "r") as stream:
    data = yaml.safe_load(stream)

(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR, KV_DIR) = (
    Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
        data["KV_DIR"],
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]

MODEL_DIR = Path(data["MODEL_DIR"]) if data.get("MODEL_DIR") else None
WIKIPEDIA_DATA_DIR = (
    Path(data["WIKIPEDIA_DATA_DIR"]) if data.get("WIKIPEDIA_DATA_DIR") else None
)


def resolve_model_path(model_name: str) -> str:
    """
    Resolve a Hugging Face model identifier to a local directory when one is configured.
    Falls back to the original identifier if no local match exists.
    """
    direct_path = Path(model_name)
    if direct_path.exists():
        return str(direct_path)

    if MODEL_DIR is None:
        return model_name

    candidates = [
        MODEL_DIR / model_name,
        MODEL_DIR / model_name.replace("/", "_"),
        MODEL_DIR / model_name.replace("/", "--"),
        MODEL_DIR / model_name.split("/")[-1],
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return model_name
