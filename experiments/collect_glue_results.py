import argparse
import csv
import json
import re
from pathlib import Path


DEFAULT_GLUE_ROOT = Path("results") / "补充实验" / "补充实验一：GLUE 、 Downstream Eval" / "GLUE"
DEFAULT_OUTPUT_NAME = "glue_results_summary.csv"
DEFAULT_GLUE_STEP = 500

GLUE_TASKS = ["sst", "mmmlu", "mrpc", "cola", "rte", "nli"]
METRICS = [
    "correct",
    "incorrect",
    "invalid",
    "correct_new",
    "incorrect_new",
    "total",
    "f1",
    "f1_new",
    "mcc",
    "time",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="gbk", errors="replace")
    return json.loads(text)


def glue_sort_key(path: Path):
    if path.name == "base_glue.json":
        return (-1, path.name)

    match = re.search(r"case_(\d+)_glue\.json$", path.name)
    if match:
        return (int(match.group(1)), path.name)

    return (10**12, path.name)


def find_run_dirs(glue_root: Path):
    return sorted(
        (
            path
            for path in glue_root.rglob("run_*")
            if path.is_dir() and (path / "glue_eval").is_dir()
        ),
        key=lambda path: str(path.relative_to(glue_root)),
    )


def get_method_name(glue_root: Path, run_dir: Path) -> str:
    rel_parts = run_dir.relative_to(glue_root).parts
    return rel_parts[0] if len(rel_parts) > 1 else ""


def get_case_start(result_file: Path) -> str:
    match = re.search(r"case_(\d+)_glue\.json$", result_file.name)
    return match.group(1) if match else ""


def get_case_count(data: dict) -> int:
    case_id = data.get("case_id", "")
    if isinstance(case_id, list):
        return len(case_id)
    return 1 if case_id != "" else 0


def get_glue_step(data: dict, result_type: str, case_index: int, step_size: int) -> int:
    if result_type == "base":
        return 0

    edit_num = data.get("edit_num", "")
    if isinstance(edit_num, int) and edit_num >= 0:
        return edit_num

    return case_index * step_size


def flatten_task_metrics(data: dict) -> dict:
    row = {}
    for task in GLUE_TASKS:
        task_metrics = data.get(task, {})
        for metric in METRICS:
            row[f"{task}_{metric}"] = task_metrics.get(metric, "")
    return row


def collect_rows(glue_root: Path, step_size: int):
    rows = []
    warnings = []

    for run_dir in find_run_dirs(glue_root):
        glue_eval_dir = run_dir / "glue_eval"
        result_files = sorted(glue_eval_dir.glob("*_glue.json"), key=glue_sort_key)

        if not result_files:
            warnings.append(f"No *_glue.json files found in {glue_eval_dir}")
            continue

        case_index = 0
        for result_file in result_files:
            try:
                data = read_json(result_file)
            except json.JSONDecodeError as exc:
                warnings.append(f"Could not parse {result_file}: {exc}")
                continue

            result_type = "base" if result_file.name == "base_glue.json" else "case"
            if result_type == "case":
                case_index += 1

            row = {
                "method": get_method_name(glue_root, run_dir),
                "run_name": run_dir.name,
                "run_rel_path": str(run_dir.relative_to(glue_root)),
                "result_file": result_file.name,
                "result_type": result_type,
                "case_start": get_case_start(result_file),
                "glue_step": get_glue_step(data, result_type, case_index, step_size),
                "step_size": step_size,
                "edit_num": data.get("edit_num", ""),
                "case_count": get_case_count(data),
            }
            row.update(flatten_task_metrics(data))
            rows.append(row)

    return rows, warnings


def fieldnames() -> list:
    fields = [
        "method",
        "run_name",
        "run_rel_path",
        "result_file",
        "result_type",
        "case_start",
        "glue_step",
        "step_size",
        "edit_num",
        "case_count",
    ]

    for task in GLUE_TASKS:
        for metric in METRICS:
            fields.append(f"{task}_{metric}")

    return fields


def main():
    parser = argparse.ArgumentParser(
        description="Collect the six GLUE task summary metrics from every *_glue.json file."
    )
    parser.add_argument(
        "--glue_root",
        type=Path,
        default=repo_root() / DEFAULT_GLUE_ROOT,
        help="Root directory that contains method folders and run_* result folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path. Defaults to <glue_root parent>/glue_results_summary.csv.",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=DEFAULT_GLUE_STEP,
        help="GLUE evaluation interval. Used only when edit_num is absent from a case file.",
    )
    args = parser.parse_args()

    glue_root = args.glue_root
    if not glue_root.is_absolute():
        glue_root = repo_root() / glue_root
    glue_root = glue_root.resolve()

    output = args.output or glue_root.parent / DEFAULT_OUTPUT_NAME
    if not output.is_absolute():
        output = repo_root() / output
    output = output.resolve()

    if not glue_root.exists():
        raise FileNotFoundError(f"GLUE root does not exist: {glue_root}")

    rows, warnings = collect_rows(glue_root, args.step_size)
    if not rows:
        raise RuntimeError(f"No GLUE result rows were collected from {glue_root}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Collected {len(rows)} GLUE result rows.")
    print(f"GLUE evaluation step size: {args.step_size}")
    print(f"Wrote CSV: {output}")
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
