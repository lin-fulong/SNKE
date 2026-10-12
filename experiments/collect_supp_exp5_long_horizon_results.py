import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, pstdev


DEFAULT_ROOT = (
    Path("results")
    / "补充实验"
    / "补充实验五（长程实验）"
)
DEFAULT_EDIT_COUNTS = (1000, 2000, 3000, 4000, 5000)

METHOD_LABELS = {
    "KGMET_sequential": "KGMET",
    "SNKE_sequential": "SNKE",
    "NMKE_sequential": "NMKE",
}

METRIC_KEYS = {
    "Eff.": "rewrite_prompts_correct",
    "Gen.": "paraphrase_prompts_correct",
    "Loc.": "neighborhood_prompts_correct",
}


def parse_edit_counts(value):
    if value.lower() == "all":
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def case_sort_key(path):
    match = re.search(r"case_(\d+)", path.name)
    return int(match.group(1)) if match else path.name


def score_bool_list(values):
    if values is None:
        return None
    if len(values) == 0:
        return None
    return sum(1.0 if item else 0.0 for item in values) / len(values)


def collect_test_dir(test_dir):
    metric_values = {metric: [] for metric in METRIC_KEYS}
    case_files = sorted(test_dir.glob("*case_*.json"), key=case_sort_key)

    for case_file in case_files:
        try:
            with case_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"[warn] Skip invalid JSON: {case_file}")
            continue

        post = data.get("post", {})
        for metric, json_key in METRIC_KEYS.items():
            score = score_bool_list(post.get(json_key))
            if score is not None:
                metric_values[metric].append(score)

    summary = {}
    for metric, values in metric_values.items():
        if values:
            summary[metric] = {
                "value": mean(values),
                "std": pstdev(values) if len(values) > 1 else 0.0,
                "num_cases": len(values),
            }
        else:
            summary[metric] = {
                "value": None,
                "std": None,
                "num_cases": 0,
            }

    return summary


def edit_count_from_summary_row(row):
    run_dir = row.get("run_dir", "")
    match = re.search(r"(\d+)_test", run_dir)
    if match:
        return int(match.group(1))
    return int(row["num_cases"])


def collect_existing_summary(summary_csv, edit_counts):
    if not summary_csv.exists():
        return {}

    summary_by_edit_count = {}
    with summary_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {
            "post_rewrite_acc",
            "post_paraphrase_acc",
            "post_neighborhood_acc",
            "num_cases",
        }
        if not required.issubset(reader.fieldnames or []):
            return {}

        for row in reader:
            edit_count = edit_count_from_summary_row(row)
            if edit_counts is not None and edit_count not in edit_counts:
                continue

            summary_by_edit_count[edit_count] = {
                "Eff.": {
                    "value": float(row["post_rewrite_acc"]) / 100.0,
                    "std": None,
                    "num_cases": int(float(row["num_cases"])),
                },
                "Gen.": {
                    "value": float(row["post_paraphrase_acc"]) / 100.0,
                    "std": None,
                    "num_cases": int(float(row["num_cases"])),
                },
                "Loc.": {
                    "value": float(row["post_neighborhood_acc"]) / 100.0,
                    "std": None,
                    "num_cases": int(float(row["num_cases"])),
                },
            }

    return summary_by_edit_count


def iter_test_dirs(run_dir, edit_counts):
    for test_dir in run_dir.glob("*_test"):
        if not test_dir.is_dir():
            continue
        try:
            edit_count = int(test_dir.name.split("_", 1)[0])
        except ValueError:
            continue
        if edit_counts is not None and edit_count not in edit_counts:
            continue
        yield edit_count, test_dir


def collect(root, edit_counts):
    long_rows = []
    wide_rows = []

    method_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    for method_dir in method_dirs:
        if method_dir.name == "summary_tables":
            continue

        method = METHOD_LABELS.get(
            method_dir.name,
            method_dir.name.removesuffix("_sequential"),
        )

        run_dirs = [path for path in sorted(method_dir.glob("run_*")) if path.is_dir()]
        if not run_dirs and any(method_dir.glob("*_test")):
            run_dirs = [method_dir]

        for run_dir in run_dirs:
            run_label = run_dir.name if run_dir != method_dir else "direct"
            summary_by_edit_count = collect_existing_summary(
                run_dir / "summary.csv",
                edit_counts,
            )

            if summary_by_edit_count:
                items = [
                    (
                        edit_count,
                        run_dir / f"{edit_count}_test",
                        summary,
                        "summary.csv",
                    )
                    for edit_count, summary in sorted(summary_by_edit_count.items())
                ]
            else:
                items = [
                    (
                        edit_count,
                        test_dir,
                        collect_test_dir(test_dir),
                        "json",
                    )
                    for edit_count, test_dir in sorted(
                        iter_test_dirs(run_dir, edit_counts),
                        key=lambda item: item[0],
                    )
                ]

            for edit_count, test_dir, summary, source_type in items:
                print(f"[collect] {method_dir.name}/{run_label}/{test_dir.name} ({source_type})")

                wide_row = {
                    "method": method,
                    "source_method_dir": method_dir.name,
                    "run": run_label,
                    "edit_count": edit_count,
                }

                for metric in METRIC_KEYS:
                    stats = summary[metric]
                    value = stats["value"]
                    std = stats["std"]
                    num_cases = stats["num_cases"]

                    long_rows.append(
                        {
                            "method": method,
                            "source_method_dir": method_dir.name,
                            "run": run_label,
                            "edit_count": edit_count,
                            "metric": metric,
                            "value": value,
                            "std": std,
                            "num_cases": num_cases,
                            "test_dir": str(test_dir),
                        }
                    )

                    wide_row[metric] = value
                    wide_row[f"{metric}_std"] = std
                    wide_row[f"{metric}_num_cases"] = num_cases

                wide_rows.append(wide_row)

    long_rows.sort(key=lambda row: (row["metric"], row["method"], row["run"], row["edit_count"]))
    wide_rows.sort(key=lambda row: (row["method"], row["run"], row["edit_count"]))
    return long_rows, wide_rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_excel(path, long_rows, wide_rows):
    try:
        import pandas as pd
    except ImportError:
        print("[warn] pandas is not installed; skip Excel export.")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(long_rows).to_excel(writer, sheet_name="long", index=False)
        pd.DataFrame(wide_rows).to_excel(writer, sheet_name="wide", index=False)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Collect supplementary experiment 5 long-horizon sequential results."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory of supplementary experiment 5 results.",
    )
    parser.add_argument(
        "--edit-counts",
        default=",".join(str(item) for item in DEFAULT_EDIT_COUNTS),
        help="Comma-separated edit counts to collect, or 'all'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <root>/summary_tables.",
    )
    parser.add_argument(
        "--no-excel",
        action="store_true",
        help="Only write CSV files.",
    )
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise FileNotFoundError(f"Result root does not exist: {root}")

    edit_counts = parse_edit_counts(args.edit_counts)
    output_dir = args.output_dir or root / "summary_tables"

    long_rows, wide_rows = collect(root, edit_counts)
    if not long_rows:
        raise RuntimeError(f"No result rows collected from {root}")

    long_csv = output_dir / "supp_exp5_long_horizon_long.csv"
    wide_csv = output_dir / "supp_exp5_long_horizon_wide.csv"
    xlsx = output_dir / "supp_exp5_long_horizon_results.xlsx"

    write_csv(
        long_csv,
        long_rows,
        [
            "method",
            "source_method_dir",
            "run",
            "edit_count",
            "metric",
            "value",
            "std",
            "num_cases",
            "test_dir",
        ],
    )
    write_csv(
        wide_csv,
        wide_rows,
        [
            "method",
            "source_method_dir",
            "run",
            "edit_count",
            "Eff.",
            "Eff._std",
            "Eff._num_cases",
            "Gen.",
            "Gen._std",
            "Gen._num_cases",
            "Loc.",
            "Loc._std",
            "Loc._num_cases",
        ],
    )

    print(f"[done] Wrote {long_csv}")
    print(f"[done] Wrote {wide_csv}")

    if not args.no_excel:
        if write_excel(xlsx, long_rows, wide_rows):
            print(f"[done] Wrote {xlsx}")


if __name__ == "__main__":
    main()
