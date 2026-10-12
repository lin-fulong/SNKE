import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_CSV = (
    Path("results")
    / "补充实验"
    / "补充实验一：GLUE 、 Downstream Eval"
    / "glue_results_summary.csv"
)
DEFAULT_OUTPUT = (
    Path("results")
    / "补充实验"
    / "补充实验一：GLUE 、 Downstream Eval"
    / "glue_f1_trends.pdf"
)

TASKS = [
    ("sst", "SST"),
    ("mmmlu", "MMLU"),
    ("mrpc", "MRPC"),
    ("cola", "CoLA"),
    ("rte", "RTE"),
    ("nli", "NLI"),
]
TASKS_WITHOUT_MMLU = [task for task in TASKS if task[0] != "mmmlu"]

SERIES_ORDER = [
    "SNKE",
    "KGMET",
    "MEMIT",
    "NMKE-MCF",
    "NMKE-ZsRE",
    "NMKE-MQuAKE",
    "NSE",
    "ROME",
]
SERIES_STYLES = {
    "SNKE": {"color": "#ff1f1f", "marker": "o", "linewidth": 2.2, "zorder": 5},
    "KGMET": {"color": "#9467bd", "marker": "o", "linewidth": 1.15, "alpha": 0.82},
    "MEMIT": {"color": "#8c564b", "marker": "o", "linewidth": 1.15, "alpha": 0.82},
    "NMKE-MCF": {"color": "#2ca02c", "marker": "o", "linewidth": 1.15, "alpha": 0.82},
    "NMKE-ZsRE": {"color": "#e377c2", "marker": "o", "linewidth": 1.15, "alpha": 0.82},
    "NMKE-MQuAKE": {"color": "#ff7f0e", "marker": "o", "linewidth": 1.15, "alpha": 0.82},
    "NSE": {"color": "#17becf", "marker": "o", "linewidth": 1.15, "alpha": 0.82},
    "ROME": {"color": "#7f7f7f", "marker": "o", "linestyle": ":", "linewidth": 1.15, "alpha": 0.82},
}
NMKE_RUN_LABELS = {
    "run_000": "NMKE-MCF",
    "run_001": "NMKE-ZsRE",
    "run_002": "NMKE-MQuAKE",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else repo_root() / path


def parse_float(value: str):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value: str):
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def read_rows(csv_path: Path) -> list:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def get_series_label(row: dict) -> str:
    if row["method"] == "NMKE":
        return NMKE_RUN_LABELS.get(row["run_name"], f"NMKE-{row['run_name']}")
    return row["method"]


def aggregate_by_series_and_step(rows: list, metric: str, scale: float, tasks: list):
    grouped = defaultdict(list)

    for row in rows:
        label = get_series_label(row)
        step = parse_int(row["glue_step"])
        if step is None:
            continue

        for task, _ in tasks:
            value = parse_float(row.get(f"{task}_{metric}", ""))
            if value is not None:
                grouped[(label, step, task)].append(value * scale)

    aggregated = defaultdict(dict)
    for (label, step, task), values in grouped.items():
        aggregated[(label, task)][step] = sum(values) / len(values)

    return aggregated


def sorted_series(rows: list):
    labels = sorted({get_series_label(row) for row in rows})
    ordered = [label for label in SERIES_ORDER if label in labels]
    ordered.extend(label for label in labels if label not in ordered)
    return ordered


def draw_plot(rows: list, metric: str, output: Path, title: str, scale: float, tasks: list):
    data = aggregate_by_series_and_step(rows, metric, scale, tasks)
    labels = sorted_series(rows)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    if len(tasks) == 5:
        fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.6), sharex=True)
        axes = axes.flatten()
        axes[-1].axis("off")
        plot_axes = axes[:-1]
    else:
        fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.6), sharex=True)
        axes = axes.flatten()
        plot_axes = axes

    for ax, (task, task_title) in zip(plot_axes, tasks):
        for label in labels:
            series = data.get((label, task), {})
            if not series:
                continue

            steps = sorted(series)
            values = [series[step] for step in steps]
            style = SERIES_STYLES.get(label, {})
            ax.plot(
                steps,
                values,
                label=label,
                markersize=3.0,
                markerfacecolor="white",
                markeredgewidth=0.8,
                **style,
            )

        ax.set_title(task_title, fontweight="bold")
        ax.set_xlabel("Number of Edits", fontweight="bold")
        ax.set_ylabel(f"{metric.upper()} Score", fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.55)
        ax.set_xlim(left=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(sorted({parse_int(row["glue_step"]) for row in rows if parse_int(row["glue_step"]) is not None}))

    handles, labels = plot_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 8),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        handlelength=1.7,
        columnspacing=1.1,
    )
    if title:
        fig.suptitle(title, fontsize=13, y=0.985, fontweight="bold")
        rect = (0, 0.08, 1, 0.94)
    else:
        rect = (0, 0.08, 1, 1)
    fig.tight_layout(rect=rect, w_pad=2.0, h_pad=2.2)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot 2x3 GLUE task trend lines from glue_results_summary.csv."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Input CSV generated by experiments/collect_glue_results.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PDF path.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        choices=["f1", "f1_new", "mcc", "time"],
        help="Metric to plot for each GLUE task.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Figure title.",
    )
    parser.add_argument(
        "--no_percent",
        action="store_true",
        help="Plot raw metric values instead of multiplying by 100.",
    )
    parser.add_argument(
        "--without_mmlu",
        action="store_true",
        help="Plot five GLUE tasks and exclude MMLU.",
    )
    args = parser.parse_args()

    csv_path = resolve_path(args.csv).resolve()
    output = resolve_path(args.output).resolve()

    rows = read_rows(csv_path)
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")

    scale = 1.0 if args.no_percent or args.metric == "time" else 100.0
    tasks = TASKS_WITHOUT_MMLU if args.without_mmlu else TASKS
    draw_plot(rows, args.metric, output, args.title, scale, tasks)
    print(f"Read {len(rows)} rows from {csv_path}")
    print(f"Wrote PDF: {output}")


if __name__ == "__main__":
    main()
