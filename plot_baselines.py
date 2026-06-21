"""Plot baseline comparison CSVs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import settings as S

BAR_COLORS = ["#388bfd", "#8b949e", "#63c174", "#d29922"]


def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def aggregate_by_agent(rows: list[dict]) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["agent"]].append(row)

    summary: dict[str, dict] = {}
    for agent, items in buckets.items():
        summary[agent] = {
            "avg_return": float(np.mean([float(r["return"]) for r in items])),
            "clear_rate": float(np.mean([int(r["clear"]) for r in items])),
            "avg_steps": float(np.mean([int(r["steps"]) for r in items])),
            "blocks_per_100_steps": float(
                np.mean([float(r["blocks_per_100_steps"]) for r in items])
            ),
        }
    return summary


def plot_baselines(csv_path: Path) -> Path:
    rows = load_csv(csv_path)
    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    summary = aggregate_by_agent(rows)
    agents = sorted(summary.keys())
    colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(agents))]

    out_dir = S.BASELINE_PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("avg_return", "Average Return", False),
        ("clear_rate", "Clear Rate (%)", True),
        ("avg_steps", "Average Steps", False),
        ("blocks_per_100_steps", "Blocks / 100 Steps", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7, 9))
    fig.suptitle(f"Baseline Comparison — {csv_path.stem}", fontsize=13, y=0.98)

    for ax, (key, title, as_percent) in zip(axes.flat, specs):
        values = [summary[a][key] for a in agents]
        display_vals = [v * 100 if as_percent else v for v in values]
        x = np.arange(len(agents))
        ax.bar(x, display_vals, width=0.45, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=15, ha="right")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(title if key != "clear_rate" else "Rate (%)")
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_file = S.BASELINE_COMPARISON_PNG
    fig.savefig(out_file, dpi=120)
    plt.close(fig)
    print(f"Saved {out_file}")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot baseline comparison CSV.")
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()
    plot_baselines(Path(args.csv))


if __name__ == "__main__":
    main()
