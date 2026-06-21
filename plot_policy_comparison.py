"""Bar-chart comparison of Random, CNN-DQN, and FollowBall policies."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import settings as S

AGENT_ORDER = ("Random", "CNN-DQN", "FollowBall")
AGENT_LABELS = {
    "Random": "Random",
    "DQN": "CNN-DQN",
    "CNN-DQN": "CNN-DQN",
    "FollowBall": "FollowBall",
}
AGENT_COLORS = {
    "Random": "#8b949e",
    "CNN-DQN": "#388bfd",
    "FollowBall": "#63c174",
}

METRICS = [
    ("avg_return", "Average Return", False),
    ("avg_broken_bricks", "Average Broken Bricks", False),
    ("avg_blocks_per_100_steps", "Blocks / 100 Steps", False),
    ("avg_steps", "Average Steps", False),
]


def load_summary(path: Path) -> dict[str, dict[str, float]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    summary: dict[str, dict[str, float]] = {}
    for row in rows:
        label = AGENT_LABELS.get(row["agent"], row["agent"])
        summary[label] = {
            key: float(row[key])
            for key, _, _ in METRICS
            if key in row
        }
    return summary


def plot_policy_comparison(
    csv_path: Path,
    out_path: Path | None = None,
    metrics: list[tuple[str, str, bool]] | None = None,
) -> Path:
    summary = load_summary(csv_path)
    missing = [agent for agent in AGENT_ORDER if agent not in summary]
    if missing:
        raise ValueError(f"Missing agents in {csv_path}: {', '.join(missing)}")

    metrics = metrics or METRICS[:3]
    agents = list(AGENT_ORDER)
    colors = [AGENT_COLORS[a] for a in agents]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]

    fig.suptitle("Policy Comparison (50 episodes, shared seeds)", fontsize=13, y=1.02)

    x = np.arange(len(agents))
    for ax, (key, title, as_percent) in zip(axes, metrics):
        values = [summary[a][key] for a in agents]
        display_vals = [v * 100 if as_percent else v for v in values]
        bars = ax.bar(x, display_vals, width=0.55, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=12, ha="right")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(title)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, value in zip(bars, display_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    out_path = out_path or (S.COMPARISON_DIR / "plots" / "policy_comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Random vs CNN-DQN vs FollowBall.")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(S.COMPARISON_SUMMARY_CSV),
        help="Summary CSV from compare_dqn_vs_baselines.py",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(S.COMPARISON_DIR / "plots" / "policy_comparison.png"),
    )
    args = parser.parse_args()
    plot_policy_comparison(Path(args.csv), Path(args.out))


if __name__ == "__main__":
    main()
