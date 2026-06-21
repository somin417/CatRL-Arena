"""Plot CEM-MPC evaluation CSVs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import settings as S

BAR_COLORS = ["#388bfd", "#8b949e", "#63c174", "#d29922", "#f85149"]


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
            "avg_broken_bricks": float(np.mean([float(r["broken_bricks"]) for r in items])),
            "blocks_per_100_steps": float(
                np.mean([float(r["blocks_per_100_steps"]) for r in items])
            ),
            "clear_rate": float(np.mean([int(r["clear"]) for r in items])),
            "avg_steps": float(np.mean([float(r["steps"]) for r in items])),
            "wall_clock_sec": float(np.mean([float(r["wall_clock_sec"]) for r in items])),
        }
    return summary


def _bar_plot(agents, values, title, ylabel, out_file, as_percent=False):
    fig, ax = plt.subplots(figsize=(5, 4))
    display = [v * 100 if as_percent else v for v in values]
    colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(agents))]
    x = np.arange(len(agents))
    ax.bar(x, display, width=0.45, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=15, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_file, dpi=120)
    plt.close(fig)
    print(f"Saved {out_file}")


def plot_cem_mpc(csv_path: Path) -> Path:
    rows = load_csv(csv_path)
    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = S.CEM_MPC_PLOTS_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = aggregate_by_agent(rows)
    agents = sorted(summary.keys())

    specs = [
        ("avg_broken_bricks", "avg_broken_bricks_by_agent.png", "Avg Broken Bricks", "Bricks"),
        ("blocks_per_100_steps", "blocks_per_100_steps_by_agent.png", "Blocks / 100 Steps", "Blk/100"),
        ("clear_rate", "clear_rate_by_agent.png", "Clear Rate", "Rate (%)", True),
        ("avg_steps", "steps_by_agent.png", "Average Steps", "Steps"),
        ("wall_clock_sec", "wall_clock_by_agent.png", "Wall Clock Time", "Seconds"),
    ]

    for spec in specs:
        key, fname, title, ylabel = spec[:4]
        as_pct = spec[4] if len(spec) > 4 else False
        if key not in summary[agents[0]]:
            print(f"WARNING: Missing field '{key}'; skipping {fname}")
            continue
        values = [summary[a][key] for a in agents]
        _bar_plot(agents, values, title, ylabel, out_dir / fname, as_percent=as_pct)

    follow_rows = {int(r["env_seed"]): float(r["broken_bricks"])
                   for r in rows if r["agent"] == "FollowBall"}
    cem_rows = {int(r["env_seed"]): float(r["broken_bricks"])
                for r in rows if r["agent"] == "CEM-MPC"}
    shared = sorted(set(follow_rows) & set(cem_rows))
    if shared:
        fig, ax = plt.subplots(figsize=(5, 5))
        fx = [follow_rows[s] for s in shared]
        cx = [cem_rows[s] for s in shared]
        ax.scatter(fx, cx, alpha=0.7, color=BAR_COLORS[0])
        lim = max(max(fx + cx, default=1), 1)
        ax.plot([0, lim], [0, lim], "--", color="#8b949e", linewidth=1)
        ax.set_xlabel("FollowBall broken bricks")
        ax.set_ylabel("CEM-MPC broken bricks")
        ax.set_title("CEM-MPC vs FollowBall (per seed)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        scatter_path = out_dir / "cem_mpc_vs_followball_scatter.png"
        fig.savefig(scatter_path, dpi=120)
        plt.close(fig)
        print(f"Saved {scatter_path}")
    else:
        print("WARNING: No shared seeds for scatter plot; skipping.")

    leaderboard_path = out_dir / "leaderboard.md"
    ranked = sorted(
        summary.items(),
        key=lambda kv: (
            -kv[1]["avg_broken_bricks"],
            -kv[1]["blocks_per_100_steps"],
            -kv[1]["clear_rate"],
            kv[1]["avg_steps"],
        ),
    )
    with leaderboard_path.open("w") as f:
        f.write("# CEM-MPC Evaluation Leaderboard\n\n")
        f.write(f"Source: `{csv_path}`\n\n")
        f.write("| Rank | Agent | Bricks | Blk/100 | Clear% | Steps |\n")
        f.write("|------|-------|--------|---------|--------|-------|\n")
        for i, (agent, stats) in enumerate(ranked, 1):
            f.write(
                f"| {i} | {agent} | {stats['avg_broken_bricks']:.2f} | "
                f"{stats['blocks_per_100_steps']:.2f} | "
                f"{stats['clear_rate']*100:.1f}% | {stats['avg_steps']:.1f} |\n"
            )
    print(f"Saved {leaderboard_path}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CEM-MPC evaluation results.")
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()
    plot_cem_mpc(Path(args.csv))


if __name__ == "__main__":
    main()
