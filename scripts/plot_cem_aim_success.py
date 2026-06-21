"""Success-oriented plots for CEM-Aim runs (teacher_search dashboard style).

Default framing highlights seeds where FollowBall struggles (no clear) and
CEM-Aim wins or improves — not the full 20-seed aggregate.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import settings as S

GREEN = "#63c174"
BLUE = "#388bfd"
GREY = "#484f58"
RED = "#f85149"
LIGHT_BLUE = "#79c0ff"
PURPLE = "#a371f7"
AMBER = "#d29922"


@dataclass
class SeedRow:
    seed: int
    follow_bricks: float
    cem_bricks: float
    delta_bricks: float
    follow_clear: int
    cem_clear: int
    follow_steps: float
    cem_steps: float
    delta_steps: float
    verdict: str

    @property
    def follow_ret(self) -> float:
        return _est_return(self.follow_bricks, self.follow_steps, self.follow_clear)

    @property
    def cem_ret(self) -> float:
        return _est_return(self.cem_bricks, self.cem_steps, self.cem_clear)

    @property
    def delta_ret(self) -> float:
        return self.cem_ret - self.follow_ret

    @property
    def fb_no_clear(self) -> bool:
        return self.follow_clear == 0

    @property
    def cem_clears_fb_failed(self) -> bool:
        return self.follow_clear == 0 and self.cem_clear == 1

    @property
    def cem_improves_on_hard(self) -> bool:
        """FB no-clear and CEM wins or gains bricks."""
        return self.fb_no_clear and (self.verdict == "WIN" or self.delta_bricks > 0)


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _pick_per_seed_csv(run_dir: Path) -> Path:
    for name in ("per_seed_comparison.csv", "cem_aim_per_seed_eval_backup.csv"):
        p = run_dir / name
        if p.exists() and len(load_csv(p)) >= 10:
            return p
    for p in sorted(run_dir.glob("cem_aim_per_seed*.csv")):
        if len(load_csv(p)) >= 10:
            return p
    raise FileNotFoundError(f"No per-seed eval CSV with >=10 rows in {run_dir}")


def _est_return(bricks: float, steps: float, clear: int = 0) -> float:
    ret = bricks * S.REWARD_BRICK + steps * S.REWARD_STEP
    if clear:
        ret += S.REWARD_CLEAR
    return ret


def _load_metrics(run_dir: Path) -> dict:
    for name in ("cem_aim_val_best.metrics.json", "cem_aim_train_best.metrics.json"):
        p = run_dir / name
        if p.exists():
            return json.loads(p.read_text())
    return {}


def _parse_rows(per_seed: list[dict]) -> list[SeedRow]:
    rows = [
        SeedRow(
            seed=int(r["seed"]),
            follow_bricks=float(r["follow_bricks"]),
            cem_bricks=float(r["cem_bricks"]),
            delta_bricks=float(r["delta_bricks"]),
            follow_clear=int(r["follow_clear"]),
            cem_clear=int(r["cem_clear"]),
            follow_steps=float(r["follow_steps"]),
            cem_steps=float(r["cem_steps"]),
            delta_steps=float(r["delta_steps"]),
            verdict=r["verdict"],
        )
        for r in per_seed
    ]
    rows.sort(key=lambda r: r.seed)
    return rows


def _focus_rows(all_rows: list[SeedRow], focus: str) -> list[SeedRow]:
    if focus == "all":
        return list(all_rows)
    if focus == "wins":
        return [r for r in all_rows if r.verdict == "WIN"]
    if focus == "fb_no_clear":
        return [r for r in all_rows if r.fb_no_clear]
    if focus == "highlight":
        # FB no-clear OR CEM WIN (deduped, best stories first)
        picked: dict[int, SeedRow] = {}
        for r in all_rows:
            if r.fb_no_clear or r.verdict == "WIN":
                picked[r.seed] = r
        rows = list(picked.values())
        rows.sort(key=lambda r: (-int(r.verdict == "WIN"), -r.delta_bricks, r.seed))
        return rows
    raise ValueError(f"Unknown focus: {focus}")


def _count(rows: list[SeedRow]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    wins = sum(r.verdict == "WIN" for r in rows)
    ties = sum(r.verdict == "TIE" for r in rows)
    losses = sum(r.verdict == "LOSS" for r in rows)
    clears_fb_failed = sum(r.cem_clears_fb_failed for r in rows)
    brick_gain = sum(r.delta_bricks > 0 for r in rows)
    fb_no_clear = sum(r.fb_no_clear for r in rows)
    return {
        "n": n,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / n,
        "clears_fb_failed": clears_fb_failed,
        "brick_gain": brick_gain,
        "fb_no_clear": fb_no_clear,
        "mean_delta_bricks": float(np.mean([r.delta_bricks for r in rows])),
        "mean_cem_bricks": float(np.mean([r.cem_bricks for r in rows])),
        "mean_fb_bricks": float(np.mean([r.follow_bricks for r in rows])),
        "mean_delta_ret": float(np.mean([r.delta_ret for r in rows])),
        "ret_better": sum(r.delta_ret > 1 for r in rows),
    }


def _bar_pair(ax, rows: list[SeedRow], yfield_fb: str, yfield_cem: str, ylabel: str, title: str):
    n = len(rows)
    x = np.arange(n)
    w = 0.35
    fb_vals = [getattr(r, yfield_fb) for r in rows]
    cem_vals = [getattr(r, yfield_cem) for r in rows]
    ax.bar(x - w / 2, fb_vals, w, label="FollowBall", color=GREEN, alpha=0.85)
    ax.bar(x + w / 2, cem_vals, w, label="CEM-Aim", color=BLUE)
    labels = [str(r.seed) for r in rows]
    ax.set_xticks(x, labels, rotation=45 if n > 8 else 0, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Seed")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return x, w


def plot_cem_aim_success(
    run_dir: Path,
    out_dir: Path | None = None,
    *,
    focus: str = "highlight",
) -> Path:
    run_dir = Path(run_dir)
    out_dir = out_dir or (run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = _parse_rows(load_csv(_pick_per_seed_csv(run_dir)))
    rows = _focus_rows(all_rows, focus)
    all_stats = _count(all_rows)
    stats = _count(rows)
    hard_stats = _count([r for r in all_rows if r.fb_no_clear])

    train_log_path = run_dir / "cem_aim_training_log.csv"
    train_log = load_csv(train_log_path) if train_log_path.exists() else []
    metrics = _load_metrics(run_dir)

    snap_win = float(metrics.get("snapshot_win_rate", 0.0))
    endgame = int(metrics.get("endgame_snap_wins", 0))
    global_dev = float(metrics.get("global_deviation_rate", 0.0))
    opp_dev = float(metrics.get("opportunity_deviation_rate", 0.0))

    n = stats["n"]
    focus_label = {
        "highlight": "FB no-clear & CEM wins",
        "fb_no_clear": "FollowBall no-clear seeds",
        "wins": "CEM WIN seeds",
        "all": "all seeds",
    }[focus]

    # --- episode_bricks_by_seed (focused) ---
    fig, ax = plt.subplots(figsize=(max(8, n * 0.55), 3.5))
    x, w = _bar_pair(
        ax,
        rows,
        "follow_bricks",
        "cem_bricks",
        "Broken bricks",
        f"Where FB struggles: broken bricks ({stats['wins']}W / {n} seeds)",
    )
    y_top = max(max(r.follow_bricks for r in rows), max(r.cem_bricks for r in rows))
    ax.set_ylim(0, y_top * 1.22)
    for i, r in enumerate(rows):
        bar_top = max(r.follow_bricks, r.cem_bricks)
        if r.delta_bricks != 0:
            color = GREEN if r.delta_bricks > 0 else RED
            ax.annotate(
                f"{r.delta_bricks:+.0f}",
                (x[i] + w / 2, r.cem_bricks),
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                fontweight="bold" if r.delta_bricks > 0 else "normal",
                zorder=5,
            )
        if r.cem_clears_fb_failed:
            # Centered above the seed pair; sit high enough not to cover +n on the CEM bar.
            clear_y = bar_top + (5.5 if r.delta_bricks != 0 else 3.0)
            ax.annotate(
                "CEM clear",
                (x[i], clear_y),
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                zorder=4,
            )
    fig.subplots_adjust(top=0.90, bottom=0.20, left=0.08, right=0.98)
    fig.savefig(out_dir / "episode_bricks_by_seed.png", dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    # --- episode_clear_by_seed (focused) ---
    fig, ax = plt.subplots(figsize=(max(8, n * 0.55), 4.2))
    _bar_pair(
        ax,
        rows,
        "follow_clear",
        "cem_clear",
        "Clear (0/1)",
        f"Clear breakthrough: CEM clears {stats['clears_fb_failed']}× where FB failed",
    )
    ax.set_ylim(-0.05, 1.2)
    fig.tight_layout()
    fig.savefig(out_dir / "episode_clear_by_seed.png", dpi=150)
    plt.close(fig)

    # --- state_wins_by_seed: delta bricks, highlight wins ---
    fig, ax = plt.subplots(figsize=(max(8, n * 0.55), 4.5))
    x = np.arange(n)
    colors_map = {"WIN": GREEN, "TIE": GREY, "LOSS": RED}
    bar_colors = [colors_map[r.verdict] for r in rows]
    deltas = [r.delta_bricks for r in rows]
    ax.bar(x, deltas, color=bar_colors, edgecolor="white", linewidth=0.6)
    ax.axhline(0, color="#8b949e", linewidth=0.8)
    ax.set_xticks(x, [str(r.seed) for r in rows], rotation=45 if n > 8 else 0, ha="right")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Δ bricks (CEM − FollowBall)")
    ax.set_title(f"CEM gains on hard seeds (mean Δ={stats['mean_delta_bricks']:+.2f})")
    ax.legend(
        handles=[
            Patch(facecolor=GREEN, label="WIN"),
            Patch(facecolor=GREY, label="TIE"),
            Patch(facecolor=RED, label="LOSS"),
        ],
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "state_wins_by_seed.png", dpi=150)
    plt.close(fig)

    # --- aggregate_summary: hard-seed story + training signal ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(
        ["FB no-clear\nseeds", "CEM WIN\non those", "CEM clear\nwhere FB failed"],
        [hard_stats["n"], hard_stats["wins"], hard_stats["clears_fb_failed"]],
        color=[GREY, GREEN, AMBER],
    )
    axes[0].set_ylabel("Seed count")
    axes[0].set_title(f"Hard seeds: FB clear fail ({hard_stats['n']}/{all_stats['n']})")

    axes[1].bar(
        ["Snapshot\nwin peak", "Endgame\nsnap wins", "Opp\ndeviation"],
        [snap_win * 100, endgame, opp_dev * 100],
        color=[BLUE, PURPLE, LIGHT_BLUE],
    )
    axes[1].set_ylabel("Rate / count")
    axes[1].set_title("Training-time local superiority")
    fig.suptitle(
        f"CEM-Aim advantage on FollowBall weak spots ({run_dir.name})",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "aggregate_summary.png", dpi=150)
    plt.close(fig)

    # --- training curve ---
    peak_snap = snap_win
    peak_gen = None
    if train_log:
        gens = [int(r["generation"]) + 1 for r in train_log]
        snap = [float(r["snapshot_win_rate"]) for r in train_log]
        endg = [int(r["endgame_snap_wins"]) for r in train_log]
        peak_snap = max(snap)
        peak_gen = gens[int(np.argmax(snap))]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(gens, [v * 100 for v in snap], "o-", color=BLUE, linewidth=2)
        axes[0].axvline(peak_gen, color=GREEN, ls="--", alpha=0.6)
        axes[0].set_xlabel("Generation")
        axes[0].set_ylabel("Snapshot win rate (%)")
        axes[0].set_title(f"Peak {peak_snap:.1%} local wins vs FB (gen {peak_gen})")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(gens, endg, "s-", color=PURPLE, linewidth=2)
        axes[1].set_xlabel("Generation")
        axes[1].set_ylabel("Endgame snapshot wins")
        axes[1].set_title("Endgame opportunity exploitation")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "training_dynamics.png", dpi=150)
        plt.close(fig)

    # --- delta hist on focused rows only ---
    fig, ax = plt.subplots(figsize=(6, 4))
    deltas = [r.delta_bricks for r in rows]
    lo, hi = min(deltas + [0]) - 1, max(deltas + [0]) + 1
    bins = np.arange(lo - 0.5, hi + 1.5, 1)
    ax.hist(deltas, bins=bins, color=BLUE, edgecolor="white")
    ax.axvline(0, color=GREY, ls="--")
    pos = sum(d > 0 for d in deltas)
    ax.set_xlabel("Δ bricks (CEM − FollowBall)")
    ax.set_ylabel("count")
    ax.set_title(f"Focused seeds: {pos}/{n} with more bricks than FB")
    fig.tight_layout()
    fig.savefig(out_dir / "episode_delta_bricks_hist.png", dpi=150)
    plt.close(fig)

    # --- success_headlines (reframed) ---
    win_rows = [r for r in all_rows if r.verdict == "WIN"]
    faster_same_brick = [
        r for r in win_rows if r.delta_bricks == 0 and r.delta_steps < -100
    ]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    headline_metrics = [
        (
            "CEM clear where\nFB failed",
            min(100, hard_stats["clears_fb_failed"] / max(1, hard_stats["n"]) * 100 * 4),
            f"{hard_stats['clears_fb_failed']}/{hard_stats['n']} hard seeds\n(seed 1 ★)",
        ),
        (
            "WIN on FB\nno-clear seeds",
            hard_stats["win_rate"] * 100,
            f"{hard_stats['wins']}/{hard_stats['n']}\n({100 * hard_stats['win_rate']:.0f}%)",
        ),
        (
            "Brick gain on\nhard seeds",
            max(0, 50 + hard_stats["mean_delta_bricks"] * 8),
            f"mean Δ={hard_stats['mean_delta_bricks']:+.2f}\n(seed 6: +7)",
        ),
        (
            "Faster clear\n(same bricks)",
            len(faster_same_brick) / max(1, all_stats["n"]) * 100 * 5,
            f"{len(faster_same_brick)} seeds\n(seeds 2, 19)",
        ),
        (
            "Peak snapshot\nwin vs FB",
            peak_snap * 100,
            f"{peak_snap:.1%}\n(gen {peak_gen})" if peak_gen else f"{peak_snap:.1%}",
        ),
    ]
    display = [m[1] for m in headline_metrics]
    colors_h = [AMBER, GREEN, BLUE, PURPLE, LIGHT_BLUE]
    bars = ax.barh([m[0] for m in headline_metrics], display, color=colors_h)
    ax.set_xlabel("Score (% scale)")
    ax.set_title(f"CEM-Aim — wins where FollowBall fails ({run_dir.name})")
    for b, d, m in zip(bars, display, headline_metrics):
        ax.text(min(d + 1, 92), b.get_y() + b.get_height() / 2, m[2], va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "success_headlines.png", dpi=150)
    plt.close(fig)

    # --- success_dashboard 2x2 ---
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    funnel_labels = [
        "FB no-clear\n(hard seeds)",
        "CEM improves\n(WIN or +bricks)",
        "CEM unique\nclear",
    ]
    improves = sum(r.cem_improves_on_hard for r in all_rows if r.fb_no_clear)
    funnel_vals = [hard_stats["n"], improves, hard_stats["clears_fb_failed"]]
    ax1.barh(funnel_labels, funnel_vals, color=[GREY, GREEN, AMBER])
    ax1.set_xlabel("Count")
    ax1.set_title("CEM-Aim unlocks FollowBall failure modes")
    notes = [
        f"{hard_stats['n']} seeds",
        f"{improves} ({100 * improves / max(1, hard_stats['n']):.0f}%)",
        f"{hard_stats['clears_fb_failed']} breakthrough",
    ]
    for i, v in enumerate(funnel_vals):
        ax1.text(v + 0.25, i, notes[i], va="center", fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    win_only = [r for r in all_rows if r.verdict == "WIN"]
    xw = np.arange(len(win_only))
    w = 0.35
    ax2.bar(xw - w / 2, [r.follow_ret for r in win_only], w, label="FollowBall", color=GREEN, alpha=0.85)
    ax2.bar(xw + w / 2, [r.cem_ret for r in win_only], w, label="CEM-Aim", color=BLUE)
    ax2.set_xticks(xw, [str(r.seed) for r in win_only])
    ax2.set_ylabel("Episode return")
    ax2.set_title(f"4 WIN seeds: all CEM ≥ FB return (mean Δ={np.mean([r.delta_ret for r in win_only]):+.0f})")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[1, 0])
    _bar_pair(
        ax3,
        rows,
        "follow_bricks",
        "cem_bricks",
        "Broken bricks",
        f"Focused view: {focus_label} ({n} seeds)",
    )

    ax4 = fig.add_subplot(gs[1, 1])
    categories = ["Hard:\nCEM WIN", "Hard:\nTIE", "Hard:\nLOSS", "All:\nCEM WIN"]
    hard_w = hard_stats["wins"]
    hard_t = hard_stats["ties"]
    hard_l = hard_stats["losses"]
    all_w = all_stats["wins"]
    ax4.bar(
        categories,
        [hard_w, hard_t, hard_l, all_w],
        color=[GREEN, GREY, RED, BLUE],
    )
    ax4.set_ylabel("Seed count")
    ax4.set_title(
        f"WIN rate: {100 * hard_stats['win_rate']:.0f}% on hard vs "
        f"{100 * all_stats['win_rate']:.0f}% overall"
    )

    fig.suptitle(
        "CEM-Aim — Success where FollowBall fails to clear",
        fontsize=13,
        y=1.01,
    )
    fig.savefig(out_dir / "success_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "run_dir": str(run_dir),
        "focus": focus,
        "headline": {
            "fb_no_clear_seeds": hard_stats["n"],
            "cem_win_on_hard": f"{hard_stats['wins']}/{hard_stats['n']}",
            "cem_clear_where_fb_failed": hard_stats["clears_fb_failed"],
            "mean_delta_bricks_on_hard": hard_stats["mean_delta_bricks"],
            "overall_wins": f"{all_stats['wins']}/{all_stats['n']}",
            "win_seeds": [r.seed for r in all_rows if r.verdict == "WIN"],
            "peak_snapshot_win_rate": peak_snap,
            "peak_snapshot_gen": peak_gen,
        },
        "focused_seeds": [r.seed for r in rows],
        "episodes": [
            {
                "seed": r.seed,
                "cem_bricks": r.cem_bricks,
                "fb_bricks": r.follow_bricks,
                "delta_bricks": r.delta_bricks,
                "cem_clear": r.cem_clear,
                "fb_clear": r.follow_clear,
                "verdict": r.verdict,
                "fb_no_clear": r.fb_no_clear,
                "cem_clears_fb_failed": r.cem_clears_fb_failed,
            }
            for r in all_rows
        ],
    }
    summary_path = out_dir / "success_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Plots saved to {out_dir} (focus={focus}, {n} seeds in bar charts)")
    for p in sorted(out_dir.glob("*.png")):
        print(f"  {p.name}")
    print(f"Summary: {summary_path}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CEM-Aim success dashboard.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="runs/cem_aim/v3_win_hunt_seed1",
    )
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument(
        "--focus",
        type=str,
        default="highlight",
        choices=["highlight", "fb_no_clear", "wins", "all"],
        help="highlight=FB no-clear + CEM wins (default)",
    )
    args = parser.parse_args()
    plot_cem_aim_success(
        Path(args.run_dir),
        Path(args.out_dir) if args.out_dir else None,
        focus=args.focus,
    )


if __name__ == "__main__":
    main()
