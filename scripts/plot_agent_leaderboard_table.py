#!/usr/bin/env python3
"""Render agent comparison leaderboard table as PNG (no Interpretation column)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents import DQNPolicyAgent, FollowBallAgent, make_agent
from catbreak_env import CatBreakEnv
from cem_aim_v3 import RolloutResult, STEP_TOLERANCE, better_than_followball, worse_than_followball
from evaluate import run_episode, summarize_rows


def _verdict(c: dict, f: dict) -> str:
    cr = RolloutResult(c["broken_bricks"], c["clear"], c["steps"], c["lives"] < 1)
    fr = RolloutResult(f["broken_bricks"], f["clear"], f["steps"], False)
    if better_than_followball(cr, fr):
        return "WIN"
    if worse_than_followball(cr, fr):
        return "LOSS"
    if (
        abs(c["broken_bricks"] - f["broken_bricks"]) < 1e-6
        and c["clear"] == f["clear"]
        and abs(c["steps"] - f["steps"]) <= STEP_TOLERANCE
    ):
        return "TIE"
    return "LOSS"


def _pairwise(agent_rows: list[dict], fb_rows: list[dict]) -> str:
    w = t = l = 0
    for c, f in zip(agent_rows, fb_rows):
        v = _verdict(c, f)
        w += v == "WIN"
        t += v == "TIE"
        l += v == "LOSS"
    return f"{w}/{t}/{l}"


def _load_cem_mpc_csv(path: Path, n_seeds: int) -> dict:
    rows = list(csv.DictReader(path.open()))
    if len(rows) < n_seeds:
        raise ValueError(f"Need {n_seeds} CEM-MPC seeds in {path}, found {len(rows)}")
    cem_rows = []
    for r in rows:
        cem_rows.append({
            "broken_bricks": float(r["cem_bricks"]),
            "clear": int(r["cem_clear"]),
            "steps": float(r["cem_steps"]),
            "blocks_per_100_steps": 100.0 * float(r["cem_bricks"]) / max(1.0, float(r["cem_steps"])),
            "return": 0.0,
        })
    s = summarize_rows(cem_rows)
    w = sum(1 for r in rows if r["verdict"] == "WIN")
    t = sum(1 for r in rows if r["verdict"] == "TIE")
    l = sum(1 for r in rows if r["verdict"] == "LOSS")
    return {
        "agent": "CEM-MPC",
        "avg_bricks": s["avg_broken_bricks"],
        "blk100": s["avg_blocks_per_100_steps"],
        "clear_pct": s["clear_rate"] * 100.0,
        "pairwise": f"{w}/{t}/{l}",
    }


def _load_cem_aim_csv(path: Path, seeds: list[int]) -> dict:
    rows = list(csv.DictReader(path.open()))
    by_seed = {int(r["seed"]): r for r in rows}
    missing = [s for s in seeds if s not in by_seed]
    if missing:
        raise ValueError(f"Missing CEM-Aim seeds {missing} in {path}")
    ordered = [by_seed[s] for s in seeds]
    cem_rows = []
    for r in ordered:
        cem_rows.append({
            "broken_bricks": float(r["cem_bricks"]),
            "clear": int(r["cem_clear"]),
            "steps": float(r["cem_steps"]),
            "blocks_per_100_steps": float(r["cem_blk100"]),
            "return": 0.0,
        })
    s = summarize_rows(cem_rows)
    w = sum(1 for r in ordered if r["verdict"] == "WIN")
    t = sum(1 for r in ordered if r["verdict"] == "TIE")
    l = sum(1 for r in ordered if r["verdict"] == "LOSS")
    return {
        "agent": "CEM-Aim",
        "avg_bricks": s["avg_broken_bricks"],
        "blk100": s["avg_blocks_per_100_steps"],
        "clear_pct": s["clear_rate"] * 100.0,
        "pairwise": f"{w}/{t}/{l}",
    }


def collect_rows(seeds: list[int], cem_csv: Path, cem_aim_csv: Path | None = None) -> list[dict]:
    env = CatBreakEnv(config={"layout": "cat"})
    fb = FollowBallAgent()
    fb_rows = [run_episode(env, fb, s) for s in seeds]

    out: list[dict] = []

    for name, agent in [("Random", make_agent("random"))]:
        rows = [run_episode(env, agent, s) for s in seeds]
        s = summarize_rows(rows)
        out.append({
            "agent": name,
            "avg_bricks": s["avg_broken_bricks"],
            "blk100": s["avg_blocks_per_100_steps"],
            "clear_pct": s["clear_rate"] * 100.0,
            "pairwise": _pairwise(rows, fb_rows),
        })

    s = summarize_rows(fb_rows)
    out.append({
        "agent": "FollowBall",
        "avg_bricks": s["avg_broken_bricks"],
        "blk100": s["avg_blocks_per_100_steps"],
        "clear_pct": s["clear_rate"] * 100.0,
        "pairwise": "—",
    })

    for label, ckpt in [("MLP-DQN", ROOT / "runs/dqn/dqn_best.pt"),
                        ("CNN-DQN", ROOT / "runs/dqn_cnn/dqn_best.pt")]:
        agent = DQNPolicyAgent(model_path=str(ckpt), fallback_to_follow=False)
        e = CatBreakEnv(config={"layout": "cat", "obs_mode": agent._dqn.obs_mode})
        rows = [run_episode(e, agent, s) for s in seeds]
        s = summarize_rows(rows)
        out.append({
            "agent": label,
            "avg_bricks": s["avg_broken_bricks"],
            "blk100": s["avg_blocks_per_100_steps"],
            "clear_pct": s["clear_rate"] * 100.0,
            "pairwise": _pairwise(rows, fb_rows),
        })
        e.close()

    env.close()
    out.append(_load_cem_mpc_csv(cem_csv, len(seeds)))
    if cem_aim_csv is not None:
        out.append(_load_cem_aim_csv(cem_aim_csv, seeds))
    return out


def render_png(rows: list[dict], out_path: Path, title: str, subtitle: str) -> None:
    headers = [
        "Agent",
        "Avg Bricks",
        "Blocks/100 Steps",
        "Clear Rate",
        "Pairwise vs\nFollowBall\n(win / tie / lose)",
    ]
    body = []
    for r in rows:
        body.append([
            r["agent"],
            f"{r['avg_bricks']:.1f}",
            f"{r['blk100']:.2f}",
            f"{r['clear_pct']:.0f}%",
            r["pairwise"],
        ])

    n = len(body)
    col_widths = [0.11, 0.10, 0.17, 0.10, 0.30]
    fig_h = 0.55 + 0.30 * n
    fig, ax = plt.subplots(figsize=(7.6, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=10, pad=4, weight="bold", y=0.98)
    ax.text(0.5, 0.90, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color="#444")

    table = ax.table(
        cellText=body,
        colLabels=headers,
        loc="upper center",
        cellLoc="center",
        colWidths=col_widths,
        bbox=[0.0, 0.0, 1.0, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.12)

    for (row, col), cell in table.get_celld().items():
        cell.PAD = 0.06
        if row == 0:
            cell.set_height(0.24)
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold", fontsize=7)
        elif row % 2 == 0:
            cell.set_facecolor("#f3f4f6")
        if col == 0 and row > 0:
            cell.set_text_props(weight="bold")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    table_bbox = table.get_window_extent(renderer)
    title_bbox = ax.title.get_window_extent(renderer)
    subtitle_bbox = ax.texts[0].get_window_extent(renderer)
    pad_x, pad_y = 12, 10
    crop_display = Bbox.from_extents(
        table_bbox.x0 - pad_x,
        min(table_bbox.y0, title_bbox.y0, subtitle_bbox.y0) - pad_y,
        table_bbox.x1 + pad_x,
        max(title_bbox.y1, subtitle_bbox.y1) + pad_y,
    )
    crop = crop_display.transformed(fig.dpi_scale_trans.inverted())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches=crop, facecolor="white")
    plt.close(fig)


def _parse_seeds(spec: str) -> list[int]:
    if ":" in spec:
        a, b = spec.split(":")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="0:4", help="seed spec e.g. 0:4 or 0:19")
    p.add_argument("--cem-csv", default=str(ROOT / "runs/cem_mpc/cem_mpc_5seed_pop256_eval.csv"))
    p.add_argument(
        "--cem-aim-csv",
        default=None,
        help="per_seed_comparison.csv from evaluate_cem_aim.py (optional CEM-Aim row)",
    )
    p.add_argument(
        "--out",
        default=str(ROOT / "runs/comparison/agent_leaderboard_table.png"),
    )
    args = p.parse_args()
    seeds = _parse_seeds(args.seeds)
    cem_csv = Path(args.cem_csv)
    cem_aim_csv = Path(args.cem_aim_csv) if args.cem_aim_csv else None
    rows = collect_rows(seeds, cem_csv, cem_aim_csv)
    lo, hi = min(seeds), max(seeds)
    title = f"CatBreak Agent Comparison (cat layout, seeds {lo}–{hi})"
    subtitle = (
        f"n={len(seeds)} | CEM-MPC: safe_eval h=25 pop=256 iter=5 | "
        f"Pairwise W/T/L vs FollowBall (±50 steps)"
    )
    if cem_aim_csv is not None:
        subtitle += " | CEM-Aim: cpr_quick_3h train_best"
    render_png(rows, Path(args.out), title, subtitle)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
