#!/usr/bin/env python3
"""Evaluate CEM-MPC on fixed env seeds in parallel; append results to CSV."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from agents import CEMMPCPolicyAgent, FollowBallAgent
from catbreak_env import CatBreakEnv
from cem_aim_v3 import RolloutResult, STEP_TOLERANCE, better_than_followball, worse_than_followball
from evaluate import run_episode, summarize_rows


FIELDS = [
    "seed", "cem_bricks", "cem_steps", "cem_clear", "follow_bricks", "follow_steps",
    "follow_clear", "verdict", "wall_clock_sec",
]


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


def _eval_seed(payload: tuple[int, dict]) -> dict:
    seed, cfg = payload
    t0 = time.time()
    env = CatBreakEnv(config={"layout": cfg["layout"]})
    planner = CEMMPCPolicyAgent(
        horizon=cfg["horizon"],
        population_size=cfg["population_size"],
        iterations=cfg["iterations"],
        elite_frac=cfg["elite_frac"],
        workers=cfg["workers"],
    )
    fb = FollowBallAgent()
    c = run_episode(env, planner, seed)
    f = run_episode(env, fb, seed)
    if hasattr(planner, "planner") and hasattr(planner.planner, "shutdown_rollout_pool"):
        planner.planner.shutdown_rollout_pool()
    env.close()
    v = _verdict(c, f)
    return {
        "seed": seed,
        "cem_bricks": c["broken_bricks"],
        "cem_steps": c["steps"],
        "cem_clear": int(c["clear"]),
        "follow_bricks": f["broken_bricks"],
        "follow_steps": f["steps"],
        "follow_clear": int(f["clear"]),
        "verdict": v,
        "wall_clock_sec": round(time.time() - t0, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="0:19", help="seed spec like 0:19")
    p.add_argument("--layout", default="cat")
    p.add_argument("--horizon", type=int, default=25)
    p.add_argument("--population-size", type=int, default=256)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--elite-frac", type=float, default=0.15)
    p.add_argument("--parallel", type=int, default=4, help="concurrent episodes")
    p.add_argument("--workers", type=int, default=5, help="CEM workers per episode")
    p.add_argument(
        "--out",
        default=str(ROOT / "runs/cem_mpc/cem_mpc_20seed_eval.csv"),
    )
    args = p.parse_args()

    if ":" in args.seeds:
        a, b = args.seeds.split(":")
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(x) for x in args.seeds.split(",")]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[int] = set()
    if out.exists():
        for row in csv.DictReader(out.open()):
            done.add(int(row["seed"]))

    cfg = {
        "layout": args.layout,
        "horizon": args.horizon,
        "population_size": args.population_size,
        "iterations": args.iterations,
        "elite_frac": args.elite_frac,
        "workers": args.workers,
    }
    todo = [s for s in seeds if s not in done]
    if not todo:
        print(f"All seeds already in {out}")
        return

    if not out.exists():
        with out.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    print(f"CEM-MPC eval | seeds={len(todo)} parallel={args.parallel} workers/ep={args.workers}")
    payloads = [(s, cfg) for s in todo]
    with ProcessPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(_eval_seed, p): p[0] for p in payloads}
        for fut in as_completed(futures):
            seed = futures[fut]
            row = fut.result()
            with out.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
            print(
                f"seed {row['seed']:2d} | cem={row['cem_bricks']} fb={row['follow_bricks']} "
                f"{row['verdict']} | {row['wall_clock_sec']:.0f}s"
            )

    rows = list(csv.DictReader(out.open()))
    if rows:
        bricks = [float(r["cem_bricks"]) for r in rows]
        steps = [float(r["cem_steps"]) for r in rows]
        clears = [int(r["cem_clear"]) for r in rows]
        blk100 = [100.0 * b / max(1.0, s) for b, s in zip(bricks, steps)]
        w = sum(1 for r in rows if r["verdict"] == "WIN")
        t = sum(1 for r in rows if r["verdict"] == "TIE")
        l = sum(1 for r in rows if r["verdict"] == "LOSS")
        print(
            f"DONE n={len(rows)} bricks={float(np.mean(bricks)):.2f} "
            f"blk100={float(np.mean(blk100)):.2f} clear={100*float(np.mean(clears)):.0f}% "
            f"W/T/L={w}/{t}/{l}"
        )


if __name__ == "__main__":
    main()
