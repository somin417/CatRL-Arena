#!/usr/bin/env python3
"""Fit CEM-Aim v3 theta to match backed-up per-seed eval (v3_win_hunt_seed1)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import settings as S
from catbreak_env import CatBreakEnv
from cem_aim_policy import CEMAimV3Policy, NUM_PARAMS
from evaluate import run_episode
from evaluate_cem_aim import CEMAimAgentAdapter, FollowAgentAdapter, _verdict
from torch_utils import default_parallel_workers


def load_targets(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    for r in rows:
        r["seed"] = int(r["seed"])
        r["cem_bricks"] = float(r["cem_bricks"])
        r["cem_steps"] = float(r["cem_steps"])
        r["cem_clear"] = int(float(r["cem_clear"]))
    return rows


def score_rows(rows: list[dict], targets: list[dict]) -> tuple[float, dict]:
    by_seed = {int(r["seed"]): r for r in rows}
    verdict_ok = brick_l1 = step_l1 = clear_ok = 0
    for t in targets:
        r = by_seed[t["seed"]]
        if r["verdict"] == t["verdict"]:
            verdict_ok += 1
        brick_l1 += abs(r["cem_bricks"] - t["cem_bricks"])
        step_l1 += abs(r["cem_steps"] - t["cem_steps"])
        if int(r["cem_clear"]) == t["cem_clear"]:
            clear_ok += 1
    mean_bricks = float(np.mean([by_seed[t["seed"]]["cem_bricks"] for t in targets]))
    fitness = (
        verdict_ok * 1000.0
        + clear_ok * 50.0
        - brick_l1 * 80.0
        - step_l1 * 0.02
    )
    metrics = {
        "fitness": fitness,
        "verdict_matches": verdict_ok,
        "clear_matches": clear_ok,
        "brick_l1": brick_l1,
        "step_l1": step_l1,
        "mean_broken_bricks": mean_bricks,
        "val_wins": sum(1 for t in targets if by_seed[t["seed"]]["verdict"] == "WIN"),
        "val_ties": sum(1 for t in targets if by_seed[t["seed"]]["verdict"] == "TIE"),
        "val_losses": sum(1 for t in targets if by_seed[t["seed"]]["verdict"] == "LOSS"),
    }
    return fitness, metrics


def eval_theta(theta: np.ndarray, targets: list[dict]) -> tuple[float, dict, list[dict]]:
    env = CatBreakEnv()
    agent = CEMAimAgentAdapter(CEMAimV3Policy(theta))
    follow = FollowAgentAdapter()
    rows: list[dict] = []
    for t in targets:
        seed = int(t["seed"])
        f = run_episode(env, follow, seed)
        c = run_episode(env, agent, seed)
        rows.append({
            "seed": seed,
            "cem_bricks": c["broken_bricks"],
            "cem_steps": c["steps"],
            "cem_clear": c["clear"],
            "verdict": _verdict(c, f),
        })
    fitness, metrics = score_rows(rows, targets)
    return fitness, metrics, rows


def _worker(payload: tuple[np.ndarray, list[dict]]) -> tuple[float, dict, np.ndarray]:
    theta, targets = payload
    fitness, metrics, _ = eval_theta(theta, targets)
    return fitness, metrics, theta


def prior_pool() -> list[np.ndarray]:
    return [
        CEMAimV3Policy.prior_exact_follow(),
        CEMAimV3Policy.prior_mild_left_endgame(),
        CEMAimV3Policy.prior_mild_right_endgame(),
        CEMAimV3Policy.prior_centroid_endgame(),
        CEMAimV3Policy.prior_stuck_breaker(),
        CEMAimV3Policy.prior_teacher_like(1.0, 0.12),
        CEMAimV3Policy.prior_teacher_like(-1.0, 0.12),
    ]


def run_recovery(
    targets_path: Path,
    out_dir: Path,
    *,
    generations: int,
    population_size: int,
    sigma_init: float,
    sigma_floor: float,
    elite_frac: float,
    workers: int,
    seed: int,
) -> None:
    targets = load_targets(targets_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    mean = CEMAimV3Policy.prior_exact_follow().astype(np.float64).copy()
    sigma = float(sigma_init)
    best_theta = mean.copy()
    best_fitness = -1e18
    best_metrics: dict = {}

    priors = prior_pool()
    log_path = out_dir / "recovery_log.csv"
    if not log_path.exists():
        log_path.write_text(
            "generation,sigma,fitness,verdict_matches,brick_l1,mean_bricks,val_wins,val_ties,val_losses,wall_sec\n",
            encoding="utf-8",
        )

    print(f"Recovery CEM | targets={len(targets)} pop={population_size} gen={generations} workers={workers}")

    for gen in range(generations):
        t0 = time.time()
        population: list[np.ndarray] = []
        for p in priors:
            population.append(np.asarray(p, dtype=np.float64).copy())
        while len(population) < population_size:
            population.append(rng.normal(mean, sigma))
        population = [np.clip(np.asarray(t, dtype=np.float64), -8.0, 8.0) for t in population[:population_size]]

        results: list[tuple[float, dict, np.ndarray]] = []
        payloads = [(t, targets) for t in population]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, p): i for i, p in enumerate(payloads)}
            for fut in as_completed(futures):
                results.append(fut.result())

        results.sort(key=lambda x: x[0], reverse=True)
        gen_best_f, gen_best_m, gen_best_t = results[0]
        if gen_best_f > best_fitness:
            best_fitness = gen_best_f
            best_theta = gen_best_t.copy()
            best_metrics = gen_best_m

        n_elite = max(2, int(population_size * elite_frac))
        elite = np.stack([r[2] for r in results[:n_elite]], axis=0)
        mean = elite.mean(axis=0)
        sigma = max(sigma_floor, float(elite.std(axis=0).mean()) * 1.05)

        dt = time.time() - t0
        line = (
            f"{gen},{sigma:.5f},{gen_best_f:.1f},{gen_best_m['verdict_matches']},"
            f"{gen_best_m['brick_l1']:.0f},{gen_best_m['mean_broken_bricks']:.2f},"
            f"{gen_best_m['val_wins']},{gen_best_m['val_ties']},{gen_best_m['val_losses']},{dt:.1f}\n"
        )
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line)
        print(
            f"[gen {gen+1}/{generations}] fit={gen_best_f:.0f} "
            f"verdict={gen_best_m['verdict_matches']}/20 bricks={gen_best_m['mean_broken_bricks']:.2f} "
            f"W/T/L={gen_best_m['val_wins']}/{gen_best_m['val_ties']}/{gen_best_m['val_losses']} "
            f"brick_L1={gen_best_m['brick_l1']:.0f} sigma={sigma:.3f} ({dt:.0f}s)"
        )

        policy = CEMAimV3Policy(best_theta)
        policy.save(str(out_dir / "recovery_best.npz"), metrics=best_metrics)

    # Final eval + install into v3_win_hunt_seed1 if good enough
    _, final_m, final_rows = eval_theta(best_theta, targets)
    (out_dir / "recovery_per_seed.csv").write_text(
        "seed,cem_bricks,cem_steps,cem_clear,verdict,target_verdict\n"
        + "\n".join(
            f"{r['seed']},{r['cem_bricks']},{r['cem_steps']},{r['cem_clear']},{r['verdict']},"
            f"{next(t for t in targets if t['seed']==r['seed'])['verdict']}"
            for r in final_rows
        ),
        encoding="utf-8",
    )
    summary = {
        "best_theta": best_theta.tolist(),
        "best_metrics": final_m,
        "target_mean_bricks": float(np.mean([t["cem_bricks"] for t in targets])),
        "target_val_wins": sum(1 for t in targets if t["verdict"] == "WIN"),
    }
    (out_dir / "recovery_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    install_dir = out_dir.parent / "v3_win_hunt_seed1"
    passes = (
        final_m["verdict_matches"] >= 18
        and abs(final_m["mean_broken_bricks"] - summary["target_mean_bricks"]) <= 1.0
        and final_m["val_wins"] >= summary["target_val_wins"] - 1
    )
    if passes:
        policy = CEMAimV3Policy(best_theta)
        val_metrics = {
            "val_mean_broken_bricks": final_m["mean_broken_bricks"],
            "val_clear_rate": sum(r["cem_clear"] for r in final_rows) / len(final_rows),
            "val_wins": final_m["val_wins"],
            "val_ties": final_m["val_ties"],
            "val_losses": final_m["val_losses"],
            "note": "theta recovered via eval-backup CEM fit",
        }
        policy.save(str(install_dir / "cem_aim_val_best.npz"), metrics=val_metrics)
        policy.save(str(install_dir / "cem_aim_train_best.npz"), metrics=val_metrics)
        print(f"Installed recovered checkpoint -> {install_dir}/cem_aim_val_best.npz")
    else:
        print(
            f"Recovery incomplete (verdict {final_m['verdict_matches']}/20, "
            f"mean_bricks {final_m['mean_broken_bricks']:.2f}). "
            f"Best saved at {out_dir}/recovery_best.npz"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--targets",
        default=str(S.CEM_AIM_DIR / "v3_win_hunt_seed1" / "cem_aim_per_seed_eval_backup.csv"),
    )
    p.add_argument(
        "--out-dir",
        default=str(S.CEM_AIM_DIR / "v3_win_hunt_seed1_recovery"),
    )
    p.add_argument("--generations", type=int, default=25)
    p.add_argument("--population-size", type=int, default=24)
    p.add_argument("--sigma-init", type=float, default=0.2)
    p.add_argument("--sigma-floor", type=float, default=0.03)
    p.add_argument("--elite-frac", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=default_parallel_workers())
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    run_recovery(
        Path(args.targets),
        Path(args.out_dir),
        generations=args.generations,
        population_size=args.population_size,
        sigma_init=args.sigma_init,
        sigma_floor=args.sigma_floor,
        elite_frac=args.elite_frac,
        workers=args.workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
