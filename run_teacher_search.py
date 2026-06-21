"""Run CEM-MPC teacher_search: oracle trajectories that beat FollowBall."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

import settings as S
from agents import CEMMPCPolicyAgent, make_agent
from catbreak_env import CatBreakEnv
from cem_mpc_teacher import TeacherSearchPlanner, beats_followball_rollout
from evaluate import env_seed_for_episode, run_episode, summarize_rows
from torch_utils import configure_torch, default_parallel_workers, get_device

TEACHER_COMPARISON_FIELDS = [
    "episode", "env_seed", "step", "followball_action", "followball_action_name",
    "mpc_action", "mpc_action_name", "mpc_sequence_prefix", "action_source",
    "is_opportunity", "advantage_score", "teacher_score", "follow_teacher_score",
    "followball_horizon_bricks", "mpc_horizon_bricks",
    "followball_rollout_bricks", "mpc_rollout_bricks",
    "followball_rollout_steps_to_next_brick", "mpc_rollout_steps_to_next_brick",
    "predicted_life_lost", "remaining_bricks", "no_brick_broken_for",
    "ball_x", "ball_y", "ball_vx", "ball_vy", "paddle_x",
    "predicted_landing_x", "target_hit_offset", "beats_followball",
]

STRESS_SEED_LIST = list(S.TEACHER_STRESS_SEEDS)


def _obs_to_storable(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        parts = [np.asarray(obs["vector"], dtype=np.float32).ravel()]
        if "grid" in obs:
            parts.append(np.asarray(obs["grid"], dtype=np.float32).ravel())
        return np.concatenate(parts)
    return np.asarray(obs, dtype=np.float32).ravel()


def build_teacher_planner_kwargs(args: argparse.Namespace) -> dict:
    workers = default_parallel_workers(args.workers)
    return {
        "mode": "teacher_search",
        "env_config": {"layout": args.layout},
        "layout": args.layout,
        "horizon": args.horizon,
        "population_size": args.population_size,
        "elite_frac": args.elite_frac,
        "iterations": args.iterations,
        "seed": args.seed,
        "teacher_margin": args.teacher_margin,
        "num_rollout_after_sequence": args.num_rollout_after_sequence,
        "focus_endgame": args.focus_endgame,
        "focus_stuck": args.focus_stuck,
        "stuck_threshold": args.stuck_threshold,
        "allow_unsafe_search": args.allow_unsafe_search,
        "workers": workers,
        "followball_floor": False,
        "verbose": args.verbose,
    }


def run_teacher_episode(
    env: CatBreakEnv,
    agent: CEMMPCPolicyAgent,
    env_seed: int,
    *,
    save_better_only: bool,
    save_snapshots: bool,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    t0 = time.perf_counter()
    obs = env.reset(seed=env_seed)
    agent.reset(seed=env_seed + 1)
    planner: TeacherSearchPlanner = agent.planner  # type: ignore[assignment]

    total_return = 0.0
    comparisons: list[dict] = []
    better_rows: list[dict] = []
    snapshots: list[dict] = []
    step = 0

    while not env.done:
        obs_before = _obs_to_storable(obs)
        state_before = env.get_state_dict()

        action = agent.act(obs, env.last_info, env=env)
        logs = agent.planner.consume_plan_log()
        log = logs[-1] if logs else {}

        obs, reward, done, info = env.step(action)
        total_return += reward
        planner.note_step_outcome(info["broken_bricks"])

        row = {
            "env_seed": env_seed,
            "step": step,
            "followball_action": log.get("followball_action", action),
            "followball_action_name": log.get("followball_action_name", "?"),
            "mpc_action": action,
            "mpc_action_name": S.ACTION_NAMES.get(action, "?"),
            "mpc_sequence_prefix": log.get("best_sequence_prefix", ""),
            "action_source": log.get("action_source", log.get("mode", "")),
            "is_opportunity": log.get("is_opportunity", 0),
            "advantage_score": log.get("advantage_score", 0.0),
            "teacher_score": log.get("teacher_score", 0.0),
            "follow_teacher_score": log.get("follow_teacher_score", 0.0),
            "followball_horizon_bricks": log.get("followball_horizon_bricks", 0),
            "mpc_horizon_bricks": log.get("mpc_horizon_bricks", 0),
            "followball_rollout_bricks": log.get("followball_rollout_bricks", 0),
            "mpc_rollout_bricks": log.get("mpc_rollout_bricks", 0),
            "followball_rollout_steps_to_next_brick": log.get(
                "followball_rollout_steps_to_next_brick", 0
            ),
            "mpc_rollout_steps_to_next_brick": log.get("mpc_rollout_steps_to_next_brick", 0),
            "predicted_life_lost": log.get("best_predicted_life_lost", 0),
            "remaining_bricks": log.get("remaining_bricks", info["remaining_bricks"]),
            "no_brick_broken_for": log.get("no_brick_broken_for", 0),
            "ball_x": log.get("ball_x", env.ball_x),
            "ball_y": log.get("ball_y", env.ball_y),
            "ball_vx": log.get("ball_vx", env.ball_vx),
            "ball_vy": log.get("ball_vy", env.ball_vy),
            "paddle_x": log.get("paddle_x", env.paddle_x),
            "predicted_landing_x": log.get("predicted_landing_x", 0.0),
            "target_hit_offset": log.get("target_hit_offset", 0.0),
            "beats_followball": log.get("beats_followball", 0),
            "obs_before": obs_before,
            "action": action,
            "reward": reward,
            "done": done,
        }
        comparisons.append(row)

        save_this = bool(row["beats_followball"]) and int(row["mpc_action"]) != int(
            row["followball_action"]
        )
        if save_better_only and save_this:
            better_rows.append(row)
        elif not save_better_only and int(row["mpc_action"]) != int(row["followball_action"]):
            better_rows.append(row)

        if save_snapshots and int(row.get("is_opportunity", 0)):
            snapshots.append({
                "env_seed": env_seed,
                "step": step,
                "obs_before": obs_before,
                "state_dict_json": json.dumps({k: v.tolist() if hasattr(v, "tolist") else v
                                               for k, v in state_before.items()}),
                "followball_action": row["followball_action"],
                "mpc_action": row["mpc_action"],
                "advantage_score": row["advantage_score"],
            })

        step += 1

    result = {
        "env_seed": env_seed,
        "return": total_return,
        "steps": info["step_count"],
        "broken_bricks": info["broken_bricks"],
        "remaining_bricks": info["remaining_bricks"],
        "clear": int(info["clear"]),
        "terminal_reason": info["terminal_reason"] or "",
        "wall_clock_sec": time.perf_counter() - t0,
    }
    return result, comparisons, better_rows, snapshots


def run_teacher_search(args: argparse.Namespace) -> dict:
    configure_torch(get_device())
    workers = default_parallel_workers(args.workers)
    print(f"teacher_search | workers={workers} | cuda={get_device().type == 'cuda'}")

    planner_kwargs = build_teacher_planner_kwargs(args)
    env = CatBreakEnv(config={"layout": args.layout})
    agent = make_agent("cem_mpc", planner_kwargs=planner_kwargs, layout=args.layout)

    seeds = args.seeds if args.seeds else [
        env_seed_for_episode(args.seed, ep) for ep in range(args.episodes)
    ]

    cem_rows: list[dict] = []
    follow_rows: list[dict] = []
    all_comparisons: list[dict] = []
    all_better: list[dict] = []
    all_snapshots: list[dict] = []

    follow_agent = make_agent("follow")

    try:
        for ep_idx, env_seed in enumerate(seeds):
            print(f"[teacher] episode {ep_idx + 1}/{len(seeds)} seed={env_seed} ...")
            cem_result, comps, better, snaps = run_teacher_episode(
                env, agent, env_seed,
                save_better_only=args.save_better_than_followball_only,
                save_snapshots=args.save_opportunity_snapshots,
            )
            follow_result = run_episode(env, follow_agent, env_seed)

            for c in comps:
                c["episode"] = ep_idx
            for b in better:
                b["episode"] = ep_idx
            for s in snaps:
                s["episode"] = ep_idx

            cem_rows.append({"episode": ep_idx, "agent": "CEM-MPC-teacher", **cem_result})
            follow_rows.append({"episode": ep_idx, "agent": "FollowBall", **follow_result})
            all_comparisons.extend(comps)
            all_better.extend(better)
            all_snapshots.extend(snaps)

            print(
                f"  CEM bricks={cem_result['broken_bricks']} clear={cem_result['clear']} | "
                f"FollowBall bricks={follow_result['broken_bricks']} clear={follow_result['clear']} | "
                f"better_states={sum(1 for c in comps if c.get('beats_followball'))}"
            )
    finally:
        if hasattr(agent, "planner") and hasattr(agent.planner, "shutdown_rollout_pool"):
            agent.planner.shutdown_rollout_pool()
        env.close()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    S.CEM_MPC_DEMOS_DIR.mkdir(parents=True, exist_ok=True)
    S.CEM_MPC_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    S.CEM_MPC_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    log_path = S.CEM_MPC_LOGS_DIR / f"cem_mpc_teacher_comparison_{ts}.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TEACHER_COMPARISON_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_comparisons)
    print(f"Comparison log: {log_path}")

    demo_path = None
    if all_better:
        demo_path = S.CEM_MPC_DEMOS_DIR / f"cem_mpc_teacher_better_{ts}.npz"
        np.savez_compressed(
            demo_path,
            obs_before=np.stack([r["obs_before"] for r in all_better]),
            action=np.array([r["action"] for r in all_better], dtype=np.int64),
            followball_action=np.array(
                [r["followball_action"] for r in all_better], dtype=np.int64
            ),
            advantage_score=np.array(
                [r["advantage_score"] for r in all_better], dtype=np.float32
            ),
            env_seed=np.array([r["env_seed"] for r in all_better], dtype=np.int64),
            episode=np.array([r["episode"] for r in all_better], dtype=np.int64),
            step=np.array([r["step"] for r in all_better], dtype=np.int64),
            reward=np.array([r["reward"] for r in all_better], dtype=np.float32),
            done=np.array([r["done"] for r in all_better], dtype=bool),
        )
        print(f"Teacher demo (better-than-FollowBall): {demo_path}")

    snap_path = None
    if all_snapshots:
        snap_path = S.CEM_MPC_SNAPSHOTS_DIR / f"opportunity_snapshots_{ts}.npz"
        np.savez_compressed(
            snap_path,
            env_seed=np.array([s["env_seed"] for s in all_snapshots], dtype=np.int64),
            episode=np.array([s["episode"] for s in all_snapshots], dtype=np.int64),
            step=np.array([s["step"] for s in all_snapshots], dtype=np.int64),
            obs_before=np.stack([s["obs_before"] for s in all_snapshots]),
            followball_action=np.array(
                [s["followball_action"] for s in all_snapshots], dtype=np.int64
            ),
            mpc_action=np.array([s["mpc_action"] for s in all_snapshots], dtype=np.int64),
            advantage_score=np.array(
                [s["advantage_score"] for s in all_snapshots], dtype=np.float32
            ),
        )
        print(f"Opportunity snapshots: {snap_path}")

    report = build_teacher_report(all_comparisons, cem_rows, follow_rows)
    report_path = S.CEM_MPC_LOGS_DIR / f"teacher_search_report_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_teacher_report(report)
    print(f"Report JSON: {report_path}")

    _warn_demo_quality(all_better, report)
    return report


def build_teacher_report(
    comparisons: list[dict],
    cem_rows: list[dict],
    follow_rows: list[dict],
) -> dict:
    total = len(comparisons)
    deviated = [c for c in comparisons if int(c["mpc_action"]) != int(c["followball_action"])]
    opp = [c for c in comparisons if int(c.get("is_opportunity", 0))]
    opp_dev = [c for c in opp if int(c["mpc_action"]) != int(c["followball_action"])]
    beats = [c for c in comparisons if int(c.get("beats_followball", 0))]
    loses = [c for c in comparisons if int(c.get("beats_followball", 0)) == 0
             and int(c.get("is_opportunity", 0))
             and int(c["mpc_action"]) != int(c["followball_action"])]

    delta_bricks = [
        int(c.get("mpc_rollout_bricks", 0)) - int(c.get("followball_rollout_bricks", 0))
        for c in comparisons if int(c.get("is_opportunity", 0))
    ]

    seed_table = []
    for cem, follow in zip(cem_rows, follow_rows):
        seed_table.append({
            "env_seed": cem["env_seed"],
            "cem_bricks": cem["broken_bricks"],
            "follow_bricks": follow["broken_bricks"],
            "cem_clear": cem["clear"],
            "follow_clear": follow["clear"],
            "delta_bricks": cem["broken_bricks"] - follow["broken_bricks"],
        })

    return {
        "total_decisions": total,
        "mpc_deviations": len(deviated),
        "deviation_rate": len(deviated) / total if total else 0.0,
        "opportunity_states": len(opp),
        "opportunity_deviation_rate": len(opp_dev) / len(opp) if opp else 0.0,
        "beats_followball_states": len(beats),
        "loses_to_followball_states": len(loses),
        "avg_delta_rollout_bricks": float(np.mean(delta_bricks)) if delta_bricks else 0.0,
        "seed_table": seed_table,
    }


def print_teacher_report(report: dict) -> None:
    print()
    print("=== Teacher Search Report ===")
    print(f"  Total decisions:           {report['total_decisions']}")
    print(f"  MPC deviations:            {report['mpc_deviations']} "
          f"({100 * report['deviation_rate']:.1f}%)")
    print(f"  Opportunity states:        {report['opportunity_states']}")
    print(f"  Opportunity deviation:     {100 * report['opportunity_deviation_rate']:.1f}%")
    print(f"  Beats FollowBall states:   {report['beats_followball_states']}")
    print(f"  Loses vs FollowBall:       {report['loses_to_followball_states']}")
    print(f"  Avg delta rollout bricks:  {report['avg_delta_rollout_bricks']:.3f}")
    print()
    print(f"  {'Seed':>6} {'CEM Brk':>8} {'FB Brk':>8} {'Δ Brk':>7} {'CEM Cl':>7} {'FB Cl':>6}")
    print("  " + "-" * 50)
    for row in report["seed_table"]:
        print(
            f"  {row['env_seed']:>6} {row['cem_bricks']:8.0f} {row['follow_bricks']:8.0f} "
            f"{row['delta_bricks']:7.0f} {row['cem_clear']:7d} {row['follow_clear']:6d}"
        )


def _warn_demo_quality(better_rows: list[dict], report: dict) -> None:
    if not better_rows:
        print("No teacher advantage found. Increase horizon/population/iterations or relax teacher_margin.")
        return
    agree = sum(
        1 for r in better_rows
        if int(r["action"]) == int(r["followball_action"])
    )
    rate = agree / len(better_rows)
    if rate > 0.95:
        print("WARNING: Teacher demo is mostly FollowBall; not useful for imitation.")


def main() -> None:
    p = argparse.ArgumentParser(description="CEM-MPC teacher_search oracle mode.")
    p.add_argument("--mode", type=str, default="teacher_search", choices=["teacher_search"])
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="Explicit env seeds (overrides --episodes)")
    p.add_argument("--layout", type=str, default=S.DEFAULT_LAYOUT,
                   choices=[S.LAYOUT_RECT, S.LAYOUT_CAT])
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--population-size", type=int, default=128)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--elite-frac", type=float, default=0.15)
    p.add_argument("--teacher-margin", type=float, default=0.0)
    p.add_argument("--num-rollout-after-sequence", type=int, default=500)
    p.add_argument("--focus-endgame", action="store_true", default=True)
    p.add_argument("--no-focus-endgame", action="store_false", dest="focus_endgame")
    p.add_argument("--focus-stuck", action="store_true", default=True)
    p.add_argument("--no-focus-stuck", action="store_false", dest="focus_stuck")
    p.add_argument("--stuck-threshold", type=int, default=800)
    p.add_argument("--save-better-than-followball-only", action="store_true", default=True)
    p.add_argument("--save-all-deviations", action="store_false",
                   dest="save_better_than_followball_only")
    p.add_argument("--save-opportunity-snapshots", action="store_true", default=False)
    p.add_argument("--save-action-comparison-log", action="store_true", default=True)
    p.add_argument("--allow-unsafe-search", action="store_true", default=False)
    p.add_argument("--workers", type=int, default=None,
                   help="parallel rollout workers (default: auto, cpu_count-2 on GPU box)")
    p.add_argument("--sequential", action="store_true", help="disable parallel rollouts")
    p.add_argument("--verbose", action="store_true", default=False)
    args = p.parse_args()
    if args.sequential:
        args.workers = 0
    run_teacher_search(args)


if __name__ == "__main__":
    main()
