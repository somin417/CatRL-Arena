"""Evaluate CEM-MPC vs Random and FollowBall baselines."""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

import settings as S
from agents import BaseAgent, CEMMPCPolicyAgent, make_agent
from catbreak_env import CatBreakEnv
from evaluate import env_seed_for_episode, summarize_rows

AGENTS = ("random", "follow", "cem_mpc")

EPISODE_FIELDS = [
    "episode",
    "env_seed",
    "agent",
    "return",
    "steps",
    "score",
    "broken_bricks",
    "remaining_bricks",
    "clear",
    "lives",
    "terminal_reason",
    "blocks_per_100_steps",
    "wall_clock_sec",
    "life_lost_count",
    "avg_plan_time_ms",
    "avg_best_plan_score",
    "avg_elite_bricks",
    "avg_entropy",
    "config_horizon",
    "config_population_size",
    "config_iterations",
    "config_elite_frac",
]

SUMMARY_FIELDS = [
    "agent",
    "episodes",
    "avg_return",
    "std_return",
    "clear_rate",
    "avg_steps",
    "avg_broken_bricks",
    "avg_remaining_bricks",
    "avg_blocks_per_100_steps",
    "avg_wall_clock_sec",
    "avg_plan_time_ms",
]

PLANNING_LOG_FIELDS = [
    "episode",
    "env_seed",
    "real_step",
    "chosen_action",
    "chosen_action_name",
    "followball_action",
    "followball_action_name",
    "mode",
    "is_safe_aim",
    "ball_descending",
    "best_sequence_prefix",
    "best_score_scalar",
    "best_score_tuple_as_string",
    "best_predicted_bricks",
    "best_predicted_life_lost",
    "elite_mean_bricks",
    "entropy_mean",
    "plan_time_ms",
    "ball_x",
    "ball_y",
    "ball_vx",
    "ball_vy",
    "paddle_x",
    "remaining_bricks",
]

SAFE_AIM_DIFF_MIN = 0.05
SAFE_AIM_DIFF_MAX = 0.15


def _obs_to_storable(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        parts = [np.asarray(obs["vector"], dtype=np.float32).ravel()]
        if "grid" in obs:
            parts.append(np.asarray(obs["grid"], dtype=np.float32).ravel())
        return np.concatenate(parts)
    return np.asarray(obs, dtype=np.float32).ravel()


def run_cem_episode(
    env: CatBreakEnv,
    agent: CEMMPCPolicyAgent,
    env_seed: int,
    planner_config: dict,
) -> tuple[dict, list[dict], list[dict]]:
    t0 = time.perf_counter()
    obs = env.reset(seed=env_seed)
    agent.reset(seed=env_seed + 1)
    total_return = 0.0
    life_lost_count = 0
    plan_times: list[float] = []
    plan_scores: list[float] = []
    elite_bricks: list[float] = []
    entropies: list[float] = []
    plan_logs: list[dict] = []
    demo_transitions: list[dict] = []
    real_step = 0

    while not env.done:
        action = agent.act(obs, env.last_info, env=env)
        step_log = agent.planner.consume_plan_log()
        for entry in step_log:
            entry["episode"] = None  # filled later
            entry["env_seed"] = env_seed
            entry["real_step"] = real_step
            entry["best_score_tuple_as_string"] = entry.pop("best_score_tuple", "")
            plan_logs.append(entry)
            plan_times.append(entry["plan_time_ms"])
            plan_scores.append(entry["best_score_scalar"])
            elite_bricks.append(entry["elite_mean_bricks"])
            entropies.append(entry["entropy_mean"])

        obs_before = _obs_to_storable(obs)
        obs, reward, done, info = env.step(action)
        total_return += reward
        if info.get("life_lost"):
            life_lost_count += 1
        log_entry = step_log[-1] if step_log else {}
        demo_transitions.append({
            "obs_before": obs_before,
            "action": action,
            "followball_action": log_entry.get("followball_action", action),
            "is_safe_aim": log_entry.get("is_safe_aim", 0),
            "mode": log_entry.get("mode", ""),
            "reward": reward,
            "obs_after": _obs_to_storable(obs),
            "done": done,
            "env_seed": env_seed,
            "step": real_step,
            "agent": agent.name,
            "broken_bricks": info["broken_bricks"],
            "clear": int(info["clear"]),
        })
        real_step += 1

    steps = info["step_count"]
    broken = info["broken_bricks"]
    blocks_per_100 = (broken / steps * 100.0) if steps > 0 else 0.0
    result = {
        "env_seed": env_seed,
        "return": total_return,
        "steps": steps,
        "score": info["score"],
        "broken_bricks": broken,
        "remaining_bricks": info["remaining_bricks"],
        "clear": int(info["clear"]),
        "lives": info["lives"],
        "terminal_reason": info["terminal_reason"] or "",
        "blocks_per_100_steps": blocks_per_100,
        "wall_clock_sec": time.perf_counter() - t0,
        "life_lost_count": life_lost_count,
        "avg_plan_time_ms": float(np.mean(plan_times)) if plan_times else 0.0,
        "avg_best_plan_score": float(np.mean(plan_scores)) if plan_scores else 0.0,
        "avg_elite_bricks": float(np.mean(elite_bricks)) if elite_bricks else 0.0,
        "avg_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "config_horizon": planner_config["horizon"],
        "config_population_size": planner_config["population_size"],
        "config_iterations": planner_config["iterations"],
        "config_elite_frac": planner_config["elite_frac"],
    }
    return result, plan_logs, demo_transitions


def run_baseline_episode(env: CatBreakEnv, agent: BaseAgent, env_seed: int) -> dict:
    from evaluate import run_episode

    row = run_episode(env, agent, env_seed)
    row.update({
        "life_lost_count": 1 if row["terminal_reason"] == "no_lives" else 0,
        "avg_plan_time_ms": 0.0,
        "avg_best_plan_score": 0.0,
        "avg_elite_bricks": 0.0,
        "avg_entropy": 0.0,
        "config_horizon": "",
        "config_population_size": "",
        "config_iterations": "",
        "config_elite_frac": "",
    })
    return row


def build_planner_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "mode": getattr(args, "mode", "safe_eval"),
        "env_config": {"layout": args.layout},
        "layout": args.layout,
        "horizon": args.horizon,
        "population_size": args.population_size,
        "elite_frac": args.elite_frac,
        "iterations": args.iterations,
        "seed": args.seed,
        "demo_teacher_mode": args.save_demo,
        "demo_teacher_stride": args.demo_teacher_stride,
        "workers": getattr(args, "workers", None),
        "layout": args.layout,
    }
    if kwargs["mode"] == "teacher_search":
        kwargs.update({
            "teacher_margin": getattr(args, "teacher_margin", 0.0),
            "num_rollout_after_sequence": getattr(args, "num_rollout_after_sequence", 500),
            "focus_endgame": getattr(args, "focus_endgame", True),
            "focus_stuck": getattr(args, "focus_stuck", True),
            "workers": getattr(args, "workers", None),
            "followball_floor": False,
        })
    return kwargs


def evaluate_cem_mpc(args: argparse.Namespace) -> tuple[Path, Path]:
    planner_kwargs = build_planner_kwargs(args)
    planner_config = {
        "horizon": args.horizon,
        "population_size": args.population_size,
        "iterations": args.iterations,
        "elite_frac": args.elite_frac,
    }
    env = CatBreakEnv(config={"layout": args.layout})

    episode_rows: list[dict] = []
    planning_rows: list[dict] = []
    demo_all: list[dict] = []

    for agent_key in AGENTS:
        if agent_key == "cem_mpc":
            agent = make_agent("cem_mpc", planner_kwargs=planner_kwargs, layout=args.layout)
        else:
            agent = make_agent(agent_key)

        for ep in range(args.episodes):
            ep_seed = env_seed_for_episode(args.seed, ep)
            if agent_key == "cem_mpc":
                print(f"[CEM-MPC] episode {ep + 1}/{args.episodes} env_seed={ep_seed} ...")
                result, plan_logs, demos = run_cem_episode(
                    env, agent, ep_seed, planner_config
                )
                for log in plan_logs:
                    log["episode"] = ep
                planning_rows.extend(plan_logs)
                for d in demos:
                    d["episode"] = ep
                demo_all.extend(demos)
                print(
                    f"  -> return={result['return']:.0f} bricks={result['broken_bricks']} "
                    f"steps={result['steps']} reason={result['terminal_reason']}"
                )
            else:
                result = run_baseline_episode(env, agent, ep_seed)

            episode_rows.append({"episode": ep, "agent": agent.name, **result})

    env.close()

    S.CEM_MPC_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ep_path = S.CEM_MPC_DIR / f"cem_mpc_episodes_{ts}.csv"
    sum_path = S.CEM_MPC_DIR / f"cem_mpc_summary_{ts}.csv"

    with ep_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows(episode_rows)

    summary_rows: list[dict] = []
    for agent_key in AGENTS:
        agent_name = {
            "random": "Random",
            "follow": "FollowBall",
            "cem_mpc": "CEM-MPC",
        }[agent_key]
        rows = [r for r in episode_rows if r["agent"] == agent_name]
        if not rows:
            continue
        base = summarize_rows(rows)
        returns = [float(r["return"]) for r in rows]
        plan_times = [float(r["avg_plan_time_ms"]) for r in rows]
        summary_rows.append({
            "agent": agent_name,
            "episodes": len(rows),
            "avg_return": base["avg_return"],
            "std_return": float(np.std(returns)) if returns else 0.0,
            "clear_rate": base["clear_rate"],
            "avg_steps": base["avg_steps"],
            "avg_broken_bricks": base["avg_broken_bricks"],
            "avg_remaining_bricks": float(np.mean([r["remaining_bricks"] for r in rows])),
            "avg_blocks_per_100_steps": base["avg_blocks_per_100_steps"],
            "avg_wall_clock_sec": float(np.mean([r["wall_clock_sec"] for r in rows])),
            "avg_plan_time_ms": float(np.mean(plan_times)),
        })

    with sum_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    if args.save_planning_log and planning_rows:
        log_path = S.CEM_MPC_DIR / f"planning_log_{ts}.csv"
        with log_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PLANNING_LOG_FIELDS)
            writer.writeheader()
            writer.writerows(planning_rows)
        print(f"Planning log: {log_path}")

    if args.save_demo and demo_all:
        S.CEM_MPC_DEMOS_DIR.mkdir(parents=True, exist_ok=True)
        demo_npz = S.CEM_MPC_DEMOS_DIR / f"cem_mpc_demo_{ts}.npz"
        demo_csv = S.CEM_MPC_DEMOS_DIR / f"cem_mpc_demo_{ts}.csv"
        np.savez_compressed(
            demo_npz,
            obs_before=np.stack([d["obs_before"] for d in demo_all]),
            action=np.array([d["action"] for d in demo_all], dtype=np.int64),
            followball_action=np.array(
                [d["followball_action"] for d in demo_all], dtype=np.int64
            ),
            is_safe_aim=np.array([d["is_safe_aim"] for d in demo_all], dtype=np.int8),
            reward=np.array([d["reward"] for d in demo_all], dtype=np.float32),
            obs_after=np.stack([d["obs_after"] for d in demo_all]),
            done=np.array([d["done"] for d in demo_all], dtype=bool),
            env_seed=np.array([d["env_seed"] for d in demo_all], dtype=np.int64),
            episode=np.array([d["episode"] for d in demo_all], dtype=np.int64),
            step=np.array([d["step"] for d in demo_all], dtype=np.int64),
        )
        demo_fields = [
            "episode", "step", "env_seed", "agent", "action", "followball_action",
            "is_safe_aim", "mode", "reward", "done", "broken_bricks", "clear",
        ]
        with demo_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=demo_fields)
            writer.writeheader()
            for d in demo_all:
                writer.writerow({k: d[k] for k in demo_fields})
        print(f"Demo data: {demo_npz}")
        print(f"Demo CSV: {demo_csv}")

    _print_leaderboard(summary_rows, ep_path, sum_path)
    demo_quality = compute_demo_quality(planning_rows, demo_all)
    _print_demo_quality(demo_quality)
    _print_success_test(summary_rows, demo_quality)
    return ep_path, sum_path


def compute_demo_quality(
    planning_rows: list[dict],
    demo_rows: list[dict],
) -> dict:
    """Measure how often CEM-MPC differs from FollowBall in safe aiming states."""
    safe_plan = [
        r for r in planning_rows
        if int(r.get("is_safe_aim", 0)) == 1
    ]
    safe_diff_plan = [
        r for r in safe_plan
        if int(r.get("chosen_action", -1)) != int(r.get("followball_action", -2))
    ]
    safe_demo = [d for d in demo_rows if int(d.get("is_safe_aim", 0)) == 1]
    safe_diff_demo = [
        d for d in safe_demo
        if int(d.get("action", -1)) != int(d.get("followball_action", -2))
    ]
    all_diff_demo = [
        d for d in demo_rows
        if int(d.get("action", -1)) != int(d.get("followball_action", -2))
    ]
    safe_n = len(safe_plan)
    safe_diff_rate = len(safe_diff_plan) / safe_n if safe_n else 0.0
    demo_safe_diff_rate = len(safe_diff_demo) / len(safe_demo) if safe_demo else 0.0
    demo_overall_diff_rate = len(all_diff_demo) / len(demo_rows) if demo_rows else 0.0
    return {
        "planning_steps": len(planning_rows),
        "safe_aim_steps": safe_n,
        "safe_aim_diff_steps": len(safe_diff_plan),
        "safe_aim_diff_rate": safe_diff_rate,
        "demo_transitions": len(demo_rows),
        "demo_safe_transitions": len(safe_demo),
        "demo_safe_diff_transitions": len(safe_diff_demo),
        "demo_safe_diff_rate": demo_safe_diff_rate,
        "demo_overall_diff_rate": demo_overall_diff_rate,
    }


def _print_demo_quality(quality: dict) -> None:
    if quality["planning_steps"] == 0 and quality["demo_transitions"] == 0:
        return
    print()
    print("Demo teacher quality:")
    print(
        f"  Safe-aim diff (planning log): "
        f"{quality['safe_aim_diff_steps']}/{quality['safe_aim_steps']} "
        f"({100 * quality['safe_aim_diff_rate']:.1f}%)  "
        f"[target {100 * SAFE_AIM_DIFF_MIN:.0f}–{100 * SAFE_AIM_DIFF_MAX:.0f}%]"
    )
    if quality["demo_transitions"]:
        print(
            f"  Safe-aim diff (demo npz):     "
            f"{quality['demo_safe_diff_transitions']}/{quality['demo_safe_transitions']} "
            f"({100 * quality['demo_safe_diff_rate']:.1f}%)"
        )
        print(
            f"  Overall demo diff:            "
            f"{100 * quality['demo_overall_diff_rate']:.1f}% of all transitions"
        )


def _print_leaderboard(summary_rows: list[dict], ep_path: Path, sum_path: Path) -> None:
    summary_rows.sort(
        key=lambda r: (
            -r["avg_broken_bricks"],
            -r["avg_blocks_per_100_steps"],
            -r["clear_rate"],
            r["avg_steps"],
        )
    )
    print(f"Episode CSV: {ep_path}")
    print(f"Summary CSV: {sum_path}")
    print()
    print(f"{'Agent':<12} {'Bricks':>8} {'Blk/100':>8} {'Clear%':>8} {'Steps':>8} {'Plan ms':>10}")
    print("-" * 60)
    for row in summary_rows:
        print(
            f"{row['agent']:<12} "
            f"{row['avg_broken_bricks']:8.2f} "
            f"{row['avg_blocks_per_100_steps']:8.2f} "
            f"{row['clear_rate'] * 100:7.1f}% "
            f"{row['avg_steps']:8.1f} "
            f"{row['avg_plan_time_ms']:10.1f}"
        )


def _print_success_test(summary_rows: list[dict], demo_quality: dict | None = None) -> None:
    by_name = {r["agent"]: r for r in summary_rows}
    cem = by_name.get("CEM-MPC")
    follow = by_name.get("FollowBall")
    if cem is None or follow is None:
        print("WARNING: Missing CEM-MPC or FollowBall in summary.")
        return

    meets_episode_floor = (
        cem["avg_broken_bricks"] >= follow["avg_broken_bricks"]
        and cem["clear_rate"] >= follow["clear_rate"]
    )
    beat_bricks = cem["avg_broken_bricks"] > follow["avg_broken_bricks"]
    beat_clear = cem["clear_rate"] > follow["clear_rate"]

    demo_quality = demo_quality or {}
    safe_diff = demo_quality.get("safe_aim_diff_rate", 0.0)
    demo_diff = demo_quality.get("demo_safe_diff_rate", safe_diff)
    diff_rate = demo_diff if demo_quality.get("demo_transitions") else safe_diff
    meets_demo_target = SAFE_AIM_DIFF_MIN <= diff_rate <= SAFE_AIM_DIFF_MAX
    has_demo_signal = diff_rate >= SAFE_AIM_DIFF_MIN

    print()
    if meets_episode_floor and meets_demo_target:
        print(
            "SUCCESS: CEM-MPC meets FollowBall episode floor and safe-aim demo diff "
            f"is in target range ({100 * diff_rate:.1f}%)."
        )
    elif meets_episode_floor and has_demo_signal:
        print(
            "PARTIAL: CEM-MPC meets FollowBall episode floor; safe-aim demo diff "
            f"={100 * diff_rate:.1f}% (target {100 * SAFE_AIM_DIFF_MIN:.0f}–"
            f"{100 * SAFE_AIM_DIFF_MAX:.0f}%)."
        )
    elif meets_episode_floor:
        print(
            "PARTIAL: CEM-MPC meets FollowBall episode floor but demo diff is too low "
            f"({100 * diff_rate:.1f}%). Increase horizon/population or check safe-aim gate."
        )
    elif beat_bricks or beat_clear:
        print(
            "SUCCESS: CEM-MPC beats FollowBall on "
            + ("broken_bricks" if beat_bricks else "clear_rate")
            + "."
        )
    else:
        print(
            "WARNING: CEM-MPC did not meet FollowBall floor. "
            f"CEM bricks={cem['avg_broken_bricks']:.1f} vs FollowBall={follow['avg_broken_bricks']:.1f}, "
            f"clear={cem['clear_rate']*100:.0f}% vs {follow['clear_rate']*100:.0f}%, "
            f"safe demo diff={100 * diff_rate:.1f}%."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CEM-MPC vs baselines.")
    parser.add_argument(
        "--mode",
        type=str,
        default="safe_eval",
        choices=["safe_eval", "teacher_search"],
        help="safe_eval: conservative floor; teacher_search: oracle demo search",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", type=str, default=S.DEFAULT_LAYOUT,
                        choices=[S.LAYOUT_RECT, S.LAYOUT_CAT])
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--population-size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--elite-frac", type=float, default=0.15)
    parser.add_argument("--save-planning-log", action="store_true", default=False)
    parser.add_argument("--save-demo", action="store_true", default=False)
    parser.add_argument(
        "--demo-teacher-stride",
        type=int,
        default=8,
        help="In demo mode, use CEM action every N safe-aim steps (~12%% at 8)",
    )
    parser.add_argument("--teacher-margin", type=float, default=0.0)
    parser.add_argument("--num-rollout-after-sequence", type=int, default=500)
    parser.add_argument("--focus-endgame", action="store_true", default=True)
    parser.add_argument("--no-focus-endgame", action="store_false", dest="focus_endgame")
    parser.add_argument("--focus-stuck", action="store_true", default=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel CEM rollout workers (auto on GPU box)")
    args = parser.parse_args()
    if args.mode == "teacher_search":
        from run_teacher_search import run_teacher_search
        run_teacher_search(args)
        return
    evaluate_cem_mpc(args)


if __name__ == "__main__":
    main()
