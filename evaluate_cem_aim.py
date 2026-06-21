"""Evaluate CEM-Aim v2/v3 vs Random and FollowBall baselines."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

import settings as S
from agents import FollowBallAgent, make_agent
from catbreak_env import CatBreakEnv
from cem_aim_policy import (
    POLICY_VERSION_V3,
    act_cem_aim_v3,
    load_cem_aim_policy,
)
from cem_aim_v3 import (
    WEAK_SEEDS,
    SAFETY_SEEDS,
    DEFAULT_STRESS_SEEDS,
    FollowAgentAdapter,
    V3AgentAdapter,
    load_snapshots,
    load_teacher_demo,
    parse_seed_spec,
    rollout_from_state,
    better_than_followball,
    worse_than_followball,
    RolloutResult,
    STEP_TOLERANCE,
)
from evaluate import run_episode, summarize_rows


PER_SEED_FIELDS = [
    "seed", "follow_bricks", "cem_bricks", "delta_bricks",
    "follow_clear", "cem_clear", "follow_steps", "cem_steps",
    "delta_steps", "follow_blk100", "cem_blk100", "verdict",
]


class CEMAimAgentAdapter:
    name = "CEM-Aim"

    def __init__(self, policy) -> None:
        self.policy = policy
        self.is_v3 = hasattr(policy, "policy_version") and policy.policy_version == POLICY_VERSION_V3

    def reset(self, seed=None) -> None:
        if hasattr(self.policy, "reset_episode"):
            self.policy.reset_episode(seed)

    def act(self, obs, info=None, env=None) -> int:
        return self.policy.act(obs, info=info, env=env)

    def note_step(self, info, env) -> None:
        if hasattr(self.policy, "note_step_after_env_step"):
            self.policy.note_step_after_env_step(info, env)

    def load(self, path: str) -> None:
        self.policy = load_cem_aim_policy(path)
        self.is_v3 = hasattr(self.policy, "policy_version")


def _verdict(c: dict, f: dict) -> str:
    cr = RolloutResult(c["broken_bricks"], c["clear"], c["steps"], c["lives"] < S.INITIAL_LIVES)
    fr = RolloutResult(f["broken_bricks"], f["clear"], f["steps"], False)
    if better_than_followball(cr, fr):
        return "WIN"
    if worse_than_followball(cr, fr):
        return "LOSS"
    if abs(c["broken_bricks"] - f["broken_bricks"]) < 1e-6 and c["clear"] == f["clear"]:
        if abs(c["steps"] - f["steps"]) <= STEP_TOLERANCE:
            return "TIE"
    return "LOSS"


def _print_seed_table(title: str, rows: list[dict]) -> None:
    if not rows:
        return
    print(f"\n{title}")
    print(f"{'Seed':>6} {'FB Brk':>7} {'CEM Brk':>8} {'ΔBrk':>6} {'Verdict':>8}")
    print("-" * 45)
    for r in sorted(rows, key=lambda x: x["seed"]):
        print(
            f"{r['seed']:6d} {r['follow_bricks']:7.0f} {r['cem_bricks']:8.0f} "
            f"{r['delta_bricks']:6.0f} {r['verdict']:8s}"
        )


def evaluate_cem_aim(args: argparse.Namespace) -> tuple[Path, Path]:
    seeds = parse_seed_spec(args.seeds) if args.seeds else [
        args.seed + i for i in range(args.episodes)
    ]
    stress = set(parse_seed_spec(args.stress_seeds)) if args.stress_seeds else set(DEFAULT_STRESS_SEEDS)

    env = CatBreakEnv(config={"layout": args.layout})
    policy = load_cem_aim_policy(args.model)
    cem_agent = CEMAimAgentAdapter(policy)
    follow = FollowAgentAdapter()
    random_agent = make_agent("random")

    per_seed_rows: list[dict] = []
    debug_rows: list[dict] = []

    global_dev = opp_dev = residual_acts = unsafe_fb = total_dbg = opp_dbg = 0
    committed_contacts = committed_shots = 0
    hit_offsets: list[float] = []

    for seed in seeds:
        f_row = run_episode(env, follow, seed)
        c_row = run_episode(env, cem_agent, seed)
        run_episode(env, random_agent, seed)

        per_seed_rows.append({
            "seed": seed,
            "follow_bricks": f_row["broken_bricks"],
            "cem_bricks": c_row["broken_bricks"],
            "delta_bricks": c_row["broken_bricks"] - f_row["broken_bricks"],
            "follow_clear": f_row["clear"],
            "cem_clear": c_row["clear"],
            "follow_steps": f_row["steps"],
            "cem_steps": c_row["steps"],
            "delta_steps": c_row["steps"] - f_row["steps"],
            "follow_blk100": f_row["blocks_per_100_steps"],
            "cem_blk100": c_row["blocks_per_100_steps"],
            "verdict": _verdict(c_row, f_row),
        })

        if cem_agent.is_v3 and hasattr(cem_agent.policy, "commitment_summary"):
            summary = cem_agent.policy.commitment_summary()
            per_seed_rows[-1]["committed_contact_rate"] = summary.get("committed_contact_rate", 0.0)
            per_seed_rows[-1]["committed_shots"] = summary.get("num_committed_shots", 0.0)
            committed_shots += int(summary.get("num_committed_shots", 0.0))
            committed_contacts += int(summary.get("committed_contacts", 0.0))
            if summary.get("mean_abs_hit_offset_on_contact", 0.0) > 0:
                hit_offsets.append(summary["mean_abs_hit_offset_on_contact"])

        if cem_agent.is_v3 and args.log_debug_actions:
            policy_ref = cem_agent.policy
            obs = env.reset(seed=seed)
            policy_ref.reset_episode(seed)
            step = 0
            while not env.done:
                action, dbg = policy_ref.act_debug(obs, env=env)
                total_dbg += 1
                if dbg["deviated_from_followball"]:
                    global_dev += 1
                if dbg["opportunity"]:
                    opp_dbg += 1
                    if dbg["deviated_from_followball"]:
                        opp_dev += 1
                    if dbg.get("committed"):
                        residual_acts += 1
                if dbg.get("committed_contact"):
                    if abs(float(dbg.get("hit_offset", 0.0))) > 0:
                        hit_offsets.append(abs(float(dbg["hit_offset"])))
                if dbg["unsafe"]:
                    unsafe_fb += 1
                debug_rows.append({"seed": seed, "step": step, **dbg})
                _, _, done, info = env.step(action)
                policy_ref.note_step_after_env_step(info, env)
                obs = env.get_obs() if not done else obs
                step += 1

    env.close()

    out_dir = Path(args.output_dir) if args.output_dir else S.CEM_AIM_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    per_path = out_dir / "per_seed_comparison.csv"
    sum_path = out_dir / f"cem_aim_summary_{ts}.csv"

    with per_path.open("w", newline="") as f:
        fields = list(PER_SEED_FIELDS)
        if per_seed_rows and "committed_contact_rate" in per_seed_rows[0]:
            fields.extend(["committed_contact_rate", "committed_shots"])
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(per_seed_rows)

    cem_rows = [{"broken_bricks": r["cem_bricks"], "clear": r["cem_clear"],
                 "steps": r["cem_steps"], "blocks_per_100_steps": r["cem_blk100"],
                 "return": 0, "wall_clock_sec": 0} for r in per_seed_rows]
    follow_rows = [{"broken_bricks": r["follow_bricks"], "clear": r["follow_clear"],
                    "steps": r["follow_steps"], "blocks_per_100_steps": r["follow_blk100"],
                    "return": 0, "wall_clock_sec": 0} for r in per_seed_rows]

    summaries = []
    for name, rows in [("FollowBall", follow_rows), ("CEM-Aim", cem_rows)]:
        s = summarize_rows(rows)
        summaries.append({"agent": name, **s})

    with sum_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["agent", "avg_broken_bricks", "clear_rate",
                                           "avg_steps", "avg_blocks_per_100_steps", "avg_return"])
        w.writeheader()
        w.writerows(summaries)

    wins = sum(1 for r in per_seed_rows if r["verdict"] == "WIN")
    ties = sum(1 for r in per_seed_rows if r["verdict"] == "TIE")
    losses = sum(1 for r in per_seed_rows if r["verdict"] == "LOSS")

    print(f"Per-seed CSV: {per_path}")
    print(f"Summary CSV: {sum_path}")
    print()
    print(f"{'Agent':<12} {'Bricks':>8} {'Clear%':>8} {'Steps':>8} {'Blk/100':>8}")
    print("-" * 50)
    for row in summaries:
        print(
            f"{row['agent']:<12} {row['avg_broken_bricks']:8.2f} "
            f"{row['clear_rate']*100:7.1f}% {row['avg_steps']:8.1f} "
            f"{row['avg_blocks_per_100_steps']:8.2f}"
        )

    print()
    print(f"Pairwise vs FollowBall: Wins={wins} Ties={ties} Losses={losses}")
    print()
    print(f"{'Seed':>6} {'FB Brk':>7} {'CEM Brk':>8} {'ΔBrk':>6} {'Verdict':>8}")
    print("-" * 45)
    for r in per_seed_rows:
        print(
            f"{r['seed']:6d} {r['follow_bricks']:7.0f} {r['cem_bricks']:8.0f} "
            f"{r['delta_bricks']:6.0f} {r['verdict']:8s}"
        )

    if cem_agent.is_v3:
        contact_rate = committed_contacts / max(1, committed_shots)
        mean_abs_hit = float(np.mean(hit_offsets)) if hit_offsets else 0.0
        print()
        print(
            f"committed_contact_rate={contact_rate:.3f} "
            f"mean_abs_hit_offset_when_committed={mean_abs_hit:.4f}"
        )
        if args.log_debug_actions:
            print(
                f"global_deviation_rate={global_dev/max(1,total_dbg):.3f} "
                f"opportunity_deviation_rate={opp_dev/max(1,opp_dbg):.3f} "
                f"committed_shot_rate={residual_acts/max(1,opp_dbg):.3f} "
                f"unsafe_fallback_rate={unsafe_fb/max(1,total_dbg):.3f}"
            )

    weak_rows = [r for r in per_seed_rows if r["seed"] in WEAK_SEEDS]
    _print_seed_table("Weak seed table (4, 6, 13, 14):", weak_rows)

    safety_rows = [r for r in per_seed_rows if r["seed"] in SAFETY_SEEDS]
    _print_seed_table("Stress seed safety table (10, 11, 19):", safety_rows)

    snap_win_rate = None
    if args.opportunity_snapshots and (args.eval_snapshots or args.log_debug_actions):
        snap_win_rate = _eval_snapshots(args, cem_agent, follow, out_dir)

    if args.teacher_demo:
        demo = load_teacher_demo(args.teacher_demo)
        if demo.deviation_samples:
            matches = 0
            for s in demo.deviation_samples[:100]:
                act = int(act_cem_aim_v3(s.obs, cem_agent.policy.theta, env=None))
                if act == s.teacher_action:
                    matches += 1
            print(f"teacher_action_match_rate={matches/max(1,len(demo.deviation_samples[:100])):.3f}")

    if snap_win_rate is not None:
        print(f"snapshot_win_rate={snap_win_rate:.3f}")

    _print_success(summaries, wins, ties, losses)

    if args.log_debug_actions and debug_rows:
        dbg_path = out_dir / "debug_actions.csv"
        with dbg_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(debug_rows[0].keys()), extrasaction="ignore")
            w.writeheader()
            w.writerows(debug_rows)
        print(f"Debug actions: {dbg_path}")

    return per_path, sum_path


def _eval_snapshots(args, cem_agent, follow, out_dir) -> float:
    snaps = load_snapshots(args.opportunity_snapshots)
    env = CatBreakEnv(config={"layout": args.layout})
    rows = []
    wins = 0
    horizon = args.snapshot_rollout_horizon
    for snap in snaps:
        c = rollout_from_state(env, snap.env_state, cem_agent, horizon)
        f = rollout_from_state(env, snap.env_state, follow, horizon)
        win = better_than_followball(c, f)
        wins += int(win)
        rows.append({
            "seed": snap.seed,
            "step": snap.step,
            "remaining_bricks": snap.remaining_bricks,
            "cem_bricks": c.broken_bricks,
            "follow_bricks": f.broken_bricks,
            "delta_bricks": c.broken_bricks - f.broken_bricks,
            "cem_steps": c.steps,
            "follow_steps": f.steps,
            "win": int(win),
        })
    env.close()
    path = out_dir / "opportunity_snapshot_eval.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["seed"])
        w.writeheader()
        w.writerows(rows)
    rate = wins / max(1, len(rows))
    print(f"Snapshot eval CSV: {path} (win_rate={rate:.3f})")
    return rate


def _print_success(summaries, wins, ties, losses) -> None:
    by_name = {r["agent"]: r for r in summaries}
    aim = by_name.get("CEM-Aim")
    follow = by_name.get("FollowBall")
    if not aim or not follow:
        return
    if aim["avg_broken_bricks"] > follow["avg_broken_bricks"]:
        print("SUCCESS: CEM-Aim beats FollowBall on avg_broken_bricks.")
    elif (
        abs(aim["avg_broken_bricks"] - follow["avg_broken_bricks"]) < 1e-6
        and aim["avg_blocks_per_100_steps"] > follow["avg_blocks_per_100_steps"]
    ):
        print("SUCCESS: same bricks, better blocks_per_100.")
    elif (
        abs(aim["avg_broken_bricks"] - follow["avg_broken_bricks"]) < 1e-6
        and aim["clear_rate"] == follow["clear_rate"]
        and aim["avg_steps"] <= follow["avg_steps"]
    ):
        print("SUCCESS: same bricks/clear, fewer or equal steps.")
    else:
        print(
            f"DIAGNOSTIC: Aim bricks={aim['avg_broken_bricks']:.1f} vs "
            f"Follow={follow['avg_broken_bricks']:.1f} | "
            f"wins={wins} ties={ties} losses={losses}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate CEM-Aim vs baselines.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--policy-version", type=str, default=None)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument("--stress-seeds", type=str, default="10,11,13,14,19")
    p.add_argument("--layout", type=str, default=S.DEFAULT_LAYOUT,
                   choices=[S.LAYOUT_RECT, S.LAYOUT_CAT])
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--compare-followball", action="store_true", default=True)
    p.add_argument("--save-per-seed", action="store_true", default=True)
    p.add_argument("--opportunity-snapshots", type=str, default=None)
    p.add_argument("--teacher-demo", type=str, default=None)
    p.add_argument("--eval-snapshots", action="store_true")
    p.add_argument("--snapshot-rollout-horizon", type=int, default=2000)
    p.add_argument("--log-debug-actions", action="store_true")
    args = p.parse_args()
    evaluate_cem_aim(args)


if __name__ == "__main__":
    main()
