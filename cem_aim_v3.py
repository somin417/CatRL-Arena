"""CEM-Aim v3 training utilities: snapshots, teacher demo, pairwise evaluation."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

import settings as S
from agents import FollowBallAgent, make_agent
from catbreak_env import CatBreakEnv, obs_vector_for_agent
from cem_aim_policy import (
    POLICY_VERSION_V3,
    CEMAimV3Policy,
    CEMAimV3EpisodeState,
    ShotMacroConfig,
    SIDE_RULES,
    TARGET_BASES,
    act_cem_aim_v3,
    exact_followball_action,
    extract_cem_aim_features,
    is_opportunity_state,
    macro_config_to_theta,
    make_cem_aim_policy,
    NUM_PARAMS,
)
from evaluate import env_seed_for_episode, run_episode, summarize_rows
from torch_utils import default_parallel_workers

DEFAULT_STRESS_SEEDS = (10, 11, 13, 14, 19)
STEP_TOLERANCE = 50


def run_episode_fast(env: CatBreakEnv, agent, env_seed: int) -> dict:
    """Headless episode rollout without per-step obs construction."""
    obs = env.reset(seed=env_seed)
    agent.reset(seed=env_seed + 1)
    total_return = 0.0
    while not env.done:
        action = agent.act(obs, env.last_info, env=env)
        reward, done, info = env.step_fast(action)
        total_return += reward
        if hasattr(agent, "note_step"):
            agent.note_step(info, env)
        if not done:
            obs = env.get_obs()
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
    }
    if hasattr(agent, "commitment_summary"):
        result.update(agent.commitment_summary())
    return result


def _aggregate_commitment_metrics(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {
            "committed_shot_rate": 0.0,
            "committed_contact_rate": 0.0,
            "committed_shot_success_rate": 0.0,
            "delta_next_brick_after_committed_shot": 0.0,
            "endgame_committed_shot_wins": 0.0,
        }
    opp_steps = sum(float(r.get("opportunity_steps", 0.0)) for r in rows)
    shots = sum(float(r.get("num_committed_shots", 0.0)) for r in rows)
    contacts = sum(float(r.get("committed_contacts", 0.0)) for r in rows)
    successes = sum(
        float(r.get("committed_shot_success_rate", 0.0)) * float(r.get("committed_contacts", 0.0))
        for r in rows
    )
    deltas = [
        float(r.get("delta_next_brick_after_committed_shot", 0.0))
        for r in rows if float(r.get("committed_contacts", 0.0)) > 0
    ]
    endgame_wins = sum(float(r.get("endgame_committed_shot_wins", 0.0)) for r in rows)
    return {
        "committed_shot_rate": shots / max(1.0, opp_steps),
        "committed_contact_rate": contacts / max(1.0, shots),
        "committed_shot_success_rate": successes / max(1.0, contacts),
        "delta_next_brick_after_committed_shot": float(np.mean(deltas)) if deltas else 0.0,
        "endgame_committed_shot_wins": endgame_wins,
    }


def parse_seed_spec(spec: str) -> list[int]:
    """Parse '0:49', '1000:1099', or '10,11,13'."""
    spec = spec.strip()
    if not spec:
        return []
    if ":" in spec:
        a, b = spec.split(":", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _obs_to_storable(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        parts = [np.asarray(obs["vector"], dtype=np.float32).ravel()]
        if "grid" in obs:
            parts.append(np.asarray(obs["grid"], dtype=np.float32).ravel())
        return np.concatenate(parts)
    return np.asarray(obs, dtype=np.float32).ravel()


@dataclass
class TeacherSample:
    obs: np.ndarray
    teacher_action: int
    followball_action: int
    advantage_score: float
    env_seed: int = -1
    step: int = -1
    teacher_offset: float = 0.0
    env_state: Optional[dict] = None


@dataclass
class TeacherDemo:
    samples: list[TeacherSample] = field(default_factory=list)
    path: str = ""

    @property
    def deviation_samples(self) -> list[TeacherSample]:
        return [
            s for s in self.samples
            if int(s.teacher_action) != int(s.followball_action)
        ]


def load_teacher_demo(path: str | Path) -> TeacherDemo:
    """Load CEM-MPC teacher npz; keep only better-than-FollowBall samples."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Teacher demo not found: {path}")

    data = np.load(path, allow_pickle=True)
    keys = set(data.files)

    def _arr(name: str, alt: str = "") -> Optional[np.ndarray]:
        if name in keys:
            return np.asarray(data[name])
        if alt and alt in keys:
            return np.asarray(data[alt])
        return None

    obs = _arr("obs_before", "obs")
    teacher = _arr("action", "teacher_action")
    if teacher is None:
        teacher = _arr("mpc_action")
    follow = _arr("followball_action")
    advantage = _arr("advantage_score")
    delta_bricks = _arr("delta_bricks")
    delta_next = _arr("delta_next_brick_steps")
    seeds = _arr("env_seed", "seed")
    steps = _arr("step")
    offsets = _arr("teacher_offset", "target_hit_offset")
    if offsets is None:
        offsets = _arr("desired_offset")
    states = _arr("env_state", "states")

    if obs is None or teacher is None:
        raise ValueError(f"Teacher demo missing obs/action keys: {path}")

    n = len(obs)
    if follow is None:
        follow = teacher.copy()
    samples: list[TeacherSample] = []

    for i in range(n):
        adv = float(advantage[i]) if advantage is not None else 0.0
        d_br = int(delta_bricks[i]) if delta_bricks is not None else 0
        d_next = int(delta_next[i]) if delta_next is not None else 0
        ta = int(teacher[i])
        fa = int(follow[i])

        beats = adv > 0 or d_br > 0 or (delta_next is not None and d_next < 0)
        if not beats and ta == fa:
            continue
        if not beats:
            continue

        env_state = None
        if states is not None:
            raw = states[i]
            if isinstance(raw, dict):
                env_state = raw
            elif isinstance(raw, np.ndarray) and raw.dtype == object:
                env_state = raw.item() if raw.ndim == 0 else raw

        off = float(offsets[i]) if offsets is not None else 0.0
        samples.append(TeacherSample(
            obs=_obs_to_storable(obs[i]),
            teacher_action=ta,
            followball_action=fa,
            advantage_score=adv,
            env_seed=int(seeds[i]) if seeds is not None else -1,
            step=int(steps[i]) if steps is not None else -1,
            teacher_offset=off,
            env_state=env_state,
        ))

    demo = TeacherDemo(samples=samples, path=str(path))
    if not samples:
        print(
            "WARNING: No better-than-FollowBall teacher samples found. "
            "Training will use opportunity rollouts only."
        )
    else:
        print(
            f"Teacher demo: {path} | {len(samples)} valid samples "
            f"({len(demo.deviation_samples)} deviations)"
        )
    return demo


def teacher_action_match_score(
    theta: np.ndarray,
    demo: TeacherDemo,
    sample_indices: Optional[np.ndarray] = None,
) -> dict:
    """Score how well theta matches CEM-MPC teacher actions on deviation states."""
    samples = demo.deviation_samples
    if sample_indices is not None:
        samples = [samples[int(i)] for i in sample_indices]
    match = deviate_wrong = committed = 0
    total = len(samples)
    if total == 0:
        return {"score": 0.0, "match_rate": 0.0, "match": 0, "total": 0}

    for s in samples:
        action, dbg = act_cem_aim_v3(
            s.obs,
            theta,
            return_debug=True,
            force_opportunity=True,
            teacher_imitate=True,
            state=CEMAimV3EpisodeState(),
        )
        if action == s.teacher_action:
            match += 1
        elif action == s.followball_action:
            deviate_wrong += 1
        if dbg.get("committed"):
            committed += 1

    rate = match / total
    score = match * 10.0 - deviate_wrong * 5.0 + committed * 2.0
    return {
        "score": score,
        "match_rate": rate,
        "match": match,
        "total": total,
        "committed": committed,
    }


def fit_theta_to_teacher_demo(
    demo: TeacherDemo,
    *,
    rng: np.random.Generator,
    iters: int = 400,
    population_size: int = 48,
    elite_frac: float = 0.2,
    sigma_init: float = 0.15,
    sigma_floor: float = 0.02,
    bc_subsample: int = 64,
) -> tuple[np.ndarray, dict]:
    """Behavior-cloning warm-start: fit 12-theta policy to CEM-MPC teacher deviations."""
    from cem_aim_policy import build_v3_prior_candidates

    if not demo.deviation_samples:
        theta = CEMAimV3Policy.prior_exact_follow().copy()
        return theta, {"match_rate": 0.0, "match": 0, "total": 0}

    offsets = [s.teacher_offset for s in demo.deviation_samples[:8] if s.teacher_offset]
    priors = build_v3_prior_candidates(teacher_offsets=offsets or None)
    mean = priors[0].copy()
    sigma = sigma_init
    best_theta = mean.copy()
    full_n = len(demo.deviation_samples)
    sub_n = min(bc_subsample, full_n)
    sub_idx = rng.choice(full_n, size=sub_n, replace=False)

    best = teacher_action_match_score(best_theta, demo, sub_idx)
    best_score = best["score"]
    full_best = teacher_action_match_score(best_theta, demo)

    print(
        f"Teacher BC | samples={full_n} (sub={sub_n}) priors={len(priors)} "
        f"start_match={full_best['match_rate']:.3f} ({full_best['match']}/{full_best['total']})"
    )

    for it in range(iters):
        pop: list[np.ndarray] = [p.copy() for p in priors[: min(len(priors), 8)]]
        n_rand = max(0, population_size - len(pop))
        if n_rand > 0:
            pop.extend(rng.normal(mean, sigma, size=(n_rand, NUM_PARAMS)))
        population = np.asarray(pop[:population_size], dtype=np.float64)

        scored: list[tuple[float, np.ndarray, dict]] = []
        for theta in population:
            metrics = teacher_action_match_score(theta, demo, sub_idx)
            scored.append((metrics["score"], theta.copy(), metrics))
        scored.sort(key=lambda x: x[0], reverse=True)

        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best_theta = scored[0][1].copy()
            best = scored[0][2]
            full_best = teacher_action_match_score(best_theta, demo)

        elite_n = max(1, int(population_size * elite_frac))
        elite = np.stack([x[1] for x in scored[:elite_n]])
        mean = elite.mean(axis=0)
        sigma = max(sigma_floor, sigma * 0.95)

        if (it + 1) % 50 == 0 or it + 1 == iters:
            print(
                f"  BC {it + 1}/{iters} full_match={full_best['match_rate']:.3f} "
                f"({full_best['match']}/{full_best['total']}) sigma={sigma:.3f}"
            )

    print(f"Teacher BC done | full_match_rate={full_best['match_rate']:.3f}")
    return best_theta, full_best


@dataclass
class Snapshot:
    env_state: dict
    obs: np.ndarray
    seed: int
    step: int
    followball_action: int
    remaining_bricks: int
    no_brick_broken_for: int
    brick_centroid_x: float
    predicted_landing_x: float
    ball_x: float
    ball_y: float
    ball_vx: float
    ball_vy: float
    paddle_x: float


def _isolated_bricks(feats: dict) -> bool:
    imbalance = abs(float(feats.get("right_frac", 0.5)) - float(feats.get("left_frac", 0.5)))
    upper = float(feats.get("upper_frac", 0.5))
    remaining = int(feats.get("remaining_bricks", 999))
    if imbalance >= 0.55 and remaining <= 15:
        return True
    if upper >= 0.6 and remaining <= 12:
        return True
    return False


def collect_opportunity_snapshots(
    seeds: list[int],
    *,
    layout: str = S.DEFAULT_LAYOUT,
    max_snapshots: int = 512,
    min_gap_steps: int = 200,
    stress_seeds: tuple[int, ...] = DEFAULT_STRESS_SEEDS,
    output: Optional[Path] = None,
) -> Path:
    """Collect FollowBall opportunity snapshots for pairwise training."""
    env = CatBreakEnv(config={"layout": layout})
    follow = FollowBallAgent()
    snapshots: list[Snapshot] = []
    last_saved_step: dict[int, int] = {}

    for seed in seeds:
        obs = env.reset(seed=seed)
        follow.reset()
        no_brick_streak = 0
        last_broken = 0
        step = 0
        env._no_brick_streak = 0

        while not env.done:
            feats = extract_cem_aim_features(obs, env.last_info, env)
            feats["no_brick_broken_for"] = no_brick_streak

            save_cond = (
                int(feats["remaining_bricks"]) <= 10
                or no_brick_streak >= 1000
                or int(feats["step_count"]) > 8000
                or seed in stress_seeds
                or _isolated_bricks(feats)
            )
            gap_ok = step - last_saved_step.get(seed, -10**9) >= min_gap_steps

            if save_cond and gap_ok and len(snapshots) < max_snapshots:
                fb_action = follow.act(obs, env.last_info, env=env)
                snapshots.append(Snapshot(
                    env_state=env.get_state_dict(),
                    obs=_obs_to_storable(obs),
                    seed=seed,
                    step=step,
                    followball_action=fb_action,
                    remaining_bricks=int(feats["remaining_bricks"]),
                    no_brick_broken_for=no_brick_streak,
                    brick_centroid_x=float(feats["brick_centroid_x"]),
                    predicted_landing_x=float(feats["predicted_landing_x"]),
                    ball_x=float(feats["ball_x"]),
                    ball_y=float(feats["ball_y"]),
                    ball_vx=float(feats["ball_vx"]),
                    ball_vy=float(feats["ball_vy"]),
                    paddle_x=float(feats["paddle_x"]),
                ))
                last_saved_step[seed] = step

            action = follow.act(obs, env.last_info, env=env)
            obs, _, done, info = env.step(action)
            if info["broken_bricks"] > last_broken:
                no_brick_streak = 0
                last_broken = info["broken_bricks"]
            else:
                no_brick_streak += 1
            env._no_brick_streak = no_brick_streak
            step += 1

    env.close()

    out_dir = S.CEM_AIM_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output or (out_dir / f"opportunity_snapshots_{ts}.npz")

    np.savez_compressed(
        out_path,
        env_state=np.array([s.env_state for s in snapshots], dtype=object),
        obs=np.stack([s.obs for s in snapshots]) if snapshots else np.zeros((0, 1)),
        seed=np.array([s.seed for s in snapshots], dtype=np.int64),
        step=np.array([s.step for s in snapshots], dtype=np.int64),
        followball_action=np.array([s.followball_action for s in snapshots], dtype=np.int64),
        remaining_bricks=np.array([s.remaining_bricks for s in snapshots], dtype=np.int64),
        no_brick_broken_for=np.array([s.no_brick_broken_for for s in snapshots], dtype=np.int64),
        brick_centroid_x=np.array([s.brick_centroid_x for s in snapshots], dtype=np.float32),
        predicted_landing_x=np.array([s.predicted_landing_x for s in snapshots], dtype=np.float32),
    )
    print(f"Collected {len(snapshots)} opportunity snapshots -> {out_path}")
    return out_path


def load_snapshots(path: str | Path) -> list[Snapshot]:
    data = np.load(path, allow_pickle=True)
    n = len(data["seed"])
    snaps = []
    for i in range(n):
        snaps.append(Snapshot(
            env_state=data["env_state"][i].item() if isinstance(data["env_state"][i], np.ndarray) else data["env_state"][i],
            obs=np.asarray(data["obs"][i]),
            seed=int(data["seed"][i]),
            step=int(data["step"][i]),
            followball_action=int(data["followball_action"][i]),
            remaining_bricks=int(data["remaining_bricks"][i]),
            no_brick_broken_for=int(data["no_brick_broken_for"][i]),
            brick_centroid_x=float(data["brick_centroid_x"][i]),
            predicted_landing_x=float(data["predicted_landing_x"][i]),
            ball_x=0.0,
            ball_y=0.0,
            ball_vx=0.0,
            ball_vy=0.0,
            paddle_x=0.0,
        ))
    return snaps


@dataclass
class RolloutResult:
    broken_bricks: int
    clear: int
    steps: int
    life_lost: bool
    steps_to_next_brick: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "broken_bricks": self.broken_bricks,
            "clear": self.clear,
            "steps": self.steps,
            "life_lost": self.life_lost,
            "steps_to_next_brick": self.steps_to_next_brick,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RolloutResult:
        return cls(
            broken_bricks=int(d["broken_bricks"]),
            clear=int(d["clear"]),
            steps=int(d["steps"]),
            life_lost=bool(d["life_lost"]),
            steps_to_next_brick=d.get("steps_to_next_brick"),
        )


DEFAULT_V3_SCORING: dict[str, float | int | bool] = {
    "stress_loss_penalty": 500.0,
    "non_stress_loss_penalty": 40.0,
    "endgame_bricks_threshold": 15,
    "endgame_speed_step_margin": 25,
    "anti_clone_snap_penalty": 250.0,
    "endgame_residual_bonus": 120.0,
    "use_endgame_aware_selection": True,
}


def build_v3_scoring(args) -> dict[str, float | int | bool]:
    """Scoring knobs for v3 CEM (endgame-aware, anti FollowBall-clone trap)."""
    return {
        "stress_loss_penalty": float(getattr(args, "stress_loss_penalty", 500.0)),
        "non_stress_loss_penalty": float(getattr(args, "non_stress_loss_penalty", 40.0)),
        "endgame_bricks_threshold": int(getattr(args, "endgame_bricks_threshold", 15)),
        "endgame_speed_step_margin": int(getattr(args, "endgame_speed_step_margin", 25)),
        "anti_clone_snap_penalty": float(getattr(args, "anti_clone_snap_penalty", 250.0)),
        "endgame_residual_bonus": float(getattr(args, "endgame_residual_bonus", 120.0)),
        "use_endgame_aware_selection": bool(
            getattr(args, "endgame_aware_selection", True)
        ),
    }


def endgame_snapshot_weight(snap: Snapshot, threshold: int) -> float:
    if snap.remaining_bricks > threshold:
        return 1.0
    if snap.remaining_bricks <= 4:
        return 3.0
    if snap.remaining_bricks <= 10:
        return 2.0
    return 1.5


def _score_snapshot_pair(
    c: RolloutResult,
    f: RolloutResult,
    snap: Snapshot,
    dbg: dict,
    scoring: dict[str, float | int | bool],
) -> tuple[float, bool, bool]:
    """Return (points, snap_win, endgame_win)."""
    threshold = int(scoring["endgame_bricks_threshold"])
    endgame = snap.remaining_bricks <= threshold
    w = endgame_snapshot_weight(snap, threshold)
    margin = int(scoring["endgame_speed_step_margin"])

    if c.life_lost and not f.life_lost:
        return -1000.0 * w, False, False
    if c.broken_bricks > f.broken_bricks:
        return 1000.0 * w, True, endgame
    if c.clear > f.clear:
        return 300.0 * w, True, endgame
    if (
        c.steps_to_next_brick is not None
        and f.steps_to_next_brick is not None
        and c.steps_to_next_brick < f.steps_to_next_brick - 5
    ):
        pts = (150.0 if endgame else 100.0) * w
        return pts, True, endgame
    if (
        endgame
        and c.broken_bricks == f.broken_bricks
        and c.clear >= f.clear
        and not c.life_lost
        and c.steps + margin < f.steps
    ):
        step_adv = f.steps - c.steps
        pts = 60.0 * w * min(step_adv / max(float(margin), 1.0), 3.0)
        return pts, True, True
    if c.broken_bricks < f.broken_bricks:
        return -300.0 * w, False, False
    if dbg.get("committed") and not c.life_lost:
        bonus = float(scoring["endgame_residual_bonus"]) if endgame else 100.0
        return bonus * w, False, False
    return 0.0, False, False


@dataclass
class TrainingCaches:
    """Precomputed FollowBall baselines reused across all theta evaluations."""

    follow_train: dict[int, dict]
    follow_val: dict[int, dict]
    follow_snapshot: list[RolloutResult]
    layout: str = S.DEFAULT_LAYOUT

    @classmethod
    def build(
        cls,
        env: CatBreakEnv,
        *,
        train_seeds: list[int],
        val_seeds: list[int],
        snapshots: list[Snapshot],
        snapshot_horizon: int,
        layout: str,
    ) -> TrainingCaches:
        follow = FollowAgentAdapter()
        print(
            f"Precomputing FollowBall baselines "
            f"({len(train_seeds)} train + {len(val_seeds)} val + {len(snapshots)} snapshots) ..."
        )
        t0 = time.perf_counter()
        follow_train = {s: run_episode_fast(env, follow, s) for s in train_seeds}
        follow_val = {s: run_episode_fast(env, follow, s) for s in val_seeds}
        follow_snapshot = [
            rollout_from_state(env, snap.env_state, follow, snapshot_horizon, fast=True)
            for snap in snapshots
        ]
        print(f"FollowBall cache ready in {time.perf_counter() - t0:.1f}s")
        return cls(
            follow_train=follow_train,
            follow_val=follow_val,
            follow_snapshot=follow_snapshot,
            layout=layout,
        )


def rollout_from_state(
    env: CatBreakEnv,
    state: dict,
    agent,
    horizon: int,
    *,
    fast: bool = True,
) -> RolloutResult:
    saved = env.get_state_dict()
    env.set_state_dict(state)
    start_broken = env.broken_bricks
    start_lives = env.lives
    steps_to_brick: Optional[int] = None
    try:
        if hasattr(agent, "reset"):
            agent.reset()
        obs = env.get_obs()
        steps = 0
        while not env.done and steps < horizon:
            action = agent.act(obs, env.last_info, env=env)
            if fast:
                _, done, info = env.step_fast(action)
            else:
                obs, _, done, info = env.step(action)
            if hasattr(agent, "note_step"):
                agent.note_step(info, env)
            steps += 1
            if info.get("bricks_broken_this_step", 0) and steps_to_brick is None:
                steps_to_brick = steps
            if fast and not done:
                obs = env.get_obs()
        return RolloutResult(
            broken_bricks=env.broken_bricks - start_broken,
            clear=int(env.terminal_reason == "cleared"),
            steps=steps,
            life_lost=env.lives < start_lives,
            steps_to_next_brick=steps_to_brick,
        )
    finally:
        env.set_state_dict(saved)


def worse_than_followball(c: RolloutResult, f: RolloutResult) -> bool:
    if c.clear < f.clear:
        return True
    if c.broken_bricks < f.broken_bricks:
        return True
    if c.broken_bricks == f.broken_bricks and c.steps > f.steps + STEP_TOLERANCE:
        return True
    if c.life_lost and not f.life_lost:
        return True
    return False


def better_than_followball(c: RolloutResult, f: RolloutResult) -> bool:
    if c.broken_bricks > f.broken_bricks:
        return True
    if c.clear > f.clear:
        return True
    if (
        c.broken_bricks == f.broken_bricks
        and c.clear >= f.clear
        and c.steps < f.steps - STEP_TOLERANCE
    ):
        return True
    if (
        c.broken_bricks == f.broken_bricks
        and c.steps_to_next_brick is not None
        and f.steps_to_next_brick is not None
        and c.steps_to_next_brick < f.steps_to_next_brick
    ):
        return True
    return False


class V3AgentAdapter:
    name = "CEM-Aim-v3"

    def __init__(
        self,
        theta: np.ndarray,
        *,
        macro_config: Optional[ShotMacroConfig] = None,
        offset_floor: float = 0.05,
    ) -> None:
        self.policy = CEMAimV3Policy(
            theta, macro_config=macro_config, offset_floor=offset_floor
        )

    def reset(self, seed=None) -> None:
        self.policy.reset_episode(seed)

    def act(self, obs, info=None, env=None) -> int:
        return self.policy.act(obs, info=info, env=env)

    def note_step(self, info, env) -> None:
        self.policy.note_step_after_env_step(info, env)

    def commitment_summary(self) -> dict[str, float]:
        return self.policy.commitment_summary()


class FollowAgentAdapter:
    name = "FollowBall"

    def __init__(self) -> None:
        self._fb = FollowBallAgent()

    def reset(self, seed=None) -> None:
        pass

    def act(self, obs, info=None, env=None) -> int:
        return self._fb.act(obs, info, env=env)


def evaluate_theta_v3(
    theta: np.ndarray,
    env: CatBreakEnv,
    *,
    train_seeds: list[int],
    val_seeds: list[int],
    snapshots: list[Snapshot],
    val_snapshots: list[Snapshot],
    teacher: Optional[TeacherDemo],
    snapshot_horizon: int,
    snapshots_per_theta: int,
    stress_seeds: tuple[int, ...],
    teacher_weight: float,
    full_episode_weight: float,
    snapshot_weight: float,
    rng: np.random.Generator,
    caches: Optional[TrainingCaches] = None,
    snapshot_indices: Optional[list[int]] = None,
    behavior_sample_seeds: int = 1,
    skip_behavior_metrics: bool = False,
    scoring: Optional[dict[str, float | int | bool]] = None,
) -> dict:
    """Evaluate one theta: full episodes + snapshot pairwise + optional teacher."""
    scoring_cfg = {**DEFAULT_V3_SCORING, **(scoring or {})}
    stress_set = set(stress_seeds)
    agent = V3AgentAdapter(theta)

    def _cand_episodes(seeds: list[int]) -> list[dict]:
        return [run_episode_fast(env, agent, s) for s in seeds]

    train_c = _cand_episodes(train_seeds)
    if caches and caches.follow_train:
        train_f = [caches.follow_train[s] for s in train_seeds]
    else:
        follow = FollowAgentAdapter()
        train_f = [run_episode_fast(env, follow, s) for s in train_seeds]

    if val_seeds:
        val_c = _cand_episodes(val_seeds)
        if caches and caches.follow_val:
            val_f = [caches.follow_val[s] for s in val_seeds]
        else:
            follow = FollowAgentAdapter()
            val_f = [run_episode_fast(env, follow, s) for s in val_seeds]
    else:
        val_c, val_f = [], []

    def _full_losses(c_rows, f_rows) -> tuple[int, int, int]:
        wins = ties = losses = 0
        for c, f in zip(c_rows, f_rows):
            cr = RolloutResult(c["broken_bricks"], c["clear"], c["steps"], c["lives"] < S.INITIAL_LIVES)
            fr = RolloutResult(f["broken_bricks"], f["clear"], f["steps"], False)
            if worse_than_followball(cr, fr):
                losses += 1
            elif better_than_followball(cr, fr):
                wins += 1
            else:
                ties += 1
        return wins, ties, losses

    tr_w, tr_t, tr_l = _full_losses(train_c, train_f)
    va_w, va_t, va_l = _full_losses(val_c, val_f) if val_c else (0, 0, 0)

    stress_loss = 0
    non_stress_loss = 0
    seed13_delta_bricks = 0
    seed13_delta_steps = 0
    for c, f in zip(train_c, train_f):
        seed = c["env_seed"]
        cr = RolloutResult(c["broken_bricks"], c["clear"], c["steps"], c["lives"] < S.INITIAL_LIVES)
        fr = RolloutResult(f["broken_bricks"], f["clear"], f["steps"], False)
        if worse_than_followball(cr, fr):
            if seed in stress_set:
                stress_loss += 1
            else:
                non_stress_loss += 1
        if seed == 13:
            seed13_delta_bricks = c["broken_bricks"] - f["broken_bricks"]
            seed13_delta_steps = c["steps"] - f["steps"]

    # Snapshot pairwise
    if snapshot_indices is not None:
        pool_idx = snapshot_indices
    elif len(snapshots) > snapshots_per_theta:
        pool_idx = rng.choice(len(snapshots), size=snapshots_per_theta, replace=False).tolist()
    else:
        pool_idx = list(range(len(snapshots)))
    pool = [snapshots[i] for i in pool_idx]

    snap_wins = snap_losses = endgame_snap_wins = 0
    snap_delta_bricks: list[float] = []
    committed_acts = opp_count = 0
    snap_score = 0.0

    for local_i, snap in enumerate(pool):
        c = rollout_from_state(env, snap.env_state, agent, snapshot_horizon, fast=True)
        if caches and caches.follow_snapshot and pool_idx[local_i] < len(caches.follow_snapshot):
            f = caches.follow_snapshot[pool_idx[local_i]]
        else:
            f = rollout_from_state(env, snap.env_state, FollowAgentAdapter(), snapshot_horizon, fast=True)
        snap_delta_bricks.append(c.broken_bricks - f.broken_bricks)

        env.set_state_dict(snap.env_state)
        obs = env.get_obs()
        snap_policy = CEMAimV3Policy(theta)
        _, dbg = snap_policy.act_debug(obs, env=env)
        if dbg["opportunity"]:
            opp_count += 1
            if dbg.get("committed"):
                committed_acts += 1

        pts, win, end_win = _score_snapshot_pair(c, f, snap, dbg, scoring_cfg)
        snap_score += pts
        if win:
            snap_wins += 1
        elif pts < 0:
            snap_losses += 1
        if end_win:
            endgame_snap_wins += 1

    if opp_count > 0 and committed_acts == 0 and snap_wins == 0:
        snap_score -= float(scoring_cfg["anti_clone_snap_penalty"])

    # Teacher imitation
    teacher_match = teacher_dev = 0
    teacher_score = 0.0
    if teacher and teacher_weight > 0:
        teacher_cap = int(scoring_cfg.get("teacher_max_samples", 248))
        for samp in teacher.deviation_samples[:teacher_cap]:
            env_state = samp.env_state
            if env_state is not None:
                env.set_state_dict(env_state)
                obs = env.get_obs()
            else:
                obs = samp.obs
            action, dbg = act_cem_aim_v3(
                obs,
                theta,
                env=env if env_state else None,
                return_debug=True,
                force_opportunity=True,
                teacher_imitate=True,
                state=CEMAimV3EpisodeState(),
            )
            teacher_dev += 1
            if action == samp.teacher_action:
                teacher_match += 1
                teacher_score += 300
            if dbg.get("committed"):
                teacher_score += 150
            if samp.teacher_offset != 0 and dbg["desired_offset"] != 0:
                if np.sign(dbg["desired_offset"]) == np.sign(samp.teacher_offset):
                    teacher_score += 100
            if action == samp.followball_action:
                teacher_score -= 50

        if teacher_dev > 0 and teacher_match == 0:
            teacher_score -= 200

    # Behavior rates (sample a few train seeds only)
    global_dev = opp_dev = residual_rate = unsafe_fb = 0.0
    commit_metrics = _aggregate_commitment_metrics(train_c)
    if not skip_behavior_metrics and train_seeds:
        total_steps = deviated = opp_steps = committed_count = unsafe_count = 0
        policy = CEMAimV3Policy(theta)
        for seed in train_seeds[: max(1, behavior_sample_seeds)]:
            obs = env.reset(seed=seed)
            policy.reset_episode()
            while not env.done:
                action, dbg = policy.act_debug(obs, env=env)
                total_steps += 1
                if dbg["deviated_from_followball"]:
                    deviated += 1
                if dbg["opportunity"]:
                    opp_steps += 1
                    if dbg["deviated_from_followball"]:
                        opp_dev += 1
                    if dbg.get("committed"):
                        committed_count += 1
                if dbg["unsafe"] and dbg["final_action"] == dbg["follow_action"]:
                    unsafe_count += 1
                _, _, done, info = env.step(action)
                policy.note_step_after_env_step(info, env)
                obs = env.get_obs() if not done else obs

        global_dev = deviated / max(1, total_steps)
        opportunity_dev = opp_dev / max(1, opp_steps)
        residual_rate = committed_count / max(1, opp_steps)
        unsafe_fb = unsafe_count / max(1, total_steps)
    else:
        opportunity_dev = 0.0
        residual_rate = committed_acts / max(1, opp_count)
    teacher_match_rate = teacher_match / max(1, teacher_dev)

    train_summary = summarize_rows(train_c)
    val_summary = summarize_rows(val_c) if val_c else {}

    total_score = (
        full_episode_weight * (
            tr_w * 100.0
            - stress_loss * float(scoring_cfg["stress_loss_penalty"])
            - non_stress_loss * float(scoring_cfg["non_stress_loss_penalty"])
        )
        + snapshot_weight * snap_score
        + teacher_weight * teacher_score
    )

    global_penalty = max(0.0, global_dev - 0.05) * 1000.0

    if scoring_cfg.get("use_endgame_aware_selection", True):
        selection_key = (
            endgame_snap_wins,
            commit_metrics["endgame_committed_shot_wins"],
            snap_wins,
            commit_metrics["committed_shot_success_rate"],
            commit_metrics["committed_shot_rate"],
            float(np.mean(snap_delta_bricks)) if snap_delta_bricks else 0.0,
            tr_w,
            teacher_match_rate,
            total_score / 1000.0,
            -stress_loss,
            commit_metrics["committed_contact_rate"],
            -non_stress_loss,
            train_summary.get("clear_rate", 0.0),
            train_summary.get("avg_broken_bricks", 0.0),
            -train_summary.get("avg_steps", 0.0) / 10000.0,
            -global_penalty,
        )
    else:
        selection_key = (
            -tr_l,
            -stress_loss,
            snap_wins,
            float(np.mean(snap_delta_bricks)) if snap_delta_bricks else 0.0,
            teacher_match_rate,
            residual_rate,
            tr_w,
            train_summary.get("clear_rate", 0.0),
            train_summary.get("avg_broken_bricks", 0.0),
            -train_summary.get("avg_steps", 0.0),
            -global_penalty,
        )

    return {
        "total_score": total_score,
        "selection_key": selection_key,
        "train_wins": tr_w,
        "train_ties": tr_t,
        "train_losses": tr_l,
        "val_wins": va_w,
        "val_ties": va_t,
        "val_losses": va_l,
        "stress_seed_loss_count": stress_loss,
        "non_stress_loss_count": non_stress_loss,
        "seed13_delta_bricks": seed13_delta_bricks,
        "seed13_delta_steps": seed13_delta_steps,
        "snapshot_win_rate": snap_wins / max(1, len(pool)),
        "endgame_snap_wins": endgame_snap_wins,
        "snapshot_mean_delta_bricks": float(np.mean(snap_delta_bricks)) if snap_delta_bricks else 0.0,
        "global_deviation_rate": global_dev,
        "opportunity_deviation_rate": opportunity_dev,
        "residual_activation_rate": residual_rate,
        "unsafe_fallback_rate": unsafe_fb,
        "teacher_action_match_rate": teacher_match_rate,
        **commit_metrics,
        "mean_broken_bricks": train_summary.get("avg_broken_bricks", 0.0),
        "clear_rate": train_summary.get("clear_rate", 0.0),
        "mean_steps": train_summary.get("avg_steps", 0.0),
        "val_mean_broken_bricks": val_summary.get("avg_broken_bricks", 0.0),
        "val_clear_rate": val_summary.get("clear_rate", 0.0),
    }


# ---------------------------------------------------------------------------
# Parallel theta evaluation (process pool with shared FollowBall cache)
# ---------------------------------------------------------------------------

_POOL_ENV: Optional[CatBreakEnv] = None
_POOL_CACHES: Optional[TrainingCaches] = None
_POOL_SNAPSHOTS: list[Snapshot] = []
_POOL_CFG: dict = {}


def _init_eval_worker(layout: str, caches: TrainingCaches, snapshots: list[Snapshot], cfg: dict) -> None:
    global _POOL_ENV, _POOL_CACHES, _POOL_SNAPSHOTS, _POOL_CFG
    _POOL_ENV = CatBreakEnv(config={"layout": layout})
    _POOL_CACHES = caches
    _POOL_SNAPSHOTS = snapshots
    _POOL_CFG = cfg


def _evaluate_theta_v3_worker(payload: tuple[np.ndarray, list[int]]) -> dict:
    theta, snap_indices = payload
    assert _POOL_ENV is not None and _POOL_CACHES is not None
    cfg = _POOL_CFG
    return evaluate_theta_v3(
        theta,
        _POOL_ENV,
        train_seeds=cfg["train_seeds"],
        val_seeds=cfg["val_seeds"],
        snapshots=_POOL_SNAPSHOTS,
        val_snapshots=[],
        teacher=cfg.get("teacher"),
        snapshot_horizon=cfg["snapshot_horizon"],
        snapshots_per_theta=cfg["snapshots_per_theta"],
        stress_seeds=tuple(cfg["stress_seeds"]),
        teacher_weight=cfg["teacher_weight"],
        full_episode_weight=cfg["full_episode_weight"],
        snapshot_weight=cfg["snapshot_weight"],
        rng=np.random.default_rng(0),
        caches=_POOL_CACHES,
        snapshot_indices=snap_indices,
        behavior_sample_seeds=cfg.get("behavior_sample_seeds", 1),
        skip_behavior_metrics=cfg.get("skip_behavior_metrics", False),
        scoring=cfg.get("scoring"),
    )


def evaluate_population_v3_parallel(
    population: np.ndarray,
    *,
    layout: str,
    caches: TrainingCaches,
    snapshots: list[Snapshot],
    snapshot_indices: list[int],
    cfg: dict,
    workers: int,
) -> list[dict]:
    if cfg.get("teacher") is not None and workers > 1:
        workers = 1
    if workers <= 1:
        env = CatBreakEnv(config={"layout": layout})
        try:
            return [
                evaluate_theta_v3(
                    population[i], env,
                    train_seeds=cfg["train_seeds"],
                    val_seeds=cfg["val_seeds"],
                    snapshots=snapshots,
                    val_snapshots=[],
                    teacher=cfg.get("teacher"),
                    snapshot_horizon=cfg["snapshot_horizon"],
                    snapshots_per_theta=cfg["snapshots_per_theta"],
                    stress_seeds=tuple(cfg["stress_seeds"]),
                    teacher_weight=cfg["teacher_weight"],
                    full_episode_weight=cfg["full_episode_weight"],
                    snapshot_weight=cfg["snapshot_weight"],
                    rng=np.random.default_rng(cfg.get("seed", 0) + i),
                    caches=caches,
                    snapshot_indices=snapshot_indices,
                    behavior_sample_seeds=cfg.get("behavior_sample_seeds", 1),
                    skip_behavior_metrics=cfg.get("skip_behavior_metrics", False),
                    scoring=cfg.get("scoring"),
                )
                for i in range(len(population))
            ]
        finally:
            env.close()

    payloads = [(population[i].copy(), snapshot_indices) for i in range(len(population))]
    results: list[Optional[dict]] = [None] * len(payloads)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_eval_worker,
        initargs=(layout, caches, snapshots, cfg),
    ) as pool:
        futures = {pool.submit(_evaluate_theta_v3_worker, p): i for i, p in enumerate(payloads)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [r for r in results if r is not None]


def sanity_followball_reproduction(
    model_path: Path,
    seeds: list[int],
    layout: str = S.DEFAULT_LAYOUT,
) -> bool:
    """Return True if v3 with disable_residual matches FollowBall."""
    data = np.load(model_path, allow_pickle=True)
    theta = np.asarray(data["theta"], dtype=np.float64)
    env = CatBreakEnv(config={"layout": layout})
    follow = FollowBallAgent()
    mismatches = 0
    total = 0
    for seed in seeds:
        obs = env.reset(seed=seed)
        while not env.done:
            fb = follow.act(obs, env.last_info, env=env)
            aim, _ = act_cem_aim_v3(
                obs, theta, env=env, return_debug=True, disable_residual=True
            )
            if aim != fb:
                mismatches += 1
            total += 1
            obs, _, done, _ = env.step(fb)
    env.close()
    ok = mismatches == 0
    if not ok:
        print(
            f"ERROR: CEM-Aim wrapper does not reproduce FollowBall "
            f"({mismatches}/{total} mismatches). Do not train yet."
        )
    else:
        print(f"FollowBall reproduction OK ({total} steps, {len(seeds)} seeds)")
    return ok


def sanity_clone_restore(env: CatBreakEnv, seed: int = 0, n: int = 50, k: int = 30) -> bool:
    """Clone/restore determinism test."""
    actions = [S.ACTION_LEFT, S.ACTION_STAY, S.ACTION_RIGHT] * 20
    env.reset(seed=seed)
    for a in actions[:n]:
        if env.done:
            break
        env.step(a)

    state = env.get_state_dict()

    def roll(env_ref, acts):
        rewards, infos, obs_list = [], [], []
        for a in acts[:k]:
            if env_ref.done:
                break
            obs, r, _, info = env_ref.step(a)
            rewards.append(r)
            infos.append(dict(info))
            obs_list.append(np.asarray(obs).copy())
        return obs_list, rewards, infos, env_ref.done

    env_a = env.clone()
    env_b = env.clone()
    env_b.set_state_dict(state)

    o1, r1, i1, d1 = roll(env_a, actions[n:n + k])
    o2, r2, i2, d2 = roll(env_b, actions[n:n + k])

    ok = (
        len(o1) == len(o2)
        and all(np.allclose(a, b) for a, b in zip(o1, o2))
        and r1 == r2
        and d1 == d2
    )
    env_a.close()
    env_b.close()
    if ok:
        print("Clone/restore determinism OK")
    else:
        print("ERROR: clone/restore mismatch")
    return ok


def train_cem_aim_v3(args) -> Path:
    """CEM search for v3 residual option policy."""
    from cem_aim_policy import build_v3_prior_candidates, CEMAimV3Policy

    run_dir = Path(args.save_dir) if args.save_dir else S.CEM_AIM_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_seeds = parse_seed_spec(args.train_seeds)
    val_seeds = parse_seed_spec(args.val_seeds) if args.val_seeds else []
    stress_seeds = tuple(parse_seed_spec(args.stress_seeds)) if args.stress_seeds else DEFAULT_STRESS_SEEDS

    snapshots: list[Snapshot] = []
    val_snapshots: list[Snapshot] = []
    if args.opportunity_snapshots:
        snapshots = load_snapshots(args.opportunity_snapshots)
        print(f"Loaded {len(snapshots)} train snapshots from {args.opportunity_snapshots}")
    if getattr(args, "val_opportunity_snapshots", None):
        val_snapshots = load_snapshots(args.val_opportunity_snapshots)

    teacher: Optional[TeacherDemo] = None
    teacher_offsets: list[float] = []
    teacher_weight = float(args.teacher_imitation_weight)
    if args.teacher_demo:
        teacher = load_teacher_demo(args.teacher_demo)
        teacher_offsets = [s.teacher_offset for s in teacher.deviation_samples[:5] if s.teacher_offset]
        if not teacher.samples:
            teacher_weight = 0.0
    elif teacher_weight > 0 and not args.teacher_demo:
        teacher_weight = 0.0

    rng = np.random.default_rng(args.seed)

    bc_iters = int(getattr(args, "teacher_bc_iters", 0))
    bc_theta: Optional[np.ndarray] = None
    if teacher and teacher.deviation_samples and bc_iters > 0:
        bc_theta, bc_metrics = fit_theta_to_teacher_demo(
            teacher,
            rng=rng,
            iters=bc_iters,
            population_size=min(48, args.population_size),
        )
        bc_path = run_dir / "cem_aim_teacher_bc_init.npz"
        CEMAimV3Policy(bc_theta).save(str(bc_path), metrics=bc_metrics)
        print(f"Saved teacher BC init -> {bc_path}")
    env = CatBreakEnv(config={"layout": args.layout})
    workers = default_parallel_workers(getattr(args, "workers", None))

    resume_theta = None
    resume_macro = None
    resume_offset_floor = 0.05
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_path}\n"
                f"Regenerate with train_cem_aim.py or see "
                f"runs/cem_aim/v3_win_hunt_seed1/RESTORE.md"
            )
        ckpt = np.load(resume_path, allow_pickle=True)
        resume_theta = np.asarray(ckpt["theta"], dtype=np.float64)
        if "offset_floor" in ckpt.files:
            resume_offset_floor = float(ckpt["offset_floor"])
        if "macro_config" in ckpt.files:
            from cem_aim_policy import ShotMacroConfig
            import json as _json
            raw = ckpt["macro_config"]
            if isinstance(raw, np.ndarray):
                raw = raw.item()
            if isinstance(raw, (bytes, str)):
                resume_macro = ShotMacroConfig.from_dict(_json.loads(raw))

    use_bc_init = bc_theta is not None and not getattr(args, "no_teacher_bc_init", False)
    if use_bc_init:
        mean = bc_theta.copy()
    else:
        mean = CEMAimV3Policy.prior_exact_follow().copy()
        if resume_theta is not None:
            mean = resume_theta.copy()
    sigma = args.sigma_init
    best_theta = mean.copy()
    best_metrics: Optional[dict] = None
    best_key: tuple = (-1,)
    val_best_theta = best_theta.copy()
    val_best_metrics: Optional[dict] = None
    val_best_key: tuple = (-1,)

    exact_prior_path = run_dir / "cem_aim_exact_follow_prior.npz"
    CEMAimV3Policy(CEMAimV3Policy.prior_exact_follow()).save(str(exact_prior_path))

    log_rows: list[dict] = []
    behavior_sample_seeds = int(getattr(args, "behavior_sample_seeds", 1))
    skip_behavior = bool(getattr(args, "skip_behavior_metrics", False))

    caches = TrainingCaches.build(
        env,
        train_seeds=train_seeds,
        val_seeds=val_seeds if args.select_by == "val" else [],
        snapshots=snapshots,
        snapshot_horizon=args.snapshot_rollout_horizon,
        layout=args.layout,
    )

    eval_cfg = {
        "train_seeds": train_seeds,
        "val_seeds": [] if args.select_by == "val" else val_seeds,
        "teacher": teacher,
        "snapshot_horizon": args.snapshot_rollout_horizon,
        "snapshots_per_theta": args.snapshots_per_theta,
        "stress_seeds": list(stress_seeds),
        "teacher_weight": teacher_weight,
        "full_episode_weight": args.full_episode_weight,
        "snapshot_weight": args.snapshot_weight,
        "behavior_sample_seeds": behavior_sample_seeds,
        "skip_behavior_metrics": skip_behavior,
        "seed": args.seed,
        "scoring": {
            **build_v3_scoring(args),
            "teacher_max_samples": int(getattr(args, "teacher_max_samples", 248)),
        },
    }

    print(
        f"CEM-Aim v3 | train_seeds={len(train_seeds)} val_seeds={len(val_seeds)} "
        f"snapshots={len(snapshots)} teacher={'yes' if teacher and teacher.samples else 'no'} "
        f"workers={workers}"
    )

    for gen in range(args.generations):
        t0 = time.perf_counter()
        priors = build_v3_prior_candidates(
            previous_best=best_theta,
            resume_theta=resume_theta if gen == 0 and not use_bc_init else None,
            teacher_offsets=teacher_offsets,
        )
        n_rand = max(0, args.population_size - len(priors))
        pop = list(priors)
        pop.extend(rng.normal(mean, sigma, size=(n_rand, NUM_PARAMS)))
        population = np.asarray(pop[: args.population_size], dtype=np.float64)

        n_pool = min(args.snapshots_per_theta, len(snapshots))
        if len(snapshots) > n_pool:
            snap_indices = rng.choice(len(snapshots), size=n_pool, replace=False).tolist()
        else:
            snap_indices = list(range(len(snapshots)))

        metrics_list = evaluate_population_v3_parallel(
            population,
            layout=args.layout,
            caches=caches,
            snapshots=snapshots,
            snapshot_indices=snap_indices,
            cfg=eval_cfg,
            workers=workers,
        )

        order = sorted(range(len(metrics_list)), key=lambda i: metrics_list[i]["selection_key"], reverse=True)
        gen_best_idx = order[0]
        gen_best_theta = population[gen_best_idx].copy()
        gen_best = metrics_list[gen_best_idx]

        if gen_best["selection_key"] > best_key:
            best_key = gen_best["selection_key"]
            best_theta = gen_best_theta
            best_metrics = gen_best

        elite_n = max(1, int(args.population_size * args.elite_frac))
        elite = population[order[:elite_n]]
        mean = (1 - args.smoothing) * mean + args.smoothing * elite.mean(axis=0)
        sigma = max(args.sigma_floor, sigma * (1 - args.smoothing * 0.5))

        CEMAimV3Policy(mean).save(str(run_dir / "cem_aim_last_mean.npz"), metrics={"generation": gen})
        CEMAimV3Policy(gen_best_theta).save(str(run_dir / "cem_aim_train_best.npz"), metrics=gen_best)

        log_rows.append({
            "generation": gen,
            "sigma": sigma,
            "train_losses": gen_best["train_losses"],
            "stress_losses": gen_best["stress_seed_loss_count"],
            "non_stress_losses": gen_best.get("non_stress_loss_count", 0),
            "snapshot_win_rate": gen_best["snapshot_win_rate"],
            "endgame_snap_wins": gen_best.get("endgame_snap_wins", 0),
            "committed_shot_rate": gen_best.get("committed_shot_rate", 0.0),
            "committed_contact_rate": gen_best.get("committed_contact_rate", 0.0),
            "residual_activation_rate": gen_best["residual_activation_rate"],
            "global_deviation_rate": gen_best["global_deviation_rate"],
            "wall_clock_sec": time.perf_counter() - t0,
        })
        print(
            f"[gen {gen + 1}/{args.generations}] stress={gen_best['stress_seed_loss_count']} "
            f"non_stress={gen_best.get('non_stress_loss_count', 0)} "
            f"snap_win={gen_best['snapshot_win_rate']:.2f} "
            f"endgame={gen_best.get('endgame_snap_wins', 0)} "
            f"commit={gen_best.get('committed_shot_rate', 0.0):.2f} "
            f"contact={gen_best.get('committed_contact_rate', 0.0):.2f} "
            f"residual={gen_best['residual_activation_rate']:.2f} "
            f"global_dev={gen_best['global_deviation_rate']:.3f} sigma={sigma:.3f}"
        )

        if val_seeds and args.select_by == "val":
            top_k = min(args.save_top_k, len(order))
            val_eval_cfg = {**eval_cfg, "val_seeds": val_seeds, "skip_behavior_metrics": True}
            top_metrics = []
            for idx in order[:top_k]:
                vm = evaluate_theta_v3(
                    population[idx], env,
                    train_seeds=[],
                    val_seeds=val_seeds,
                    snapshots=snapshots,
                    val_snapshots=val_snapshots,
                    teacher=teacher,
                    snapshot_horizon=args.snapshot_rollout_horizon,
                    snapshots_per_theta=min(32, args.snapshots_per_theta),
                    stress_seeds=stress_seeds,
                    teacher_weight=teacher_weight,
                    full_episode_weight=args.full_episode_weight,
                    snapshot_weight=args.snapshot_weight,
                    rng=rng,
                    caches=caches,
                    snapshot_indices=snap_indices[: min(32, len(snap_indices))],
                    behavior_sample_seeds=0,
                    skip_behavior_metrics=True,
                    scoring=eval_cfg.get("scoring"),
                )
                top_metrics.append((idx, vm))

            val_order = sorted(top_metrics, key=lambda x: x[1]["selection_key"], reverse=True)
            vidx, vmetrics = val_order[0]
            vtheta = population[vidx].copy()

            passes = (
                vmetrics["stress_seed_loss_count"] == 0
                and vmetrics["global_deviation_rate"] <= args.max_global_deviation_rate
                and (
                    vmetrics.get("committed_shot_rate", 0.0) >= args.min_opportunity_activation_rate
                    or vmetrics.get("committed_contact_rate", 0.0) > 0.0
                    or vmetrics.get("endgame_snap_wins", 0) >= 1
                    or vmetrics.get("endgame_committed_shot_wins", 0) >= 1
                    or vmetrics.get("snapshot_win_rate", 0.0) > 0.0
                )
            )
            if teacher and teacher.deviation_samples:
                passes = passes and vmetrics["teacher_action_match_rate"] >= 0.10

            if passes and vmetrics["selection_key"] > val_best_key:
                val_best_key = vmetrics["selection_key"]
                val_best_theta = vtheta
                val_best_metrics = vmetrics
                CEMAimV3Policy(val_best_theta).save(str(run_dir / "cem_aim_val_best.npz"), metrics=vmetrics)

    env.close()

    with (run_dir / "cem_aim_training_log.csv").open("w", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()) if log_rows else ["generation"])
        w.writeheader()
        w.writerows(log_rows)

    meta = {
        "policy_version": POLICY_VERSION_V3,
        "objective": args.objective,
        "train_seeds": train_seeds,
        "val_seeds": val_seeds,
        "stress_seeds": list(stress_seeds),
        "teacher_demo": args.teacher_demo,
        "best_metrics": best_metrics,
        "timestamp": datetime.now().isoformat(),
    }
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"Training done -> {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# Deterministic shot-macro grid search (metric rescue, no CEM training)
# ---------------------------------------------------------------------------

WEAK_SEEDS = (4, 6, 13, 14)
SAFETY_SEEDS = (10, 11, 19)
GRID_REMAINING_GATES = (1, 2, 3, 4, 6, 10, 15)
GRID_STEP_GATES = (6000, 7000, 8000, 9000)
GRID_NO_BRICK_GATES = (400, 800, 1200, 1600, 2400)
GRID_OFFSET_MAGS = (0.10, 0.18, 0.25, 0.35, 0.45, 0.55)
GRID_SIDE_RULES = SIDE_RULES
GRID_TARGET_BASES = TARGET_BASES


def iter_shot_macro_grid(quick: bool = False) -> list[ShotMacroConfig]:
    import itertools

    if quick:
        rem_vals = (3, 10)
        step_vals = (8000,)
        nob_vals = (1200,)
        off_vals = (0.18, 0.35)
        side_vals = ("toward_centroid", "left")
        base_vals = ("predicted_landing",)
    else:
        rem_vals = GRID_REMAINING_GATES
        step_vals = GRID_STEP_GATES
        nob_vals = GRID_NO_BRICK_GATES
        off_vals = GRID_OFFSET_MAGS
        side_vals = GRID_SIDE_RULES
        base_vals = GRID_TARGET_BASES

    configs: list[ShotMacroConfig] = []
    for rem, step, nob, off, side, base in itertools.product(
        rem_vals, step_vals, nob_vals, off_vals, side_vals, base_vals,
    ):
        floor = max(f for f in (0.08, 0.12, 0.18, 0.25) if f <= off + 1e-9)
        configs.append(ShotMacroConfig(
            remaining_gate=int(rem),
            step_gate=int(step),
            no_brick_gate=int(nob),
            offset_mag=float(off),
            offset_floor=float(floor),
            side_rule=str(side),
            target_base=str(base),
        ))
    return configs


def _snapshot_weight(snap: Snapshot) -> float:
    w = 1.0
    if snap.seed in WEAK_SEEDS:
        w *= 3.0
    if snap.remaining_bricks <= 15:
        w *= 2.0
    return w


def _macro_agent(macro: ShotMacroConfig) -> V3AgentAdapter:
    theta = macro_config_to_theta(macro)
    return V3AgentAdapter(theta, macro_config=macro, offset_floor=macro.offset_floor)


def _score_stage1_snapshot(
    cem: RolloutResult,
    fb: RolloutResult,
    snap: Snapshot,
    commit: dict[str, float],
) -> float:
    w = _snapshot_weight(snap)
    score = 0.0
    if cem.life_lost and not fb.life_lost:
        return -500.0 * w
    if cem.broken_bricks > fb.broken_bricks:
        score += 300.0 * w
    elif (
        cem.broken_bricks == fb.broken_bricks
        and cem.steps_to_next_brick is not None
        and fb.steps_to_next_brick is not None
        and cem.steps_to_next_brick < fb.steps_to_next_brick
    ):
        score += 120.0 * w
    if commit.get("committed_contacts", 0.0) > 0 and commit.get("mean_abs_hit_offset_on_contact", 0.0) >= 0.04:
        score += 80.0 * w
    if cem.broken_bricks < fb.broken_bricks:
        score -= 200.0 * w
    return score


def evaluate_macro_stage1(
    macro: ShotMacroConfig,
    env: CatBreakEnv,
    snapshots: list[Snapshot],
    fb_rollouts: list[RolloutResult],
    horizon: int,
    snap_indices: Optional[list[int]] = None,
) -> dict:
    agent = _macro_agent(macro)
    if snap_indices is None:
        snap_indices = list(range(len(snapshots)))
    total = 0.0
    wins = 0
    contacts = 0
    for idx in snap_indices:
        snap = snapshots[idx]
        fb = fb_rollouts[idx]
        cem = rollout_from_state(env, snap.env_state, agent, horizon, fast=True)
        commit = agent.commitment_summary()
        pts = _score_stage1_snapshot(cem, fb, snap, commit)
        total += pts
        if better_than_followball(cem, fb):
            wins += 1
        contacts += int(commit.get("committed_contacts", 0.0))
    n = max(1, len(snap_indices))
    return {
        "stage1_score": total,
        "snapshot_wins": wins,
        "snapshot_win_rate": wins / n,
        "committed_contacts": contacts,
    }


def evaluate_macro_stage2(
    macro: ShotMacroConfig,
    env: CatBreakEnv,
    seeds: list[int],
    fb_cache: dict[int, dict],
) -> dict:
    agent = _macro_agent(macro)
    rows = []
    for seed in seeds:
        cem = run_episode_fast(env, agent, seed)
        fb = fb_cache[seed]
        cr = RolloutResult(cem["broken_bricks"], cem["clear"], cem["steps"], cem["lives"] < S.INITIAL_LIVES)
        fr = RolloutResult(fb["broken_bricks"], fb["clear"], fb["steps"], False)
        if worse_than_followball(cr, fr):
            verdict = "LOSS"
        elif better_than_followball(cr, fr):
            verdict = "WIN"
        else:
            verdict = "TIE"
        rows.append({
            "seed": seed,
            "cem_bricks": cem["broken_bricks"],
            "fb_bricks": fb["broken_bricks"],
            "delta_bricks": cem["broken_bricks"] - fb["broken_bricks"],
            "cem_steps": cem["steps"],
            "fb_steps": fb["steps"],
            "delta_steps": cem["steps"] - fb["steps"],
            "verdict": verdict,
            "committed_contact_rate": cem.get("committed_contact_rate", 0.0),
            "mean_abs_hit_offset_on_contact": cem.get("mean_abs_hit_offset_on_contact", 0.0),
            "committed_shot_rate": cem.get("committed_shot_rate", 0.0),
            "num_committed_shots": cem.get("num_committed_shots", 0.0),
            "committed_contacts": cem.get("committed_contacts", 0.0),
        })

    weak = [r for r in rows if r["seed"] in WEAK_SEEDS]
    safety = [r for r in rows if r["seed"] in SAFETY_SEEDS]
    wins = sum(1 for r in rows if r["verdict"] == "WIN")
    losses = sum(1 for r in rows if r["verdict"] == "LOSS")
    total_delta_bricks = sum(r["delta_bricks"] for r in rows)
    total_delta_steps = sum(r["delta_steps"] for r in rows)
    weak_delta = sum(r["delta_bricks"] for r in weak)
    safety_loss = any(r["verdict"] == "LOSS" for r in safety)
    seed14_ok = all(
        r["verdict"] != "LOSS" for r in rows if r["seed"] == 14
    )
    hard_reject = safety_loss or not seed14_ok
    commit = _aggregate_commitment_metrics(rows)
    hit_vals = [
        float(r.get("mean_abs_hit_offset_on_contact", 0.0))
        for r in rows if float(r.get("committed_contacts", 0.0)) > 0
    ]

    return {
        "rows": rows,
        "wins": wins,
        "losses": losses,
        "total_delta_bricks": total_delta_bricks,
        "total_delta_steps": total_delta_steps,
        "weak_seed_delta_bricks": weak_delta,
        "hard_reject": hard_reject,
        "committed_contact_rate": commit["committed_contact_rate"],
        "mean_abs_hit_offset": float(np.mean(hit_vals)) if hit_vals else 0.0,
    }


def _weighted_snapshot_indices(snapshots: list[Snapshot], max_snaps: int, rng: np.random.Generator) -> list[int]:
    if max_snaps <= 0 or max_snaps >= len(snapshots):
        return list(range(len(snapshots)))
    weights = np.array([_snapshot_weight(s) for s in snapshots], dtype=np.float64)
    weights /= weights.sum()
    chosen = rng.choice(len(snapshots), size=max_snaps, replace=False, p=weights)
    return sorted(int(i) for i in chosen)


_POOL_GRID_ENV: Optional[CatBreakEnv] = None
_POOL_GRID_SNAPS: list[Snapshot] = []
_POOL_GRID_FB: list[RolloutResult] = []
_POOL_GRID_HORIZON = 1200
_POOL_GRID_INDICES: list[int] = []


def _init_grid_worker(
    layout: str,
    snapshots: list[Snapshot],
    fb_rollouts: list[RolloutResult],
    horizon: int,
    snap_indices: list[int],
) -> None:
    global _POOL_GRID_ENV, _POOL_GRID_SNAPS, _POOL_GRID_FB, _POOL_GRID_HORIZON, _POOL_GRID_INDICES
    _POOL_GRID_ENV = CatBreakEnv(config={"layout": layout})
    _POOL_GRID_SNAPS = snapshots
    _POOL_GRID_FB = fb_rollouts
    _POOL_GRID_HORIZON = horizon
    _POOL_GRID_INDICES = snap_indices


def _grid_stage1_worker(payload: tuple[int, dict]) -> dict:
    idx, macro_dict = payload
    macro = ShotMacroConfig.from_dict(macro_dict)
    assert _POOL_GRID_ENV is not None
    s1 = evaluate_macro_stage1(
        macro, _POOL_GRID_ENV, _POOL_GRID_SNAPS, _POOL_GRID_FB,
        _POOL_GRID_HORIZON, _POOL_GRID_INDICES,
    )
    return {**macro.to_dict(), **s1, "config_id": idx}


def _passes_save_criteria(stage1: dict, stage2: dict, fb_loss_baseline: int = 8) -> bool:
    if stage2["hard_reject"]:
        return False
    if stage2["total_delta_bricks"] > 0:
        return True
    if stage2["total_delta_bricks"] == 0 and stage2["total_delta_steps"] < -50:
        return True
    if stage2["weak_seed_delta_bricks"] > 0:
        return True
    if (
        stage1["snapshot_win_rate"] > 0.0
        and stage2["losses"] <= 3
        and stage2["losses"] < fb_loss_baseline
    ):
        return True
    return False


def grid_search_shot_macros(args) -> Path:
    """Two-stage deterministic grid over shot macros -> 12-theta checkpoint."""
    import csv
    import itertools

    run_dir = Path(args.save_dir) if args.save_dir else S.CEM_AIM_DIR / "shot_macro_grid"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.opportunity_snapshots:
        raise ValueError("--opportunity-snapshots required for grid search")

    snapshots = load_snapshots(args.opportunity_snapshots)
    stage1_horizon = int(getattr(args, "grid_stage1_horizon", 1200))
    stage2_seeds = parse_seed_spec(getattr(args, "grid_episode_seeds", "0:19"))
    top_k = int(getattr(args, "grid_top_k", 50))
    max_stage1_snaps = int(getattr(args, "grid_stage1_max_snapshots", 128))
    workers = default_parallel_workers(getattr(args, "workers", None))
    rng = np.random.default_rng(int(getattr(args, "seed", 0)))

    env = CatBreakEnv(config={"layout": args.layout})
    follow = FollowAgentAdapter()
    snap_indices = _weighted_snapshot_indices(snapshots, max_stage1_snaps, rng)
    print(
        f"Precomputing FollowBall snapshot rollouts "
        f"({len(snapshots)} total, stage1 uses {len(snap_indices)} x horizon={stage1_horizon}) ..."
    )
    t0 = time.perf_counter()
    fb_snap_rollouts = [
        rollout_from_state(env, s.env_state, follow, stage1_horizon, fast=True)
        for s in snapshots
    ]
    fb_episode_cache = {s: run_episode_fast(env, follow, s) for s in stage2_seeds}
    print(f"FollowBall cache ready in {time.perf_counter() - t0:.1f}s")

    configs = iter_shot_macro_grid(quick=bool(getattr(args, "grid_quick", False)))
    print(
        f"Shot macro grid search | configs={len(configs)} "
        f"stage1_snaps={len(snap_indices)} workers={workers}"
    )

    stage1_rows: list[dict] = []
    if workers <= 1:
        for i, macro in enumerate(configs):
            s1 = evaluate_macro_stage1(
                macro, env, snapshots, fb_snap_rollouts, stage1_horizon, snap_indices
            )
            stage1_rows.append({**macro.to_dict(), **s1, "config_id": i})
            if (i + 1) % 200 == 0 or i + 1 == len(configs):
                print(f"  stage1 {i + 1}/{len(configs)}")
    else:
        payloads = [(i, macro.to_dict()) for i, macro in enumerate(configs)]
        results: list[Optional[dict]] = [None] * len(payloads)
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_grid_worker,
            initargs=(args.layout, snapshots, fb_snap_rollouts, stage1_horizon, snap_indices),
        ) as pool:
            futures = {pool.submit(_grid_stage1_worker, p): p[0] for p in payloads}
            done = 0
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                done += 1
                if done % 500 == 0 or done == len(payloads):
                    print(f"  stage1 {done}/{len(payloads)}")
        stage1_rows = [r for r in results if r is not None]

    stage1_rows.sort(key=lambda r: (r["stage1_score"], r["snapshot_wins"], r["committed_contacts"]), reverse=True)
    top50 = stage1_rows[:top_k]

    top50_path = run_dir / "shot_macro_grid_top50.csv"
    with top50_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stage1_rows[0].keys()))
        w.writeheader()
        w.writerows(top50)

    full_path = run_dir / "shot_macro_grid.csv"
    with full_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stage1_rows[0].keys()))
        w.writeheader()
        w.writerows(stage1_rows)

    saved: list[dict] = []
    best_row: Optional[dict] = None
    best_key: tuple = (-1,)

    for rank, s1row in enumerate(top50):
        macro = ShotMacroConfig.from_dict(s1row)
        s2 = evaluate_macro_stage2(macro, env, stage2_seeds, fb_episode_cache)
        merged = {
            **macro.to_dict(),
            **{k: s1row[k] for k in ("stage1_score", "snapshot_win_rate", "snapshot_wins")},
            **{k: s2[k] for k in (
                "wins", "losses", "total_delta_bricks", "total_delta_steps",
                "weak_seed_delta_bricks", "hard_reject", "committed_contact_rate",
                "mean_abs_hit_offset",
            )},
            "rank": rank,
            "passes_save": _passes_save_criteria(s1row, s2),
        }
        saved.append(merged)
        key = (
            int(merged["passes_save"]),
            s2["total_delta_bricks"],
            -s2["total_delta_steps"],
            s2["weak_seed_delta_bricks"],
            s1row["stage1_score"],
        )
        if key > best_key:
            best_key = key
            best_row = merged
            theta = macro_config_to_theta(macro)
            agent = CEMAimV3Policy(theta, macro_config=macro, offset_floor=macro.offset_floor)
            agent.save(str(run_dir / "shot_macro_grid_best.npz"), metrics=merged)

        per_seed_path = run_dir / f"shot_macro_rank{rank:02d}_per_seed.csv"
        with per_seed_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(s2["rows"][0].keys()))
            w.writeheader()
            w.writerows(s2["rows"])

    env.close()

    if best_row:
        print(
            f"Grid best: stage1={best_row['stage1_score']:.1f} "
            f"snap_win={best_row['snapshot_win_rate']:.3f} "
            f"Δbricks={best_row['total_delta_bricks']} "
            f"contact={best_row['committed_contact_rate']:.3f} "
            f"passes={best_row['passes_save']}"
        )
    print(f"Saved -> {run_dir}")
    print(f"  {full_path.name} ({len(stage1_rows)} configs)")
    print(f"  {top50_path.name}")
    print(f"  shot_macro_grid_best.npz")
    return run_dir
