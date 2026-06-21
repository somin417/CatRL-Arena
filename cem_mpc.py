"""Cross-Entropy Method Model Predictive Control for CatBreak."""

from __future__ import annotations

import math
import time
from typing import Any, Optional

import numpy as np

import settings as S
from agents import FollowBallAgent, followball_action_from_norm
from catbreak_env import CatBreakEnv


def predict_landing_x(
    ball_x: float,
    ball_y: float,
    ball_vx: float,
    ball_vy: float,
    target_y: float,
    field_width: float = S.FIELD_WIDTH,
    ball_radius: float = S.BALL_RADIUS,
    max_bounces: int = 24,
) -> float:
    """Approximate x where the ball reaches target_y with wall reflections."""
    x, y, vx, vy = float(ball_x), float(ball_y), float(ball_vx), float(ball_vy)
    if abs(vy) < 1e-6:
        return x
    for _ in range(max_bounces):
        if vy > 0 and y < target_y:
            t_land = (target_y - y) / vy
            if t_land >= 0:
                return x + vx * t_land
        t_candidates = []
        if vy < 0:
            t_top = (ball_radius - y) / vy if vy < 0 else float("inf")
            if t_top > 0:
                t_candidates.append(t_top)
        if vy > 0:
            t_down = (target_y - y) / vy
            if t_down > 0:
                return x + vx * t_down
        if vx > 0:
            t_r = (field_width - ball_radius - x) / vx
            if t_r > 1e-9:
                t_candidates.append(t_r)
        elif vx < 0:
            t_l = (ball_radius - x) / vx
            if t_l > 1e-9:
                t_candidates.append(t_l)
        if not t_candidates:
            break
        t_hit = min(t_candidates)
        x += vx * t_hit
        y += vy * t_hit
        if x - ball_radius < 0:
            x = ball_radius
            vx = abs(vx)
        elif x + ball_radius > field_width:
            x = field_width - ball_radius
            vx = -abs(vx)
        if y - ball_radius < 0:
            y = ball_radius
            vy = abs(vy)
    return x


def _brick_field_mid_y(env: CatBreakEnv) -> float:
    rects = [
        env._layout.brick_rect(r, c)
        for r in range(env.brick_rows)
        for c in range(env.brick_cols)
        if env._bricks[r, c]
    ]
    if not rects:
        return env.paddle_y / 2.0
    ys = [ry + rh / 2.0 for _, ry, _, rh in rects]
    return float(np.mean(ys))


def _min_distance_to_alive_brick(env: CatBreakEnv) -> float:
    centers = []
    for row in range(env.brick_rows):
        for col in range(env.brick_cols):
            if not env._bricks[row, col]:
                continue
            rx, ry, rw, rh = env._layout.brick_rect(row, col)
            centers.append((rx + rw / 2.0, ry + rh / 2.0))
    if not centers:
        return 0.0
    centers_arr = np.asarray(centers, dtype=np.float64)
    dx = env.ball_x - centers_arr[:, 0]
    dy = env.ball_y - centers_arr[:, 1]
    return float(np.hypot(dx, dy).min())


def is_safe_aim_state(ball_vy: float, ball_y: float) -> bool:
    """Ball rising in the upper half — safe to deviate from pure tracking."""
    return ball_vy < 0 and ball_y < S.FIELD_HEIGHT * 0.5


def _tunnel_progress_bonus(env: CatBreakEnv, ball_y_min: float) -> float:
    if env.brick_rows == 0:
        return 0.0
    brick_bottom = max(
        env._layout.brick_rect(r, c)[1] + env._layout.brick_rect(r, c)[3]
        for r in range(env.brick_rows)
        for c in range(env.brick_cols)
    )
    if ball_y_min < brick_bottom * 0.5:
        return 1.0
    return 0.0


def simulate_sequence(
    env: CatBreakEnv,
    state_dict: dict,
    actions: np.ndarray | list[int],
    gamma: float = 1.0,
    extra_rollout_steps: int = 0,
    follow_agent: Optional[FollowBallAgent] = None,
) -> dict:
    """Roll out actions from a copied state; restore env afterward."""
    saved = env.get_state_dict()
    env.set_state_dict(state_dict)
    start_ball_vy = float(state_dict.get("ball_vy", 0.0))
    start_ball_y = float(state_dict.get("ball_y", 0.0))
    start_remaining = int(np.asarray(state_dict["bricks"], dtype=bool).sum())
    follow_agent = follow_agent or FollowBallAgent()
    fb_threshold = float(getattr(follow_agent, "threshold", S.FOLLOW_BALL_THRESHOLD))
    try:
        bricks_start = env.broken_bricks
        lives_start = env.lives
        total_reward = 0.0
        steps_simulated = 0
        life_lost = False
        cleared = False
        upper_hits = 0
        ball_y_min = env.ball_y
        mid_y = _brick_field_mid_y(env)
        columns_hit: set[int] = set()
        steps_to_next_brick: Optional[int] = None
        target_hit_offset = 0.0

        for t, action in enumerate(actions):
            if env.done:
                break
            reward, done, info = env.step_fast(int(action))
            total_reward += reward * (gamma ** t)
            steps_simulated += 1
            ball_y_min = min(ball_y_min, env.ball_y)
            if info.get("life_lost"):
                life_lost = True
            if info.get("clear"):
                cleared = True
            bricks_this = info.get("bricks_broken_this_step", 0)
            if bricks_this and steps_to_next_brick is None:
                steps_to_next_brick = t + 1
            if bricks_this and env.ball_y < mid_y:
                upper_hits += bricks_this
            if bricks_this and getattr(env, "_last_paddle_collision", None):
                target_hit_offset = float(env._last_paddle_collision.get("hit_offset", 0.0))

        horizon_bricks = env.broken_bricks - bricks_start
        extra_steps = 0
        extra_bricks = 0
        if extra_rollout_steps > 0 and not env.done and not life_lost:
            for _ in range(extra_rollout_steps):
                if env.done:
                    break
                a = followball_action_from_norm(
                    env.ball_x / S.FIELD_WIDTH,
                    env.paddle_x / S.FIELD_WIDTH,
                    fb_threshold,
                )
                reward, done, info = env.step_fast(a)
                total_reward += reward
                extra_steps += 1
                steps_simulated += 1
                if info.get("life_lost"):
                    life_lost = True
                if info.get("clear"):
                    cleared = True
                bricks_this = info.get("bricks_broken_this_step", 0)
                if bricks_this and steps_to_next_brick is None:
                    steps_to_next_brick = len(actions) + extra_steps
                if bricks_this and env.ball_y < mid_y:
                    upper_hits += bricks_this

        # Re-scan columns hit from brick diff (approximate via remaining)
        bricks_broken = env.broken_bricks - bricks_start
        extra_bricks = bricks_broken - horizon_bricks
        remaining = int(env._bricks.sum())
        remaining_reduction = start_remaining - remaining
        landing_x = predict_landing_x(
            env.ball_x, env.ball_y, env.ball_vx, env.ball_vy, env.paddle_y
        )
        tunnel_bonus = _tunnel_progress_bonus(env, ball_y_min)
        min_brick_dist = _min_distance_to_alive_brick(env)
        paddle_unreachable = max(
            0.0, abs(float(env.paddle_x) - landing_x) - S.PADDLE_WIDTH
        ) / S.FIELD_WIDTH

        start_mask = np.asarray(state_dict["bricks"], dtype=bool)
        broken_mask = start_mask & ~env._bricks
        for col in range(env.brick_cols):
            if broken_mask[:, col].any():
                columns_hit.add(col)

        if steps_to_next_brick is None:
            steps_to_next_brick = steps_simulated + 1

        rollout_info = {
            "bricks_broken_during_sequence": bricks_broken,
            "horizon_bricks_broken": horizon_bricks,
            "extra_rollout_bricks": extra_bricks,
            "life_lost": life_lost or (env.lives < lives_start),
            "done": env.done,
            "clear": cleared or bool(env.terminal_reason == "cleared"),
            "steps_simulated": steps_simulated,
            "horizon_steps": len(actions),
            "extra_rollout_steps": extra_steps,
            "total_env_reward": total_reward,
            "remaining_bricks": remaining,
            "remaining_reduction": remaining_reduction,
            "ball_y": env.ball_y,
            "ball_vy": env.ball_vy,
            "ball_x": env.ball_x,
            "ball_vx": env.ball_vx,
            "paddle_x": env.paddle_x,
            "predicted_landing_x": landing_x,
            "min_distance_to_alive_brick": min_brick_dist,
            "upper_brick_hit_bonus": upper_hits,
            "tunnel_progress_bonus": tunnel_bonus,
            "unique_columns_hit": len(columns_hit),
            "steps_to_next_brick": steps_to_next_brick,
            "paddle_unreachable_risk": paddle_unreachable,
            "target_hit_offset": target_hit_offset,
        }
        score_tuple = CEMMPCPlanner.score_rollout(
            rollout_info, start_ball_vy=start_ball_vy, start_ball_y=start_ball_y
        )
        rollout_info["score_tuple"] = score_tuple
        rollout_info["score_scalar"] = CEMMPCPlanner.score_to_scalar(score_tuple)
        return rollout_info
    finally:
        env.set_state_dict(saved)


class CEMMPCPlanner:
    """Categorical CEM over discrete action sequences with lexicographic scoring."""

    def __init__(
        self,
        env_config: Optional[dict] = None,
        horizon: int = 20,
        population_size: int = 128,
        elite_frac: float = 0.15,
        iterations: int = 4,
        action_dim: int = S.N_ACTIONS,
        gamma: float = 1.0,
        prob_floor: float = 0.05,
        smoothing: float = 0.25,
        temperature: float = 1.0,
        warm_start: bool = True,
        include_followball_seed: bool = True,
        followball_floor: bool = True,
        demo_teacher_mode: bool = False,
        demo_teacher_stride: int = 8,
        include_stay_sequence: bool = True,
        include_action_repeats: bool = True,
        sequence_repeat: int = 1,
        seed: int = 0,
        verbose: bool = False,
        mode: str = "safe_eval",
        workers: Optional[int] = None,
        layout: str = S.DEFAULT_LAYOUT,
    ) -> None:
        self.mode = str(mode)
        self.workers = workers
        self.layout = layout
        self.env_config = env_config or {"layout": S.DEFAULT_LAYOUT}
        self.horizon = int(horizon)
        self.population_size = int(population_size)
        self.elite_frac = float(elite_frac)
        self.iterations = int(iterations)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.prob_floor = float(prob_floor)
        self.smoothing = float(smoothing)
        self.temperature = float(temperature)
        self.warm_start = bool(warm_start)
        self.include_followball_seed = bool(include_followball_seed)
        self.followball_floor = bool(followball_floor)
        self.demo_teacher_mode = bool(demo_teacher_mode)
        self.demo_teacher_stride = max(2, int(demo_teacher_stride))
        self.include_stay_sequence = bool(include_stay_sequence)
        self.include_action_repeats = bool(include_action_repeats)
        self.sequence_repeat = max(1, int(sequence_repeat))
        self.verbose = bool(verbose)
        self.rng = np.random.default_rng(seed)
        self._probs: Optional[np.ndarray] = None
        self._follow_agent = FollowBallAgent()
        self._last_plan_info: dict = {}
        self._plan_log: list[dict] = []
        self._planning_env: Optional[CatBreakEnv] = None
        self._safe_aim_step_count = 0
        self._rollout_pool: Optional[Any] = None

    def _get_rollout_pool(self) -> Optional[Any]:
        from torch_utils import default_parallel_workers

        workers = default_parallel_workers(self.workers)
        if workers <= 1:
            return None
        if self._rollout_pool is None:
            from cem_mpc_parallel import ParallelRolloutPool

            self._rollout_pool = ParallelRolloutPool(
                workers, self.layout, self.env_config
            )
        return self._rollout_pool

    def shutdown_rollout_pool(self) -> None:
        if self._rollout_pool is not None:
            self._rollout_pool.shutdown()
            self._rollout_pool = None

    def _planning_env_for(self, env: CatBreakEnv) -> CatBreakEnv:
        if self._planning_env is None:
            self._planning_env = env.clone()
        return self._planning_env

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._probs = None
        self._last_plan_info = {}
        self._plan_log = []
        self._safe_aim_step_count = 0

    @staticmethod
    def _ball_descending(start_ball_vy: float, start_ball_y: float) -> bool:
        """True when the paddle should prioritize intercept over brick chasing."""
        return start_ball_vy > 0 or start_ball_y > S.FIELD_HEIGHT * 0.55

    @staticmethod
    def score_rollout(
        rollout_info: dict,
        start_ball_vy: float = 0.0,
        start_ball_y: float = 0.0,
    ) -> tuple:
        survival = int(not rollout_info["life_lost"])
        bricks = int(rollout_info["bricks_broken_during_sequence"])
        cleared = int(rollout_info["clear"])
        remaining = -int(rollout_info["remaining_bricks"])
        upper = int(rollout_info.get("upper_brick_hit_bonus", 0))
        tunnel = float(rollout_info.get("tunnel_progress_bonus", 0.0))
        align = -abs(
            float(rollout_info["paddle_x"]) - float(rollout_info["predicted_landing_x"])
        )
        steps = -int(rollout_info["steps_simulated"])
        reward = float(rollout_info["total_env_reward"]) * 0.001

        min_dist = -float(rollout_info.get("min_distance_to_alive_brick", 0.0))
        if CEMMPCPlanner._ball_descending(start_ball_vy, start_ball_y):
            # Ball falling: never trade a miss for a brick in the horizon window.
            return (survival, align, cleared, bricks, remaining, upper, tunnel, steps, reward)
        # Ball rising: survival first, then brick progress / aiming toward clusters.
        return (survival, bricks, cleared, remaining, upper, tunnel, min_dist, align, steps, reward)

    @staticmethod
    def score_to_scalar(score_tuple: tuple) -> float:
        weights = [1e6, 1e5, 1e4, 1e3, 1e2, 10.0, 1.0, 0.1, 0.01]
        return float(sum(w * v for w, v in zip(weights, score_tuple)))

    def _init_probs(self) -> np.ndarray:
        if self.warm_start and self._probs is not None and self._probs.shape[0] == self.horizon:
            shifted = np.zeros((self.horizon, self.action_dim), dtype=np.float64)
            shifted[:-1] = self._probs[1:]
            shifted[-1] = self._probs[-1]
            return self._normalize_probs(shifted)
        uniform = np.ones((self.horizon, self.action_dim), dtype=np.float64) / self.action_dim
        return self._normalize_probs(uniform)

    def _normalize_probs(self, probs: np.ndarray) -> np.ndarray:
        probs = np.clip(probs, self.prob_floor, 1.0)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
        return probs / row_sums

    def sample_action_sequences(self, probs: np.ndarray) -> np.ndarray:
        p = probs
        if self.temperature != 1.0:
            p = np.power(probs, 1.0 / self.temperature)
            p = p / p.sum(axis=1, keepdims=True)
        sequences = np.empty((self.population_size, self.horizon), dtype=np.int64)
        for t in range(self.horizon):
            sequences[:, t] = self.rng.choice(
                self.action_dim, size=self.population_size, p=p[t]
            )
        return sequences

    def update_distribution(
        self,
        probs: np.ndarray,
        elite_sequences: np.ndarray,
        elite_scores: np.ndarray,
    ) -> np.ndarray:
        del elite_scores  # lexicographic ranking already applied
        elite_count = max(1, elite_sequences.shape[0])
        elite_freq = np.zeros_like(probs)
        for t in range(self.horizon):
            counts = np.bincount(elite_sequences[:, t], minlength=self.action_dim)
            elite_freq[t] = counts / elite_count
        new_probs = (1.0 - self.smoothing) * probs + self.smoothing * elite_freq
        return self._normalize_probs(new_probs)

    def evaluate_sequence(
        self,
        env: CatBreakEnv,
        start_state: dict,
        action_seq: np.ndarray | list[int],
    ) -> dict:
        actions = np.asarray(action_seq, dtype=np.int64)
        if self.sequence_repeat > 1:
            expanded = []
            for a in actions:
                expanded.extend([int(a)] * self.sequence_repeat)
            actions = np.array(expanded[: self.horizon], dtype=np.int64)
        return simulate_sequence(env, start_state, actions, gamma=self.gamma)

    def _followball_sequence(
        self,
        planning_env: CatBreakEnv,
        start_state: dict,
    ) -> np.ndarray:
        planning_env.set_state_dict(start_state)
        seq = []
        for _ in range(self.horizon):
            if planning_env.done:
                seq.append(S.ACTION_STAY)
            else:
                obs = planning_env.get_obs()
                a = self._follow_agent.act(obs, planning_env.last_info, env=planning_env)
                seq.append(a)
                planning_env.step(a)
        return np.array(seq, dtype=np.int64)

    def _deterministic_candidates(
        self,
        env: CatBreakEnv,
        start_state: dict,
        planning_env: CatBreakEnv,
    ) -> list[np.ndarray]:
        H = self.horizon
        candidates: list[np.ndarray] = []

        if self.include_followball_seed:
            candidates.append(self._followball_sequence(planning_env, start_state))

        if self.include_stay_sequence:
            candidates.append(np.full(H, S.ACTION_STAY, dtype=np.int64))
            candidates.append(np.full(H, S.ACTION_LEFT, dtype=np.int64))
            candidates.append(np.full(H, S.ACTION_RIGHT, dtype=np.int64))

        alt = np.array([S.ACTION_LEFT if t % 2 == 0 else S.ACTION_RIGHT for t in range(H)])
        candidates.append(alt)

        if self.include_action_repeats:
            for base in (
                [S.ACTION_LEFT, S.ACTION_LEFT, S.ACTION_STAY],
                [S.ACTION_RIGHT, S.ACTION_RIGHT, S.ACTION_STAY],
                [S.ACTION_LEFT, S.ACTION_STAY, S.ACTION_STAY],
                [S.ACTION_RIGHT, S.ACTION_STAY, S.ACTION_STAY],
            ):
                seq = (base * ((H // len(base)) + 1))[:H]
                candidates.append(np.array(seq, dtype=np.int64))

        return candidates

    def plan(self, env: CatBreakEnv) -> dict:
        t0 = time.perf_counter()
        start_state = env.get_state_dict()
        planning_env = self._planning_env_for(env)
        start_vy = float(start_state.get("ball_vy", 0.0))
        start_by = float(start_state.get("ball_y", 0.0))
        probs = self._init_probs()

        follow_seq: Optional[np.ndarray] = None
        follow_score: Optional[tuple] = None
        follow_result: Optional[dict] = None
        if self.include_followball_seed:
            follow_seq = self._followball_sequence(planning_env, start_state)
            follow_result = self.evaluate_sequence(planning_env, start_state, follow_seq)
            follow_score = follow_result["score_tuple"]

        best_seq = follow_seq if follow_seq is not None else np.full(
            self.horizon, S.ACTION_STAY, dtype=np.int64
        )
        best_result = follow_result or self.evaluate_sequence(planning_env, start_state, best_seq)
        best_score = follow_score or best_result["score_tuple"]
        used_follow_floor = False
        elite_bricks_all: list[float] = []
        entropy_all: list[float] = []

        n_elite = max(1, int(math.ceil(self.population_size * self.elite_frac)))

        for it in range(self.iterations):
            sequences = self.sample_action_sequences(probs)
            det = self._deterministic_candidates(env, start_state, planning_env)
            if det:
                n_det = min(len(det), self.population_size)
                sequences[:n_det] = np.array(det[:n_det])

            if self.workers and self.workers > 1 and len(sequences) > 4:
                from cem_mpc_parallel import evaluate_sequences_parallel

                results = evaluate_sequences_parallel(
                    self.env_config,
                    start_state,
                    sequences,
                    layout=self.layout,
                    gamma=self.gamma,
                    workers=self.workers,
                    pool=self._get_rollout_pool(),
                )
                scores = [r["score_tuple"] for r in results]
            else:
                scores = []
                results = []
                for seq in sequences:
                    result = self.evaluate_sequence(planning_env, start_state, seq)
                    scores.append(result["score_tuple"])
                    results.append(result)

            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            elite_idx = order[:n_elite]
            elite_sequences = sequences[elite_idx]
            elite_scores = np.array([self.score_to_scalar(scores[i]) for i in elite_idx])
            elite_bricks_all.append(
                float(np.mean([
                    scores[i][3] if self._ball_descending(start_vy, start_by) else scores[i][1]
                    for i in elite_idx
                ]))
            )
            entropy_all.append(float(self._mean_entropy(probs)))

            if scores[order[0]] > best_score:
                best_score = scores[order[0]]
                best_seq = sequences[order[0]].copy()
                best_result = results[order[0]]

            probs = self.update_distribution(probs, elite_sequences, elite_scores)
            if self.verbose:
                print(
                    f"  CEM iter {it + 1}/{self.iterations} "
                    f"best_surv={best_score[0]} best_bricks={best_score[3] if self._ball_descending(start_vy, start_by) else best_score[1]} "
                    f"elite_mean={elite_bricks_all[-1]:.2f}"
                )

        if self.followball_floor and follow_seq is not None and follow_score is not None:
            if follow_score > best_score:
                best_seq = follow_seq.copy()
                best_score = follow_score
                best_result = follow_result  # type: ignore[assignment]
                used_follow_floor = True

        self._probs = probs.copy()
        plan_time_ms = (time.perf_counter() - t0) * 1000.0
        prefix = " ".join(S.ACTION_NAMES.get(int(a), "?") for a in best_seq[:5])
        aim_signal = False
        min_dist_gain = 0.0
        descending = self._ball_descending(start_vy, start_by)
        if follow_result is not None and follow_score is not None and follow_seq is not None:
            if int(best_seq[0]) != int(follow_seq[0]):
                min_dist_gain = (
                    float(follow_result.get("min_distance_to_alive_brick", 0.0))
                    - float(best_result.get("min_distance_to_alive_brick", 0.0))
                )
                best_prefix = self._teacher_score_prefix(best_score, descending)
                follow_prefix = self._teacher_score_prefix(follow_score, descending)
                aim_signal = best_prefix > follow_prefix
        plan_info = {
            "action": int(best_seq[0]),
            "best_sequence": best_seq,
            "best_score_tuple": best_score,
            "best_score_scalar": self.score_to_scalar(best_score),
            "best_predicted_bricks": int(best_result["bricks_broken_during_sequence"]),
            "best_predicted_life_lost": bool(best_result["life_lost"]),
            "follow_score_tuple": follow_score,
            "aim_signal": aim_signal,
            "min_dist_gain": min_dist_gain,
            "follow_predicted_bricks": int(
                follow_result["bricks_broken_during_sequence"] if follow_result else 0
            ),
            "follow_predicted_life_lost": bool(
                follow_result["life_lost"] if follow_result else False
            ),
            "elite_mean_bricks": float(np.mean(elite_bricks_all)) if elite_bricks_all else 0.0,
            "entropy_mean": float(np.mean(entropy_all)) if entropy_all else 0.0,
            "plan_time_ms": plan_time_ms,
            "best_sequence_prefix": prefix,
            "used_followball_floor": used_follow_floor,
            "ball_descending": self._ball_descending(start_vy, start_by),
        }
        self._last_plan_info = plan_info
        return plan_info

    @staticmethod
    def _teacher_score_prefix(score_tuple: tuple, descending: bool) -> tuple:
        """Brick/aim-relevant prefix; excludes align/steps/reward tie-breakers."""
        if descending:
            return score_tuple[:7]
        return score_tuple[:7]

    @staticmethod
    def _aim_comparison_tuple(result: dict) -> tuple:
        """Brick-focused rollout comparison for safe-state aiming decisions."""
        return (
            int(not result["life_lost"]),
            int(result["bricks_broken_during_sequence"]),
            int(result.get("upper_brick_hit_bonus", 0)),
            float(result.get("tunnel_progress_bonus", 0.0)),
            -float(result.get("min_distance_to_alive_brick", 0.0)),
        )

    def _mean_entropy(self, probs: np.ndarray) -> float:
        p = np.clip(probs, 1e-12, 1.0)
        return float((-np.sum(p * np.log(p), axis=1)).mean())

    def _choose_planned_action(
        self,
        env: CatBreakEnv,
        follow_action: int,
        cem_action: int,
        plan_info: dict,
    ) -> tuple[int, str]:
        if plan_info["best_predicted_life_lost"]:
            return follow_action, "fallback_life"

        if cem_action == follow_action:
            return follow_action, "follow_default"

        follow_score = plan_info.get("follow_score_tuple")
        best_score = plan_info["best_score_tuple"]
        if follow_score is not None and best_score < follow_score:
            return follow_action, "follow_floor"

        safe = is_safe_aim_state(env.ball_vy, env.ball_y)
        if safe:
            self._safe_aim_step_count += 1

        if safe and plan_info.get("aim_signal"):
            return cem_action, "cem_aim"

        follow_bricks = int(plan_info.get("follow_predicted_bricks", 0))
        if plan_info["best_predicted_bricks"] > follow_bricks:
            return cem_action, "cem_aim"

        if (
            self.demo_teacher_mode
            and safe
            and follow_score is not None
            and best_score >= follow_score
            and self._safe_aim_step_count % self.demo_teacher_stride == 0
        ):
            return cem_action, "cem_teacher"

        return follow_action, "follow_default"

    def _append_plan_log(
        self,
        env: CatBreakEnv,
        chosen: int,
        follow_action: int,
        plan_info: dict,
    ) -> None:
        self._plan_log.append({
            "chosen_action": chosen,
            "chosen_action_name": S.ACTION_NAMES.get(chosen, "?"),
            "followball_action": follow_action,
            "followball_action_name": S.ACTION_NAMES.get(follow_action, "?"),
            "mode": plan_info.get("mode", "unknown"),
            "is_safe_aim": int(is_safe_aim_state(env.ball_vy, env.ball_y)),
            "ball_descending": int(plan_info.get("ball_descending", False)),
            "best_sequence_prefix": plan_info.get("best_sequence_prefix", ""),
            "best_score_scalar": plan_info.get("best_score_scalar", 0.0),
            "best_score_tuple": str(plan_info.get("best_score_tuple", ())),
            "best_predicted_bricks": plan_info.get("best_predicted_bricks", 0),
            "best_predicted_life_lost": int(bool(plan_info.get("best_predicted_life_lost", False))),
            "elite_mean_bricks": plan_info.get("elite_mean_bricks", 0.0),
            "entropy_mean": plan_info.get("entropy_mean", 0.0),
            "plan_time_ms": plan_info.get("plan_time_ms", 0.0),
            "ball_x": env.ball_x,
            "ball_y": env.ball_y,
            "ball_vx": env.ball_vx,
            "ball_vy": env.ball_vy,
            "paddle_x": env.paddle_x,
            "remaining_bricks": int(env._bricks.sum()),
        })

    def act(
        self,
        obs: Any,
        info: Optional[dict] = None,
        env: Optional[CatBreakEnv] = None,
    ) -> int:
        if env is None:
            raise ValueError("CEM-MPC requires env for planning.")

        follow_action = self._follow_agent.act(obs, info, env=env)

        # Descending / lower half: pure tracking (INITIAL_LIVES=1 — never gamble).
        if env.ball_vy > 0 or env.ball_y > S.FIELD_HEIGHT * 0.5:
            plan_info = {
                "action": follow_action,
                "used_followball_floor": True,
                "mode": "track",
                "best_sequence_prefix": S.ACTION_NAMES.get(follow_action, "?"),
                "best_score_scalar": 0.0,
                "best_score_tuple": (),
                "best_predicted_bricks": 0,
                "best_predicted_life_lost": False,
                "elite_mean_bricks": 0.0,
                "entropy_mean": 0.0,
                "plan_time_ms": 0.0,
                "ball_descending": True,
            }
            self._last_plan_info = plan_info
            self._append_plan_log(env, follow_action, follow_action, plan_info)
            return follow_action

        plan_info = self.plan(env)
        cem_action = int(plan_info["action"])
        chosen, mode = self._choose_planned_action(env, follow_action, cem_action, plan_info)
        plan_info["mode"] = mode
        plan_info["action"] = chosen
        plan_info["ball_descending"] = False
        self._last_plan_info = plan_info
        self._append_plan_log(env, chosen, follow_action, plan_info)
        return chosen

    def consume_plan_log(self) -> list[dict]:
        log = self._plan_log
        self._plan_log = []
        return log

    @property
    def last_plan_info(self) -> dict:
        return self._last_plan_info
