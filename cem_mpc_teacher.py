"""CEM-MPC teacher_search mode: discover trajectories that beat FollowBall."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

import settings as S
from agents import FollowBallAgent
from catbreak_env import CatBreakEnv
from cem_mpc import CEMMPCPlanner, is_safe_aim_state, predict_landing_x, simulate_sequence
from cem_mpc_parallel import evaluate_sequences_parallel


@dataclass(frozen=True)
class TeacherScoreWeights:
    clear: float = 1000.0
    bricks: float = 80.0
    unique_columns: float = 20.0
    upper: float = 15.0
    remaining_reduction: float = 10.0
    steps: float = 0.02
    life_lost: float = 5000.0
    unreachable: float = 30.0


def compute_teacher_score(rollout: dict, weights: TeacherScoreWeights | None = None) -> float:
    w = weights or TeacherScoreWeights()
    return (
        w.clear * float(rollout.get("clear", False))
        + w.bricks * float(rollout.get("bricks_broken_during_sequence", 0))
        + w.unique_columns * float(rollout.get("unique_columns_hit", 0))
        + w.upper * float(rollout.get("upper_brick_hit_bonus", 0))
        + w.remaining_reduction * float(rollout.get("remaining_reduction", 0))
        - w.steps * float(rollout.get("steps_simulated", 0))
        - w.life_lost * float(rollout.get("life_lost", False))
        - w.unreachable * float(rollout.get("paddle_unreachable_risk", 0.0))
    )


def beats_followball_rollout(mpc: dict, follow: dict) -> bool:
    mpc_b = int(mpc.get("bricks_broken_during_sequence", 0))
    fb_b = int(follow.get("bricks_broken_during_sequence", 0))
    if mpc_b > fb_b:
        return True
    if bool(mpc.get("clear")) and not bool(follow.get("clear")):
        return True
    if mpc_b == fb_b and mpc_b > 0:
        mpc_t = int(mpc.get("steps_to_next_brick", 10**9))
        fb_t = int(follow.get("steps_to_next_brick", 10**9))
        if mpc_t < fb_t:
            return True
    return False


def _brick_column_centers(env: CatBreakEnv) -> list[tuple[int, float]]:
    cols: list[tuple[int, float]] = []
    for col in range(env.brick_cols):
        alive = [
            env._layout.brick_rect(row, col)
            for row in range(env.brick_rows)
            if env._bricks[row, col]
        ]
        if alive:
            cx = float(np.mean([rx + rw / 2 for rx, _, rw, _ in alive]))
            cols.append((col, cx))
    return cols


def _target_paddle_x_for_column(env: CatBreakEnv, col_x: float) -> float:
    landing = predict_landing_x(
        env.ball_x, env.ball_y, env.ball_vx, env.ball_vy, env.paddle_y
    )
    offset_frac = float(np.clip((col_x - landing) / (S.PADDLE_WIDTH / 2), -0.85, 0.85))
    return landing + offset_frac * (S.PADDLE_WIDTH / 2)


def _move_toward(paddle_x: float, target_x: float, steps: int) -> list[int]:
    seq = []
    px = paddle_x
    for _ in range(steps):
        if px < target_x - S.PADDLE_WIDTH * 0.15:
            seq.append(S.ACTION_RIGHT)
            px += S.PADDLE_SPEED * S.FIXED_DT
        elif px > target_x + S.PADDLE_WIDTH * 0.15:
            seq.append(S.ACTION_LEFT)
            px -= S.PADDLE_SPEED * S.FIXED_DT
        else:
            seq.append(S.ACTION_STAY)
    return seq


def _edge_hit_sequence(
    env: CatBreakEnv,
    start_state: dict,
    horizon: int,
    edge: str,
) -> np.ndarray:
    """Macro: move paddle to hit ball on left/center/right edge of paddle."""
    planning_env = CatBreakEnv(config={"layout": start_state.get("layout", S.DEFAULT_LAYOUT)})
    planning_env.set_state_dict(start_state)
    landing = predict_landing_x(
        planning_env.ball_x, planning_env.ball_y,
        planning_env.ball_vx, planning_env.ball_vy, planning_env.paddle_y,
    )
    frac = {"left": -0.75, "center": 0.0, "right": 0.75}[edge]
    target = landing + frac * (S.PADDLE_WIDTH / 2)
    moves = _move_toward(planning_env.paddle_x, target, min(horizon, 8))
    seq = moves + [S.ACTION_STAY] * max(0, horizon - len(moves))
    planning_env.close()
    return np.array(seq[:horizon], dtype=np.int64)


def is_opportunity_state(
    env: CatBreakEnv,
    env_seed: int,
    *,
    focus_endgame: bool,
    focus_stuck: bool,
    no_brick_streak: int,
    stuck_threshold: int = 800,
) -> bool:
    remaining = int(env._bricks.sum())
    if focus_endgame and remaining <= 10:
        return True
    if focus_stuck and no_brick_streak >= stuck_threshold:
        return True
    if env.step_count > 8000 and remaining <= 15:
        return True
    if no_brick_streak >= 1000:
        return True
    if env_seed in S.TEACHER_STRESS_SEEDS:
        return True
    cols = _brick_column_centers(env)
    if len(cols) >= 2:
        xs = [c[1] for c in cols]
        if max(xs) - min(xs) > S.FIELD_WIDTH * 0.35:
            return True
    mid_y = S.FIELD_HEIGHT * 0.45
    upper = sum(
        1 for row in range(env.brick_rows) for col in range(env.brick_cols)
        if env._bricks[row, col]
        and env._layout.brick_rect(row, col)[1] < mid_y
    )
    if upper >= max(1, remaining // 2):
        return True
    return is_safe_aim_state(env.ball_vy, env.ball_y)


class TeacherSearchPlanner(CEMMPCPlanner):
    """Oracle planner: search for shots that beat FollowBall; save as teacher data."""

    def __init__(
        self,
        *,
        teacher_margin: float = 0.0,
        num_rollout_after_sequence: int = 500,
        focus_endgame: bool = True,
        focus_stuck: bool = True,
        stuck_threshold: int = 800,
        allow_unsafe_search: bool = False,
        workers: Optional[int] = None,
        layout: str = S.DEFAULT_LAYOUT,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("mode", "teacher_search")
        kwargs.setdefault("followball_floor", False)
        kwargs.setdefault("demo_teacher_mode", False)
        super().__init__(**kwargs)
        self.teacher_margin = float(teacher_margin)
        self.num_rollout_after_sequence = int(num_rollout_after_sequence)
        self.focus_endgame = bool(focus_endgame)
        self.focus_stuck = bool(focus_stuck)
        self.stuck_threshold = int(stuck_threshold)
        self.allow_unsafe_search = bool(allow_unsafe_search)
        self.workers = workers
        self.layout = layout
        self._no_brick_streak = 0
        self._last_broken = 0
        self._teacher_comparisons: list[dict] = []
        self._teacher_better: list[dict] = []
        self._snapshots: list[dict] = []

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed=seed)
        self._no_brick_streak = 0
        self._last_broken = 0
        self._teacher_comparisons = []
        self._teacher_better = []
        self._snapshots = []

    def note_step_outcome(self, broken_bricks: int) -> None:
        if broken_bricks > self._last_broken:
            self._no_brick_streak = 0
        else:
            self._no_brick_streak += 1
        self._last_broken = broken_bricks

    def evaluate_sequence(
        self,
        env: CatBreakEnv,
        start_state: dict,
        action_seq: np.ndarray | list[int],
    ) -> dict:
        result = simulate_sequence(
            env, start_state, action_seq, gamma=self.gamma,
            extra_rollout_steps=self.num_rollout_after_sequence,
            follow_agent=self._follow_agent,
        )
        result["teacher_score"] = compute_teacher_score(result)
        return result

    def _evaluate_batch(
        self,
        start_state: dict,
        sequences: np.ndarray,
        env: CatBreakEnv,
    ) -> list[dict]:
        if self.workers and self.workers > 1 and len(sequences) > 4:
            results = evaluate_sequences_parallel(
                self.env_config,
                start_state,
                sequences,
                layout=self.layout,
                gamma=self.gamma,
                extra_rollout_steps=self.num_rollout_after_sequence,
                workers=self.workers,
                pool=self._get_rollout_pool(),
            )
        else:
            results = [
                self.evaluate_sequence(env, start_state, sequences[i])
                for i in range(len(sequences))
            ]
        for r in results:
            if "teacher_score" not in r:
                r["teacher_score"] = compute_teacher_score(r)
        return results

    def _macro_candidates(
        self,
        env: CatBreakEnv,
        start_state: dict,
        planning_env: CatBreakEnv,
    ) -> list[np.ndarray]:
        H = self.horizon
        cands: list[np.ndarray] = list(
            super()._deterministic_candidates(env, start_state, planning_env)
        )
        for edge in ("left", "center", "right"):
            cands.append(_edge_hit_sequence(env, start_state, H, edge))

        cols = _brick_column_centers(env)
        if cols:
            isolated = min(cols, key=lambda c: len([
                1 for _, cx in cols if abs(cx - c[1]) < S.PADDLE_WIDTH
            ]))
            target = _target_paddle_x_for_column(env, isolated[1])
            hold = _move_toward(env.paddle_x, target, min(6, H))
            track = []
            planning_env.set_state_dict(start_state)
            for _ in range(H - len(hold)):
                if planning_env.done:
                    track.append(S.ACTION_STAY)
                else:
                    obs = planning_env.get_obs()
                    track.append(
                        self._follow_agent.act(obs, planning_env.last_info, env=planning_env)
                    )
                    planning_env.step(track[-1])
            cands.append(np.array((hold + track)[:H], dtype=np.int64))

            left_cols = [c for c in cols if c[1] < S.FIELD_WIDTH * 0.45]
            right_cols = [c for c in cols if c[1] > S.FIELD_WIDTH * 0.55]
            if left_cols:
                tx = _target_paddle_x_for_column(env, float(np.mean([c[1] for c in left_cols])))
                cands.append(np.array(_move_toward(env.paddle_x, tx, H), dtype=np.int64))
            if right_cols:
                tx = _target_paddle_x_for_column(env, float(np.mean([c[1] for c in right_cols])))
                cands.append(np.array(_move_toward(env.paddle_x, tx, H), dtype=np.int64))

        for hold_n, edge in ((3, "left"), (3, "right")):
            edge_seq = _edge_hit_sequence(env, start_state, hold_n, edge)
            rest = np.full(H - hold_n, S.ACTION_STAY, dtype=np.int64)
            cands.append(np.concatenate([edge_seq[:hold_n], rest]))

        return cands

    def plan(self, env: CatBreakEnv) -> dict:
        t0 = time.perf_counter()
        start_state = env.get_state_dict()
        start_state["layout"] = self.layout
        planning_env = self._planning_env_for(env)
        probs = self._init_probs()

        follow_seq = self._followball_sequence(planning_env, start_state)
        follow_result = self.evaluate_sequence(planning_env, start_state, follow_seq)
        follow_score = follow_result["teacher_score"]

        best_seq = follow_seq.copy()
        best_result = follow_result
        best_score = follow_score
        n_elite = max(1, int(math.ceil(self.population_size * self.elite_frac)))

        for it in range(self.iterations):
            sequences = self.sample_action_sequences(probs)
            macros = self._macro_candidates(env, start_state, planning_env)
            n_macro = min(len(macros), self.population_size)
            sequences[:n_macro] = np.array(macros[:n_macro])

            results = self._evaluate_batch(start_state, sequences, planning_env)
            scores = [r["teacher_score"] for r in results]
            advantages = [s - follow_score for s in scores]

            order = sorted(
                range(len(scores)),
                key=lambda i: (advantages[i], scores[i]),
                reverse=True,
            )
            elite_idx = order[:n_elite]
            elite_sequences = sequences[elite_idx]

            if advantages[order[0]] > self.teacher_margin:
                if scores[order[0]] > best_score or advantages[order[0]] > 0:
                    best_score = scores[order[0]]
                    best_seq = sequences[order[0]].copy()
                    best_result = results[order[0]]

            elite_scores = np.array([scores[i] for i in elite_idx])
            probs = self.update_distribution(probs, elite_sequences, elite_scores)

        advantage = best_score - follow_score
        plan_info = {
            "action": int(best_seq[0]),
            "best_sequence": best_seq,
            "follow_sequence": follow_seq,
            "teacher_score": best_score,
            "follow_teacher_score": follow_score,
            "advantage_score": advantage,
            "best_predicted_bricks": int(best_result.get("horizon_bricks_broken", 0)),
            "follow_predicted_bricks": int(follow_result.get("horizon_bricks_broken", 0)),
            "mpc_rollout_bricks": int(best_result.get("bricks_broken_during_sequence", 0)),
            "followball_rollout_bricks": int(follow_result.get("bricks_broken_during_sequence", 0)),
            "mpc_rollout_steps_to_next_brick": int(best_result.get("steps_to_next_brick", 0)),
            "followball_rollout_steps_to_next_brick": int(follow_result.get("steps_to_next_brick", 0)),
            "best_predicted_life_lost": bool(best_result.get("life_lost", False)),
            "follow_predicted_life_lost": bool(follow_result.get("life_lost", False)),
            "predicted_landing_x": float(best_result.get("predicted_landing_x", 0)),
            "target_hit_offset": float(best_result.get("target_hit_offset", 0)),
            "plan_time_ms": (time.perf_counter() - t0) * 1000.0,
            "best_sequence_prefix": " ".join(
                S.ACTION_NAMES.get(int(a), "?") for a in best_seq[:5]
            ),
            "beats_followball": beats_followball_rollout(best_result, follow_result),
            "ball_descending": False,
        }
        self._last_plan_info = plan_info
        return plan_info

    def _choose_teacher_action(
        self,
        follow_action: int,
        mpc_action: int,
        plan_info: dict,
    ) -> tuple[int, str]:
        if plan_info["best_predicted_life_lost"] and not self.allow_unsafe_search:
            return follow_action, "fallback_life"
        if plan_info["advantage_score"] <= self.teacher_margin:
            return follow_action, "follow_margin"
        if plan_info.get("beats_followball"):
            return mpc_action, "teacher_beat"
        if plan_info["advantage_score"] > 0:
            return mpc_action, "teacher_advantage"
        return follow_action, "follow_default"

    def act(
        self,
        obs: Any,
        info: Optional[dict] = None,
        env: Optional[CatBreakEnv] = None,
    ) -> int:
        if env is None:
            raise ValueError("CEM-MPC requires env for planning.")

        follow_action = self._follow_agent.act(obs, info, env=env)
        env_seed = int(getattr(env, "_last_seed", 0) or 0)

        if env.ball_vy > 0 or env.ball_y > S.FIELD_HEIGHT * 0.5:
            chosen = follow_action
            mode = "track"
            plan_info = {"mode": mode, "action_source": "followball_track"}
            self._append_teacher_log(env, chosen, follow_action, plan_info, opportunity=False)
            return chosen

        opportunity = is_opportunity_state(
            env, env_seed,
            focus_endgame=self.focus_endgame,
            focus_stuck=self.focus_stuck,
            no_brick_streak=self._no_brick_streak,
            stuck_threshold=self.stuck_threshold,
        )

        if not opportunity:
            plan_info = {"mode": "cheap_follow", "action_source": "followball_cheap"}
            self._append_teacher_log(env, follow_action, follow_action, plan_info, opportunity=False)
            return follow_action

        plan_info = self.plan(env)
        mpc_action = int(plan_info["action"])
        chosen, mode = self._choose_teacher_action(follow_action, mpc_action, plan_info)
        plan_info["mode"] = mode
        plan_info["action"] = chosen
        plan_info["action_source"] = mode
        self._append_teacher_log(env, chosen, follow_action, plan_info, opportunity=True)
        return chosen

    def _append_teacher_log(
        self,
        env: CatBreakEnv,
        chosen: int,
        follow_action: int,
        plan_info: dict,
        opportunity: bool,
    ) -> None:
        entry = {
            "chosen_action": chosen,
            "chosen_action_name": S.ACTION_NAMES.get(chosen, "?"),
            "followball_action": follow_action,
            "followball_action_name": S.ACTION_NAMES.get(follow_action, "?"),
            "mode": plan_info.get("mode", ""),
            "action_source": plan_info.get("action_source", plan_info.get("mode", "")),
            "is_opportunity": int(opportunity),
            "is_safe_aim": int(is_safe_aim_state(env.ball_vy, env.ball_y)),
            "advantage_score": plan_info.get("advantage_score", 0.0),
            "teacher_score": plan_info.get("teacher_score", 0.0),
            "follow_teacher_score": plan_info.get("follow_teacher_score", 0.0),
            "mpc_rollout_bricks": plan_info.get("mpc_rollout_bricks", 0),
            "followball_rollout_bricks": plan_info.get("followball_rollout_bricks", 0),
            "best_predicted_bricks": plan_info.get("best_predicted_bricks", 0),
            "follow_predicted_bricks": plan_info.get("follow_predicted_bricks", 0),
            "mpc_horizon_bricks": plan_info.get("best_predicted_bricks", 0),
            "followball_horizon_bricks": plan_info.get("follow_predicted_bricks", 0),
            "best_predicted_life_lost": int(bool(plan_info.get("best_predicted_life_lost", False))),
            "beats_followball": int(plan_info.get("beats_followball", False)),
            "best_sequence_prefix": plan_info.get("best_sequence_prefix", ""),
            "plan_time_ms": plan_info.get("plan_time_ms", 0.0),
            "ball_x": env.ball_x,
            "ball_y": env.ball_y,
            "ball_vx": env.ball_vx,
            "ball_vy": env.ball_vy,
            "paddle_x": env.paddle_x,
            "predicted_landing_x": plan_info.get(
                "predicted_landing_x",
                predict_landing_x(env.ball_x, env.ball_y, env.ball_vx, env.ball_vy, env.paddle_y),
            ),
            "target_hit_offset": plan_info.get("target_hit_offset", 0.0),
            "remaining_bricks": int(env._bricks.sum()),
            "no_brick_broken_for": self._no_brick_streak,
        }
        self._plan_log.append(entry)
        self._teacher_comparisons.append(entry.copy())
        if plan_info.get("beats_followball") and chosen != follow_action:
            self._teacher_better.append(entry.copy())

    def consume_teacher_data(self) -> tuple[list[dict], list[dict]]:
        comps = self._teacher_comparisons
        better = self._teacher_better
        self._teacher_comparisons = []
        self._teacher_better = []
        return comps, better

    def add_snapshot(self, record: dict) -> None:
        self._snapshots.append(record)

    def consume_snapshots(self) -> list[dict]:
        snaps = self._snapshots
        self._snapshots = []
        return snaps
