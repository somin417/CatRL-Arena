"""CEM-Aim v2: fast learned policy with FollowBall safety prior."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

import settings as S
from catbreak_env import obs_vector_for_agent
from cem_mpc import predict_landing_x

NUM_PARAMS = 12

# Fixed danger feature weights (normalized coordinates).
DANGER_C1 = 1.5
DANGER_C2 = 2.0
DANGER_C3 = 1.5

PADDLE_HALF_WIDTH_NORM = (S.PADDLE_WIDTH / 2.0) / S.FIELD_WIDTH


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def _brick_stats_from_matrix(
    bricks: np.ndarray,
    layout,
) -> tuple[float, float, float, float, float]:
    """Return centroid_x/y (norm), left_frac, right_frac, upper_frac."""
    rows, cols = bricks.shape
    alive_centers: list[tuple[float, float]] = []
    left_count = right_count = upper_count = total = 0
    mid_x = S.FIELD_WIDTH / 2.0
    mid_y = S.FIELD_HEIGHT / 2.0

    for row in range(rows):
        for col in range(cols):
            if not bricks[row, col]:
                continue
            rx, ry, rw, rh = layout.brick_rect(row, col)
            cx = rx + rw / 2.0
            cy = ry + rh / 2.0
            alive_centers.append((cx, cy))
            total += 1
            if cx < mid_x:
                left_count += 1
            else:
                right_count += 1
            if cy < mid_y:
                upper_count += 1

    if total == 0:
        return 0.5, 0.5, 0.5, 0.5, 0.5

    xs = [c[0] for c in alive_centers]
    ys = [c[1] for c in alive_centers]
    left_frac = left_count / total
    right_frac = right_count / total
    upper_frac = upper_count / total
    return (
        float(np.mean(xs) / S.FIELD_WIDTH),
        float(np.mean(ys) / S.FIELD_HEIGHT),
        left_frac,
        right_frac,
        upper_frac,
    )


def compute_features(
    obs: Any,
    env: Optional[object] = None,
) -> dict[str, float]:
    """Build normalized policy features without mutating env."""
    vec = obs_vector_for_agent(obs)
    ball_x = float(vec[0])
    ball_y = float(vec[1])
    ball_vx = float(vec[2])
    ball_vy = float(vec[3])
    paddle_x = float(vec[4])
    paddle_vx = float(vec[5])

    brick_centroid_x = 0.5
    brick_centroid_y = 0.5
    left_brick_fraction = 0.5
    right_brick_fraction = 0.5
    upper_brick_fraction = 0.5

    if env is not None:
        bricks = np.asarray(env._bricks, dtype=bool)
        layout = env._layout
        (
            brick_centroid_x,
            brick_centroid_y,
            left_brick_fraction,
            right_brick_fraction,
            upper_brick_fraction,
        ) = _brick_stats_from_matrix(bricks, layout)
        landing_world = predict_landing_x(
            env.ball_x,
            env.ball_y,
            env.ball_vx,
            env.ball_vy,
            env.paddle_y,
        )
    else:
        brick_n = len(vec) - 8
        if brick_n > 0:
            brick_flat = vec[6 : 6 + brick_n] > 0.5
            if brick_flat.any():
                rows = int(np.sqrt(brick_n))
                while rows > 1 and brick_n % rows != 0:
                    rows -= 1
                cols = brick_n // max(rows, 1)
                bricks = brick_flat.reshape(rows, cols)
                try:
                    from cat_layout import get_layout

                    layout = get_layout(S.DEFAULT_LAYOUT)
                    if layout.rows == rows and layout.cols == cols:
                        (
                            brick_centroid_x,
                            brick_centroid_y,
                            left_brick_fraction,
                            right_brick_fraction,
                            upper_brick_fraction,
                        ) = _brick_stats_from_matrix(bricks, layout)
                except Exception:
                    pass
        landing_world = predict_landing_x(
            ball_x * S.FIELD_WIDTH,
            ball_y * S.FIELD_HEIGHT,
            ball_vx * S.BALL_SPEED_MAX,
            ball_vy * S.BALL_SPEED_MAX,
            S.FIELD_HEIGHT - S.PADDLE_Y_OFFSET,
        )

    predicted_landing_x = float(np.clip(landing_world / S.FIELD_WIDTH, 0.0, 1.0))
    danger_level = (
        DANGER_C1 * ball_y
        + DANGER_C2 * abs(predicted_landing_x - paddle_x)
        + DANGER_C3 * float(ball_vy > 0)
    )

    return {
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_vx": ball_vx,
        "ball_vy": ball_vy,
        "paddle_x": paddle_x,
        "paddle_vx": paddle_vx,
        "predicted_landing_x": predicted_landing_x,
        "brick_centroid_x": brick_centroid_x,
        "brick_centroid_y": brick_centroid_y,
        "left_brick_fraction": left_brick_fraction,
        "right_brick_fraction": right_brick_fraction,
        "upper_brick_fraction": upper_brick_fraction,
        "danger_level": danger_level,
        "paddle_half_width_normalized": PADDLE_HALF_WIDTH_NORM,
    }


class CEMAimPolicy:
    """12-parameter safety-gated aiming policy (no runtime simulation)."""

    def __init__(self, theta: Optional[np.ndarray] = None) -> None:
        if theta is None:
            self.theta = np.zeros(NUM_PARAMS, dtype=np.float64)
        else:
            arr = np.asarray(theta, dtype=np.float64).ravel()
            if arr.size != NUM_PARAMS:
                raise ValueError(f"CEMAimPolicy expects {NUM_PARAMS} params, got {arr.size}")
            self.theta = arr.copy()

    def copy(self) -> CEMAimPolicy:
        return CEMAimPolicy(self.theta)

    def act(
        self,
        obs: Any,
        info: Optional[dict] = None,
        env: Optional[object] = None,
    ) -> int:
        feats = compute_features(obs, env=env)
        theta = self.theta

        raw_offset = (
            theta[0]
            + theta[1] * (feats["brick_centroid_x"] - 0.5)
            + theta[2]
            * (feats["right_brick_fraction"] - feats["left_brick_fraction"])
            + theta[3] * feats["ball_vx"]
            + theta[4] * (feats["brick_centroid_x"] - feats["ball_x"])
            + theta[5] * feats["upper_brick_fraction"]
            + theta[6]
            * np.sign(feats["brick_centroid_x"] - feats["predicted_landing_x"])
        )

        max_offset = 0.85 * float(sigmoid(theta[7]))
        desired_offset = float(np.tanh(raw_offset) * max_offset)

        danger = feats["danger_level"]
        safety = float(sigmoid(theta[8] * danger + theta[9]))
        desired_offset *= 1.0 - safety

        if feats["ball_vy"] < 0:
            rest_blend = float(sigmoid(theta[10]))
            target_paddle_center = (
                rest_blend * feats["brick_centroid_x"]
                + (1.0 - rest_blend) * 0.5
            )
        else:
            target_paddle_center = (
                feats["predicted_landing_x"]
                - desired_offset * feats["paddle_half_width_normalized"]
            )

        threshold = 0.015 + 0.05 * float(sigmoid(theta[11]))
        delta = feats["paddle_x"] - target_paddle_center

        if delta < -threshold:
            return S.ACTION_RIGHT
        if delta > threshold:
            return S.ACTION_LEFT
        return S.ACTION_STAY

    def save(self, path: str) -> None:
        np.savez_compressed(path, theta=self.theta, num_params=NUM_PARAMS, version=2)

    @classmethod
    def load(cls, path: str) -> CEMAimPolicy:
        data = np.load(path, allow_pickle=False)
        return cls(theta=data["theta"])

    @staticmethod
    def prior_follow_like() -> np.ndarray:
        """Near FollowBall: tiny offset, strong safety gate."""
        theta = np.zeros(NUM_PARAMS, dtype=np.float64)
        theta[7] = -6.0
        theta[8] = 4.0
        theta[9] = -1.0
        theta[10] = -4.0
        theta[11] = 0.0
        return theta

    @staticmethod
    def prior_left_aim() -> np.ndarray:
        theta = np.zeros(NUM_PARAMS, dtype=np.float64)
        theta[1] = -1.5
        theta[6] = -1.0
        theta[7] = 1.0
        theta[8] = 2.0
        theta[9] = -2.5
        theta[10] = 2.0
        return theta

    @staticmethod
    def prior_right_aim() -> np.ndarray:
        theta = np.zeros(NUM_PARAMS, dtype=np.float64)
        theta[1] = 1.5
        theta[6] = 1.0
        theta[7] = 1.0
        theta[8] = 2.0
        theta[9] = -2.5
        theta[10] = -2.0
        return theta

    @staticmethod
    def prior_centroid_aim() -> np.ndarray:
        theta = np.zeros(NUM_PARAMS, dtype=np.float64)
        theta[1] = 2.0
        theta[4] = 1.5
        theta[7] = 0.5
        theta[8] = 2.0
        theta[9] = -2.0
        theta[10] = 4.0
        return theta


def build_prior_candidates(
    include_follow: bool = True,
    include_targeted: bool = True,
    previous_best: Optional[np.ndarray] = None,
    mpc_best: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    """Fixed prior thetas injected every CEM generation."""
    priors: list[np.ndarray] = []
    if include_follow:
        priors.append(CEMAimPolicy.prior_follow_like())
    if include_targeted:
        priors.extend([
            CEMAimPolicy.prior_left_aim(),
            CEMAimPolicy.prior_right_aim(),
            CEMAimPolicy.prior_centroid_aim(),
        ])
    if previous_best is not None:
        priors.append(np.asarray(previous_best, dtype=np.float64).copy())
    if mpc_best is not None:
        priors.append(np.asarray(mpc_best, dtype=np.float64).copy())
    return priors


# ---------------------------------------------------------------------------
# CEM-Aim v3: residual option policy (opportunity-gated, FollowBall-safe)
# ---------------------------------------------------------------------------

POLICY_VERSION_V2 = "cem_aim_v2"
POLICY_VERSION_V3 = "cem_aim_v3_residual_option"
FOLLOWBALL_THRESHOLD = S.FOLLOW_BALL_THRESHOLD


def exact_followball_action(
    obs_or_features: Any,
    threshold: float = FOLLOWBALL_THRESHOLD,
    env: Optional[object] = None,
) -> int:
    """Reproduce FollowBall baseline exactly (normalized coordinates)."""
    if isinstance(obs_or_features, dict):
        ball_x = float(obs_or_features.get("ball_x", 0.5))
        paddle_x = float(obs_or_features.get("paddle_x", 0.5))
    else:
        vec = obs_vector_for_agent(obs_or_features)
        ball_x = float(vec[0])
        paddle_x = float(vec[4])
    return move_toward_target(paddle_x, ball_x, threshold)


def move_toward_target(
    paddle_x: float,
    target_x: float,
    threshold: float = FOLLOWBALL_THRESHOLD,
) -> int:
    if paddle_x > target_x + threshold:
        return S.ACTION_LEFT
    if paddle_x < target_x - threshold:
        return S.ACTION_RIGHT
    return S.ACTION_STAY


def _estimate_time_to_paddle(
    ball_y_norm: float,
    ball_vy_world: float,
    paddle_y_norm: float,
) -> float:
    """Steps until ball reaches paddle row; inf if not descending."""
    if ball_vy_world <= 1e-9:
        return float("inf")
    ball_y_world = ball_y_norm * S.FIELD_HEIGHT
    paddle_y_world = paddle_y_norm * S.FIELD_HEIGHT
    if ball_y_world <= paddle_y_world:
        return float("inf")
    dt_steps = (ball_y_world - paddle_y_world) / (ball_vy_world * S.FIXED_DT)
    return max(0.0, float(dt_steps))


def extract_cem_aim_features(
    obs: Any,
    info: Optional[dict] = None,
    env: Optional[object] = None,
) -> dict[str, float]:
    """Robust feature dict for CEM-Aim v3 (safe defaults on missing fields)."""
    base = compute_features(obs, env=env)
    vec = obs_vector_for_agent(obs)
    remaining_bricks = int(base.get("remaining_bricks", 0))
    step_count = 0
    no_brick_broken_for = 0
    max_total_bricks = S.CAT_LAYOUT_BRICKS if hasattr(S, "CAT_LAYOUT_BRICKS") else 82

    if env is not None:
        remaining_bricks = int(env._bricks.sum())
        step_count = int(env.step_count)
        max_total_bricks = int(getattr(env, "_initial_brick_count", remaining_bricks))
        no_brick_broken_for = int(getattr(env, "_no_brick_streak", 0))
    elif len(vec) > 8:
        brick_n = len(vec) - 8
        brick_flat = vec[6 : 6 + brick_n] > 0.5
        remaining_bricks = int(brick_flat.sum())
        if len(vec) > 6 + brick_n:
            step_count = int(vec[7 + brick_n] * S.MAX_STEPS)

    if info:
        remaining_bricks = int(info.get("remaining_bricks", remaining_bricks))
        step_count = int(info.get("step_count", step_count))
        no_brick_broken_for = int(info.get("no_brick_broken_for", no_brick_broken_for))

    left_frac = float(base["left_brick_fraction"])
    right_frac = float(base["right_brick_fraction"])
    upper_frac = float(base["upper_brick_fraction"])
    ball_vy = float(base["ball_vy"])
    ball_y = float(base["ball_y"])
    if env is not None:
        paddle_y_norm = float(env.paddle_y) / S.FIELD_HEIGHT
        ttp = _estimate_time_to_paddle(ball_y, env.ball_vy, paddle_y_norm)
    else:
        paddle_y_norm = (S.FIELD_HEIGHT - S.PADDLE_Y_OFFSET) / S.FIELD_HEIGHT
        ttp = _estimate_time_to_paddle(
            ball_y, ball_vy * S.BALL_SPEED_MAX, paddle_y_norm
        )

    return {
        **base,
        "remaining_bricks": remaining_bricks,
        "max_total_bricks": max_total_bricks,
        "left_frac": left_frac,
        "right_frac": right_frac,
        "upper_frac": upper_frac,
        "right_frac_minus_left_frac": right_frac - left_frac,
        "step_count": step_count,
        "no_brick_broken_for": no_brick_broken_for,
        "time_to_paddle": ttp,
        "paddle_half_width_norm": float(base["paddle_half_width_normalized"]),
    }


def is_opportunity_state(
    features: dict[str, float],
    force_opportunity: bool = False,
    stress_seed: bool = False,
) -> bool:
    if force_opportunity or stress_seed:
        return True
    remaining = int(features.get("remaining_bricks", 999))
    step_count = int(features.get("step_count", 0))
    no_brick = int(features.get("no_brick_broken_for", 0))
    left_frac = float(features.get("left_frac", 0.5))
    right_frac = float(features.get("right_frac", 0.5))
    upper_frac = float(features.get("upper_frac", 0.5))
    imbalance = abs(right_frac - left_frac)

    if remaining <= 4:
        return True
    if step_count > 8000 and remaining <= 10:
        return True
    if no_brick >= 1200:
        return True
    if imbalance >= 0.45 and remaining <= 15:
        return True
    if (left_frac >= 0.75 or right_frac >= 0.75) and remaining <= 15:
        return True
    if upper_frac >= 0.55 and remaining <= 12:
        return True
    return False


def is_unsafe_for_residual(features: dict[str, float]) -> bool:
    ball_vy = float(features.get("ball_vy", 0.0))
    ball_y = float(features.get("ball_y", 0.0))
    paddle_x = float(features.get("paddle_x", 0.5))
    landing = float(features.get("predicted_landing_x", 0.5))
    centroid = float(features.get("brick_centroid_x", 0.5))
    remaining = int(features.get("remaining_bricks", 1))
    ttp = float(features.get("time_to_paddle", float("inf")))

    if remaining <= 0:
        return True
    if ball_vy > 0 and ball_y > 0.45:
        return True
    if ball_vy > 0 and ttp < 25.0:
        return True
    if not np.isfinite(landing) or landing < 0.0 or landing > 1.0:
        return True
    if abs(landing - paddle_x) > 0.12:
        return True
    if not np.isfinite(centroid):
        return True
    return False


COMMIT_HIT_OFFSET_MIN = 0.04
COMMIT_MISS_BALL_X_GAP = 0.18
COMMIT_REACH_MARGIN_STEPS = 3.0
COMMITTED_THRESHOLD = 0.025
OFFSET_RANGE = 0.25
MACRO_OFFSET_FLOORS = (0.08, 0.12, 0.18, 0.25)

SIDE_RULES = (
    "toward_centroid",
    "away_from_centroid",
    "with_ball_vx",
    "against_ball_vx",
    "left",
    "right",
    "sparse_side",
    "dense_side",
)

TARGET_BASES = (
    "predicted_landing",
    "ball_x",
    "predicted_landing_plus_ball_vx_lead",
)


@dataclass
class ShotMacroConfig:
    """Deterministic endgame shot macro mapped to 12-theta checkpoints."""

    remaining_gate: int = 10
    step_gate: int = 8000
    no_brick_gate: int = 1200
    offset_mag: float = 0.25
    offset_floor: float = 0.10
    side_rule: str = "toward_centroid"
    target_base: str = "predicted_landing"

    def to_dict(self) -> dict:
        return {
            "remaining_gate": self.remaining_gate,
            "step_gate": self.step_gate,
            "no_brick_gate": self.no_brick_gate,
            "offset_mag": self.offset_mag,
            "offset_floor": self.offset_floor,
            "side_rule": self.side_rule,
            "target_base": self.target_base,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ShotMacroConfig:
        return cls(
            remaining_gate=int(d.get("remaining_gate", 10)),
            step_gate=int(d.get("step_gate", 8000)),
            no_brick_gate=int(d.get("no_brick_gate", 1200)),
            offset_mag=float(d.get("offset_mag", 0.25)),
            offset_floor=float(d.get("offset_floor", 0.10)),
            side_rule=str(d.get("side_rule", "toward_centroid")),
            target_base=str(d.get("target_base", "predicted_landing")),
        )


@dataclass
class CEMAimV3EpisodeState:
    """Per-episode shot commitment state for v3 residual policy."""

    committed: bool = False
    committed_target_x: float = 0.5
    committed_offset: float = 0.0
    committed_side: int = 0
    committed_start_step: int = -1
    committed_until_contact: bool = False
    last_ball_vy: float = 0.0
    last_contact_step: int = -1
    num_committed_shots: int = 0
    num_successful_committed_shots: int = 0
    steps_since_commit: int = 0
    bricks_at_commit: int = 0
    endgame_at_commit: bool = False
    opportunity_steps: int = 0
    total_steps: int = 0
    committed_contacts: int = 0
    delta_bricks_after_commit: list[int] = field(default_factory=list)
    endgame_committed_wins: int = 0
    post_contact_ball_vx: float = 0.0
    post_contact_ball_vy: float = 0.0
    last_hit_offset: float = 0.0
    steps_to_next_brick_after_contact: list[int] = field(default_factory=list)
    last_committed_cancel: bool = False
    last_committed_contact: bool = False
    _contact_step_marker: int = -1
    _bricks_at_contact: int = 0

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]

    def summary(self) -> dict[str, float]:
        hit_offsets = [abs(self.last_hit_offset)] if self.committed_contacts else []
        return {
            "committed_shot_rate": self.num_committed_shots / max(1, self.opportunity_steps),
            "committed_contact_rate": self.committed_contacts / max(1, self.num_committed_shots),
            "committed_shot_success_rate": (
                self.num_successful_committed_shots / max(1, self.committed_contacts)
            ),
            "delta_next_brick_after_committed_shot": float(
                np.mean(self.delta_bricks_after_commit)
            ) if self.delta_bricks_after_commit else 0.0,
            "endgame_committed_shot_wins": float(self.endgame_committed_wins),
            "num_committed_shots": float(self.num_committed_shots),
            "committed_contacts": float(self.committed_contacts),
            "opportunity_steps": float(self.opportunity_steps),
            "mean_abs_hit_offset_on_contact": float(np.mean(hit_offsets)) if hit_offsets else 0.0,
            "steps_to_next_brick_after_contact": float(
                np.mean(self.steps_to_next_brick_after_contact)
            ) if self.steps_to_next_brick_after_contact else 0.0,
        }


def _commitment_debug(state: CEMAimV3EpisodeState) -> dict[str, Any]:
    return {
        "committed": state.committed,
        "committed_target_x": state.committed_target_x,
        "committed_offset": state.committed_offset,
        "committed_side": state.committed_side,
        "committed_start_step": state.committed_start_step,
        "committed_until_contact": state.committed_until_contact,
        "steps_since_commit": state.steps_since_commit,
        "num_committed_shots": state.num_committed_shots,
        "num_successful_committed_shots": state.num_successful_committed_shots,
        "commitment_cancelled": state.last_committed_cancel,
        "committed_cancel": state.last_committed_cancel,
        "contact_detected": state.last_committed_contact,
        "committed_contact": state.last_committed_contact,
        "hit_offset": state.last_hit_offset,
        "post_contact_ball_vx": state.post_contact_ball_vx,
        "post_contact_ball_vy": state.post_contact_ball_vy,
        "steps_to_next_brick_after_contact": (
            float(np.mean(state.steps_to_next_brick_after_contact))
            if state.steps_to_next_brick_after_contact else 0.0
        ),
        "meaningful_commit": False,
    }


def _cancel_commitment(state: CEMAimV3EpisodeState) -> None:
    state.committed = False
    state.committed_until_contact = False
    state.steps_since_commit = 0


def _paddle_steps_to_reach_norm(paddle_x: float, target_x: float) -> float:
    dist_world = abs(target_x - paddle_x) * S.FIELD_WIDTH
    step_delta = max(S.PADDLE_SPEED * S.FIXED_DT, 1e-6)
    return dist_world / step_delta


def is_emergency_unsafe_committed(
    features: dict[str, float],
    state: CEMAimV3EpisodeState,
    env: Optional[object] = None,
) -> bool:
    """Emergency cancel only for invalid targets or obvious unrecoverable miss."""
    target = float(state.committed_target_x)
    if not np.isfinite(target) or target < 0.0 or target > 1.0:
        return True

    landing = float(features.get("predicted_landing_x", 0.5))
    if not np.isfinite(landing) or landing < 0.0 or landing > 1.0:
        return True

    ball_y = float(features.get("ball_y", 0.0))
    ball_vy = float(features.get("ball_vy", 0.0))
    ball_x = float(features.get("ball_x", 0.5))

    if env is not None:
        paddle_y_norm = float(getattr(env, "paddle_y", S.FIELD_HEIGHT - S.PADDLE_Y_OFFSET)) / S.FIELD_HEIGHT
    else:
        paddle_y_norm = (S.FIELD_HEIGHT - S.PADDLE_Y_OFFSET) / S.FIELD_HEIGHT

    # Ball already passed paddle while descending.
    if ball_vy > 0.0 and ball_y > paddle_y_norm + 0.02:
        return True

    # Obvious miss: ball descending near paddle row but far from committed target.
    if ball_vy > 0.0 and ball_y > paddle_y_norm - 0.08:
        if abs(ball_x - target) > COMMIT_MISS_BALL_X_GAP:
            return True

    return False


def _detect_paddle_contact(
    state: CEMAimV3EpisodeState,
    env: Optional[object],
    info: Optional[dict],
) -> bool:
    if info and bool(info.get("paddle_hit", 0)):
        return True
    if env is None:
        return False
    ball_vy = float(getattr(env, "ball_vy", 0.0))
    ball_y = float(getattr(env, "ball_y", 0.0))
    paddle_y = float(getattr(env, "paddle_y", S.FIELD_HEIGHT - S.PADDLE_Y_OFFSET))
    if state.last_ball_vy > 0.0 and ball_vy < 0.0:
        if ball_y >= paddle_y - S.BALL_RADIUS - 6.0:
            return True
    return False


def _pick_offset_floor(offset_mag: float) -> float:
    eligible = [f for f in MACRO_OFFSET_FLOORS if f <= offset_mag + 1e-9]
    return eligible[-1] if eligible else MACRO_OFFSET_FLOORS[0]


def is_macro_opportunity(feats: dict[str, float], macro: ShotMacroConfig) -> bool:
    remaining = int(feats.get("remaining_bricks", 999))
    step_count = int(feats.get("step_count", 0))
    no_brick = int(feats.get("no_brick_broken_for", 0))
    return (
        remaining <= macro.remaining_gate
        or step_count >= macro.step_gate
        or no_brick >= macro.no_brick_gate
    )


def _macro_side(feats: dict[str, float], side_rule: str) -> int:
    centroid = float(feats.get("brick_centroid_x", 0.5))
    landing = float(feats.get("predicted_landing_x", 0.5))
    ball_vx = float(feats.get("ball_vx", 0.0))
    left_frac = float(feats.get("left_frac", feats.get("left_brick_fraction", 0.5)))
    right_frac = float(feats.get("right_frac", feats.get("right_brick_fraction", 0.5)))

    if side_rule == "toward_centroid":
        side = int(np.sign(centroid - landing))
    elif side_rule == "away_from_centroid":
        side = int(np.sign(landing - centroid))
    elif side_rule == "with_ball_vx":
        side = int(np.sign(ball_vx)) or 1
    elif side_rule == "against_ball_vx":
        side = -int(np.sign(ball_vx)) or -1
    elif side_rule == "left":
        side = -1
    elif side_rule == "right":
        side = 1
    elif side_rule == "sparse_side":
        side = -1 if left_frac < right_frac else 1
    elif side_rule == "dense_side":
        side = 1 if left_frac < right_frac else -1
    else:
        side = int(np.sign(centroid - landing))
    if side == 0:
        side = 1
    return side


def _macro_target_base(feats: dict[str, float], target_base: str) -> float:
    landing = float(feats.get("predicted_landing_x", 0.5))
    ball_x = float(feats.get("ball_x", 0.5))
    ball_vx = float(feats.get("ball_vx", 0.0))
    if target_base == "ball_x":
        return ball_x
    if target_base == "predicted_landing_plus_ball_vx_lead":
        return float(np.clip(landing + 0.12 * ball_vx, 0.0, 1.0))
    return landing


def _compute_macro_target(
    feats: dict[str, float],
    macro: ShotMacroConfig,
) -> tuple[float, float, int]:
    side = _macro_side(feats, macro.side_rule)
    offset_mag = max(macro.offset_mag, macro.offset_floor)
    desired_offset = side * offset_mag
    base = _macro_target_base(feats, macro.target_base)
    target = base - desired_offset * float(feats.get("paddle_half_width_norm", PADDLE_HALF_WIDTH_NORM))
    return float(np.clip(target, 0.0, 1.0)), desired_offset, side


def macro_config_to_theta(macro: ShotMacroConfig) -> np.ndarray:
    """Encode a shot macro as a valid 12-theta vector."""
    floor = macro.offset_floor if macro.offset_floor > 0 else _pick_offset_floor(macro.offset_mag)
    extra = max(0.0, macro.offset_mag - floor)
    sig = float(np.clip(extra / OFFSET_RANGE, 1e-6, 1.0 - 1e-6))
    theta = np.zeros(NUM_PARAMS, dtype=np.float64)
    theta[0] = 8.0
    theta[1] = -4.0 * (macro.remaining_gate / 82.0)
    theta[2] = 4.0 * (macro.no_brick_gate / 2000.0)
    theta[3] = 2.0
    theta[4] = 1.5
    theta[5] = 1.0
    side = _macro_side(
        {
            "brick_centroid_x": 0.3,
            "predicted_landing_x": 0.5,
            "ball_vx": 0.2,
            "left_frac": 0.3,
            "right_frac": 0.7,
        },
        macro.side_rule,
    )
    theta[6] = float(side) * 3.0
    theta[7] = float(side) * 2.0
    theta[8] = float(side) * 1.0 if macro.side_rule in ("with_ball_vx", "against_ball_vx") else 0.0
    theta[9] = float(side) * 2.0
    theta[10] = float(np.log(sig / (1.0 - sig)))
    theta[11] = 4.0
    macro.offset_floor = floor
    return theta


def _compute_residual_target(
    feats: dict[str, float],
    theta: np.ndarray,
    *,
    offset_floor: float = 0.05,
    offset_range: float = OFFSET_RANGE,
) -> tuple[float, float, int]:
    target_side_score = (
        theta[6] * (float(feats["brick_centroid_x"]) - 0.5)
        + theta[7] * float(feats["right_frac_minus_left_frac"])
        + theta[8] * float(feats["ball_vx"])
        + theta[9] * np.sign(float(feats["brick_centroid_x"]) - float(feats["predicted_landing_x"]))
    )
    side = int(np.sign(target_side_score))
    if side == 0:
        side = int(np.sign(float(feats["brick_centroid_x"]) - float(feats["predicted_landing_x"])))
    if side == 0:
        side = 1

    offset_mag = offset_floor + offset_range * float(sigmoid(theta[10]))
    edge_mix = float(sigmoid(theta[11]))
    desired_offset = side * offset_mag * (0.5 + 0.5 * edge_mix)
    target = float(feats["predicted_landing_x"]) - desired_offset * float(
        feats["paddle_half_width_norm"]
    )
    return float(np.clip(target, 0.0, 1.0)), desired_offset, side


def _start_commitment(
    state: CEMAimV3EpisodeState,
    feats: dict[str, float],
    target: float,
    desired_offset: float,
    side: int,
) -> None:
    state.committed = True
    state.committed_until_contact = True
    state.committed_target_x = float(target)
    state.committed_offset = float(desired_offset)
    state.committed_side = int(side)
    state.committed_start_step = int(feats.get("step_count", 0))
    state.num_committed_shots += 1
    state.steps_since_commit = 0
    state.bricks_at_commit = int(feats.get("remaining_bricks", 0))
    state.endgame_at_commit = state.bricks_at_commit <= 15


def note_cem_aim_v3_after_step(
    state: CEMAimV3EpisodeState,
    env: Optional[object],
    info: Optional[dict],
) -> dict[str, Any]:
    """Update commitment state after env.step (contact detection)."""
    state.last_committed_cancel = False
    state.last_committed_contact = False

    contact_dbg: dict[str, Any] = {
        "contact_detected": False,
        "committed_contact": False,
        "committed_cancel": False,
        "hit_offset": 0.0,
        "post_contact_ball_vx": 0.0,
        "post_contact_ball_vy": 0.0,
        "steps_to_next_brick_after_contact": 0.0,
        "meaningful_commit": False,
    }
    state.total_steps += 1
    bricks_before = int(info.get("remaining_bricks", 0)) if info else 0
    if env is not None:
        bricks_before = int(getattr(env, "_bricks", np.array([])).sum())

    if _detect_paddle_contact(state, env, info):
        contact_dbg["contact_detected"] = True
        contact_dbg["committed_contact"] = True
        state.last_committed_contact = True
        hit_offset = float(info.get("hit_offset", 0.0)) if info else 0.0
        collision = getattr(env, "_last_paddle_collision", None) if env else None
        if collision:
            contact_dbg["post_contact_ball_vx"] = float(collision.get("ball_vx_after", 0.0))
            contact_dbg["post_contact_ball_vy"] = float(collision.get("ball_vy_after", 0.0))
            hit_offset = float(collision.get("hit_offset", hit_offset))
            state.post_contact_ball_vx = contact_dbg["post_contact_ball_vx"]
            state.post_contact_ball_vy = contact_dbg["post_contact_ball_vy"]
        state.last_hit_offset = hit_offset
        contact_dbg["hit_offset"] = hit_offset

        was_committed = state.committed or state.committed_until_contact
        if was_committed:
            state.committed_contacts += 1
            bricks_now = int(info.get("remaining_bricks", 0)) if info else 0
            if env is not None:
                bricks_now = int(getattr(env, "_bricks", np.array([])).sum())
            delta_bricks = max(0, state.bricks_at_commit - bricks_now)
            state.delta_bricks_after_commit.append(delta_bricks)
            meaningful = (
                abs(hit_offset) >= COMMIT_HIT_OFFSET_MIN
                or delta_bricks > 0
                or abs(state.committed_offset) >= COMMIT_HIT_OFFSET_MIN
            )
            contact_dbg["meaningful_commit"] = meaningful
            if meaningful:
                state.num_successful_committed_shots += 1
            if state.endgame_at_commit and delta_bricks > 0:
                state.endgame_committed_wins += 1
            state.last_contact_step = int(info.get("step_count", -1)) if info else -1
            state._contact_step_marker = state.total_steps
            state._bricks_at_contact = bricks_now
            _cancel_commitment(state)

    if state._contact_step_marker >= 0:
        bricks_now = int(info.get("remaining_bricks", 0)) if info else bricks_before
        if env is not None:
            bricks_now = int(getattr(env, "_bricks", np.array([])).sum())
        if bricks_now < state._bricks_at_contact:
            steps_after = state.total_steps - state._contact_step_marker
            state.steps_to_next_brick_after_contact.append(steps_after)
            contact_dbg["steps_to_next_brick_after_contact"] = float(steps_after)
            state._contact_step_marker = -1

    if env is not None:
        state.last_ball_vy = float(getattr(env, "ball_vy", 0.0))
    return contact_dbg


def act_cem_aim_v3(
    obs: Any,
    theta: np.ndarray,
    info: Optional[dict] = None,
    env: Optional[object] = None,
    return_debug: bool = False,
    force_opportunity: bool = False,
    disable_residual: bool = False,
    stress_seed: bool = False,
    teacher_imitate: bool = False,
    state: Optional[CEMAimV3EpisodeState] = None,
    macro_config: Optional[ShotMacroConfig] = None,
    offset_floor: float = 0.05,
) -> int | tuple[int, dict]:
    """CEM-Aim v3 residual option policy with shot commitment."""
    if state is None:
        state = CEMAimV3EpisodeState()

    feats = extract_cem_aim_features(obs, info=info, env=env)
    follow_action = exact_followball_action(feats)
    if macro_config is not None:
        opportunity = is_macro_opportunity(feats, macro_config)
    else:
        opportunity = is_opportunity_state(feats, force_opportunity, stress_seed)
    if opportunity:
        state.opportunity_steps += 1

    debug: dict[str, Any] = {
        "policy_version": POLICY_VERSION_V3,
        "follow_action": follow_action,
        "opportunity": opportunity,
        "unsafe": is_unsafe_for_residual(feats),
        "residual_activated": False,
        "gate_logit": 0.0,
        "desired_offset": 0.0,
        "target": feats["ball_x"],
        "final_action": follow_action,
        "deviated_from_followball": False,
        **_commitment_debug(state),
        **{k: feats.get(k, 0.0) for k in (
            "ball_x", "ball_y", "ball_vx", "ball_vy", "paddle_x",
            "predicted_landing_x", "brick_centroid_x", "remaining_bricks",
            "no_brick_broken_for", "time_to_paddle",
        )},
    }

    if state.committed and state.committed_until_contact and not disable_residual:
        state.steps_since_commit += 1
        debug["residual_activated"] = True
        debug["committed"] = True
        if is_emergency_unsafe_committed(feats, state, env=env):
            _cancel_commitment(state)
            state.last_committed_cancel = True
            debug["commitment_cancelled"] = True
            debug["committed_cancel"] = True
            debug["unsafe"] = True
            debug["committed"] = False
            debug["final_action"] = follow_action
            return (follow_action, debug) if return_debug else follow_action

        action = move_toward_target(
            float(feats["paddle_x"]),
            state.committed_target_x,
            threshold=COMMITTED_THRESHOLD,
        )
        debug["target"] = state.committed_target_x
        debug["desired_offset"] = state.committed_offset
        debug["final_action"] = action
        debug["deviated_from_followball"] = int(action) != int(follow_action)
        debug["meaningful_commit"] = state.committed_contacts > 0
        return (action, debug) if return_debug else action

    unsafe = bool(debug["unsafe"])
    if disable_residual or not opportunity or (unsafe and not teacher_imitate):
        debug["final_action"] = follow_action
        return (follow_action, debug) if return_debug else follow_action

    if macro_config is not None:
        residual_activated = True
        debug["gate_logit"] = 8.0
    else:
        max_bricks = max(1, int(feats.get("max_total_bricks", 82)))
        remaining_norm = float(feats["remaining_bricks"]) / max_bricks
        no_brick_norm = float(np.clip(feats["no_brick_broken_for"] / 2000.0, 0.0, 1.0))
        centroid_gap = abs(float(feats["brick_centroid_x"]) - float(feats["predicted_landing_x"]))
        imbalance = abs(float(feats["right_frac"]) - float(feats["left_frac"]))

        gate_logit = (
            theta[0]
            + theta[1] * remaining_norm
            + theta[2] * no_brick_norm
            + theta[3] * imbalance
            + theta[4] * float(feats["upper_frac"])
            + theta[5] * centroid_gap
        )
        residual_activated = float(sigmoid(gate_logit)) > 0.5
        debug["gate_logit"] = float(gate_logit)

    debug["residual_activated"] = residual_activated

    if not residual_activated:
        debug["final_action"] = follow_action
        return (follow_action, debug) if return_debug else follow_action

    if macro_config is not None:
        target, desired_offset, side = _compute_macro_target(feats, macro_config)
        floor = macro_config.offset_floor
    else:
        target, desired_offset, side = _compute_residual_target(
            feats, theta, offset_floor=offset_floor
        )
        floor = offset_floor

    _start_commitment(state, feats, target, desired_offset, side)
    action = move_toward_target(
        float(feats["paddle_x"]), target, threshold=COMMITTED_THRESHOLD
    )

    debug["desired_offset"] = desired_offset
    debug["target"] = target
    debug["offset_floor"] = floor
    debug["committed"] = True
    debug["committed_target_x"] = target
    debug["committed_offset"] = desired_offset
    debug["committed_side"] = side
    debug["committed_until_contact"] = True
    debug["final_action"] = action
    debug["deviated_from_followball"] = int(action) != int(follow_action)
    return (action, debug) if return_debug else action


class CEMAimV3Policy:
    """12-theta residual option policy (v3)."""

    policy_version = POLICY_VERSION_V3

    def __init__(
        self,
        theta: Optional[np.ndarray] = None,
        macro_config: Optional[ShotMacroConfig] = None,
        offset_floor: float = 0.05,
    ) -> None:
        if theta is None:
            self.theta = CEMAimV3Policy.prior_exact_follow().copy()
        else:
            arr = np.asarray(theta, dtype=np.float64).ravel()
            if arr.size != NUM_PARAMS:
                raise ValueError(f"CEMAimV3Policy expects {NUM_PARAMS} params, got {arr.size}")
            self.theta = arr.copy()
        self.macro_config = macro_config
        self.offset_floor = float(offset_floor)
        self._episode_state = CEMAimV3EpisodeState()

    def reset_episode(self, seed: Optional[int] = None) -> None:
        self._episode_state.reset()

    def note_step_after_env_step(
        self,
        info: Optional[dict] = None,
        env: Optional[object] = None,
    ) -> dict[str, Any]:
        return note_cem_aim_v3_after_step(self._episode_state, env, info)

    @property
    def episode_state(self) -> CEMAimV3EpisodeState:
        return self._episode_state

    def commitment_summary(self) -> dict[str, float]:
        return self._episode_state.summary()

    def act(
        self,
        obs: Any,
        info: Optional[dict] = None,
        env: Optional[object] = None,
        **kwargs: Any,
    ) -> int:
        result = act_cem_aim_v3(
            obs,
            self.theta,
            info=info,
            env=env,
            state=self._episode_state,
            macro_config=self.macro_config,
            offset_floor=self.offset_floor,
            **kwargs,
        )
        if isinstance(result, tuple):
            return int(result[0])
        return int(result)

    def act_debug(self, obs: Any, info=None, env=None, **kwargs) -> tuple[int, dict]:
        return act_cem_aim_v3(
            obs,
            self.theta,
            info=info,
            env=env,
            return_debug=True,
            state=self._episode_state,
            macro_config=self.macro_config,
            offset_floor=self.offset_floor,
            **kwargs,
        )

    def save(self, path: str, metrics: Optional[dict] = None) -> None:
        payload = {
            "theta": self.theta,
            "num_params": NUM_PARAMS,
            "version": 3,
            "policy_version": POLICY_VERSION_V3,
            "offset_floor": self.offset_floor,
        }
        if self.macro_config is not None:
            payload["macro_config"] = json.dumps(self.macro_config.to_dict())
        np.savez_compressed(path, **payload)
        if metrics is not None:
            out = dict(metrics)
            if self.macro_config is not None:
                out["macro_config"] = self.macro_config.to_dict()
            Path(path).with_suffix(".metrics.json").write_text(
                json.dumps(out, indent=2, default=str), encoding="utf-8"
            )

    @classmethod
    def load(cls, path: str) -> CEMAimV3Policy:
        data = np.load(path, allow_pickle=True)
        macro = None
        if "macro_config" in data.files:
            raw = data["macro_config"]
            if isinstance(raw, np.ndarray):
                raw = raw.item() if raw.ndim == 0 else raw.tolist()
            if isinstance(raw, (bytes, str)):
                macro = ShotMacroConfig.from_dict(json.loads(raw))
        offset_floor = float(data["offset_floor"]) if "offset_floor" in data.files else 0.05
        return cls(theta=data["theta"], macro_config=macro, offset_floor=offset_floor)

    @staticmethod
    def prior_exact_follow() -> np.ndarray:
        """Gate strongly off -> exact FollowBall."""
        theta = np.zeros(NUM_PARAMS, dtype=np.float64)
        theta[0] = -8.0
        return theta

    @staticmethod
    def prior_mild_left_endgame() -> np.ndarray:
        theta = np.zeros(NUM_PARAMS, dtype=np.float64)
        theta[0] = 2.0
        theta[1] = 3.0
        theta[6] = -1.5
        theta[7] = -1.0
        theta[10] = 0.5
        return theta

    @staticmethod
    def prior_mild_right_endgame() -> np.ndarray:
        theta = np.zeros(NUM_PARAMS, dtype=np.float64)
        theta[0] = 2.0
        theta[1] = 3.0
        theta[6] = 1.5
        theta[7] = 1.0
        theta[10] = 0.5
        return theta

    @staticmethod
    def prior_centroid_endgame() -> np.ndarray:
        theta = np.zeros(NUM_PARAMS, dtype=np.float64)
        theta[0] = 2.5
        theta[1] = 4.0
        theta[5] = 2.0
        theta[6] = 2.0
        theta[10] = 1.0
        return theta

    @staticmethod
    def prior_stuck_breaker() -> np.ndarray:
        theta = np.zeros(NUM_PARAMS, dtype=np.float64)
        theta[0] = 1.5
        theta[2] = 4.0
        theta[3] = 2.0
        theta[10] = 1.5
        return theta

    @staticmethod
    def prior_teacher_like(offset_sign: float = 1.0, offset_mag: float = 0.15) -> np.ndarray:
        theta = np.zeros(NUM_PARAMS, dtype=np.float64)
        theta[0] = 3.0
        theta[6] = offset_sign * 2.0
        theta[10] = float(np.clip(offset_mag * 4.0, 0.0, 3.0))
        theta[11] = 1.0
        return theta


def build_v3_prior_candidates(
    previous_best: Optional[np.ndarray] = None,
    resume_theta: Optional[np.ndarray] = None,
    teacher_offsets: Optional[list[float]] = None,
) -> list[np.ndarray]:
    priors = [
        CEMAimV3Policy.prior_exact_follow(),
        CEMAimV3Policy.prior_mild_left_endgame(),
        CEMAimV3Policy.prior_mild_right_endgame(),
        CEMAimV3Policy.prior_centroid_endgame(),
        CEMAimV3Policy.prior_stuck_breaker(),
    ]
    if teacher_offsets:
        for off in teacher_offsets[:3]:
            sign = 1.0 if off >= 0 else -1.0
            priors.append(CEMAimV3Policy.prior_teacher_like(sign, abs(float(off))))
    if previous_best is not None:
        priors.append(np.asarray(previous_best, dtype=np.float64).copy())
    if resume_theta is not None:
        priors.append(np.asarray(resume_theta, dtype=np.float64).copy())
    return priors


def load_cem_aim_policy(path: str):
    """Load v2 or v3 policy from checkpoint."""
    data = np.load(path, allow_pickle=True)
    version = str(data.get("policy_version", POLICY_VERSION_V2))
    if version == POLICY_VERSION_V3 or int(data.get("version", 2)) == 3:
        return CEMAimV3Policy.load(path)
    return CEMAimPolicy.load(path)


def make_cem_aim_policy(
    policy_version: str,
    theta: Optional[np.ndarray] = None,
):
    if policy_version == POLICY_VERSION_V3:
        return CEMAimV3Policy(theta)
    return CEMAimPolicy(theta or CEMAimPolicy.prior_follow_like())
