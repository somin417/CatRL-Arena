"""Diagnose paddle-ball collision physics controllability."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

import settings as S
from catbreak_env import CatBreakEnv
from evaluate import env_seed_for_episode

COLLISION_FIELDS = [
    "episode",
    "env_seed",
    "step",
    "ball_x",
    "paddle_center_x",
    "hit_offset",
    "paddle_vx",
    "ball_vx_before",
    "ball_vy_before",
    "ball_vx_after",
    "ball_vy_after",
]


def collect_collisions(env: CatBreakEnv, env_seed: int, episode: int) -> list[dict]:
    rows: list[dict] = []
    obs = env.reset(seed=env_seed)
    rng = np.random.default_rng(env_seed + 7)
    while not env.done:
        action = int(rng.integers(0, S.N_ACTIONS))
        obs, _, done, _ = env.step(action)
        collision = env._last_paddle_collision
        if collision is not None:
            rows.append({
                "episode": episode,
                "env_seed": env_seed,
                **collision,
            })
    return rows


def analyze_warnings(rows: list[dict]) -> list[str]:
    warnings: list[str] = []
    if not rows:
        warnings.append("No paddle collisions recorded in diagnostic episodes.")
        return warnings

    offsets = np.array([r["hit_offset"] for r in rows], dtype=np.float64)
    vx_after = np.array([r["ball_vx_after"] for r in rows], dtype=np.float64)
    paddle_vx = np.array([r["paddle_vx"] for r in rows], dtype=np.float64)

    if len(np.unique(np.round(offsets, 2))) > 2:
        corr = np.corrcoef(offsets, vx_after)[0, 1] if len(offsets) > 2 else 0.0
        if not np.isfinite(corr) or abs(corr) < 0.15:
            warnings.append(
                "ball_vx_after appears almost independent of hit_offset "
                f"(corr={corr:.3f}). Paddle aiming may be ineffective."
            )

    if len(np.unique(np.round(paddle_vx, 1))) > 2:
        corr_spin = np.corrcoef(paddle_vx, vx_after)[0, 1] if len(paddle_vx) > 2 else 0.0
        if not np.isfinite(corr_spin) or abs(corr_spin) < 0.05:
            warnings.append(
                "paddle_vx appears ignored in bounce physics "
                f"(corr={corr_spin:.3f})."
            )

    vy_flips = [
        r for r in rows
        if abs(r["ball_vy_after"]) > 1e-6
        and abs(r["ball_vx_after"]) < 0.05 * abs(r["ball_vy_after"])
        and abs(r["hit_offset"]) > 0.3
    ]
    if len(vy_flips) > len(rows) * 0.5:
        warnings.append(
            "Many paddle hits at large |hit_offset| produce near-vertical rebounds "
            "(little horizontal control)."
        )

    if not S.PADDLE_ANGLE_CONTROL:
        warnings.append(
            "PADDLE_ANGLE_CONTROL is False — enable it in settings.py for aiming."
        )

    return warnings


def diagnose(episodes: int, seed: int, layout: str = S.DEFAULT_LAYOUT) -> Path:
    env = CatBreakEnv(config={"layout": layout})
    all_rows: list[dict] = []
    for ep in range(episodes):
        ep_seed = env_seed_for_episode(seed, ep)
        all_rows.extend(collect_collisions(env, ep_seed, ep))
    env.close()

    S.DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = S.DIAGNOSTICS_DIR / f"physics_collisions_{ts}.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLLISION_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Recorded {len(all_rows)} paddle collisions across {episodes} episodes.")
    print(f"Saved: {out_path}")
    print(f"PADDLE_ANGLE_CONTROL={S.PADDLE_ANGLE_CONTROL}  "
          f"MAX_ANGLE={S.PADDLE_MAX_BOUNCE_ANGLE_DEG}deg  "
          f"SPIN={S.PADDLE_SPIN_STRENGTH}")

    for w in analyze_warnings(all_rows):
        print(f"WARNING: {w}")

    if not analyze_warnings(all_rows):
        print("Physics controllability looks OK (offset and spin affect rebound).")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose CatBreak paddle physics.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", type=str, default=S.DEFAULT_LAYOUT,
                        choices=[S.LAYOUT_RECT, S.LAYOUT_CAT])
    args = parser.parse_args()
    diagnose(args.episodes, args.seed, args.layout)


if __name__ == "__main__":
    main()
