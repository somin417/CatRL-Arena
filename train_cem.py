"""Placeholder CEM training script for CatBreak RL Arena.

TODO(CEM):
- Parameterize a linear or small MLP policy over obs -> action logits.
- Sample policy parameter vectors, evaluate on env rollouts, refit elite set.
- No gradients required — search in weight space.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import settings as S
from agents import FollowBallAgent
from catbreak_env import CatBreakEnv


def smoke_train(episodes: int, seed: int, save_dir: Path) -> None:
    env = CatBreakEnv()
    agent = FollowBallAgent()
    rng = np.random.default_rng(seed)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "train_cem_smoke.csv"

    rows = []
    for ep in range(episodes):
        ep_seed = int(rng.integers(0, 1_000_000))
        obs = env.reset(seed=ep_seed)
        agent.reset(seed=ep_seed + 1)
        total_return = 0.0
        while not env.done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            total_return += reward
        rows.append({
            "episode": ep,
            "return": total_return,
            "steps": info["step_count"],
            "broken_bricks": info["broken_bricks"],
            "clear": int(info["clear"]),
        })

    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    env.close()
    print(f"CEM placeholder smoke train done. Log: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default=str(S.RUNS_DIR / "cem_smoke"))
    args = parser.parse_args()
    smoke_train(args.episodes, args.seed, Path(args.save_dir))


if __name__ == "__main__":
    main()
