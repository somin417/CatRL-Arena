"""Evaluate a trained CatBreak DQN checkpoint."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import settings as S
from agent_dqn import CatBreakDQNAgent
from catbreak_env import CatBreakEnv, layout_from_checkpoint, obs_mode_from_checkpoint
from evaluate import env_seed_for_episode, print_summary, summarize_rows
from torch_utils import configure_torch, get_device

EVAL_FIELDS = [
    "episode", "env_seed", "agent", "return", "steps", "score",
    "broken_bricks", "remaining_bricks", "clear", "lives",
    "terminal_reason", "blocks_per_100_steps", "wall_clock_sec",
]


def evaluate_checkpoint(
    model_path: Path,
    episodes: int,
    seed: int,
    layout: str | None = None,
    obs_mode: str | None = None,
    save_dir: Path | None = None,
) -> Path:
    device = configure_torch(get_device())
    agent = CatBreakDQNAgent.from_checkpoint(model_path, device=device)
    if layout is None:
        layout = layout_from_checkpoint(model_path)
    if obs_mode is None:
        obs_mode = obs_mode_from_checkpoint(model_path)
    env = CatBreakEnv(config={"layout": layout, "obs_mode": obs_mode})

    rows = []
    for ep in range(episodes):
        t0 = time.perf_counter()
        env_seed = env_seed_for_episode(seed, ep)
        obs = env.reset(seed=env_seed)
        total_return = 0.0
        while not env.done:
            action = agent.greedy_action(obs)
            obs, reward, done, info = env.step(action)
            total_return += reward
        steps = info["step_count"]
        broken = info["broken_bricks"]
        rows.append({
            "episode": ep,
            "env_seed": env_seed,
            "agent": "DQN",
            "return": total_return,
            "steps": steps,
            "score": info["score"],
            "broken_bricks": broken,
            "remaining_bricks": info["remaining_bricks"],
            "clear": int(info["clear"]),
            "lives": info["lives"],
            "terminal_reason": info["terminal_reason"] or "",
            "blocks_per_100_steps": (broken / steps * 100.0) if steps > 0 else 0.0,
            "wall_clock_sec": time.perf_counter() - t0,
        })

    env.close()

    out_dir = save_dir or S.DQN_EVAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dqn_eval.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print_summary("DQN", episodes, summarize_rows(rows), out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained DQN checkpoint.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default=str(S.DQN_EVAL_DIR))
    parser.add_argument("--layout", type=str, default=None, choices=[S.LAYOUT_RECT, S.LAYOUT_CAT])
    parser.add_argument("--obs-mode", type=str, default=None,
                        choices=[S.OBS_MODE_VECTOR, S.OBS_MODE_GRID, S.OBS_MODE_HYBRID])
    args = parser.parse_args()
    evaluate_checkpoint(
        Path(args.model),
        args.episodes,
        args.seed,
        args.layout,
        args.obs_mode,
        Path(args.save_dir),
    )


if __name__ == "__main__":
    main()
