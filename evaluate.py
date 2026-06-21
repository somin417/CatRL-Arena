"""Headless evaluation for CatBreak RL Arena agents."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import settings as S
from agents import SUPPORTED_AGENT_NAMES, BaseAgent, make_agent, _normalize_agent_name
from catbreak_env import CatBreakEnv

EVAL_FIELDNAMES = [
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
]

PLACEHOLDER_AGENTS = ("ppo", "cem")


def env_seed_for_episode(base_seed: int, episode: int) -> int:
    return base_seed + episode


def run_episode(env: CatBreakEnv, agent: BaseAgent, env_seed: int) -> dict:
    t0 = time.perf_counter()
    obs = env.reset(seed=env_seed)
    agent.reset(seed=env_seed + 1)
    total_return = 0.0
    while not env.done:
        action = agent.act(obs, env.last_info, env=env)
        obs, reward, done, info = env.step(action)
        if hasattr(agent, "note_step"):
            agent.note_step(info, env)
        total_return += reward
    steps = info["step_count"]
    broken = info["broken_bricks"]
    blocks_per_100 = (broken / steps * 100.0) if steps > 0 else 0.0
    return {
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
    }


def summarize_rows(rows: list[dict]) -> dict:
    returns = [r["return"] for r in rows]
    steps = [r["steps"] for r in rows]
    clears = [r["clear"] for r in rows]
    broken = [r["broken_bricks"] for r in rows]
    blocks = [r["blocks_per_100_steps"] for r in rows]
    return {
        "avg_return": float(np.mean(returns)) if returns else 0.0,
        "avg_steps": float(np.mean(steps)) if steps else 0.0,
        "clear_rate": float(np.mean(clears)) if clears else 0.0,
        "avg_broken_bricks": float(np.mean(broken)) if broken else 0.0,
        "avg_blocks_per_100_steps": float(np.mean(blocks)) if blocks else 0.0,
    }


def print_summary(agent_name: str, episodes: int, summary: dict, csv_path: Path) -> None:
    print(f"Agent: {agent_name}")
    print(f"Episodes: {episodes}")
    print(f"CSV: {csv_path}")
    print(f"Average return: {summary['avg_return']:.3f}")
    print(f"Average steps: {summary['avg_steps']:.1f}")
    print(f"Clear rate: {summary['clear_rate'] * 100:.1f}%")
    print(f"Average broken bricks: {summary['avg_broken_bricks']:.2f}")
    print(f"Average blocks per 100 steps: {summary['avg_blocks_per_100_steps']:.2f}")


def evaluate(
    agent_name: str,
    episodes: int,
    seed: int,
    layout: str = S.DEFAULT_LAYOUT,
    out_dir: Path | None = None,
) -> tuple[Path, list[dict]]:
    key = _normalize_agent_name(agent_name)
    if key in PLACEHOLDER_AGENTS:
        print(
            f"Agent '{agent_name}' is not implemented yet. "
            f"Train with train_{key}.py and plug weights into RLPolicyAgent."
        )
        raise SystemExit(0)

    agent = make_agent(key)
    env = CatBreakEnv(config={"layout": layout})

    rows: list[dict] = []
    for ep in range(episodes):
        ep_seed = env_seed_for_episode(seed, ep)
        result = run_episode(env, agent, ep_seed)
        rows.append({"episode": ep, "agent": agent.name, **result})

    env.close()

    out_dir = out_dir or S.RUNS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_{key}.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print_summary(agent.name, episodes, summarize_rows(rows), out_path)
    return out_path, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CatBreak agents (headless).")
    parser.add_argument("--agent", type=str, default="random")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", type=str, default=S.DEFAULT_LAYOUT, choices=[S.LAYOUT_RECT, S.LAYOUT_CAT])
    args = parser.parse_args()
    evaluate(args.agent, args.episodes, args.seed, args.layout)


if __name__ == "__main__":
    main()
