"""Compare Random vs FollowBall baselines with fair shared seeds."""

from __future__ import annotations

import argparse
import csv

import settings as S
from evaluate import EVAL_FIELDNAMES, env_seed_for_episode, run_episode, summarize_rows
from agents import make_agent
from catbreak_env import CatBreakEnv

BASELINE_AGENTS = ("random", "follow")


def compare_baselines(episodes: int, seed: int, layout: str = S.DEFAULT_LAYOUT) -> tuple:
    env = CatBreakEnv(config={"layout": layout})
    print(
        f"Layout: {layout}  |  obs_dim={env.obs_dim}  |  "
        f"bricks={env._layout.total_bricks}"
    )
    episode_rows: list[dict] = []
    summary_rows: list[dict] = []

    for agent_key in BASELINE_AGENTS:
        agent = make_agent(agent_key)
        agent_episode_rows: list[dict] = []
        for ep in range(episodes):
            ep_seed = env_seed_for_episode(seed, ep)
            result = run_episode(env, agent, ep_seed)
            row = {"episode": ep, "agent": agent.name, **result}
            episode_rows.append(row)
            agent_episode_rows.append(row)
        summary = summarize_rows(agent_episode_rows)
        summary_rows.append({"agent": agent.name, **summary})

    env.close()

    S.BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    ep_path = S.BASELINE_EPISODES_CSV
    sum_path = S.BASELINE_SUMMARY_CSV

    with ep_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(episode_rows)

    summary_fields = [
        "agent", "avg_return", "avg_steps", "clear_rate",
        "avg_broken_bricks", "avg_blocks_per_100_steps",
    ]
    with sum_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_rows.sort(
        key=lambda r: (
            -r["clear_rate"],
            -r["avg_blocks_per_100_steps"],
            -r["avg_return"],
        )
    )

    print(f"Episodes per agent: {episodes}")
    print(f"Base seed: {seed}")
    print(f"Episode CSV: {ep_path}")
    print(f"Summary CSV: {sum_path}")
    print()
    print(f"{'Agent':<12} {'Clear%':>8} {'Blk/100':>8} {'Return':>10} {'Steps':>8}")
    print("-" * 50)
    for row in summary_rows:
        print(
            f"{row['agent']:<12} "
            f"{row['clear_rate'] * 100:7.1f}% "
            f"{row['avg_blocks_per_100_steps']:8.2f} "
            f"{row['avg_return']:10.2f} "
            f"{row['avg_steps']:8.1f}"
        )

    plot_path = None
    try:
        from plot_baselines import plot_baselines

        plot_path = plot_baselines(ep_path)
    except Exception as exc:
        print(f"WARNING: Could not generate plot: {exc}")

    return ep_path, sum_path, plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Random vs FollowBall baselines.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", type=str, default=S.DEFAULT_LAYOUT, choices=[S.LAYOUT_RECT, S.LAYOUT_CAT])
    args = parser.parse_args()
    compare_baselines(args.episodes, args.seed, args.layout)


if __name__ == "__main__":
    main()
