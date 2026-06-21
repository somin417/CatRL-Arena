"""Compare Random, FollowBall, and trained DQN with fair shared seeds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import settings as S
from agents import DQNPolicyAgent, make_agent
from catbreak_env import CatBreakEnv, layout_from_checkpoint, obs_mode_from_checkpoint
from compare_baselines import BASELINE_AGENTS
from evaluate import EVAL_FIELDNAMES, env_seed_for_episode, run_episode, summarize_rows


def compare_dqn_vs_baselines(
    model_path: str | None,
    episodes: int,
    seed: int,
    layout: str = S.DEFAULT_LAYOUT,
) -> tuple[Path, Path]:
    episode_rows: list[dict] = []
    summary_rows: list[dict] = []

    agent_keys = list(BASELINE_AGENTS) + ["dqn"]
    for agent_key in agent_keys:
        if agent_key == "dqn":
            agent = DQNPolicyAgent(model_path=model_path, fallback_to_follow=False)
            if agent._dqn is None:
                print("WARNING: DQN checkpoint unavailable; skipping DQN in comparison.")
                continue
            obs_mode = agent._dqn.obs_mode
            if model_path and Path(model_path).exists():
                layout = layout_from_checkpoint(model_path)
                obs_mode = obs_mode_from_checkpoint(model_path)
        else:
            agent = make_agent(agent_key)
            obs_mode = S.OBS_MODE_VECTOR
        env = CatBreakEnv(config={"layout": layout, "obs_mode": obs_mode})
        per_agent_rows: list[dict] = []
        for ep in range(episodes):
            ep_seed = env_seed_for_episode(seed, ep)
            result = run_episode(env, agent, ep_seed)
            row = {"episode": ep, "agent": agent.name, **result}
            episode_rows.append(row)
            per_agent_rows.append(row)
        summary_rows.append({"agent": agent.name, **summarize_rows(per_agent_rows)})
        env.close()

    S.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    ep_path = S.COMPARISON_EPISODES_CSV
    sum_path = S.COMPARISON_SUMMARY_CSV

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
    print(f"{'Agent':<24} {'Clear%':>8} {'Blk/100':>8} {'Return':>10} {'Steps':>8}")
    print("-" * 62)
    for row in summary_rows:
        print(
            f"{row['agent']:<24} "
            f"{row['clear_rate'] * 100:7.1f}% "
            f"{row['avg_blocks_per_100_steps']:8.2f} "
            f"{row['avg_return']:10.2f} "
            f"{row['avg_steps']:8.1f}"
        )

    try:
        from plot_baselines import plot_baselines

        plot_baselines(ep_path)
    except Exception as exc:
        print(f"WARNING: Could not generate plot: {exc}")

    return ep_path, sum_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DQN vs baselines.")
    parser.add_argument("--model", type=str, default=str(S.DQN_BEST_CKPT))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", type=str, default=S.DEFAULT_LAYOUT, choices=[S.LAYOUT_RECT, S.LAYOUT_CAT])
    args = parser.parse_args()
    compare_dqn_vs_baselines(args.model, args.episodes, args.seed, args.layout)


if __name__ == "__main__":
    main()
