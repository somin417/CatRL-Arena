"""Train DQN on CatBreakEnv."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import settings as S
from agent_dqn import CatBreakDQNAgent
from catbreak_env import CatBreakEnv
from evaluate import env_seed_for_episode, summarize_rows
from torch_utils import configure_torch, default_batch_size, get_device

TRAIN_FIELDS = [
    "episode", "env_seed", "obs_mode", "network_type", "dueling", "double_dqn",
    "per", "n_step", "lexicographic", "return", "steps", "score", "broken_bricks",
    "remaining_bricks", "clear", "lives", "terminal_reason", "epsilon",
    "loss", "td_error_abs_mean", "q_mean", "q_max", "replay_size",
    "global_step", "wall_clock_sec",
]

EVAL_FIELDS = [
    "eval_index", "episode", "avg_return", "clear_rate", "avg_steps",
    "avg_broken_bricks", "avg_blocks_per_100_steps",
]


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return end
    frac = min(1.0, step / decay_steps)
    return start + frac * (end - start)


def resolve_obs_mode(args: argparse.Namespace) -> str:
    obs_mode = args.obs_mode
    if args.network_type == "cnn" and obs_mode == S.OBS_MODE_VECTOR:
        obs_mode = S.OBS_MODE_HYBRID
    if args.network_type == "mlp" and obs_mode != S.OBS_MODE_VECTOR:
        raise ValueError("MLP network requires --obs-mode vector.")
    if args.lexicographic and args.network_type != "cnn":
        raise ValueError("Lexicographic DQN requires --network-type cnn.")
    return obs_mode


def build_agent(env: CatBreakEnv, args: argparse.Namespace) -> CatBreakDQNAgent:
    device = configure_torch(get_device())
    batch_size = default_batch_size(args.network_type, args.batch_size, device)
    if batch_size != args.batch_size:
        print(f"Using GPU batch size {batch_size} (requested {args.batch_size}).")
    common = dict(
        n_actions=env.n_actions,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=batch_size,
        replay_capacity=args.replay_capacity,
        min_replay_size=args.min_replay_size,
        target_update_freq=args.target_update_freq,
        tau=args.tau,
        double_dqn=args.double_dqn,
        dueling=args.dueling,
        per=args.per,
        seed=args.seed,
        device=device,
        obs_mode=args.obs_mode,
        network_type=args.network_type,
        n_step=args.n_step,
        lexicographic=args.lexicographic,
        beta_time=args.beta_time,
        brick_tolerance=args.brick_tolerance,
    )
    if args.network_type == "cnn":
        return CatBreakDQNAgent(
            grid_shape=env.grid_shape,
            vector_dim=env.hybrid_vector_dim,
            **common,
        )
    return CatBreakDQNAgent(obs_dim=env.obs_dim, **common)


def run_greedy_eval(
    agent: CatBreakDQNAgent,
    env: CatBreakEnv,
    seed: int,
    episodes: int,
) -> list[dict]:
    rows = []
    for ep in range(episodes):
        env_seed = seed + 100_000 + ep
        obs = env.reset(seed=env_seed)
        total_return = 0.0
        while not env.done:
            action = agent.greedy_action(obs)
            obs, reward, done, info = env.step(action)
            total_return += reward
        steps = info["step_count"]
        broken = info["broken_bricks"]
        rows.append({
            "return": total_return,
            "steps": steps,
            "broken_bricks": broken,
            "clear": int(info["clear"]),
            "blocks_per_100_steps": (broken / steps * 100.0) if steps > 0 else 0.0,
        })
    return rows


def better_eval(candidate: dict, best: dict | None) -> bool:
    if best is None:
        return True
    key = lambda d: (d["clear_rate"], d["avg_blocks_per_100_steps"], d["avg_return"])
    return key(candidate) > key(best)


def train(args: argparse.Namespace) -> tuple[Path, Path]:
    run_dir = Path(args.save_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    args.obs_mode = resolve_obs_mode(args)
    device = configure_torch(get_device())
    config = vars(args).copy()
    config["device"] = str(device)
    config["effective_batch_size"] = default_batch_size(
        args.network_type, args.batch_size, device
    )
    with (run_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    if args.max_steps is not None:
        S.MAX_STEPS = int(args.max_steps)

    env_config = {"layout": args.layout, "obs_mode": args.obs_mode}
    env = CatBreakEnv(config=env_config)
    agent = build_agent(env, args)

    train_rows = []
    eval_rows = []
    best_eval_summary = None

    for ep in range(args.episodes):
        t0 = time.perf_counter()
        env_seed = env_seed_for_episode(args.seed, ep)
        obs = env.reset(seed=env_seed)
        done = False
        episode_return = 0.0
        ep_loss = ep_td = ep_qmean = ep_qmax = 0.0
        ep_train_steps = 0

        while not done:
            epsilon = linear_epsilon(
                agent.global_step,
                args.epsilon_start,
                args.epsilon_end,
                args.epsilon_decay_steps,
            )
            action = agent.act(obs, epsilon)
            next_obs, reward, done, info = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, done, info=info)
            stats = agent.train_step()
            if stats:
                ep_loss += stats.get("loss", 0.0)
                ep_td += stats.get("td_error_abs_mean", 0.0)
                ep_qmean += stats.get("q_mean", 0.0)
                ep_qmax += stats.get("q_max", 0.0)
                ep_train_steps += 1
            obs = next_obs
            episode_return += reward

        agent.episode_done()
        denom = max(1, ep_train_steps)
        train_rows.append({
            "episode": ep,
            "env_seed": env_seed,
            "obs_mode": args.obs_mode,
            "network_type": args.network_type,
            "dueling": int(args.dueling),
            "double_dqn": int(args.double_dqn),
            "per": int(args.per),
            "n_step": args.n_step,
            "lexicographic": int(args.lexicographic),
            "return": episode_return,
            "steps": info["step_count"],
            "score": info["score"],
            "broken_bricks": info["broken_bricks"],
            "remaining_bricks": info["remaining_bricks"],
            "clear": int(info["clear"]),
            "lives": info["lives"],
            "terminal_reason": info["terminal_reason"] or "",
            "epsilon": linear_epsilon(
                agent.global_step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps
            ),
            "loss": ep_loss / denom,
            "td_error_abs_mean": ep_td / denom,
            "q_mean": ep_qmean / denom,
            "q_max": ep_qmax / denom,
            "replay_size": len(agent.replay_buffer),
            "global_step": agent.global_step,
            "wall_clock_sec": time.perf_counter() - t0,
        })

        if (ep + 1) % args.eval_freq == 0:
            agent.eval_mode()
            eval_ep_rows = run_greedy_eval(agent, env, args.seed, args.eval_episodes)
            summary = summarize_rows(eval_ep_rows)
            eval_rows.append({"eval_index": ep + 1, "episode": ep + 1, **summary})
            if better_eval(summary, best_eval_summary):
                best_eval_summary = summary
                agent.save(
                    run_dir / "dqn_best.pt",
                    extra={
                        "eval": summary,
                        "layout": args.layout,
                        "obs_mode": args.obs_mode,
                        "network_type": args.network_type,
                    },
                )
            agent.train_mode()
            print(
                f"[eval ep {ep + 1}] clear={summary['clear_rate']*100:.1f}% "
                f"return={summary['avg_return']:.2f} "
                f"blk/100={summary['avg_blocks_per_100_steps']:.2f}"
            )

        if (ep + 1) % max(1, args.episodes // 10) == 0 or ep == 0:
            print(
                f"episode {ep + 1}/{args.episodes} return={episode_return:.2f} "
                f"steps={info['step_count']} replay={len(agent.replay_buffer)}"
            )

    save_extra = {
        "layout": args.layout,
        "obs_mode": args.obs_mode,
        "network_type": args.network_type,
    }
    agent.save(run_dir / "dqn_last.pt", extra=save_extra)
    if best_eval_summary is None:
        agent.save(run_dir / "dqn_best.pt", extra=save_extra)

    with (run_dir / "train_episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAIN_FIELDS)
        writer.writeheader()
        writer.writerows(train_rows)

    if eval_rows:
        with (run_dir / "eval_history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EVAL_FIELDS)
            writer.writeheader()
            writer.writerows(eval_rows)

    env.close()
    best_path = run_dir / "dqn_best.pt"
    print(f"Training done. Run dir: {run_dir}")
    print(f"Best checkpoint: {best_path}")
    return run_dir, best_path


def default_save_dir(network_type: str, per: bool) -> Path:
    if network_type == "cnn":
        return S.DQN_CNN_PER_DIR if per else S.DQN_CNN_DIR
    return S.DQN_PER_DIR if per else S.DQN_DIR


def launch_demo(model_path: Path, seed: int) -> None:
    """Open ui_duel with the trained DQN on the right panel."""
    from ui_duel import main as run_ui_duel

    print(f"Launching demo UI with DQN ({model_path})...")
    run_ui_duel(agent_name="dqn", model_path=str(model_path), seed=seed)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train DQN on CatBreakEnv")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", type=str, default=None,
                   help="output run directory (default: runs/dqn_cnn for CNN, runs/dqn_per with --per, else runs/dqn)")
    p.add_argument("--obs-mode", type=str, default=S.DEFAULT_OBS_MODE,
                   choices=[S.OBS_MODE_VECTOR, S.OBS_MODE_GRID, S.OBS_MODE_HYBRID])
    p.add_argument("--network-type", type=str, default="mlp", choices=["mlp", "cnn"])
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--replay-capacity", type=int, default=100_000)
    p.add_argument("--min-replay-size", type=int, default=1000)
    p.add_argument("--target-update-freq", type=int, default=1000)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--double-dqn", dest="double_dqn", action="store_true", default=True)
    p.add_argument("--no-double-dqn", dest="double_dqn", action="store_false")
    p.add_argument("--dueling", dest="dueling", action="store_true", default=True)
    p.add_argument("--no-dueling", dest="dueling", action="store_false")
    p.add_argument("--per", dest="per", action="store_true", default=False)
    p.add_argument("--no-per", dest="per", action="store_false")
    p.add_argument("--n-step", type=int, default=1)
    p.add_argument("--lexicographic", action="store_true", default=False)
    p.add_argument("--beta-time", type=float, default=0.2)
    p.add_argument("--brick-tolerance", type=float, default=0.05)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay-steps", type=int, default=50_000)
    p.add_argument("--eval-freq", type=int, default=25)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--layout", type=str, default=S.DEFAULT_LAYOUT, choices=[S.LAYOUT_RECT, S.LAYOUT_CAT])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--demo", dest="demo", action="store_true", default=True,
                   help="launch ui_duel.py with best checkpoint after training (default)")
    p.add_argument("--no-demo", dest="demo", action="store_false",
                   help="skip post-training demo UI")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.save_dir is None:
        args.save_dir = str(default_save_dir(args.network_type, args.per))
    if args.quick:
        args.episodes = min(args.episodes, 5)
        args.min_replay_size = min(args.min_replay_size, 20)
        args.batch_size = min(args.batch_size, 8)
        args.eval_freq = 1
        args.eval_episodes = 2
        args.epsilon_decay_steps = min(args.epsilon_decay_steps, 200)
        args.demo = False
    _, best_path = train(args)
    if args.demo and best_path.exists():
        launch_demo(best_path, args.seed)


if __name__ == "__main__":
    main()
