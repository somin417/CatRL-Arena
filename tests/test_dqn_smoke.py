"""Smoke tests for CatBreak DQN pipeline."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import settings as S
from agent_dqn import CatBreakDQNAgent
from catbreak_env import CatBreakEnv


def test_import_and_instantiate():
    import torch

    env = CatBreakEnv()
    agent = CatBreakDQNAgent(obs_dim=env.obs_dim, n_actions=env.n_actions, min_replay_size=5)
    assert agent is not None
    env.close()


def test_act_and_train_step():
    env = CatBreakEnv()
    agent = CatBreakDQNAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        batch_size=8,
        min_replay_size=5,
        seed=0,
    )
    obs = env.reset(seed=0)
    action = agent.act(obs, epsilon=1.0)
    assert action in (0, 1, 2)

    rng = np.random.default_rng(0)
    for _ in range(25):
        a = int(rng.integers(0, 3))
        obs2, reward, done, _ = env.step(a)
        agent.store_transition(obs, a, reward, obs2, done)
        obs = obs2 if not done else env.reset(seed=int(rng.integers(0, 1000)))

    stats = agent.train_step()
    assert isinstance(stats, dict)
    assert "loss" in stats
    env.close()


def test_save_load_checkpoint():
    env = CatBreakEnv()
    agent = CatBreakDQNAgent(obs_dim=env.obs_dim, n_actions=env.n_actions, min_replay_size=1)
    obs = env.reset(seed=0)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "test_dqn.pt"
        agent.save(ckpt)
        loaded = CatBreakDQNAgent.from_checkpoint(ckpt)
        a1 = agent.greedy_action(obs)
        a2 = loaded.greedy_action(obs)
        assert a1 in (0, 1, 2)
        assert a2 in (0, 1, 2)
    env.close()


def test_train_quick_subprocess():
    save_dir = ROOT / "runs" / "dqn_smoke"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "train_dqn.py"),
            "--episodes", "2",
            "--quick",
            "--save-dir", str(save_dir),
            "--seed", "0",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    run_dir = save_dir
    assert (run_dir / "dqn_last.pt").exists()


def test_evaluate_dqn_subprocess():
    save_dir = ROOT / "runs" / "dqn_smoke"
    model = save_dir / "dqn_last.pt"
    if not model.exists():
        test_train_quick_subprocess()
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "evaluate_dqn.py"),
            "--model", str(model),
            "--episodes", "2",
            "--seed", "0",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (ROOT / "runs" / "dqn_eval" / "dqn_eval.csv").exists()


def run_all():
    tests = [
        test_import_and_instantiate,
        test_act_and_train_step,
        test_save_load_checkpoint,
        test_train_quick_subprocess,
        test_evaluate_dqn_subprocess,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
