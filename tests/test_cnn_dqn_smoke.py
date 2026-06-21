"""Smoke tests for Phase 4 CNN/hybrid DQN."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import settings as S
from agent_dqn import CatBreakDQNAgent
from catbreak_env import CatBreakEnv
from networks import CatBreakCNNQNet, LexicographicCatBreakCNNQNet


def test_hybrid_reset_shape():
    env = CatBreakEnv(config={"obs_mode": S.OBS_MODE_HYBRID})
    obs = env.reset(seed=0)
    assert isinstance(obs, dict)
    assert obs["grid"].shape == env.grid_shape
    assert obs["vector"].shape == (env.hybrid_vector_dim,)
    assert obs["grid"][0].sum() > 0
    assert obs["grid"][1].sum() > 0
    assert obs["grid"][3].sum() > 0
    env.close()


def test_cnn_forward():
    grid_shape = (S.GRID_CHANNELS, S.GRID_H, S.GRID_W)
    net = CatBreakCNNQNet(grid_shape, S.HYBRID_VECTOR_DIM, S.N_ACTIONS, dueling=True)
    grid = torch.randn(4, *grid_shape)
    vector = torch.randn(4, S.HYBRID_VECTOR_DIM)
    out = net(grid, vector)
    assert out.shape == (4, S.N_ACTIONS)


def test_lexicographic_cnn_forward():
    grid_shape = (S.GRID_CHANNELS, S.GRID_H, S.GRID_W)
    net = LexicographicCatBreakCNNQNet(grid_shape, S.HYBRID_VECTOR_DIM, S.N_ACTIONS)
    grid = torch.randn(2, *grid_shape)
    vector = torch.randn(2, S.HYBRID_VECTOR_DIM)
    q_brick, q_time = net(grid, vector)
    assert q_brick.shape == (2, S.N_ACTIONS)
    assert q_time.shape == (2, S.N_ACTIONS)


def test_cnn_agent_train_step():
    env = CatBreakEnv(config={"obs_mode": S.OBS_MODE_HYBRID})
    agent = CatBreakDQNAgent(
        n_actions=env.n_actions,
        network_type="cnn",
        grid_shape=env.grid_shape,
        vector_dim=env.hybrid_vector_dim,
        obs_mode=S.OBS_MODE_HYBRID,
        batch_size=8,
        min_replay_size=5,
        seed=0,
    )
    obs = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(30):
        action = int(rng.integers(0, 3))
        next_obs, reward, done, info = env.step(action)
        agent.store_transition(obs, action, reward, next_obs, done, info=info)
        obs = next_obs if not done else env.reset(seed=int(rng.integers(0, 1000)))
    stats = agent.train_step()
    assert "loss" in stats
    env.close()


def test_vector_mlp_still_works():
    env = CatBreakEnv()
    agent = CatBreakDQNAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        min_replay_size=5,
        batch_size=8,
    )
    obs = env.reset(seed=0)
    action = agent.act(obs, epsilon=0.0)
    assert action in (0, 1, 2)
    env.close()


def test_cnn_train_quick_subprocess():
    save_dir = ROOT / "runs" / "dqn_cnn_smoke"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "train_dqn.py"),
            "--obs-mode", "hybrid",
            "--network-type", "cnn",
            "--episodes", "2",
            "--quick",
            "--no-demo",
            "--save-dir", str(save_dir),
            "--seed", "0",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert (save_dir / "dqn_last.pt").exists()


def test_evaluate_cnn_checkpoint():
    save_dir = ROOT / "runs" / "dqn_cnn_smoke"
    model = save_dir / "dqn_last.pt"
    if not model.exists():
        test_cnn_train_quick_subprocess()
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
    assert result.returncode == 0, result.stderr + result.stdout


def run_all():
    tests = [
        test_hybrid_reset_shape,
        test_cnn_forward,
        test_lexicographic_cnn_forward,
        test_cnn_agent_train_step,
        test_vector_mlp_still_works,
        test_cnn_train_quick_subprocess,
        test_evaluate_cnn_checkpoint,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
