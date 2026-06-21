"""Smoke tests for CatBreakEnv."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings as S
from catbreak_env import CatBreakEnv


def test_reset_obs_shape():
    env = CatBreakEnv()
    obs = env.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (env.obs_dim,)
    assert obs.dtype == np.float32
    env.close()


def test_step_types():
    env = CatBreakEnv()
    env.reset(seed=0)
    obs, reward, done, info = env.step(S.ACTION_STAY)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert "score" in info
    assert "terminal_reason" in info
    env.close()


def test_same_seed_same_obs():
    e1 = CatBreakEnv()
    e2 = CatBreakEnv()
    o1 = e1.reset(seed=123)
    o2 = e2.reset(seed=123)
    assert np.allclose(o1, o2)
    e1.close()
    e2.close()


def test_random_rollout_100_steps():
    env = CatBreakEnv()
    obs = env.reset(seed=7)
    rng = np.random.default_rng(7)
    for _ in range(100):
        if env.done:
            obs = env.reset(seed=int(rng.integers(0, 1000)))
        action = int(rng.integers(0, S.N_ACTIONS))
        obs, reward, done, info = env.step(action)
        assert obs.shape == (env.obs_dim,)
    env.close()


def test_all_bricks_cleared_sets_done():
    env = CatBreakEnv()
    env.reset(seed=0)
    env._bricks[:] = False
    _, _, done, info = env.step(S.ACTION_STAY)
    assert done is True
    assert info["clear"] is True
    assert info["terminal_reason"] == "cleared"
    env.close()


def test_invalid_action_raises():
    env = CatBreakEnv()
    env.reset(seed=0)
    try:
        env.step(99)
        assert False, "Expected ValueError"
    except ValueError:
        pass
    env.close()


def run_all():
    tests = [
        test_reset_obs_shape,
        test_step_types,
        test_same_seed_same_obs,
        test_random_rollout_100_steps,
        test_all_bricks_cleared_sets_done,
        test_invalid_action_raises,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
