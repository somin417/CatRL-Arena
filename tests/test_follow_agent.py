"""Tests for FollowBallAgent and Phase 3-lite evaluation scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import settings as S
from agents import FollowBallAgent, HeuristicAgent, RandomAgent, make_agent
from catbreak_env import CatBreakEnv


def _obs(ball_x: float, paddle_x: float) -> np.ndarray:
    env = CatBreakEnv()
    env.reset(seed=0)
    obs = env.get_obs()
    obs[0] = ball_x
    obs[4] = paddle_x
    env.close()
    return obs


def test_make_agent_follow():
    agent = make_agent("follow")
    assert isinstance(agent, FollowBallAgent)
    assert agent.name == "FollowBall"


def test_make_agent_heuristic_alias():
    agent = make_agent("heuristic")
    assert isinstance(agent, FollowBallAgent)


def test_follow_ball_left():
    agent = FollowBallAgent(threshold=0.035)
    obs = _obs(ball_x=0.1, paddle_x=0.5)
    assert agent.act(obs) == S.ACTION_LEFT


def test_follow_ball_right():
    agent = FollowBallAgent(threshold=0.035)
    obs = _obs(ball_x=0.5, paddle_x=0.1)
    assert agent.act(obs) == S.ACTION_RIGHT


def test_follow_ball_stay():
    agent = FollowBallAgent(threshold=0.035)
    obs = _obs(ball_x=0.5, paddle_x=0.51)
    assert agent.act(obs) == S.ACTION_STAY


def test_random_valid_actions():
    agent = RandomAgent()
    agent.reset(seed=0)
    obs = _obs(0.5, 0.5)
    for _ in range(20):
        action = agent.act(obs)
        assert action in (S.ACTION_LEFT, S.ACTION_STAY, S.ACTION_RIGHT)


def test_parse_obs():
    env = CatBreakEnv()
    obs = env.reset(seed=0)
    parsed = env.parse_obs(obs)
    assert set(parsed.keys()) == {
        "ball_x", "ball_y", "ball_vx", "ball_vy",
        "paddle_x", "paddle_vx", "bricks", "lives", "step_count",
    }
    assert parsed["bricks"].shape == (env.brick_rows * env.brick_cols,)
    env.close()


def test_evaluate_follow_headless():
    result = subprocess.run(
        [sys.executable, str(ROOT / "evaluate.py"), "--agent", "follow", "--episodes", "2", "--seed", "0"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Average return" in result.stdout


def test_compare_baselines_headless():
    result = subprocess.run(
        [sys.executable, str(ROOT / "compare_baselines.py"), "--episodes", "2", "--seed", "0"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "baseline_episodes.csv" in result.stdout
    assert (ROOT / "runs" / "baselines" / "baseline_episodes.csv").exists()


def run_all():
    tests = [
        test_make_agent_follow,
        test_make_agent_heuristic_alias,
        test_follow_ball_left,
        test_follow_ball_right,
        test_follow_ball_stay,
        test_random_valid_actions,
        test_parse_obs,
        test_evaluate_follow_headless,
        test_compare_baselines_headless,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
