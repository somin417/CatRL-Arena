"""Smoke tests for CEM-MPC planner."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import settings as S
from agents import make_agent
from catbreak_env import CatBreakEnv
from cem_mpc import CEMMPCPlanner, simulate_sequence


def test_import_and_act():
    env = CatBreakEnv()
    obs = env.reset(seed=0)
    planner = CEMMPCPlanner(
        horizon=5, population_size=8, iterations=1, seed=0
    )
    action = planner.act(obs, env=env)
    assert action in (S.ACTION_LEFT, S.ACTION_STAY, S.ACTION_RIGHT)
    env.close()


def test_plan_does_not_mutate_env():
    env = CatBreakEnv()
    env.reset(seed=0)
    before = env.get_state_dict()
    planner = CEMMPCPlanner(horizon=5, population_size=8, iterations=1, seed=0)
    planner.plan(env)
    after = env.get_state_dict()
    for key in ("ball_x", "ball_y", "paddle_x", "broken_bricks", "step_count", "done"):
        assert before[key] == after[key]
    assert np.array_equal(before["bricks"], after["bricks"])
    env.close()


def test_evaluate_sequence_fields():
    env = CatBreakEnv()
    state = env.reset(seed=1)
    state_dict = env.get_state_dict()
    result = simulate_sequence(env, state_dict, [S.ACTION_STAY] * 3)
    assert "bricks_broken_during_sequence" in result
    assert "life_lost" in result
    assert "score_tuple" in result
    env.close()


def test_categorical_probs_sum_to_one():
    planner = CEMMPCPlanner(horizon=8, population_size=4, iterations=1, seed=0)
    probs = planner._init_probs()
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_probability_floor():
    planner = CEMMPCPlanner(
        horizon=5, population_size=8, iterations=1, prob_floor=0.05, seed=0
    )
    probs = planner._init_probs()
    assert np.all(probs >= planner.prob_floor - 1e-9)
    elite = np.array([[0, 1, 2, 1, 0], [2, 2, 2, 2, 2]])
    updated = planner.update_distribution(probs, elite, np.zeros(2))
    assert np.all(updated >= planner.prob_floor - 1e-9)
    assert np.allclose(updated.sum(axis=1), 1.0, atol=1e-6)


def test_evaluate_cem_mpc_cli():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "evaluate_cem_mpc.py"),
            "--episodes", "2",
            "--seed", "0",
            "--horizon", "5",
            "--population-size", "8",
            "--iterations", "1",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert "cem_mpc_episodes_" in result.stdout or list(S.CEM_MPC_DIR.glob("cem_mpc_episodes_*.csv"))


def test_evaluate_cem_mpc_creates_csv():
    csvs = sorted(S.CEM_MPC_DIR.glob("cem_mpc_episodes_*.csv"))
    assert csvs, "Expected episode CSV under runs/cem_mpc/"
    summaries = sorted(S.CEM_MPC_DIR.glob("cem_mpc_summary_*.csv"))
    assert summaries, "Expected summary CSV under runs/cem_mpc/"


def test_evaluate_follow_still_works():
    result = subprocess.run(
        [sys.executable, str(ROOT / "evaluate.py"), "--agent", "follow", "--episodes", "2", "--seed", "0"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stderr


def test_ui_duel_imports():
    spec = importlib.util.spec_from_file_location("ui_duel", ROOT / "ui_duel.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert hasattr(module, "main")


def run_all():
    tests = [
        test_import_and_act,
        test_plan_does_not_mutate_env,
        test_evaluate_sequence_fields,
        test_categorical_probs_sum_to_one,
        test_probability_floor,
        test_evaluate_cem_mpc_cli,
        test_evaluate_cem_mpc_creates_csv,
        test_evaluate_follow_still_works,
        test_ui_duel_imports,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
