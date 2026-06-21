"""Smoke tests for CEM-Aim v2/v3."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import settings as S
from agents import FollowBallAgent, make_agent
from catbreak_env import CatBreakEnv
from cem_aim_policy import (
    CEMAimPolicy,
    CEMAimV3Policy,
    NUM_PARAMS,
    act_cem_aim_v3,
    compute_features,
    exact_followball_action,
    extract_cem_aim_features,
)
from cem_aim_v3 import sanity_clone_restore, sanity_followball_reproduction


def test_num_params():
    assert NUM_PARAMS == 12


def test_v2_act_with_env():
    env = CatBreakEnv()
    obs = env.reset(seed=0)
    policy = CEMAimPolicy(CEMAimPolicy.prior_follow_like())
    for _ in range(20):
        action = policy.act(obs, env=env)
        assert action in (S.ACTION_LEFT, S.ACTION_STAY, S.ACTION_RIGHT)
        obs, _, done, _ = env.step(action)
        if done:
            break
    env.close()


def test_v3_act_with_env():
    env = CatBreakEnv()
    obs = env.reset(seed=0)
    policy = CEMAimV3Policy()
    for _ in range(20):
        action = policy.act(obs, env=env)
        assert action in (S.ACTION_LEFT, S.ACTION_STAY, S.ACTION_RIGHT)
        obs, _, done, _ = env.step(action)
        if done:
            break
    env.close()


def test_exact_followball():
    env = CatBreakEnv()
    obs = env.reset(seed=0)
    follow = FollowBallAgent()
    for _ in range(100):
        fb = follow.act(obs, env.last_info, env=env)
        feats = extract_cem_aim_features(obs, env=env)
        exact = exact_followball_action(feats)
        assert fb == exact
        obs, _, done, _ = env.step(fb)
        if done:
            break
    env.close()


def test_v3_followball_reproduction():
    env = CatBreakEnv()
    run_dir = ROOT / "runs" / "cem_aim_smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "cem_aim_exact_follow_prior.npz"
    CEMAimV3Policy(CEMAimV3Policy.prior_exact_follow()).save(str(path))
    assert sanity_followball_reproduction(path, [0, 1, 2], S.DEFAULT_LAYOUT)


def test_clone_restore():
    env = CatBreakEnv()
    assert sanity_clone_restore(env, seed=0)
    env.close()


def test_v3_residual_nontrivial():
    from cem_aim_policy import is_unsafe_for_residual
    from agents import FollowBallAgent

    env = CatBreakEnv()
    obs = env.reset(seed=10)
    follow = FollowBallAgent()
    env._no_brick_streak = 1500
    for _ in range(2000):
        feats = extract_cem_aim_features(obs, env=env)
        if not is_unsafe_for_residual(feats):
            break
        obs, _, done, _ = env.step(follow.act(obs, env.last_info, env=env))
        if done:
            break

    activated = 0
    for theta in (
        CEMAimV3Policy.prior_mild_left_endgame(),
        CEMAimV3Policy.prior_stuck_breaker(),
    ):
        _, dbg = act_cem_aim_v3(
            obs, theta, env=env, return_debug=True, force_opportunity=True
        )
        if dbg["residual_activated"]:
            activated += 1
    assert activated >= 1
    env.close()


def test_v3_commitment_holds_target():
    """Committed target should persist across steps until paddle contact."""
    env = CatBreakEnv()
    policy = CEMAimV3Policy(CEMAimV3Policy.prior_mild_left_endgame())
    obs = env.reset(seed=10)
    policy.reset_episode()
    env._no_brick_streak = 1500

    committed_target = None
    for _ in range(500):
        action, dbg = policy.act_debug(
            obs, env=env, force_opportunity=True
        )
        if dbg.get("committed") and committed_target is None:
            committed_target = dbg["committed_target_x"]
            break
        obs, _, done, info = env.step(action)
        policy.note_step_after_env_step(info, env)
        if done:
            break

    assert committed_target is not None
    held = 0
    for _ in range(120):
        action, dbg = policy.act_debug(obs, env=env, force_opportunity=True)
        if dbg.get("committed"):
            assert abs(dbg["committed_target_x"] - committed_target) < 1e-6
            held += 1
        obs, _, done, info = env.step(action)
        policy.note_step_after_env_step(info, env)
        if done or dbg.get("contact_detected"):
            break
    assert held >= 3
    env.close()


def test_v3_endgame_snapshot_scoring():
    from cem_aim_v3 import DEFAULT_V3_SCORING, RolloutResult, Snapshot, _score_snapshot_pair

    snap = Snapshot(
        env_state={},
        obs=np.zeros(8, dtype=np.float32),
        seed=0,
        step=100,
        followball_action=1,
        remaining_bricks=8,
        no_brick_broken_for=0,
        brick_centroid_x=0.3,
        predicted_landing_x=0.5,
        ball_x=0.5,
        ball_y=0.2,
        ball_vx=0.0,
        ball_vy=1.0,
        paddle_x=0.5,
    )
    c = RolloutResult(broken_bricks=2, clear=0, steps=100, life_lost=False, steps_to_next_brick=50)
    f = RolloutResult(broken_bricks=2, clear=0, steps=180, life_lost=False, steps_to_next_brick=80)
    pts, win, end_win = _score_snapshot_pair(
        c, f, snap, {"residual_activated": True}, DEFAULT_V3_SCORING
    )
    assert win
    assert end_win
    assert pts > 0.0


def test_v3_endgame_selection_beats_exact_follow():
    """Endgame-aware selection should prefer deviation priors over exact FollowBall clone."""
    from cem_aim_v3 import TrainingCaches, evaluate_theta_v3, load_snapshots, parse_seed_spec

    snap_dir = ROOT / "runs" / "cem_aim"
    snaps = list(snap_dir.glob("opportunity_snapshots_*.npz"))
    if not snaps:
        return

    snapshots = load_snapshots(snaps[0])
    endgame_idx = [
        i for i, s in enumerate(snapshots) if s.remaining_bricks <= 15
    ][:24]
    if len(endgame_idx) < 8:
        return

    endgame_snaps = [snapshots[i] for i in endgame_idx]
    env = CatBreakEnv()
    train_seeds = parse_seed_spec("0:4")
    caches = TrainingCaches.build(
        env,
        train_seeds=train_seeds,
        val_seeds=[],
        snapshots=endgame_snaps,
        snapshot_horizon=400,
        layout=S.DEFAULT_LAYOUT,
    )
    snap_idx = list(range(len(endgame_snaps)))
    exact = evaluate_theta_v3(
        CEMAimV3Policy.prior_exact_follow(),
        env,
        train_seeds=train_seeds,
        val_seeds=[],
        snapshots=endgame_snaps,
        val_snapshots=[],
        teacher=None,
        snapshot_horizon=400,
        snapshots_per_theta=len(endgame_snaps),
        stress_seeds=(10, 11, 13, 14, 19),
        teacher_weight=0.0,
        full_episode_weight=1.0,
        snapshot_weight=1.0,
        rng=np.random.default_rng(0),
        caches=caches,
        snapshot_indices=snap_idx,
        skip_behavior_metrics=True,
    )
    mild = evaluate_theta_v3(
        CEMAimV3Policy.prior_mild_left_endgame(),
        env,
        train_seeds=train_seeds,
        val_seeds=[],
        snapshots=endgame_snaps,
        val_snapshots=[],
        teacher=None,
        snapshot_horizon=400,
        snapshots_per_theta=len(endgame_snaps),
        stress_seeds=(10, 11, 13, 14, 19),
        teacher_weight=0.0,
        full_episode_weight=1.0,
        snapshot_weight=1.0,
        rng=np.random.default_rng(0),
        caches=caches,
        snapshot_indices=snap_idx,
        skip_behavior_metrics=True,
    )
    env.close()
    assert mild["selection_key"] > exact["selection_key"]
    assert mild["residual_activation_rate"] > 0.0


def test_train_v2_quick_subprocess():
    save_dir = ROOT / "runs" / "cem_aim_smoke_v2"
    result = subprocess.run(
        [
            sys.executable, str(ROOT / "train_cem_aim.py"),
            "--quick", "--policy-version", "cem_aim_v2",
            "--generations", "1", "--population-size", "4",
            "--episodes-per-theta", "1", "--save-dir", str(save_dir),
        ],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert (save_dir / "cem_aim_best.npz").exists()


def test_train_v3_quick_subprocess():
    save_dir = ROOT / "runs" / "cem_aim_smoke_v3"
    snap_dir = ROOT / "runs" / "cem_aim"
    snaps = list(snap_dir.glob("opportunity_snapshots_*.npz"))
    snap_arg = ["--opportunity-snapshots", str(snaps[0])] if snaps else []
    cmd = [
        sys.executable, str(ROOT / "train_cem_aim.py"),
        "--quick", "--policy-version", "cem_aim_v3_residual_option",
        "--generations", "1", "--population-size", "4",
        "--train-seeds", "0:1", "--save-dir", str(save_dir),
        *snap_arg,
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr + result.stdout
    assert (save_dir / "cem_aim_train_best.npz").exists()


def test_make_agent():
    agent = make_agent("cem_aim")
    env = CatBreakEnv()
    obs = env.reset(seed=0)
    action = agent.act(obs, env=env)
    assert action in (0, 1, 2)
    env.close()


def run_all():
    tests = [
        test_num_params,
        test_v2_act_with_env,
        test_v3_act_with_env,
        test_exact_followball,
        test_v3_followball_reproduction,
        test_clone_restore,
        test_v3_residual_nontrivial,
        test_v3_commitment_holds_target,
        test_v3_endgame_snapshot_scoring,
        test_v3_endgame_selection_beats_exact_follow,
        test_train_v2_quick_subprocess,
        test_make_agent,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
