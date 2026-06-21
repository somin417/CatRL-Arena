"""Train CEM-Aim v2/v3 policy via cross-entropy search."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import settings as S
from catbreak_env import CatBreakEnv
from cem_aim_policy import (
    CEMAimPolicy,
    NUM_PARAMS,
    POLICY_VERSION_V3,
    build_prior_candidates,
)
from cem_aim_v3 import (
    WEAK_SEEDS,
    SAFETY_SEEDS,
    DEFAULT_STRESS_SEEDS,
    collect_opportunity_snapshots,
    grid_search_shot_macros,
    parse_seed_spec,
    sanity_clone_restore,
    sanity_followball_reproduction,
    train_cem_aim_v3,
)
from evaluate import env_seed_for_episode, run_episode, summarize_rows
from torch_utils import configure_torch, default_parallel_workers, get_device


def lexicographic_key(metrics: dict) -> tuple:
    return (
        metrics["mean_broken_bricks"],
        metrics["clear_rate"],
        metrics["mean_blocks_per_100_steps"],
        metrics["mean_return"],
        -metrics["mean_steps"],
    )


def sort_key_with_imitation(
    metrics: dict,
    imitation_agreement: float,
    imitation_weight: float,
    use_imitation: bool,
) -> tuple:
    base = lexicographic_key(metrics)
    if not use_imitation:
        return base + (0.0,)
    tie = imitation_weight * imitation_agreement
    return base + (tie,)


class _PolicyAgentAdapter:
    name = "CEM-Aim"

    def __init__(self, policy: CEMAimPolicy) -> None:
        self.policy = policy

    def reset(self, seed: Optional[int] = None) -> None:
        pass

    def act(self, obs, info=None, env=None) -> int:
        return self.policy.act(obs, info=info, env=env)


def evaluate_theta(
    theta: np.ndarray,
    env: CatBreakEnv,
    episodes: int,
    base_seed: int,
) -> dict:
    policy = CEMAimPolicy(theta)
    rows = []
    for ep in range(episodes):
        ep_seed = env_seed_for_episode(base_seed, ep)
        row = run_episode(env, _PolicyAgentAdapter(policy), ep_seed)
        rows.append(row)
    summary = summarize_rows(rows)
    return {
        "mean_broken_bricks": summary["avg_broken_bricks"],
        "clear_rate": summary["clear_rate"],
        "mean_blocks_per_100_steps": summary["avg_blocks_per_100_steps"],
        "mean_return": summary["avg_return"],
        "mean_steps": summary["avg_steps"],
    }


def _evaluate_theta_worker(payload: tuple[np.ndarray, str, int, int]) -> dict:
    theta, layout, episodes, base_seed = payload
    env = CatBreakEnv(config={"layout": layout})
    try:
        return evaluate_theta(theta, env, episodes, base_seed)
    finally:
        env.close()


def evaluate_population_parallel(
    population: np.ndarray,
    layout: str,
    episodes: int,
    base_seed: int,
    workers: int,
    on_eval_complete: Callable[[int, dict], None] | None = None,
) -> list[dict]:
    if workers <= 1:
        env = CatBreakEnv(config={"layout": layout})
        try:
            results = []
            for i, theta in enumerate(population):
                metrics = evaluate_theta(theta, env, episodes, base_seed + i)
                results.append(metrics)
                if on_eval_complete is not None:
                    on_eval_complete(i, metrics)
            return results
        finally:
            env.close()

    payloads = [
        (theta.copy(), layout, episodes, base_seed + i)
        for i, theta in enumerate(population)
    ]
    results: list[Optional[dict]] = [None] * len(payloads)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_evaluate_theta_worker, payload): idx
            for idx, payload in enumerate(payloads)
        }
        for future in as_completed(futures):
            idx = futures[future]
            metrics = future.result()
            results[idx] = metrics
            if on_eval_complete is not None:
                on_eval_complete(idx, metrics)
    return [r for r in results if r is not None]


class TrainingProgressReporter:
    def __init__(
        self,
        total_evals: int,
        generations: int,
        population_size: int,
        interval_pct: float = 5.0,
    ) -> None:
        self.total_evals = max(1, total_evals)
        self.generations = generations
        self.population_size = population_size
        self.interval_pct = interval_pct
        self.completed = 0
        self.next_pct = interval_pct
        self.t0 = time.perf_counter()
        self.current_gen = 0
        self.gen_eval_done = 0
        self.gen_running_best: Optional[dict] = None
        self.global_best_metrics: Optional[dict] = None
        self.global_best_key: tuple = (-1.0,)
        self.current_sigma = 0.0

    def start_generation(self, gen: int, sigma: float) -> None:
        self.current_gen = gen
        self.gen_eval_done = 0
        self.gen_running_best = None
        self.current_sigma = sigma

    def note_eval(self, metrics: dict) -> None:
        self.completed += 1
        self.gen_eval_done += 1
        key = lexicographic_key(metrics)
        if self.gen_running_best is None or key > lexicographic_key(self.gen_running_best):
            self.gen_running_best = metrics
        if key > self.global_best_key:
            self.global_best_key = key
            self.global_best_metrics = metrics.copy()

        pct_done = 100.0 * self.completed / self.total_evals
        while pct_done >= self.next_pct - 1e-9 and self.next_pct <= 100.0:
            self._print_milestone(self.next_pct)
            self.next_pct += self.interval_pct

    def _print_milestone(self, milestone_pct: float) -> None:
        elapsed = time.perf_counter() - self.t0
        remaining = self.total_evals - self.completed
        eta = (elapsed / max(1, self.completed)) * remaining if remaining > 0 else 0.0
        gen_best = self.gen_running_best or {}
        global_best = self.global_best_metrics or {}
        print(
            f"[progress {milestone_pct:5.1f}%] "
            f"eval {self.completed}/{self.total_evals} | "
            f"gen {self.current_gen + 1}/{self.generations} "
            f"({self.gen_eval_done}/{self.population_size}) | "
            f"gen_best broken={gen_best.get('mean_broken_bricks', 0.0):.2f} "
            f"clear={gen_best.get('clear_rate', 0.0)*100:.0f}% | "
            f"global_best broken={global_best.get('mean_broken_bricks', 0.0):.2f} "
            f"clear={global_best.get('clear_rate', 0.0)*100:.0f}% | "
            f"sigma={self.current_sigma:.3f} | "
            f"elapsed={elapsed:.0f}s eta={eta:.0f}s"
        )


class MpcDemoImitation:
    def __init__(self, demo_path: Path, max_samples: int = 512, seed: int = 0) -> None:
        data = np.load(demo_path, allow_pickle=False)
        self.actions = np.asarray(data["action"], dtype=np.int64)
        obs_before = np.asarray(data["obs_before"], dtype=np.float32)
        n = len(self.actions)
        rng = np.random.default_rng(seed)
        if n > max_samples:
            idx = rng.choice(n, size=max_samples, replace=False)
            self.actions = self.actions[idx]
            obs_before = obs_before[idx]
        self.obs_list = [obs_before[i] for i in range(len(self.actions))]

    def agreement(self, theta: np.ndarray) -> float:
        if len(self.obs_list) == 0:
            return 0.0
        policy = CEMAimPolicy(theta)
        matches = sum(
            1
            for obs, target in zip(self.obs_list, self.actions)
            if policy.act(obs, env=None) == int(target)
        )
        return matches / len(self.actions)

    def fit_best_prior(self, candidates: list[np.ndarray]) -> Optional[np.ndarray]:
        if not candidates:
            return None
        return max(candidates, key=self.agreement).copy()


def sample_population(
    mean: np.ndarray,
    sigma: float,
    population_size: int,
    rng: np.random.Generator,
    priors: list[np.ndarray],
) -> np.ndarray:
    n_random = max(0, population_size - len(priors))
    samples = rng.normal(mean, sigma, size=(n_random, NUM_PARAMS))
    population = [p.copy() for p in priors]
    population.extend(samples)
    if len(population) < population_size:
        extra = rng.normal(mean, sigma, size=(population_size - len(population), NUM_PARAMS))
        population.extend(extra)
    return np.asarray(population[:population_size], dtype=np.float64)


def update_elite_mean(
    population: np.ndarray,
    scores: list[tuple],
    elite_frac: float,
    smoothing: float,
    prev_mean: np.ndarray,
) -> np.ndarray:
    elite_n = max(1, int(len(population) * elite_frac))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    elite_idx = order[:elite_n]
    elite_mean = population[elite_idx].mean(axis=0)
    return (1.0 - smoothing) * prev_mean + smoothing * elite_mean


def train_cem_aim_v2(args: argparse.Namespace) -> Path:
    run_dir = Path(args.save_dir) if args.save_dir else _default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    device = configure_torch(get_device())
    workers = default_parallel_workers(args.workers)
    print(f"Torch device: {device} (CEM-Aim rollouts stay on CPU)")

    rng = np.random.default_rng(args.seed)
    mean = CEMAimPolicy.prior_follow_like()
    sigma = args.sigma_init
    best_theta = mean.copy()
    best_metrics: Optional[dict] = None
    best_key: tuple = (-1.0,)

    imitation: Optional[MpcDemoImitation] = None
    mpc_best_theta: Optional[np.ndarray] = None
    if args.mpc_demo:
        demo_path = Path(args.mpc_demo)
        imitation = MpcDemoImitation(demo_path, seed=args.seed)
        prior_cands = build_prior_candidates(include_follow=True, include_targeted=True)
        mpc_best_theta = imitation.fit_best_prior(prior_cands)

    gen_rows: list[dict] = []
    total_evals = args.generations * args.population_size
    progress = TrainingProgressReporter(total_evals, args.generations, args.population_size)

    for gen in range(args.generations):
        use_imitation = imitation is not None and gen < args.mpc_imitation_generations
        priors = build_prior_candidates(
            include_follow=args.include_follow_init,
            include_targeted=args.include_targeted_init,
            previous_best=best_theta,
            mpc_best=mpc_best_theta if gen < args.mpc_imitation_generations else None,
        )
        population = sample_population(mean, sigma, args.population_size, rng, priors)
        progress.start_generation(gen, sigma)

        rollout_metrics = evaluate_population_parallel(
            population, args.layout, args.episodes_per_theta,
            args.seed + gen * 1000, workers,
            on_eval_complete=lambda _i, m: progress.note_eval(m),
        )

        scored = []
        metrics_list = []
        for i, metrics in enumerate(rollout_metrics):
            imitation_agreement = imitation.agreement(population[i]) if imitation else 0.0
            key = sort_key_with_imitation(
                metrics, imitation_agreement, args.mpc_imitation_weight, use_imitation
            )
            scored.append(key)
            metrics_list.append({**metrics, "imitation_agreement": imitation_agreement})

        gen_best_idx = max(range(len(scored)), key=lambda i: scored[i])
        gen_best_theta = population[gen_best_idx]
        gen_best_metrics = metrics_list[gen_best_idx]
        if lexicographic_key(gen_best_metrics) > best_key:
            best_key = lexicographic_key(gen_best_metrics)
            best_theta = gen_best_theta.copy()
            best_metrics = gen_best_metrics

        mean = update_elite_mean(population, scored, args.elite_frac, args.smoothing, mean)
        sigma = max(args.sigma_min, sigma * (1 - args.smoothing * 0.5))
        gen_rows.append({"generation": gen, **gen_best_metrics})

    best_path = run_dir / "cem_aim_best.npz"
    CEMAimPolicy(best_theta).save(str(best_path))
    CEMAimPolicy(mean).save(str(run_dir / "cem_aim_last.npz"))
    with (run_dir / "generation_history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(gen_rows[0].keys()))
        writer.writeheader()
        writer.writerows(gen_rows)
    return run_dir


def _default_run_dir() -> Path:
    return S.CEM_AIM_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train CEM-Aim on CatBreakEnv")
    p.add_argument("--policy-version", type=str, default="cem_aim_v2",
                   choices=["cem_aim_v2", POLICY_VERSION_V3])
    p.add_argument("--objective", type=str, default="pairwise_followball_plus_opportunity")
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--population-size", type=int, default=64)
    p.add_argument("--episodes-per-theta", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layout", type=str, default=S.DEFAULT_LAYOUT,
                   choices=[S.LAYOUT_RECT, S.LAYOUT_CAT])
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--sigma-init", type=float, default=0.5)
    p.add_argument("--sigma-min", type=float, default=0.05, dest="sigma_min")
    p.add_argument("--sigma-floor", type=float, default=0.05)
    p.add_argument("--smoothing", type=float, default=0.25)
    p.add_argument("--elite-frac", type=float, default=0.2)
    p.add_argument("--mpc-demo", type=str, default=None)
    p.add_argument("--teacher-demo", type=str, default=None)
    p.add_argument("--teacher-imitation-weight", type=float, default=-1.0)
    p.add_argument("--teacher-bc-iters", type=int, default=0,
                   help="CEM steps to BC-fit theta to teacher demo before training (v3)")
    p.add_argument("--teacher-max-samples", type=int, default=248,
                   help="Max teacher deviation samples per theta eval (v3)")
    p.add_argument("--no-teacher-bc-init", action="store_true",
                   help="Do not replace CEM mean with teacher BC fit (v3)")
    p.add_argument("--mpc-imitation-weight", type=float, default=0.05)
    p.add_argument("--mpc-imitation-generations", type=int, default=5)
    p.add_argument("--include-follow-init", action="store_true", default=True)
    p.add_argument("--no-include-follow-init", dest="include_follow_init", action="store_false")
    p.add_argument("--include-targeted-init", action="store_true", default=True)
    p.add_argument("--no-include-targeted-init", dest="include_targeted_init", action="store_false")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--sequential", action="store_true")
    p.add_argument("--progress-interval", type=float, default=5.0)
    p.add_argument("--quick", action="store_true")
    # v3 seeds
    p.add_argument("--train-seeds", type=str, default="0:4")
    p.add_argument("--val-seeds", type=str, default="5:9")
    p.add_argument("--stress-seeds", type=str, default="10,11,13,14,19")
    p.add_argument("--opportunity-snapshots", type=str, default=None)
    p.add_argument("--snapshot-rollout-horizon", type=int, default=2000)
    p.add_argument("--snapshots-per-theta", type=int, default=64)
    p.add_argument("--select-by", type=str, default="train", choices=["train", "val"])
    p.add_argument("--save-top-k", type=int, default=5)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--full-episode-weight", type=float, default=1.0)
    p.add_argument("--snapshot-weight", type=float, default=1.0)
    p.add_argument("--max-global-deviation-rate", type=float, default=0.05)
    p.add_argument("--min-opportunity-activation-rate", type=float, default=0.05)
    p.add_argument("--behavior-sample-seeds", type=int, default=1,
                   help="train seeds sampled for behavior-rate metrics (v3)")
    p.add_argument("--skip-behavior-metrics", action="store_true",
                   help="skip behavior-rate episode logging (v3, faster)")
    # v3 endgame-aware scoring (anti FollowBall-clone trap)
    p.add_argument("--stress-loss-penalty", type=float, default=500.0,
                   help="episode penalty per stress-seed loss vs FollowBall (v3)")
    p.add_argument("--non-stress-loss-penalty", type=float, default=40.0,
                   help="soft penalty per non-stress train loss (v3)")
    p.add_argument("--endgame-bricks-threshold", type=int, default=15,
                   help="remaining bricks <= threshold counts as endgame snapshot (v3)")
    p.add_argument("--endgame-speed-step-margin", type=int, default=25,
                   help="min step advantage for endgame speed win in snapshot rollouts (v3)")
    p.add_argument("--anti-clone-snap-penalty", type=float, default=250.0,
                   help="snapshot score penalty when opportunity exists but no residual/wins (v3)")
    p.add_argument("--endgame-residual-bonus", type=float, default=120.0,
                   help="snapshot bonus for endgame residual activation without life loss (v3)")
    p.add_argument("--no-endgame-aware-selection", dest="endgame_aware_selection",
                   action="store_false", default=True,
                   help="revert to legacy loss-first selection key (v3)")
    # snapshot collection
    p.add_argument("--collect-opportunity-snapshots", action="store_true")
    p.add_argument("--snapshot-seeds", type=str, default="0:19")
    p.add_argument("--snapshot-output", type=str, default=None)
    p.add_argument("--max-snapshots", type=int, default=512)
    p.add_argument("--snapshot-min-gap-steps", type=int, default=200)
    # sanity
    p.add_argument("--sanity-followball", action="store_true")
    p.add_argument("--sanity-clone-restore", action="store_true")
    p.add_argument("--grid-search-shot-macros", action="store_true",
                   help="Run deterministic shot-macro grid search (no CEM training)")
    p.add_argument("--grid-stage1-horizon", type=int, default=1200,
                   help="Snapshot rollout horizon for grid stage 1")
    p.add_argument("--grid-episode-seeds", type=str, default="0:19",
                   help="Full-episode seeds for grid stage 2 vetting")
    p.add_argument("--grid-top-k", type=int, default=50,
                   help="Top configs kept after grid stage 1")
    p.add_argument("--grid-quick", action="store_true",
                   help="Tiny grid subset for smoke testing")
    p.add_argument("--grid-stage1-max-snapshots", type=int, default=128,
                   help="Weighted snapshot subsample count for grid stage 1")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.sequential:
        args.workers = 0
    if args.teacher_imitation_weight < 0:
        args.teacher_imitation_weight = 5.0 if args.teacher_demo else 0.0
    if args.teacher_demo and args.teacher_bc_iters == 0:
        args.teacher_bc_iters = 50
    if args.quick:
        args.generations = min(args.generations, 2)
        args.population_size = min(args.population_size, 8)
        args.episodes_per_theta = min(args.episodes_per_theta, 1)
        args.snapshots_per_theta = min(args.snapshots_per_theta, 8)
        args.snapshot_rollout_horizon = min(args.snapshot_rollout_horizon, 500)

    if args.sanity_clone_restore:
        env = CatBreakEnv()
        ok = sanity_clone_restore(env)
        env.close()
        raise SystemExit(0 if ok else 1)

    if args.sanity_followball:
        from cem_aim_policy import CEMAimV3Policy
        run_dir = Path(args.save_dir) if args.save_dir else S.CEM_AIM_DIR
        run_dir.mkdir(parents=True, exist_ok=True)
        prior_path = run_dir / "cem_aim_exact_follow_prior.npz"
        CEMAimV3Policy(CEMAimV3Policy.prior_exact_follow()).save(str(prior_path))
        seeds = parse_seed_spec(args.snapshot_seeds or "0:19")
        ok = sanity_followball_reproduction(prior_path, seeds, args.layout)
        raise SystemExit(0 if ok else 1)

    if args.collect_opportunity_snapshots:
        seeds = parse_seed_spec(args.snapshot_seeds)
        stress = tuple(parse_seed_spec(args.stress_seeds))
        out = Path(args.snapshot_output) if args.snapshot_output else None
        collect_opportunity_snapshots(
            seeds, layout=args.layout, max_snapshots=args.max_snapshots,
            min_gap_steps=args.snapshot_min_gap_steps, stress_seeds=stress, output=out,
        )
        return

    t0 = time.perf_counter()

    if args.grid_search_shot_macros:
        grid_search_shot_macros(args)
        print(f"Wall clock: {time.perf_counter() - t0:.1f}s")
        return

    if args.policy_version == POLICY_VERSION_V3:
        train_cem_aim_v3(args)
    else:
        train_cem_aim_v2(args)
    print(f"Wall clock: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
