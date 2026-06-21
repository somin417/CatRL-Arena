"""Parallel batch rollout evaluation for CEM-MPC (CPU workers on GPU workstations)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Optional

import numpy as np

from catbreak_env import CatBreakEnv
from cem_mpc import simulate_sequence
from torch_utils import default_parallel_workers

_worker_env: Optional[CatBreakEnv] = None


def _worker_init(env_config: dict, layout: str) -> None:
    """Create one reusable env per worker process."""
    global _worker_env
    _worker_env = CatBreakEnv(config={**env_config, "layout": layout})


def _worker_evaluate(payload: tuple[int, dict, np.ndarray, float, int]) -> tuple[int, dict]:
    """Evaluate one sequence using the worker-local env."""
    idx, start_state, actions, gamma, extra_rollout = payload
    assert _worker_env is not None
    result = simulate_sequence(
        _worker_env,
        start_state,
        actions,
        gamma=gamma,
        extra_rollout_steps=extra_rollout,
    )
    return idx, result


class ParallelRolloutPool:
    """Persistent process pool with one CatBreakEnv per worker."""

    def __init__(self, workers: int, layout: str, env_config: dict) -> None:
        self.workers = int(workers)
        self.layout = layout
        self.env_config = dict(env_config)
        self._executor: Optional[ProcessPoolExecutor] = None

    def _ensure_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.workers,
                initializer=_worker_init,
                initargs=(self.env_config, self.layout),
            )
        return self._executor

    def evaluate(
        self,
        start_state: dict,
        sequences: np.ndarray,
        *,
        gamma: float = 1.0,
        extra_rollout_steps: int = 0,
    ) -> list[dict]:
        seqs = np.asarray(sequences, dtype=np.int64)
        n = len(seqs)
        if n == 0:
            return []

        executor = self._ensure_executor()
        payloads = [
            (i, start_state, seqs[i], gamma, extra_rollout_steps)
            for i in range(n)
        ]
        results: list[Optional[dict]] = [None] * n
        futures = {executor.submit(_worker_evaluate, p): p[0] for p in payloads}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
        return [r for r in results if r is not None]

    def shutdown(self, wait: bool = True) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None


def evaluate_sequences_parallel(
    env_config: dict,
    start_state: dict,
    sequences: np.ndarray,
    *,
    layout: str,
    gamma: float = 1.0,
    extra_rollout_steps: int = 0,
    workers: Optional[int] = None,
    pool: Optional[ParallelRolloutPool] = None,
) -> list[dict]:
    """Evaluate many action sequences in parallel; order matches input rows."""
    seqs = np.asarray(sequences, dtype=np.int64)
    n = len(seqs)
    if n == 0:
        return []
    workers = default_parallel_workers(workers)

    if pool is not None:
        return pool.evaluate(
            start_state,
            seqs,
            gamma=gamma,
            extra_rollout_steps=extra_rollout_steps,
        )

    if workers <= 1 or n == 1:
        env = CatBreakEnv(config={**env_config, "layout": layout})
        try:
            return [
                simulate_sequence(
                    env, start_state, seqs[i], gamma=gamma,
                    extra_rollout_steps=extra_rollout_steps,
                )
                for i in range(n)
            ]
        finally:
            env.close()

    ephemeral = ParallelRolloutPool(workers, layout, env_config)
    try:
        return ephemeral.evaluate(
            start_state,
            seqs,
            gamma=gamma,
            extra_rollout_steps=extra_rollout_steps,
        )
    finally:
        ephemeral.shutdown()
