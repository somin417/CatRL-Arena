"""Experience replay buffers (vector and hybrid observations)."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import torch

ObsType = Union[np.ndarray, dict[str, np.ndarray]]


def is_hybrid_obs(obs: Any) -> bool:
    return isinstance(obs, dict) and "grid" in obs and "vector" in obs


def clone_obs(obs: ObsType) -> ObsType:
    if is_hybrid_obs(obs):
        return {
            "grid": np.asarray(obs["grid"], dtype=np.float32).copy(),
            "vector": np.asarray(obs["vector"], dtype=np.float32).copy(),
        }
    return np.asarray(obs, dtype=np.float32).copy()


def stack_obs(batch_obs: list[ObsType]) -> ObsType:
    if is_hybrid_obs(batch_obs[0]):
        return {
            "grid": np.stack([b["grid"] for b in batch_obs]),
            "vector": np.stack([b["vector"] for b in batch_obs]),
        }
    return np.stack(batch_obs)


def _to_device(
    tensor: torch.Tensor,
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> torch.Tensor:
    if tensor.device == device:
        return tensor
    return tensor.to(device, non_blocking=non_blocking)


def _batch_tensor(
    data,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    pin = device.type == "cuda"
    tensor = torch.as_tensor(data, dtype=dtype)
    if pin:
        tensor = tensor.pin_memory()
    return tensor.to(device, non_blocking=pin)


def obs_to_tensors(
    obs: ObsType,
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
    if is_hybrid_obs(obs):
        grid = torch.as_tensor(obs["grid"], dtype=torch.float32)
        vector = torch.as_tensor(obs["vector"], dtype=torch.float32)
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        return {
            "grid": _to_device(grid, device, non_blocking=non_blocking),
            "vector": _to_device(vector, device, non_blocking=non_blocking),
        }
    tensor = torch.as_tensor(obs, dtype=torch.float32)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return _to_device(tensor, device, non_blocking=non_blocking)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: Optional[int] = None) -> None:
        self.capacity = int(capacity)
        self.buffer: list = []
        self.pos = 0
        self.rng = np.random.default_rng(seed)
        self.hybrid = False

    def push(
        self,
        state: ObsType,
        action: int,
        reward: float,
        next_state: ObsType,
        done: bool,
        reward_brick: Optional[float] = None,
        reward_time: Optional[float] = None,
    ) -> None:
        if is_hybrid_obs(state):
            self.hybrid = True
        transition = (
            clone_obs(state),
            int(action),
            float(reward),
            clone_obs(next_state),
            float(done),
            reward_brick,
            reward_time,
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, device: torch.device):
        n = len(self.buffer)
        if n == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        size = min(batch_size, n)
        indices = self.rng.choice(n, size=size, replace=False)
        batch = [self.buffer[i] for i in indices]
        pin = device.type == "cuda"
        states = obs_to_tensors(
            stack_obs([b[0] for b in batch]), device, non_blocking=pin
        )
        actions = _batch_tensor([[b[1]] for b in batch], dtype=torch.long, device=device)
        rewards = _batch_tensor([b[2] for b in batch], dtype=torch.float32, device=device)
        next_states = obs_to_tensors(
            stack_obs([b[3] for b in batch]), device, non_blocking=pin
        )
        dones = _batch_tensor([b[4] for b in batch], dtype=torch.float32, device=device)
        reward_brick = _batch_tensor(
            [b[5] if b[5] is not None else 0.0 for b in batch],
            dtype=torch.float32,
            device=device,
        )
        reward_time = _batch_tensor(
            [b[6] if b[6] is not None else 0.0 for b in batch],
            dtype=torch.float32,
            device=device,
        )
        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            None,
            reward_brick,
            reward_time,
        )

    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(capacity, seed=seed)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)

    def push(
        self,
        state: ObsType,
        action: int,
        reward: float,
        next_state: ObsType,
        done: bool,
        td_error: Optional[float] = None,
        reward_brick: Optional[float] = None,
        reward_time: Optional[float] = None,
    ) -> None:
        prio = float(td_error) if td_error is not None else 1.0
        if len(self.buffer) > 0:
            prio = max(prio, float(self.priorities[: len(self.buffer)].max()))
        else:
            prio = max(prio, 1.0)
        self.priorities[self.pos] = prio
        super().push(
            state, action, reward, next_state, done, reward_brick, reward_time
        )

    def sample(self, batch_size: int, device: torch.device):
        n = len(self.buffer)
        if n == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        size = min(batch_size, n)

        if n < size:
            return super().sample(batch_size, device)

        prios = self.priorities[:n].astype(np.float64)
        if not np.isfinite(prios).all() or prios.sum() <= 0:
            return super().sample(batch_size, device)

        probs = prios ** self.alpha
        prob_sum = probs.sum()
        if prob_sum <= 0 or not np.isfinite(prob_sum):
            return super().sample(batch_size, device)
        probs /= prob_sum

        indices = self.rng.choice(n, size=size, replace=False, p=probs)
        batch = [self.buffer[i] for i in indices]
        pin = device.type == "cuda"
        states = obs_to_tensors(
            stack_obs([b[0] for b in batch]), device, non_blocking=pin
        )
        actions = _batch_tensor([[b[1]] for b in batch], dtype=torch.long, device=device)
        rewards = _batch_tensor([b[2] for b in batch], dtype=torch.float32, device=device)
        next_states = obs_to_tensors(
            stack_obs([b[3] for b in batch]), device, non_blocking=pin
        )
        dones = _batch_tensor([b[4] for b in batch], dtype=torch.float32, device=device)
        reward_brick = _batch_tensor(
            [b[5] if b[5] is not None else 0.0 for b in batch],
            dtype=torch.float32,
            device=device,
        )
        reward_time = _batch_tensor(
            [b[6] if b[6] is not None else 0.0 for b in batch],
            dtype=torch.float32,
            device=device,
        )

        weights = (n * probs[indices]) ** (-self.beta)
        w_max = weights.max()
        if w_max > 0:
            weights /= w_max
        else:
            weights = np.ones_like(weights)
        weights_t = _batch_tensor(weights, dtype=torch.float32, device=device)
        self.beta = min(1.0, self.beta + self.beta_increment)
        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            weights_t,
            reward_brick,
            reward_time,
        )

    def update_priorities(self, indices, td_errors) -> None:
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()
        priorities = np.abs(np.asarray(td_errors, dtype=np.float32)) + self.epsilon
        for idx, prio in zip(indices, priorities):
            self.priorities[int(idx)] = float(prio)
