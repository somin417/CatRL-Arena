"""CatBreak DQN agent (MLP vector + CNN hybrid, optional lexicographic heads)."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import settings as S
from networks import make_q_network
from replay_buffer import PrioritizedReplayBuffer, ReplayBuffer, obs_to_tensors
from torch_utils import configure_torch, get_device


ObsType = Union[np.ndarray, dict[str, np.ndarray]]


def decompose_rewards(info: Optional[dict], scalar_reward: float) -> tuple[float, float]:
    """Lexicographic reward components: brick count and time/death."""
    if info is None:
        return 0.0, float(scalar_reward)
    r_brick = float(info.get("bricks_broken_this_step", 0))
    r_time = -1.0
    if info.get("terminal_reason") == "no_lives":
        r_time -= 50.0
    return r_brick, r_time


class CatBreakDQNAgent:
    def __init__(
        self,
        n_actions: int,
        obs_dim: Optional[int] = None,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        lr: float = 5e-4,
        batch_size: int = 128,
        replay_capacity: int = 100_000,
        min_replay_size: int = 1000,
        target_update_freq: int = 1000,
        tau: Optional[float] = None,
        double_dqn: bool = True,
        dueling: bool = True,
        per: bool = False,
        grad_clip_norm: float = 10.0,
        seed: int = 0,
        device: Optional[torch.device] = None,
        obs_mode: str = S.OBS_MODE_VECTOR,
        network_type: str = "mlp",
        grid_shape: Optional[tuple[int, int, int]] = None,
        vector_dim: Optional[int] = None,
        n_step: int = 1,
        lexicographic: bool = False,
        beta_time: float = 0.2,
        brick_tolerance: float = 0.05,
    ) -> None:
        self.n_actions = int(n_actions)
        self.obs_dim = int(obs_dim) if obs_dim is not None else None
        self.hidden_dim = int(hidden_dim)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.min_replay_size = int(min_replay_size)
        self.target_update_freq = int(target_update_freq)
        self.tau = tau
        self.double_dqn = bool(double_dqn)
        self.dueling = bool(dueling)
        self.per = bool(per)
        self.grad_clip_norm = float(grad_clip_norm)
        self.obs_mode = obs_mode
        self.network_type = network_type
        self.grid_shape = grid_shape
        self.vector_dim = vector_dim
        self.n_step = max(1, int(n_step))
        self.lexicographic = bool(lexicographic)
        self.beta_time = float(beta_time)
        self.brick_tolerance = float(brick_tolerance)

        self.device = configure_torch(device or get_device())
        self.rng = np.random.default_rng(seed)

        if network_type == "cnn":
            if grid_shape is None or vector_dim is None:
                raise ValueError("CNN agent requires grid_shape and vector_dim.")
            self.qnet = make_q_network(
                n_actions,
                network_type="cnn",
                dueling=dueling and not lexicographic,
                grid_shape=grid_shape,
                vector_dim=vector_dim,
                lexicographic=lexicographic,
            ).to(self.device)
            self.target_qnet = make_q_network(
                n_actions,
                network_type="cnn",
                dueling=dueling and not lexicographic,
                grid_shape=grid_shape,
                vector_dim=vector_dim,
                lexicographic=lexicographic,
            ).to(self.device)
        else:
            if self.obs_dim is None:
                raise ValueError("MLP agent requires obs_dim.")
            self.qnet = make_q_network(
                n_actions,
                network_type="mlp",
                obs_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                dueling=self.dueling,
            ).to(self.device)
            self.target_qnet = make_q_network(
                n_actions,
                network_type="mlp",
                obs_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                dueling=self.dueling,
            ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.qnet.parameters(), lr=lr, weight_decay=0.01
        )

        if self.per:
            self.replay_buffer: ReplayBuffer = PrioritizedReplayBuffer(
                replay_capacity, seed=seed
            )
        else:
            self.replay_buffer = ReplayBuffer(replay_capacity, seed=seed)

        self.global_step = 0
        self.train_updates = 0
        self._n_step_buffer: deque = deque()
        self.hard_update()

    def _forward_q(
        self, net: nn.Module, obs: Union[torch.Tensor, dict[str, torch.Tensor]]
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(obs, dict):
            return net(obs["grid"], obs["vector"])
        return net(obs)

    def _q_values_for_action(
        self, net: nn.Module, obs_t: Union[torch.Tensor, dict[str, torch.Tensor]]
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        with torch.inference_mode():
            return self._forward_q(net, obs_t)

    def _select_lexicographic(
        self, q_brick: torch.Tensor, q_time: torch.Tensor
    ) -> int:
        qb = q_brick.squeeze(0)
        qt = q_time.squeeze(0)
        max_brick = qb.max()
        brick_mask = qb >= max_brick - self.brick_tolerance
        return int(qt.masked_fill(~brick_mask, float("-inf")).argmax().item())

    def greedy_action(self, obs: ObsType) -> int:
        obs_t = obs_to_tensors(obs, self.device)
        if self.lexicographic:
            q_brick, q_time = self._q_values_for_action(self.qnet, obs_t)
            return self._select_lexicographic(q_brick, q_time)
        q_values = self._q_values_for_action(self.qnet, obs_t)
        assert isinstance(q_values, torch.Tensor)
        return int(q_values.argmax(dim=1).item())

    def act(self, obs: ObsType, epsilon: float = 0.0) -> int:
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return self.greedy_action(obs)

    def _commit_n_step(self, length: int) -> None:
        window = list(self._n_step_buffer)[:length]
        if not window:
            return
        obs0, action0, _, _, _, _, _ = window[0]
        reward_sum = r_brick_sum = r_time_sum = 0.0
        next_obs = window[-1][5]
        done = window[-1][6]
        for i, (_, _, r, rb, rt, nobs, d) in enumerate(window):
            reward_sum += (self.gamma ** i) * r
            r_brick_sum += (self.gamma ** i) * rb
            r_time_sum += (self.gamma ** i) * rt
            if d:
                next_obs = nobs
                done = True
                break
        self._push_transition(
            obs0, action0, reward_sum, next_obs, done, r_brick_sum, r_time_sum
        )

    def store_transition(
        self,
        obs: ObsType,
        action: int,
        reward: float,
        next_obs: ObsType,
        done: bool,
        info: Optional[dict] = None,
    ) -> None:
        r_brick, r_time = decompose_rewards(info, reward)
        if self.n_step <= 1:
            self._push_transition(obs, action, reward, next_obs, done, r_brick, r_time)
            return

        self._n_step_buffer.append(
            (obs, action, reward, r_brick, r_time, next_obs, done)
        )
        if len(self._n_step_buffer) >= self.n_step:
            self._commit_n_step(self.n_step)
            self._n_step_buffer.popleft()
        if done:
            while self._n_step_buffer:
                self._commit_n_step(len(self._n_step_buffer))
                self._n_step_buffer.popleft()

    def episode_done(self) -> None:
        """Flush any remaining partial n-step transitions."""
        while self._n_step_buffer:
            self._commit_n_step(len(self._n_step_buffer))
            self._n_step_buffer.popleft()

    def _push_transition(
        self,
        obs: ObsType,
        action: int,
        reward: float,
        next_obs: ObsType,
        done: bool,
        reward_brick: float,
        reward_time: float,
    ) -> None:
        if self.per:
            self.replay_buffer.push(
                obs,
                action,
                reward,
                next_obs,
                done,
                td_error=1.0,
                reward_brick=reward_brick,
                reward_time=reward_time,
            )
        else:
            self.replay_buffer.push(
                obs,
                action,
                reward,
                next_obs,
                done,
                reward_brick=reward_brick,
                reward_time=reward_time,
            )
        self.global_step += 1

    def _next_actions_lexicographic(
        self,
        qnet: nn.Module,
        next_states: Union[torch.Tensor, dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        q_brick, q_time = self._forward_q(qnet, next_states)
        max_brick = q_brick.max(dim=1, keepdim=True).values
        brick_mask = q_brick >= max_brick - self.brick_tolerance
        masked_time = q_time.masked_fill(~brick_mask, float("-inf"))
        return masked_time.argmax(dim=1, keepdim=True)

    def train_step(self) -> dict:
        if len(self.replay_buffer) < self.min_replay_size:
            return {}

        self.qnet.train()
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            weights,
            reward_brick,
            reward_time,
        ) = self.replay_buffer.sample(self.batch_size, self.device)

        if self.lexicographic:
            return self._train_step_lexicographic(
                states,
                actions,
                reward_brick,
                reward_time,
                next_states,
                dones,
                indices,
                weights,
            )

        pred_all = self._forward_q(self.qnet, states)
        assert isinstance(pred_all, torch.Tensor)
        pred = pred_all.gather(1, actions).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                next_q_online = self._forward_q(self.qnet, next_states)
                assert isinstance(next_q_online, torch.Tensor)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target = self._forward_q(self.target_qnet, next_states)
                assert isinstance(next_q_target, torch.Tensor)
                target_q = next_q_target.gather(1, next_actions).squeeze(1)
            else:
                next_q_target = self._forward_q(self.target_qnet, next_states)
                assert isinstance(next_q_target, torch.Tensor)
                target_q = next_q_target.max(dim=1)[0]
            target = rewards + self.gamma * target_q * (1.0 - dones)

        td_error = target - pred
        if weights is not None:
            loss = (F.smooth_l1_loss(pred, target, reduction="none") * weights).mean()
        else:
            loss = F.smooth_l1_loss(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        if self.per:
            self.replay_buffer.update_priorities(indices, td_error)

        self._maybe_update_target()
        return self._stats_dict(loss, td_error, pred_all)

    def _train_step_lexicographic(
        self,
        states,
        actions,
        reward_brick,
        reward_time,
        next_states,
        dones,
        indices,
        weights,
    ) -> dict:
        q_brick, q_time = self._forward_q(self.qnet, states)
        pred_brick = q_brick.gather(1, actions).squeeze(1)
        pred_time = q_time.gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_actions = self._next_actions_lexicographic(self.qnet, next_states)
            t_brick, t_time = self._forward_q(self.target_qnet, next_states)
            target_brick = t_brick.gather(1, next_actions).squeeze(1)
            target_time = t_time.gather(1, next_actions).squeeze(1)
            y_brick = reward_brick + self.gamma * target_brick * (1.0 - dones)
            y_time = reward_time + self.gamma * target_time * (1.0 - dones)

        td_brick = y_brick - pred_brick
        td_time = y_time - pred_time
        if weights is not None:
            loss_brick = (F.smooth_l1_loss(pred_brick, y_brick, reduction="none") * weights).mean()
            loss_time = (F.smooth_l1_loss(pred_time, y_time, reduction="none") * weights).mean()
        else:
            loss_brick = F.smooth_l1_loss(pred_brick, y_brick)
            loss_time = F.smooth_l1_loss(pred_time, y_time)
        loss = loss_brick + self.beta_time * loss_time
        td_error = td_brick

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        if self.per:
            self.replay_buffer.update_priorities(indices, td_error)

        self._maybe_update_target()
        stats = self._stats_dict(loss, td_error, q_brick)
        stats["loss_brick"] = float(loss_brick.item())
        stats["loss_time"] = float(loss_time.item())
        return stats

    def _maybe_update_target(self) -> None:
        self.train_updates += 1
        if self.tau is None:
            if self.train_updates % self.target_update_freq == 0:
                self.hard_update()
        else:
            self.soft_update()

    def _stats_dict(self, loss, td_error, pred_all) -> dict:
        return {
            "loss": float(loss.item()),
            "td_error_abs_mean": float(td_error.abs().mean().item()),
            "q_mean": float(pred_all.mean().item()),
            "q_max": float(pred_all.max().item()),
            "replay_size": len(self.replay_buffer),
            "global_step": self.global_step,
        }

    def hard_update(self) -> None:
        self.target_qnet.load_state_dict(self.qnet.state_dict())

    def soft_update(self) -> None:
        assert self.tau is not None
        for target_param, param in zip(
            self.target_qnet.parameters(), self.qnet.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def _checkpoint_meta(self) -> dict[str, Any]:
        return {
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "hidden_dim": self.hidden_dim,
            "gamma": self.gamma,
            "double_dqn": self.double_dqn,
            "dueling": self.dueling,
            "per": self.per,
            "global_step": self.global_step,
            "train_updates": self.train_updates,
            "obs_mode": self.obs_mode,
            "network_type": self.network_type,
            "grid_shape": self.grid_shape,
            "vector_dim": self.vector_dim,
            "n_step": self.n_step,
            "lexicographic": self.lexicographic,
            "beta_time": self.beta_time,
            "brick_tolerance": self.brick_tolerance,
        }

    def save(self, path: str | Path, extra: Optional[dict] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "qnet_state_dict": self.qnet.state_dict(),
            "target_qnet_state_dict": self.target_qnet.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "extra": extra or {},
            **self._checkpoint_meta(),
        }
        torch.save(payload, path)

    def load(self, path: str | Path, map_location=None) -> None:
        path = Path(path)
        if map_location is None:
            map_location = self.device
        payload = torch.load(path, map_location=map_location, weights_only=False)
        self.obs_dim = payload.get("obs_dim")
        if self.obs_dim is not None:
            self.obs_dim = int(self.obs_dim)
        self.n_actions = int(payload["n_actions"])
        self.hidden_dim = int(payload.get("hidden_dim", self.hidden_dim))
        self.gamma = float(payload.get("gamma", self.gamma))
        self.double_dqn = bool(payload.get("double_dqn", self.double_dqn))
        self.dueling = bool(payload.get("dueling", self.dueling))
        self.per = bool(payload.get("per", self.per))
        self.global_step = int(payload.get("global_step", 0))
        self.train_updates = int(payload.get("train_updates", 0))
        self.obs_mode = payload.get("obs_mode", self.obs_mode)
        self.network_type = payload.get("network_type", self.network_type)
        self.grid_shape = tuple(payload["grid_shape"]) if payload.get("grid_shape") else self.grid_shape
        self.vector_dim = payload.get("vector_dim", self.vector_dim)
        if self.vector_dim is not None:
            self.vector_dim = int(self.vector_dim)
        self.n_step = int(payload.get("n_step", self.n_step))
        self.lexicographic = bool(payload.get("lexicographic", self.lexicographic))
        self.beta_time = float(payload.get("beta_time", self.beta_time))
        self.brick_tolerance = float(payload.get("brick_tolerance", self.brick_tolerance))

        if self.network_type == "cnn":
            if self.grid_shape is None or self.vector_dim is None:
                raise ValueError("CNN checkpoint missing grid_shape or vector_dim.")
            self.qnet = make_q_network(
                self.n_actions,
                network_type="cnn",
                dueling=self.dueling and not self.lexicographic,
                grid_shape=self.grid_shape,
                vector_dim=self.vector_dim,
                lexicographic=self.lexicographic,
            ).to(self.device)
            self.target_qnet = make_q_network(
                self.n_actions,
                network_type="cnn",
                dueling=self.dueling and not self.lexicographic,
                grid_shape=self.grid_shape,
                vector_dim=self.vector_dim,
                lexicographic=self.lexicographic,
            ).to(self.device)
        else:
            if self.obs_dim is None:
                raise ValueError("MLP checkpoint missing obs_dim.")
            self.qnet = make_q_network(
                self.n_actions,
                network_type="mlp",
                obs_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                dueling=self.dueling,
            ).to(self.device)
            self.target_qnet = make_q_network(
                self.n_actions,
                network_type="mlp",
                obs_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                dueling=self.dueling,
            ).to(self.device)

        self.qnet.load_state_dict(payload["qnet_state_dict"])
        self.target_qnet.load_state_dict(payload["target_qnet_state_dict"])
        if "optimizer_state_dict" in payload:
            try:
                self.optimizer.load_state_dict(payload["optimizer_state_dict"])
            except Exception:
                pass

    def train_mode(self) -> None:
        self.qnet.train()

    def eval_mode(self) -> None:
        self.qnet.eval()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: Optional[torch.device] = None,
    ) -> "CatBreakDQNAgent":
        device = configure_torch(device or get_device())
        payload = torch.load(path, map_location=device, weights_only=False)
        network_type = payload.get("network_type", "mlp")
        agent = cls(
            n_actions=int(payload["n_actions"]),
            obs_dim=payload.get("obs_dim"),
            hidden_dim=int(payload.get("hidden_dim", 128)),
            gamma=float(payload.get("gamma", 0.99)),
            double_dqn=bool(payload.get("double_dqn", True)),
            dueling=bool(payload.get("dueling", True)),
            per=bool(payload.get("per", False)),
            device=device,
            obs_mode=payload.get("obs_mode", S.OBS_MODE_VECTOR),
            network_type=network_type,
            grid_shape=tuple(payload["grid_shape"]) if payload.get("grid_shape") else None,
            vector_dim=payload.get("vector_dim"),
            n_step=int(payload.get("n_step", 1)),
            lexicographic=bool(payload.get("lexicographic", False)),
            beta_time=float(payload.get("beta_time", 0.2)),
            brick_tolerance=float(payload.get("brick_tolerance", 0.05)),
        )
        agent.load(path, map_location=device)
        agent.eval_mode()
        return agent
