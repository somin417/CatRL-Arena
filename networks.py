"""Q-networks for CatBreak DQN (MLP + CNN hybrid)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class MLPQNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DuelingQNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feat = self.feature(obs)
        value = self.value_head(feat)
        advantage = self.advantage_head(feat)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))


class _CNNFeatureFusion(nn.Module):
    """Shared CNN + vector trunk for hybrid observations."""

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        vector_dim: int,
        fused_dim: int = 256,
    ) -> None:
        super().__init__()
        channels, _, _ = grid_shape
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, grid_shape[1], grid_shape[2])
            cnn_flat = int(torch.numel(self.cnn(dummy)) // dummy.shape[0])

        self.vector_branch = nn.Sequential(
            nn.Linear(vector_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(cnn_flat + 64, fused_dim),
            nn.ReLU(),
        )
        self.fused_dim = fused_dim

    def forward(self, grid: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn(grid).view(grid.size(0), -1)
        vec_feat = self.vector_branch(vector)
        return self.fusion(torch.cat([cnn_feat, vec_feat], dim=-1))


class CatBreakCNNQNet(nn.Module):
    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        vector_dim: int,
        n_actions: int,
        dueling: bool = True,
        fused_dim: int = 256,
    ) -> None:
        super().__init__()
        self.dueling = bool(dueling)
        self.trunk = _CNNFeatureFusion(grid_shape, vector_dim, fused_dim=fused_dim)
        if self.dueling:
            self.value_head = nn.Linear(fused_dim, 1)
            self.advantage_head = nn.Linear(fused_dim, n_actions)
        else:
            self.q_head = nn.Linear(fused_dim, n_actions)

    def forward(self, grid: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        feat = self.trunk(grid, vector)
        if self.dueling:
            value = self.value_head(feat)
            advantage = self.advantage_head(feat)
            return value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return self.q_head(feat)


class LexicographicCatBreakCNNQNet(nn.Module):
    """Two-headed CNN Q-network for lexicographic action selection."""

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        vector_dim: int,
        n_actions: int,
        fused_dim: int = 256,
    ) -> None:
        super().__init__()
        self.trunk = _CNNFeatureFusion(grid_shape, vector_dim, fused_dim=fused_dim)
        self.brick_value = nn.Linear(fused_dim, 1)
        self.brick_advantage = nn.Linear(fused_dim, n_actions)
        self.time_value = nn.Linear(fused_dim, 1)
        self.time_advantage = nn.Linear(fused_dim, n_actions)

    def _dueling(self, value: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))

    def forward(
        self, grid: torch.Tensor, vector: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.trunk(grid, vector)
        q_brick = self._dueling(self.brick_value(feat), self.brick_advantage(feat))
        q_time = self._dueling(self.time_value(feat), self.time_advantage(feat))
        return q_brick, q_time


def make_q_network(
    n_actions: int,
    network_type: str = "mlp",
    obs_dim: Optional[int] = None,
    hidden_dim: int = 128,
    dueling: bool = False,
    grid_shape: Optional[Tuple[int, int, int]] = None,
    vector_dim: Optional[int] = None,
    lexicographic: bool = False,
) -> nn.Module:
    if network_type == "cnn":
        if grid_shape is None or vector_dim is None:
            raise ValueError("CNN network requires grid_shape and vector_dim.")
        if lexicographic:
            return LexicographicCatBreakCNNQNet(grid_shape, vector_dim, n_actions)
        return CatBreakCNNQNet(
            grid_shape, vector_dim, n_actions, dueling=dueling
        )
    if obs_dim is None:
        raise ValueError("MLP network requires obs_dim.")
    if dueling:
        return DuelingQNet(obs_dim, n_actions, hidden_dim)
    return MLPQNet(obs_dim, n_actions, hidden_dim)
