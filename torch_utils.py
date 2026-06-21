"""Shared PyTorch device helpers for CatBreak RL."""

from __future__ import annotations

import os

import torch

_CONFIGURED = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return CUDA device when available and requested, otherwise CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_torch(device: torch.device | None = None) -> torch.device:
    """Apply one-time CUDA performance settings and return the active device."""
    global _CONFIGURED
    dev = device or get_device()
    if not _CONFIGURED:
        if dev.type == "cuda":
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        _CONFIGURED = True
    return dev


def default_batch_size(network_type: str, base_batch_size: int, device: torch.device) -> int:
    """Scale replay batch size on GPU for better utilization."""
    if device.type != "cuda":
        return base_batch_size
    if network_type == "cnn":
        return max(base_batch_size, 256)
    return max(base_batch_size, 512)


def default_parallel_workers(requested: int | None = None, reserve_for_gpu: bool = True) -> int:
    """Pick rollout worker count for CPU-bound CEM search on GPU workstations."""
    cpus = os.cpu_count() or 4
    if requested is not None and requested > 0:
        return min(requested, cpus)
    if requested == 0:
        return 1
    if reserve_for_gpu and torch.cuda.is_available():
        return max(1, cpus - 2)
    return max(1, cpus // 2)
