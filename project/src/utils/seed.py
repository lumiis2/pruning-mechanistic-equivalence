"""Reproducibility helpers."""

from __future__ import annotations

import random

import torch


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch and request deterministic kernels."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
