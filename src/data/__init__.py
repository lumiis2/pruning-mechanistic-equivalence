"""Synthetic data components for the Shape--Color experiment."""

from .dataset import ShapeColorDataset, create_splits
from .visualization import save_four_groups_figure

__all__ = [
    "ShapeColorDataset",
    "create_splits",
    "save_four_groups_figure",
]
