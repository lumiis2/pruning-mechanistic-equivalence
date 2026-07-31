"""Deterministic synthetic Shape--Color datasets and canonical splits."""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .shapes import render_shape


class ShapeColorExample(TypedDict):
    image: Tensor
    label: int
    shape: int
    color: int
    aligned: bool
    group: int


class ShapeColorDataset(Dataset[ShapeColorExample]):
    """A fully deterministic Shape--Color split.

    Conventions:
        shape 0 = circle; shape 1 = square.
        color 0 = red; color 1 = blue.
        label = shape; aligned = (color == label); group = 2 * label + color.

        group 0 = red circle, class 0, aligned.
        group 1 = blue circle, class 0, conflicting.
        group 2 = red square, class 1, conflicting.
        group 3 = blue square, class 1, aligned.

    Size and position are sampled independently of shape, color, and label.
    The sampled circle radius also defines an equal-continuous-area square.
    """

    def __init__(
        self,
        split: str,
        seed: int,
        group_counts: tuple[int, int, int, int],
        image_size: int,
        min_shape_size: float,
        max_shape_size: float,
        variant: str,
    ) -> None:
        if len(group_counts) != 4 or any(count < 0 for count in group_counts):
            raise ValueError("group_counts must contain four non-negative counts")
        if not 0 < min_shape_size <= max_shape_size < image_size:
            raise ValueError("invalid shape-size range")
        if variant not in {"original", "color_only", "shape_only"}:
            raise ValueError(f"unknown dataset variant: {variant}")

        self.split = split
        self.seed = int(seed)
        self.image_size = image_size
        self.variant = variant
        size = sum(group_counts)

        groups = torch.cat(
            [torch.full((count,), group, dtype=torch.int64) for group, count in enumerate(group_counts)]
        )
        generator = torch.Generator().manual_seed(self.seed)
        groups = groups[torch.randperm(size, generator=generator)]

        # All nuisance variables are drawn without consulting group membership.
        shape_sizes = torch.empty(size).uniform_(min_shape_size, max_shape_size, generator=generator)
        radii = shape_sizes / 2.0
        margins = radii + 0.5
        centers_x = margins + torch.rand(size, generator=generator) * (image_size - 2 * margins)
        centers_y = margins + torch.rand(size, generator=generator) * (image_size - 2 * margins)

        self.groups = groups
        self.radii = radii
        self.centers_x = centers_x
        self.centers_y = centers_y

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> ShapeColorExample:
        group = int(self.groups[index])
        shape = group // 2
        color = group % 2
        label = shape
        aligned = color == label
        image = render_shape(
            shape=1 if self.variant == "color_only" else shape,
            color=color,
            center_x=float(self.centers_x[index]),
            center_y=float(self.centers_y[index]),
            radius=float(self.radii[index]),
            image_size=self.image_size,
            grayscale=self.variant == "shape_only",
        )
        return {
            "image": image,
            "label": label,
            "shape": shape,
            "color": color,
            "aligned": aligned,
            "group": group,
        }

    def render_counterfactual(
        self, index: int, *, swap_color: bool = False, swap_shape: bool = False
    ) -> Tensor:
        """Render an intervention while holding all nuisance variables fixed."""
        if swap_color == swap_shape:
            raise ValueError("swap exactly one of color or shape")
        group = int(self.groups[index])
        shape = group // 2
        color = group % 2
        if swap_shape:
            shape = 1 - shape
        if swap_color:
            color = 1 - color
        return render_shape(
            shape=1 if self.variant == "color_only" else shape,
            color=color,
            center_x=float(self.centers_x[index]),
            center_y=float(self.centers_y[index]),
            radius=float(self.radii[index]),
            image_size=self.image_size,
            grayscale=self.variant == "shape_only",
        )


def _biased_group_counts(size: int, alignment_probability: float) -> tuple[int, int, int, int]:
    if size % 2 or not 0.0 <= alignment_probability <= 1.0:
        raise ValueError("biased split size must be even and probability must be in [0, 1]")
    per_class = size // 2
    aligned_per_class = round(per_class * alignment_probability)
    conflicting_per_class = per_class - aligned_per_class
    return aligned_per_class, conflicting_per_class, conflicting_per_class, aligned_per_class


def create_splits(
    seed: int, data_config: dict[str, int | float], variant: str = "original"
) -> dict[str, ShapeColorDataset]:
    """Create configured biased train/validation and exactly balanced test splits."""
    test_size = int(data_config["test_size"])
    if test_size % 4:
        raise ValueError("test_size must be divisible by four")
    group_counts = {
        "train_biased": _biased_group_counts(
            int(data_config["train_size"]), float(data_config["train_alignment_probability"])
        ),
        "val_biased": _biased_group_counts(
            int(data_config["val_size"]), float(data_config["val_alignment_probability"])
        ),
        "test_balanced": (test_size // 4,) * 4,
    }
    return {
        name: ShapeColorDataset(
            split=name,
            seed=seed + offset,
            group_counts=group_counts[name],
            image_size=int(data_config["image_size"]),
            min_shape_size=float(data_config["min_shape_size"]),
            max_shape_size=float(data_config["max_shape_size"]),
            variant=variant,
        )
        for offset, name in enumerate(group_counts)
    }
