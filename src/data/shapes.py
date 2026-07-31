"""Rasterization utilities for the synthetic Shape--Color dataset."""

from __future__ import annotations

import math

import torch
from torch import Tensor

IMAGE_SIZE = 32

BACKGROUND_RGB = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
RED_RGB = torch.tensor([0.9, 0.15, 0.15], dtype=torch.float32)
BLUE_RGB = torch.tensor([0.15, 0.15, 0.9], dtype=torch.float32)
GRAY_RGB = torch.tensor([0.4, 0.4, 0.4], dtype=torch.float32)


def render_shape(
    shape: int,
    color: int,
    center_x: float,
    center_y: float,
    radius: float,
    image_size: int = IMAGE_SIZE,
    grayscale: bool = False,
) -> Tensor:
    """Render one RGB image as a float tensor in ``[0, 1]``.

    ``radius`` is the circle radius. The square side is ``sqrt(pi) * radius``,
    so the two shapes have the same continuous area for a given sampled size.
    Pixel centers are used for rasterization and no class-specific outline or
    texture is added.
    """
    if shape not in (0, 1):
        raise ValueError(f"shape must be 0 (circle) or 1 (square), got {shape}")
    if color not in (0, 1):
        raise ValueError(f"color must be 0 (red) or 1 (blue), got {color}")

    coordinates = torch.arange(image_size, dtype=torch.float32) + 0.5
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")

    if shape == 0:
        mask = (xx - center_x).square() + (yy - center_y).square() <= radius**2
    else:
        half_side = math.sqrt(math.pi) * radius / 2.0
        mask = (xx - center_x).abs().le(half_side) & (yy - center_y).abs().le(half_side)

    foreground = GRAY_RGB if grayscale else (RED_RGB if color == 0 else BLUE_RGB)
    image = BACKGROUND_RGB[:, None, None].expand(3, image_size, image_size).clone()
    image[:, mask] = foreground[:, None]
    return image
