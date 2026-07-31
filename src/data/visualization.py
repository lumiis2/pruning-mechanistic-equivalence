"""Visualization helpers for Shape--Color data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .dataset import ShapeColorDataset

SHAPE_NAMES = {0: "circle", 1: "square"}
COLOR_NAMES = {0: "red", 1: "blue"}


def save_four_groups_figure(
    dataset: ShapeColorDataset,
    output_path: str | Path = "outputs/sanity_check/four_groups.png",
) -> Path:
    """Save one example from each group in canonical group order."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = []
    for group in range(4):
        index = int((dataset.groups == group).nonzero(as_tuple=False)[0])
        examples.append(dataset[index])

    figure, axes = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True)
    for axis, example in zip(axes, examples):
        axis.imshow(example["image"].permute(1, 2, 0).numpy())
        alignment = "aligned" if example["aligned"] else "conflicting"
        axis.set_title(
            f'{SHAPE_NAMES[example["shape"]]} / {COLOR_NAMES[example["color"]]}\n'
            f'class {example["label"]} | {alignment} | group {example["group"]}',
            fontsize=9,
        )
        axis.axis("off")

    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path
