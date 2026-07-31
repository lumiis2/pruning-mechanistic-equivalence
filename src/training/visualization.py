"""Training-history visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def save_training_curves(history: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Plot losses, accuracies, alignment accuracies, and shortcut gap."""
    if not history:
        raise ValueError("cannot plot an empty training history")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes[0, 0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0, 0].plot(epochs, [row["val_biased_loss"] for row in history], label="validation")
    axes[0, 0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")

    axes[0, 1].plot(epochs, [row["train_accuracy"] for row in history], label="train")
    axes[0, 1].plot(
        epochs,
        [row["val_biased_overall_accuracy"] for row in history],
        label="validation overall",
    )
    axes[0, 1].set(title="Overall accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(-0.02, 1.02))

    axes[0, 2].plot(
        epochs, [row["val_biased_aligned_accuracy"] for row in history], label="aligned"
    )
    axes[0, 2].plot(
        epochs,
        [row["val_biased_conflicting_accuracy"] for row in history],
        label="conflicting",
    )
    axes[0, 2].set(title="Validation by alignment", xlabel="Epoch", ylabel="Accuracy", ylim=(-0.02, 1.02))

    axes[1, 0].plot(
        epochs, [row["val_biased_shortcut_gap"] for row in history], label="shortcut gap"
    )
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set(title="Validation shortcut gap", xlabel="Epoch", ylabel="Aligned − conflicting")

    axes[1, 1].plot(
        epochs, [row["test_balanced_aligned_accuracy"] for row in history], label="aligned"
    )
    axes[1, 1].plot(
        epochs,
        [row["test_balanced_conflicting_accuracy"] for row in history],
        label="conflicting",
    )
    axes[1, 1].set(title="Balanced test by alignment", xlabel="Epoch", ylabel="Accuracy", ylim=(-0.02, 1.02))

    axes[1, 2].plot(
        epochs, [row["test_balanced_shortcut_gap"] for row in history], label="shortcut gap"
    )
    axes[1, 2].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 2].set(title="Balanced-test shortcut gap", xlabel="Epoch", ylabel="Aligned − conflicting")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path
