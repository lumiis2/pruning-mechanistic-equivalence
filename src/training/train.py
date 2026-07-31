"""Training loop for the dense Shape--Color sanity-check model."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .evaluate import evaluate_model
from src.pruning import apply_masks_, prunable_parameters


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device | str,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    checkpoint_path: str | Path,
    checkpoint_epochs: list[int],
    sensitivity_evaluator: Callable[[nn.Module], dict[str, float]] | None = None,
    epoch_checkpoint_dir: str | Path | None = None,
    parameter_masks: dict[str, torch.Tensor] | None = None,
) -> list[dict[str, Any]]:
    """Train with Adam and early-stop on biased-validation loss.

    The checkpoint is overwritten only when validation loss improves, so it
    always contains the best validation-loss state observed during this run.
    """
    device = torch.device(device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    gradient_hooks = []
    if parameter_masks is not None:
        apply_masks_(model, parameter_masks)
        for name, parameter in prunable_parameters(model).items():
            mask = parameter_masks[name].to(device=parameter.device, dtype=parameter.dtype)
            gradient_hooks.append(parameter.register_hook(lambda gradient, mask=mask: gradient * mask))
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if epoch_checkpoint_dir is not None:
        epoch_checkpoint_dir = Path(epoch_checkpoint_dir)
        epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    initial_val_metrics = evaluate_model(model, val_loader, device, criterion)
    initial_test_metrics = evaluate_model(model, test_loader, device, criterion)
    initial_row: dict[str, Any] = {
            "epoch": 0,
            "train_loss": None,
            "train_accuracy": None,
            "val_biased_loss": initial_val_metrics["loss"],
            "val_biased_overall_accuracy": initial_val_metrics["overall_accuracy"],
            "val_biased_aligned_accuracy": initial_val_metrics["aligned_accuracy"],
            "val_biased_conflicting_accuracy": initial_val_metrics["conflicting_accuracy"],
            "val_biased_worst_group_accuracy": initial_val_metrics["worst_group_accuracy"],
            "val_biased_shortcut_gap": initial_val_metrics["shortcut_gap"],
            "test_balanced_loss": initial_test_metrics["loss"],
            "test_balanced_overall_accuracy": initial_test_metrics["overall_accuracy"],
            "test_balanced_aligned_accuracy": initial_test_metrics["aligned_accuracy"],
            "test_balanced_conflicting_accuracy": initial_test_metrics["conflicting_accuracy"],
            "test_balanced_worst_group_accuracy": initial_test_metrics["worst_group_accuracy"],
            "test_balanced_shortcut_gap": initial_test_metrics["shortcut_gap"],
        }
    if sensitivity_evaluator is not None:
        initial_row.update(sensitivity_evaluator(model))
    history.append(initial_row)
    if epoch_checkpoint_dir is not None:
        torch.save(
            {"epoch": 0, "model_state_dict": model.state_dict()},
            epoch_checkpoint_dir / "epoch_00.pt",
        )
    print(
        "epoch=00 before_update "
        f"test_acc={initial_test_metrics['overall_accuracy']:.4f} "
        f"aligned={initial_test_metrics['aligned_accuracy']:.4f} "
        f"conflicting={initial_test_metrics['conflicting_accuracy']:.4f} "
        f"gap={initial_test_metrics['shortcut_gap']:.4f}"
    )

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_count = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if parameter_masks is not None:
                apply_masks_(model, parameter_masks)

            batch_size = labels.numel()
            running_loss += float(loss.item()) * batch_size
            running_correct += int(logits.argmax(dim=1).eq(labels).sum().item())
            running_count += batch_size

        val_metrics = evaluate_model(model, val_loader, device, criterion)
        test_metrics = evaluate_model(model, test_loader, device, criterion)
        epoch_metrics: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": running_loss / running_count,
            "train_accuracy": running_correct / running_count,
            "val_biased_loss": val_metrics["loss"],
            "val_biased_overall_accuracy": val_metrics["overall_accuracy"],
            "val_biased_aligned_accuracy": val_metrics["aligned_accuracy"],
            "val_biased_conflicting_accuracy": val_metrics["conflicting_accuracy"],
            "val_biased_worst_group_accuracy": val_metrics["worst_group_accuracy"],
            "val_biased_shortcut_gap": val_metrics["shortcut_gap"],
            "test_balanced_loss": test_metrics["loss"],
            "test_balanced_overall_accuracy": test_metrics["overall_accuracy"],
            "test_balanced_aligned_accuracy": test_metrics["aligned_accuracy"],
            "test_balanced_conflicting_accuracy": test_metrics["conflicting_accuracy"],
            "test_balanced_worst_group_accuracy": test_metrics["worst_group_accuracy"],
            "test_balanced_shortcut_gap": test_metrics["shortcut_gap"],
        }
        if sensitivity_evaluator is not None:
            epoch_metrics.update(sensitivity_evaluator(model))
        history.append(epoch_metrics)
        print(
            f"epoch={epoch:02d} train_loss={epoch_metrics['train_loss']:.4f} "
            f"train_acc={epoch_metrics['train_accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['overall_accuracy']:.4f} "
            f"aligned={val_metrics['aligned_accuracy']:.4f} "
            f"conflicting={val_metrics['conflicting_accuracy']:.4f} "
            f"gap={val_metrics['shortcut_gap']:.4f}"
            f" test_acc={test_metrics['overall_accuracy']:.4f} "
            f"test_gap={test_metrics['shortcut_gap']:.4f}"
        )

        if epoch in checkpoint_epochs:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict()},
                checkpoint_path.parent / f"epoch_{epoch:02d}.pt",
            )
        if epoch_checkpoint_dir is not None:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict()},
                epoch_checkpoint_dir / f"epoch_{epoch:02d}.pt",
            )

        val_loss = float(val_metrics["loss"])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            last_required_epoch = max(checkpoint_epochs, default=0)
            if epochs_without_improvement >= patience and epoch >= last_required_epoch:
                print(f"early_stopping epoch={epoch} patience={patience}")
                break

    for hook in gradient_hooks:
        hook.remove()
    return history
