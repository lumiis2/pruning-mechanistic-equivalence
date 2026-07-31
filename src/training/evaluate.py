"""Loss and behavioral metrics, including Shape--Color group metrics."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import ShapeColorDataset


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device | str,
    criterion: nn.Module | None = None,
) -> dict[str, Any]:
    """Evaluate a classifier and return only JSON-serializable values.

    Missing group/alignment accuracies are represented by ``None``. The worst
    group is computed over groups that are present, avoiding divisions by zero.
    """
    model.eval()
    device = torch.device(device)
    criterion = criterion or nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    group_correct = [0, 0, 0, 0]
    group_counts = [0, 0, 0, 0]
    aligned_correct = 0
    aligned_count = 0
    conflicting_correct = 0
    conflicting_count = 0

    for batch in data_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        groups = batch["group"].to(device)
        aligned = batch["aligned"].to(device=device, dtype=torch.bool)

        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        correct = predictions.eq(labels)
        batch_size = labels.numel()

        total_loss += float(loss.item()) * batch_size
        total_correct += int(correct.sum().item())
        total_count += batch_size

        aligned_correct += int(correct[aligned].sum().item())
        aligned_count += int(aligned.sum().item())
        conflicting_mask = ~aligned
        conflicting_correct += int(correct[conflicting_mask].sum().item())
        conflicting_count += int(conflicting_mask.sum().item())

        for group in range(4):
            mask = groups.eq(group)
            group_correct[group] += int(correct[mask].sum().item())
            group_counts[group] += int(mask.sum().item())

    def safe_accuracy(correct_count: int, count: int) -> float | None:
        return correct_count / count if count else None

    overall_accuracy = safe_accuracy(total_correct, total_count)
    aligned_accuracy = safe_accuracy(aligned_correct, aligned_count)
    conflicting_accuracy = safe_accuracy(conflicting_correct, conflicting_count)
    group_accuracies = [safe_accuracy(group_correct[g], group_counts[g]) for g in range(4)]
    present_group_accuracies = [value for value in group_accuracies if value is not None]
    shortcut_gap = (
        aligned_accuracy - conflicting_accuracy
        if aligned_accuracy is not None and conflicting_accuracy is not None
        else None
    )

    metrics: dict[str, Any] = {
        "loss": total_loss / total_count if total_count else None,
        "overall_accuracy": overall_accuracy,
        "aligned_accuracy": aligned_accuracy,
        "conflicting_accuracy": conflicting_accuracy,
        "worst_group_accuracy": min(present_group_accuracies) if present_group_accuracies else None,
        "shortcut_gap": shortcut_gap,
        "group_counts": {str(group): group_counts[group] for group in range(4)},
    }
    metrics.update({f"group_{group}_accuracy": group_accuracies[group] for group in range(4)})
    return metrics


@torch.no_grad()
def collect_predictions(
    model: nn.Module, data_loader: DataLoader, device: torch.device | str
) -> list[dict[str, Any]]:
    """Collect ordered per-sample predictions for an unshuffled data loader."""
    model.eval()
    device = torch.device(device)
    rows: list[dict[str, Any]] = []
    sample_index = 0
    for batch in data_loader:
        images = batch["image"].to(device)
        probabilities = torch.softmax(model(images), dim=1).cpu()
        predictions = probabilities.argmax(dim=1)
        batch_size = predictions.numel()
        for offset in range(batch_size):
            rows.append(
                {
                    "sample_index": sample_index,
                    "true_label": int(batch["label"][offset]),
                    "predicted_label": int(predictions[offset]),
                    "probability_class_0": float(probabilities[offset, 0]),
                    "probability_class_1": float(probabilities[offset, 1]),
                    "shape": int(batch["shape"][offset]),
                    "color": int(batch["color"][offset]),
                    "aligned": bool(batch["aligned"][offset]),
                    "group": int(batch["group"][offset]),
                }
            )
            sample_index += 1
    return rows


class _CounterfactualDataset(Dataset):
    def __init__(self, dataset: ShapeColorDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.dataset[index]
        return {
            "image": example["image"],
            "color_swapped_image": self.dataset.render_counterfactual(index, swap_color=True),
            "shape_swapped_image": self.dataset.render_counterfactual(index, swap_shape=True),
        }


@torch.no_grad()
def evaluate_counterfactual_sensitivity(
    model: nn.Module,
    dataset: ShapeColorDataset,
    device: torch.device | str,
    batch_size: int,
    num_workers: int,
) -> dict[str, float]:
    """Measure output sensitivity to isolated color and shape swaps.

    Sensitivity is mean absolute change in class-1 probability. Prediction flip
    rate is also returned as a complementary, threshold-dependent statistic.
    """
    model.eval()
    device = torch.device(device)
    loader = DataLoader(
        _CounterfactualDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    color_probability_change = 0.0
    shape_probability_change = 0.0
    color_prediction_flips = 0
    shape_prediction_flips = 0
    count = 0
    for batch in loader:
        original = batch["image"].to(device)
        color_swapped = batch["color_swapped_image"].to(device)
        shape_swapped = batch["shape_swapped_image"].to(device)
        original_probabilities = torch.softmax(model(original), dim=1)
        color_probabilities = torch.softmax(model(color_swapped), dim=1)
        shape_probabilities = torch.softmax(model(shape_swapped), dim=1)
        batch_count = original.shape[0]
        color_probability_change += float(
            (original_probabilities[:, 1] - color_probabilities[:, 1]).abs().sum().item()
        )
        shape_probability_change += float(
            (original_probabilities[:, 1] - shape_probabilities[:, 1]).abs().sum().item()
        )
        original_predictions = original_probabilities.argmax(dim=1)
        color_prediction_flips += int(
            original_predictions.ne(color_probabilities.argmax(dim=1)).sum().item()
        )
        shape_prediction_flips += int(
            original_predictions.ne(shape_probabilities.argmax(dim=1)).sum().item()
        )
        count += batch_count
    return {
        "color_swap_sensitivity": color_probability_change / count,
        "shape_swap_sensitivity": shape_probability_change / count,
        "color_swap_prediction_flip_rate": color_prediction_flips / count,
        "shape_swap_prediction_flip_rate": shape_prediction_flips / count,
    }
