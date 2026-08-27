"""Balanced probes and causal Shape--Color intervention measurements."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import ShapeColorDataset

LAYERS = ("conv1", "conv2", "hidden")


class CounterfactualDataset(Dataset):
    """Original examples paired with factor-isolated counterfactuals."""

    def __init__(self, dataset: ShapeColorDataset, indices: list[int] | None = None) -> None:
        self.dataset = dataset
        self.indices = indices if indices is not None else list(range(len(dataset)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, offset: int) -> dict[str, Tensor]:
        index = self.indices[offset]
        example = self.dataset[index]
        return {
            "image": example["image"],
            "color_image": self.dataset.render_counterfactual(index, swap_color=True),
            "shape_image": self.dataset.render_counterfactual(index, swap_shape=True),
            "shape": torch.tensor(example["shape"], dtype=torch.long),
            "color": torch.tensor(example["color"], dtype=torch.long),
            "group": torch.tensor(example["group"], dtype=torch.long),
        }


def balanced_indices(dataset: ShapeColorDataset, limit: int | None = None, seed: int = 0) -> list[int]:
    """Select an equal number from each of the four groups deterministically."""
    generator = torch.Generator().manual_seed(seed)
    per_group = min(int(dataset.groups.eq(group).sum()) for group in range(4))
    if limit is not None:
        per_group = min(per_group, limit // 4)
    selected: list[int] = []
    for group in range(4):
        candidates = torch.where(dataset.groups.eq(group))[0]
        order = torch.randperm(len(candidates), generator=generator)
        selected.extend(int(index) for index in candidates[order[:per_group]])
    return selected


def compress_activation(activation: Tensor, spatial_grid: int) -> Tensor:
    if activation.ndim == 4:
        activation = F.adaptive_avg_pool2d(activation, (spatial_grid, spatial_grid))
    return activation.flatten(start_dim=1).cpu()


@torch.no_grad()
def collect_activations(
    model: nn.Module, loader: DataLoader, device: torch.device, spatial_grid: int
) -> dict[str, Tensor]:
    model.eval()
    values: dict[str, list[Tensor]] = {layer: [] for layer in LAYERS}
    labels: dict[str, list[Tensor]] = {"shape": [], "color": [], "group": []}
    for batch in loader:
        _, activations = model(batch["image"].to(device), return_activations=True)
        for layer in LAYERS:
            values[layer].append(compress_activation(activations[layer], spatial_grid))
        for label in labels:
            labels[label].append(batch[label].cpu())
    return {**{key: torch.cat(parts) for key, parts in values.items()},
            **{key: torch.cat(parts) for key, parts in labels.items()}}


def stratified_splits(groups: Tensor, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    splits = [[], [], []]
    for group in range(4):
        indices = torch.where(groups.eq(group))[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        train_end, val_end = int(0.6 * len(indices)), int(0.8 * len(indices))
        for target, part in zip(splits, (indices[:train_end], indices[train_end:val_end], indices[val_end:])):
            target.append(part)
    return tuple(torch.cat(parts) for parts in splits)  # type: ignore[return-value]


def train_balanced_probe(
    features: Tensor, labels: Tensor, groups: Tensor, seed: int,
    epochs: int, learning_rate: float = 0.05, weight_decay: float = 1e-3,
) -> dict[str, Any]:
    """Train a deterministic linear probe and return held-out balanced accuracy."""
    train_idx, val_idx, test_idx = stratified_splits(groups, seed)
    mean = features[train_idx].mean(0, keepdim=True)
    scale = features[train_idx].std(0, keepdim=True).clamp_min(1e-6)
    normalized = (features - mean) / scale
    torch.manual_seed(seed)
    probe = nn.Linear(features.shape[1], 2)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state, best_val = None, -1.0
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(normalized[train_idx]), labels[train_idx])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val = float(probe(normalized[val_idx]).argmax(1).eq(labels[val_idx]).float().mean())
        if val > best_val:
            best_val = val
            best_state = {key: value.detach().clone() for key, value in probe.state_dict().items()}
    probe.load_state_dict(best_state)
    with torch.no_grad():
        predictions = probe(normalized[test_idx]).argmax(1)
    group_accuracy = {}
    for group in range(4):
        mask = groups[test_idx].eq(group)
        group_accuracy[str(group)] = float(predictions[mask].eq(labels[test_idx][mask]).float().mean())
    aligned_accuracy = (group_accuracy["0"] + group_accuracy["3"]) / 2
    conflicting_accuracy = (group_accuracy["1"] + group_accuracy["2"]) / 2
    return {
        "validation_accuracy": best_val,
        "test_accuracy": float(predictions.eq(labels[test_idx]).float().mean()),
        "test_aligned_accuracy": aligned_accuracy,
        "test_conflicting_accuracy": conflicting_accuracy,
        "test_worst_group_accuracy": min(group_accuracy.values()),
        "test_shortcut_gap": aligned_accuracy - conflicting_accuracy,
        "test_group_accuracy": group_accuracy,
        "train_count": len(train_idx), "validation_count": len(val_idx), "test_count": len(test_idx),
    }


def _effects(reference: Tensor, intervention: Tensor, labels: Tensor) -> dict[str, float]:
    ref_prob = torch.softmax(reference, 1).gather(1, labels[:, None]).squeeze(1)
    int_prob = torch.softmax(intervention, 1).gather(1, labels[:, None]).squeeze(1)
    other = 1 - labels
    ref_margin = reference.gather(1, labels[:, None]).squeeze(1) - reference.gather(1, other[:, None]).squeeze(1)
    int_margin = intervention.gather(1, labels[:, None]).squeeze(1) - intervention.gather(1, other[:, None]).squeeze(1)
    probability_change = int_prob - ref_prob
    margin_change = int_margin - ref_margin
    return {
        "probability_signed_change_sum": float(probability_change.sum()),
        "probability_absolute_change_sum": float(probability_change.abs().sum()),
        "logit_margin_signed_change_sum": float(margin_change.sum()),
        "logit_margin_absolute_change_sum": float(margin_change.abs().sum()),
        "prediction_flip_count": int(intervention.argmax(1).ne(reference.argmax(1)).sum()),
        "count": len(labels),
    }


def _accumulate_effects(
    totals: dict[str, dict[str, dict[str, float]]],
    key: str,
    reference: Tensor,
    intervention: Tensor,
    labels: Tensor,
    groups: Tensor,
) -> None:
    for scope, mask in [("overall", torch.ones_like(groups, dtype=torch.bool))] + [
        (str(group), groups.eq(group)) for group in range(4)
    ]:
        if not bool(mask.any()):
            continue
        for metric, value in _effects(reference[mask], intervention[mask], labels[mask]).items():
            totals[key][scope][metric] += value


def _finalize_effects(values: dict[str, float]) -> dict[str, float | int]:
    count = values["count"]
    return {
        "target_probability_signed_change": values["probability_signed_change_sum"] / count,
        "target_probability_absolute_change": values["probability_absolute_change_sum"] / count,
        "target_logit_margin_signed_change": values["logit_margin_signed_change_sum"] / count,
        "target_logit_margin_absolute_change": values["logit_margin_absolute_change_sum"] / count,
        "prediction_flip_rate": values["prediction_flip_count"] / count,
        "sample_count": int(count),
    }


@torch.no_grad()
def counterfactual_and_patching_metrics(
    model: nn.Module, loader: DataLoader, device: torch.device,
    channel_patching: bool = True,
) -> dict[str, dict[str, Any]]:
    """Measure input, full-layer control, and individual-channel patch effects."""
    model.eval()
    totals: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    for batch in loader:
        labels = batch["shape"].to(device)
        groups = batch["group"].to(device)
        original_logits, original_acts = model(batch["image"].to(device), return_activations=True)
        for factor in ("color", "shape"):
            swapped_logits, swapped_acts = model(batch[f"{factor}_image"].to(device), return_activations=True)
            key = f"input_{factor}"
            _accumulate_effects(totals, key, original_logits, swapped_logits, labels, groups)
            for layer in LAYERS:
                patched = model(batch["image"].to(device), activation_replacements={layer: swapped_acts[layer]})
                key = f"full_patch_{factor}_{layer}"
                _accumulate_effects(totals, key, original_logits, patched, labels, groups)
                if channel_patching:
                    channel_count = original_acts[layer].shape[1]
                    for channel in range(channel_count):
                        replacement = original_acts[layer].clone()
                        replacement[:, channel] = swapped_acts[layer][:, channel]
                        patched = model(
                            batch["image"].to(device),
                            activation_replacements={layer: replacement},
                        )
                        key = f"channel_patch_{factor}_{layer}_{channel:02d}"
                        _accumulate_effects(totals, key, original_logits, patched, labels, groups)
    results = {}
    for key, scoped_values in totals.items():
        results[key] = _finalize_effects(scoped_values["overall"])
        results[key]["group_metrics"] = {
            str(group): _finalize_effects(scoped_values[str(group)]) for group in range(4)
        }
    return results
