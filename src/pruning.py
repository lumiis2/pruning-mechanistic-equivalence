"""Minimal mask utilities for the one-shot magnitude-pruning experiment."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def prunable_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    """Return Conv2d/Linear weights; biases are intentionally excluded."""
    result: dict[str, nn.Parameter] = {}
    for module_name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            name = f"{module_name}.weight" if module_name else "weight"
            result[name] = module.weight
    return result


def _masks_from_scores(scores: dict[str, Tensor], sparsity: float) -> dict[str, Tensor]:
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    names = list(scores)
    flattened = torch.cat([scores[name].detach().flatten().cpu() for name in names])
    keep_count = round((1.0 - sparsity) * flattened.numel())
    kept_indices = torch.topk(flattened, keep_count, largest=True, sorted=False).indices
    flat_mask = torch.zeros_like(flattened, dtype=torch.bool)
    flat_mask[kept_indices] = True
    masks: dict[str, Tensor] = {}
    start = 0
    for name in names:
        count = scores[name].numel()
        masks[name] = flat_mask[start : start + count].reshape(scores[name].shape)
        start += count
    return masks


def global_magnitude_masks(model: nn.Module, sparsity: float) -> dict[str, Tensor]:
    """Keep the globally largest absolute Conv/Linear weights."""
    return _masks_from_scores(
        {name: parameter.detach().abs() for name, parameter in prunable_parameters(model).items()},
        sparsity,
    )


def global_random_masks(model: nn.Module, sparsity: float, seed: int) -> dict[str, Tensor]:
    """Keep the same global number of weights using seeded random scores."""
    generator = torch.Generator().manual_seed(seed)
    return _masks_from_scores(
        {
            name: torch.rand(parameter.shape, generator=generator)
            for name, parameter in prunable_parameters(model).items()
        },
        sparsity,
    )


def iterative_global_magnitude_prune(
    model: nn.Module,
    current_masks: dict[str, Tensor],
    pruning_fraction: float,
) -> dict[str, Tensor]:
    """Prune a fraction of the weights that remain active, globally.

    Previously pruned weights can never return. Ties are resolved by
    ``torch.topk`` and the resulting masks live on CPU for portable storage.
    """
    if not 0.0 < pruning_fraction < 1.0:
        raise ValueError("pruning_fraction must be in (0, 1)")
    parameters = prunable_parameters(model)
    if set(parameters) != set(current_masks):
        raise ValueError("mask names do not match prunable model parameters")

    active_scores: list[Tensor] = []
    locations: list[tuple[str, Tensor]] = []
    for name, parameter in parameters.items():
        mask = current_masks[name].detach().bool().cpu()
        flat_active = mask.flatten().nonzero(as_tuple=False).flatten()
        active_scores.append(parameter.detach().abs().cpu().flatten()[flat_active])
        locations.append((name, flat_active))

    scores = torch.cat(active_scores)
    active_count = scores.numel()
    prune_count = round(pruning_fraction * active_count)
    keep_count = active_count - prune_count
    if keep_count < 1:
        raise ValueError("pruning_fraction would remove all remaining weights")
    kept_active_indices = torch.topk(
        scores, keep_count, largest=True, sorted=False
    ).indices
    flat_kept = torch.zeros(active_count, dtype=torch.bool)
    flat_kept[kept_active_indices] = True

    new_masks: dict[str, Tensor] = {}
    offset = 0
    for name, flat_active in locations:
        old_mask = current_masks[name].detach().bool().cpu()
        new_flat = torch.zeros(old_mask.numel(), dtype=torch.bool)
        count = flat_active.numel()
        new_flat[flat_active[flat_kept[offset : offset + count]]] = True
        new_masks[name] = new_flat.reshape(old_mask.shape)
        offset += count
    return new_masks


def apply_masks_(model: nn.Module, masks: dict[str, Tensor]) -> None:
    parameters = prunable_parameters(model)
    if set(parameters) != set(masks):
        raise ValueError("mask names do not match prunable model parameters")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.mul_(masks[name].to(device=parameter.device, dtype=parameter.dtype))


def mask_statistics(masks: dict[str, Tensor]) -> dict[str, int | float]:
    total = sum(mask.numel() for mask in masks.values())
    kept = sum(int(mask.sum().item()) for mask in masks.values())
    return {"total_weights": total, "kept_weights": kept, "pruned_weights": total - kept, "sparsity": 1 - kept / total}


def layer_mask_statistics(
    masks: dict[str, Tensor],
) -> dict[str, dict[str, int | float]]:
    """Return JSON-serializable mask statistics for every prunable layer."""
    return {name: mask_statistics({name: mask}) for name, mask in masks.items()}
