import torch

from src.models import SmallCNN
from src.pruning import (
    apply_masks_,
    global_magnitude_masks,
    global_random_masks,
    iterative_global_magnitude_prune,
    layer_mask_statistics,
    mask_statistics,
)


def test_global_masks_have_exact_requested_sparsity():
    model = SmallCNN(hidden_dim=32)
    for masks in (global_magnitude_masks(model, 0.5), global_random_masks(model, 0.5, seed=7)):
        statistics = mask_statistics(masks)
        assert statistics["kept_weights"] * 2 == statistics["total_weights"]
        assert statistics["sparsity"] == 0.5


def test_applying_masks_zeros_exactly_the_pruned_weights():
    model = SmallCNN(hidden_dim=32)
    masks = global_random_masks(model, 0.5, seed=7)
    apply_masks_(model, masks)
    named_parameters = dict(model.named_parameters())
    for name, mask in masks.items():
        assert torch.count_nonzero(named_parameters[name][~mask]) == 0


def test_random_masks_are_seeded():
    model = SmallCNN(hidden_dim=32)
    first = global_random_masks(model, 0.5, seed=9)
    second = global_random_masks(model, 0.5, seed=9)
    assert all(torch.equal(first[name], second[name]) for name in first)


def test_iterative_pruning_removes_fraction_of_remaining_without_regrowth():
    model = SmallCNN(hidden_dim=32)
    initial = {
        name: torch.ones_like(parameter, dtype=torch.bool)
        for name, parameter in dict(model.named_parameters()).items()
        if name.endswith("weight")
    }
    first = iterative_global_magnitude_prune(model, initial, 0.2)
    second = iterative_global_magnitude_prune(model, first, 0.2)

    initial_stats = mask_statistics(initial)
    first_stats = mask_statistics(first)
    second_stats = mask_statistics(second)
    assert first_stats["kept_weights"] == initial_stats["kept_weights"] - round(
        0.2 * initial_stats["kept_weights"]
    )
    assert second_stats["kept_weights"] == first_stats["kept_weights"] - round(
        0.2 * first_stats["kept_weights"]
    )
    assert all(torch.all(second[name] <= first[name]) for name in first)
    assert set(layer_mask_statistics(second)) == set(second)
