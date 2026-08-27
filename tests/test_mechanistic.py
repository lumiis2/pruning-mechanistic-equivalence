import json

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.analysis.mechanistic import (
    CounterfactualDataset,
    balanced_indices,
    counterfactual_and_patching_metrics,
    stratified_splits,
    train_balanced_probe,
)
from src.data.dataset import ShapeColorDataset


def test_balanced_indices_select_equal_groups():
    dataset = ShapeColorDataset("test", 4, (10, 10, 10, 10), 32, 10, 16, "original")
    indices = balanced_indices(dataset, limit=20, seed=3)
    groups = dataset.groups[indices]
    assert len(indices) == 20
    assert [int(groups.eq(group).sum()) for group in range(4)] == [5, 5, 5, 5]


def test_stratified_splits_keep_every_group():
    groups = torch.arange(4).repeat_interleave(10)
    splits = stratified_splits(groups, seed=2)
    assert [len(split) for split in splits] == [24, 8, 8]
    for split in splits:
        assert set(groups[split].tolist()) == {0, 1, 2, 3}


def test_balanced_probe_learns_linearly_separable_target():
    groups = torch.arange(4).repeat_interleave(20)
    labels = groups // 2
    features = torch.stack((labels.float(), 1 - labels.float()), dim=1)
    result = train_balanced_probe(features, labels, groups, seed=1, epochs=20)
    assert result["test_accuracy"] == 1.0
    assert result["test_aligned_accuracy"] == 1.0
    assert result["test_conflicting_accuracy"] == 1.0
    assert result["test_worst_group_accuracy"] == 1.0
    assert result["test_shortcut_gap"] == 0.0


def test_balanced_probe_reports_shortcut_metrics():
    groups = torch.arange(4).repeat_interleave(20)
    labels = groups // 2
    color = groups % 2
    features = torch.stack((color.float(), 1 - color.float()), dim=1)
    result = train_balanced_probe(features, labels, groups, seed=1, epochs=20)
    assert result["test_accuracy"] == 0.5
    assert result["test_aligned_accuracy"] == 1.0
    assert result["test_conflicting_accuracy"] == 0.0
    assert result["test_worst_group_accuracy"] == 0.0
    assert result["test_shortcut_gap"] == 1.0


class ColorHead(nn.Module):
    """Tiny deterministic model exposing the mechanistic activation interface."""

    def forward(self, images, return_activations=False, activation_replacements=None):
        hidden = images.mean(dim=(2, 3))
        if activation_replacements and "hidden" in activation_replacements:
            hidden = activation_replacements["hidden"]
        logits = torch.stack((hidden[:, 0] - hidden[:, 2], hidden[:, 2] - hidden[:, 0]), dim=1)
        activations = {
            "conv1": images,
            "conv2": images,
            "hidden": hidden,
        }
        return (logits, activations) if return_activations else logits


def test_causal_metrics_are_signed_grouped_and_json_serializable():
    dataset = ShapeColorDataset("test", 4, (2, 2, 2, 2), 32, 10, 16, "original")
    loader = DataLoader(CounterfactualDataset(dataset), batch_size=8, shuffle=False)
    metrics = counterfactual_and_patching_metrics(
        ColorHead(), loader, torch.device("cpu"), channel_patching=False
    )
    color = metrics["input_color"]
    assert color["sample_count"] == 8
    assert color["target_probability_absolute_change"] > 0
    assert color["prediction_flip_rate"] == 1.0
    assert set(color["group_metrics"]) == {"0", "1", "2", "3"}
    assert all(group["sample_count"] == 2 for group in color["group_metrics"].values())
    assert color["group_metrics"]["0"]["target_probability_signed_change"] < 0
    assert color["group_metrics"]["1"]["target_probability_signed_change"] > 0
    assert color["group_metrics"]["2"]["target_logit_margin_signed_change"] > 0
    assert color["group_metrics"]["3"]["target_logit_margin_signed_change"] < 0
    json.dumps(metrics)
