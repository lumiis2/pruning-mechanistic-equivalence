import torch

from src.analysis.mechanistic import balanced_indices, stratified_splits, train_balanced_probe
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
    assert result["test_worst_group_accuracy"] == 1.0
