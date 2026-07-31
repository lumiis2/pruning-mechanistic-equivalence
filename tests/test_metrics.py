import json

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.training import evaluate_model, select_functional_epochs


class FourGroupDataset(Dataset):
    def __init__(self):
        self.examples = []
        # Predictions are correct for aligned groups 0 and 3, and incorrect
        # for conflicting groups 1 and 2.
        predictions = [0, 1, 0, 1]
        for group, prediction in enumerate(predictions):
            label = group // 2
            logits = torch.full((2,), -4.0)
            logits[prediction] = 4.0
            self.examples.append(
                {
                    "image": logits,
                    "label": label,
                    "shape": label,
                    "color": group % 2,
                    "aligned": group in (0, 3),
                    "group": group,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def test_group_metrics_and_json_serialization():
    metrics = evaluate_model(nn.Identity(), DataLoader(FourGroupDataset(), batch_size=4), "cpu")
    assert metrics["overall_accuracy"] == 0.5
    assert metrics["aligned_accuracy"] == 1.0
    assert metrics["conflicting_accuracy"] == 0.0
    assert metrics["worst_group_accuracy"] == 0.0
    assert metrics["shortcut_gap"] == 1.0
    assert metrics["group_counts"] == {"0": 1, "1": 1, "2": 1, "3": 1}
    assert metrics["group_0_accuracy"] == 1.0
    assert metrics["group_1_accuracy"] == 0.0
    assert metrics["group_2_accuracy"] == 0.0
    assert metrics["group_3_accuracy"] == 1.0
    json.dumps(metrics)


def test_missing_groups_do_not_divide_by_zero():
    dataset = FourGroupDataset()
    dataset.examples = dataset.examples[:1]
    metrics = evaluate_model(nn.Identity(), DataLoader(dataset, batch_size=1), "cpu")
    assert metrics["conflicting_accuracy"] is None
    assert metrics["shortcut_gap"] is None
    assert metrics["group_1_accuracy"] is None


def test_functional_epochs_are_selected_from_behavior_not_fixed_time():
    gaps = [0.0, 0.95, 0.96, 0.72, 0.48, 0.08]
    conflicting = [0.5, 0.0, 0.0, 0.28, 0.52, 0.93]
    history = [
        {
            "epoch": epoch,
            "test_balanced_shortcut_gap": gap,
            "test_balanced_conflicting_accuracy": conflict,
        }
        for epoch, (gap, conflict) in enumerate(zip(gaps, conflicting))
    ]
    selected = select_functional_epochs(history, 0.9, 0.5, 0.9)
    assert selected == {"shortcut": 1, "late_shortcut": 2, "transition": 4, "robust": 5}
