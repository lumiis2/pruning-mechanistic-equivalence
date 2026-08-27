import csv

import pytest

from src.analysis.model_registry import find_matched_pairs, trajectory_summary


def test_trajectory_summary_supports_training_history_schema(tmp_path):
    path = tmp_path / "history.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "test_balanced_shortcut_gap"])
        writer.writeheader()
        writer.writerows([
            {"epoch": 0, "test_balanced_shortcut_gap": 0.0},
            {"epoch": 1, "test_balanced_shortcut_gap": 1.0},
            {"epoch": 2, "test_balanced_shortcut_gap": 0.9},
            {"epoch": 3, "test_balanced_shortcut_gap": 0.1},
        ])
    assert trajectory_summary(path, 0.8, 0.2) == (1, 3, 2)


def test_matched_pairs_use_all_tolerances_and_rank_closest_first():
    base = {"model_family": "dense", "seed": 1, "sparsity": 0.0,
            "mask_selection_checkpoint": "none", "rewinding_checkpoint": "none", "imp_round": None}
    rows = [
        {**base, "model_id": "a", "overall_accuracy": 0.99, "worst_group_accuracy": 0.98, "shortcut_gap": 0.01},
        {**base, "model_id": "b", "seed": 2, "overall_accuracy": 0.99, "worst_group_accuracy": 0.98, "shortcut_gap": 0.01},
        {**base, "model_id": "c", "seed": 3, "overall_accuracy": 0.985, "worst_group_accuracy": 0.97, "shortcut_gap": 0.02},
        {**base, "model_id": "d", "seed": 4, "overall_accuracy": 0.95, "worst_group_accuracy": 0.98, "shortcut_gap": 0.01},
    ]
    pairs = find_matched_pairs(rows)
    assert (pairs[0]["model_a"], pairs[0]["model_b"]) == ("a", "b")
    assert all("d" not in (pair["model_a"], pair["model_b"]) for pair in pairs)


def test_matching_rejects_nonpositive_tolerance():
    with pytest.raises(ValueError, match="positive"):
        find_matched_pairs([], accuracy_tolerance=0)
