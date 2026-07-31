"""Semantic checkpoint selection from behavioral learning curves."""

from __future__ import annotations

from typing import Any


def select_functional_epochs(
    history: list[dict[str, Any]],
    shortcut_gap: float,
    transition_gap: float,
    robust_conflicting_accuracy: float,
) -> dict[str, int | None]:
    """Select seed-specific epochs using predeclared behavioral criteria."""
    trained = [row for row in history if int(row["epoch"]) > 0]
    shortcut_rows = [row for row in trained if row["test_balanced_shortcut_gap"] > shortcut_gap]
    robust_rows = [
        row
        for row in trained
        if row["test_balanced_conflicting_accuracy"] > robust_conflicting_accuracy
    ]
    first_shortcut = int(shortcut_rows[0]["epoch"]) if shortcut_rows else None
    late_shortcut = int(shortcut_rows[-1]["epoch"]) if shortcut_rows else None
    robust = int(robust_rows[0]["epoch"]) if robust_rows else None

    transition_candidates = trained
    if first_shortcut is not None and robust is not None:
        transition_candidates = [
            row for row in trained if first_shortcut <= int(row["epoch"]) <= robust
        ]
    transition = (
        int(
            min(
                transition_candidates,
                key=lambda row: abs(row["test_balanced_shortcut_gap"] - transition_gap),
            )["epoch"]
        )
        if transition_candidates
        else None
    )
    return {
        "shortcut": first_shortcut,
        "late_shortcut": late_shortcut,
        "transition": transition,
        "robust": robust,
    }
