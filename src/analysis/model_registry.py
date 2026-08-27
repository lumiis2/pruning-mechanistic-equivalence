"""Build a unified registry and matched cohorts from saved experiment artifacts."""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import yaml

REGISTRY_FIELDS = [
    "model_id", "model_family", "seed", "sparsity", "mask_selection_checkpoint",
    "rewinding_checkpoint", "imp_round", "overall_accuracy", "aligned_accuracy",
    "conflicting_accuracy", "worst_group_accuracy", "shortcut_gap",
    "shortcut_acquisition_epoch", "shortcut_escape_epoch", "shortcut_phase_duration",
    "checkpoint_path", "mask_path", "trajectory_path",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def trajectory_summary(path: Path, shortcut_threshold: float, escape_threshold: float) -> tuple[int | None, int | None, int]:
    """Apply one definition of shortcut acquisition/escape across experiment families."""
    rows = [row for row in read_csv(path) if int(row["epoch"]) > 0]
    if not rows:
        raise ValueError(f"trajectory has no trained epochs: {path}")
    gap_key = "test_balanced_shortcut_gap" if "test_balanced_shortcut_gap" in rows[0] else "shortcut_gap"
    acquisition = next((int(row["epoch"]) for row in rows if float(row[gap_key]) > shortcut_threshold), None)
    escape = next((int(row["epoch"]) for row in rows if acquisition is not None and int(row["epoch"]) > acquisition and float(row[gap_key]) < escape_threshold), None)
    duration = sum(float(row[gap_key]) > shortcut_threshold for row in rows)
    return acquisition, escape, duration


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def model_row(*, root: Path, model_id: str, family: str, seed: int, sparsity: float,
              mask_selection: str, rewind: str, imp_round: int | None, checkpoint: Path,
              mask: Path | None, trajectory: Path, metrics: Path,
              shortcut_threshold: float, escape_threshold: float) -> dict[str, Any]:
    required = [checkpoint, trajectory, metrics] + ([mask] if mask else [])
    missing = [str(path) for path in required if path is not None and not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{model_id} is missing: {', '.join(missing)}")
    values = read_json(metrics)
    acquisition, escape, duration = trajectory_summary(trajectory, shortcut_threshold, escape_threshold)
    return {
        "model_id": model_id, "model_family": family, "seed": seed, "sparsity": sparsity,
        "mask_selection_checkpoint": mask_selection, "rewinding_checkpoint": rewind,
        "imp_round": imp_round, "overall_accuracy": float(values["overall_accuracy"]),
        "aligned_accuracy": float(values["aligned_accuracy"]),
        "conflicting_accuracy": float(values["conflicting_accuracy"]),
        "worst_group_accuracy": float(values["worst_group_accuracy"]),
        "shortcut_gap": float(values["shortcut_gap"]),
        "shortcut_acquisition_epoch": acquisition, "shortcut_escape_epoch": escape,
        "shortcut_phase_duration": duration, "checkpoint_path": _relative(checkpoint, root),
        "mask_path": _relative(mask, root) if mask else "", "trajectory_path": _relative(trajectory, root),
    }


def add_masked_conditions(rows: list[dict[str, Any]], root: Path, experiment_root: Path,
                          seeds: Iterable[int], sparsity: float, shortcut_threshold: float,
                          escape_threshold: float) -> None:
    for seed in seeds:
        seed_dir = experiment_root / f"seed_{seed}"
        if not seed_dir.is_dir():
            raise FileNotFoundError(f"missing experiment directory: {seed_dir}")
        for directory in sorted(path for path in seed_dir.iterdir() if path.is_dir()):
            selection = directory.name.removesuffix("_mask")
            rows.append(model_row(
                root=root, model_id=f"oneshot{round(100 * sparsity)}-{selection}-s{seed}",
                family="one_shot", seed=seed, sparsity=sparsity, mask_selection=selection,
                rewind="init", imp_round=None, checkpoint=directory / "best_model.pt",
                mask=directory / "mask.pt", trajectory=directory / "training_history.csv",
                metrics=directory / "final_metrics.json", shortcut_threshold=shortcut_threshold,
                escape_threshold=escape_threshold,
            ))


def build_registry(root: Path, config_path: Path) -> list[dict[str, Any]]:
    """Return one row per saved model endpoint without duplicating model weights."""
    root = root.resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    shortcut_threshold = float(config["one_shot_pruning"]["shortcut_phase_gap_threshold"])
    escape_threshold = float(config["one_shot_pruning"]["escape_gap_threshold"])
    rows: list[dict[str, Any]] = []
    dense_root = root / config["multi_seed"]["output_dir"]
    for seed in map(int, config["multi_seed"]["seeds"]):
        directory = dense_root / f"seed_{seed}"
        rows.append(model_row(
            root=root, model_id=f"dense-s{seed}", family="dense", seed=seed, sparsity=0.0,
            mask_selection="none", rewind="none", imp_round=None,
            checkpoint=directory / "best_model.pt", mask=None,
            trajectory=directory / "training_history.csv", metrics=directory / "final_metrics.json",
            shortcut_threshold=shortcut_threshold, escape_threshold=escape_threshold,
        ))
    pruning = config["one_shot_pruning"]
    add_masked_conditions(rows, root, root / pruning["output_dir"], map(int, pruning["seeds"]),
                          float(pruning["sparsity"]), shortcut_threshold, escape_threshold)
    sweep_root = root / config["sparsity_sweep"]["output_dir"]
    for sparsity in map(float, config["sparsity_sweep"]["sparsities"]):
        if sparsity == float(pruning["sparsity"]):
            continue
        add_masked_conditions(rows, root, sweep_root / f"sparsity_{round(100 * sparsity)}",
                              map(int, pruning["seeds"]), sparsity,
                              shortcut_threshold, escape_threshold)
    rewinding = config["rewinding"]
    rewind_root = root / rewinding["output_dir"]
    for seed in map(int, rewinding["seeds"]):
        seed_dir = rewind_root / f"seed_{seed}"
        for state in map(str, rewinding["states"]):
            directory = seed_dir / state
            rows.append(model_row(
                root=root, model_id=f"rewind80-{state}-s{seed}", family="rewinding", seed=seed,
                sparsity=float(rewinding["sparsity"]), mask_selection=str(rewinding["mask_stage"]),
                rewind=state, imp_round=None, checkpoint=directory / "best_model.pt",
                mask=seed_dir / "fixed_mask.pt", trajectory=directory / "training_history.csv",
                metrics=directory / "final_metrics.json", shortcut_threshold=shortcut_threshold,
                escape_threshold=escape_threshold,
            ))
    imp = config["imp"]
    imp_root = root / imp["output_dir"]
    for seed in map(int, imp["seeds"]):
        for round_index in range(1, int(imp["rounds"]) + 1):
            directory = imp_root / f"seed_{seed}" / f"round_{round_index:02d}"
            summary_path = directory / "summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(f"missing IMP summary: {summary_path}")
            summary = read_json(summary_path)
            rows.append(model_row(
                root=root, model_id=f"imp-r{round_index:02d}-s{seed}", family="imp", seed=seed,
                sparsity=float(summary["sparsity"]), mask_selection=str(imp["mask_selection"]),
                rewind=str(imp["rewind_state"]), imp_round=round_index,
                checkpoint=directory / "best_model.pt", mask=directory / "mask.pt",
                trajectory=directory / "trajectory.csv", metrics=directory / "final_metrics.json",
                shortcut_threshold=shortcut_threshold, escape_threshold=escape_threshold,
            ))
    return sorted(rows, key=lambda row: row["model_id"])


def find_matched_pairs(rows: list[dict[str, Any]], *, accuracy_tolerance: float = 0.01,
                       worst_group_tolerance: float = 0.02,
                       shortcut_gap_tolerance: float = 0.02) -> list[dict[str, Any]]:
    """Return every pair inside explicit tolerances, ranked by normalized distance."""
    if min(accuracy_tolerance, worst_group_tolerance, shortcut_gap_tolerance) <= 0:
        raise ValueError("matching tolerances must be positive")
    pairs = []
    for left, right in combinations(rows, 2):
        deltas = {
            "accuracy_difference": abs(float(left["overall_accuracy"]) - float(right["overall_accuracy"])),
            "worst_group_difference": abs(float(left["worst_group_accuracy"]) - float(right["worst_group_accuracy"])),
            "shortcut_gap_difference": abs(float(left["shortcut_gap"]) - float(right["shortcut_gap"])),
        }
        if (deltas["accuracy_difference"] <= accuracy_tolerance
                and deltas["worst_group_difference"] <= worst_group_tolerance
                and deltas["shortcut_gap_difference"] <= shortcut_gap_tolerance):
            keys = ("model_family", "seed", "sparsity", "mask_selection_checkpoint",
                    "rewinding_checkpoint", "imp_round")
            differences = [f"{key}:{left[key]} vs {right[key]}" for key in keys if left[key] != right[key]]
            score = (deltas["accuracy_difference"] / accuracy_tolerance
                     + deltas["worst_group_difference"] / worst_group_tolerance
                     + deltas["shortcut_gap_difference"] / shortcut_gap_tolerance)
            pairs.append({"model_a": left["model_id"], "model_b": right["model_id"],
                          "sparsity_a": left["sparsity"], "sparsity_b": right["sparsity"],
                          **deltas, "match_score": score,
                          "structural_or_trajectory_difference": "; ".join(differences)})
    return sorted(pairs, key=lambda row: (row["match_score"], row["model_a"], row["model_b"]))


def write_csv(rows: list[dict[str, Any]], path: Path, fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
