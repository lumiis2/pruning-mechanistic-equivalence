#!/usr/bin/env python3
"""Run and aggregate the configured small one-shot sparsity sweep."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "sanity_check.yaml")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_trajectories(
    rows: list[dict[str, str]], sparsity: float, escape_threshold: float, shortcut_threshold: float
) -> list[dict]:
    """Recompute functional metrics consistently, including shortcut acquisition."""
    summaries = []
    keys = sorted({(int(row["seed"]), row["condition"]) for row in rows})
    for seed, condition in keys:
        trained = sorted(
            (row for row in rows if int(row["seed"]) == seed and row["condition"] == condition and int(row["epoch"]) > 0),
            key=lambda row: int(row["epoch"]),
        )
        acquisition = next((int(row["epoch"]) for row in trained if float(row["shortcut_gap"]) > shortcut_threshold), None)
        escape = next(
            (int(row["epoch"]) for row in trained if acquisition is not None and int(row["epoch"]) > acquisition and float(row["shortcut_gap"]) < escape_threshold),
            None,
        )
        final = trained[-1]
        summaries.append({
            "sparsity": sparsity,
            "seed": seed,
            "condition": condition,
            "shortcut_acquisition_epoch": acquisition,
            "escape_epoch": escape,
            "shortcut_phase_duration": sum(float(row["shortcut_gap"]) > shortcut_threshold for row in trained),
            "shortcut_gap_area": sum(float(row["shortcut_gap"]) for row in trained),
            "final_epoch": int(final["epoch"]),
            "final_conflicting_accuracy": float(final["conflicting_accuracy"]),
            "final_worst_group_accuracy": float(final["worst_group_accuracy"]),
            "final_shortcut_gap": float(final["shortcut_gap"]),
        })
    return summaries


def main() -> None:
    args = parse_args()
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    sweep = config["sparsity_sweep"]
    output_dir = resolve_path(sweep["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = resolve_path(config["one_shot_pruning"]["output_dir"])
    all_summaries, all_trajectories = [], []

    for raw_sparsity in sweep["sparsities"]:
        sparsity = float(raw_sparsity)
        if sparsity == float(config["one_shot_pruning"]["sparsity"]):
            result_dir = baseline_dir
        elif not (output_dir / f"sparsity_{round(100 * sparsity):02d}" / "comparison_summary.csv").exists():
            result_dir = output_dir / f"sparsity_{round(100 * sparsity):02d}"
            subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "run_one_shot_pruning.py"),
                 "--config", str(args.config), "--sparsity", str(sparsity),
                 "--output-dir", str(result_dir)],
                check=True,
            )
        else:
            result_dir = output_dir / f"sparsity_{round(100 * sparsity):02d}"
        trajectories = read_csv(result_dir / "all_trajectories.csv")
        all_summaries.extend(summarize_trajectories(
            trajectories, sparsity,
            float(config["one_shot_pruning"]["escape_gap_threshold"]),
            float(config["one_shot_pruning"]["shortcut_phase_gap_threshold"]),
        ))
        all_trajectories.extend({"sparsity": sparsity, **row} for row in trajectories)

    write_csv(all_summaries, output_dir / "sweep_summary.csv")
    write_csv(all_trajectories, output_dir / "sweep_trajectories.csv")
    conditions = ["dense", "random_mask", "shortcut_mask", "transition_mask", "robust_mask"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for axis, (metric, title) in zip(axes, [
        ("shortcut_phase_duration", "Shortcut-phase duration"),
        ("shortcut_gap_area", "Shortcut-gap area"),
        ("final_shortcut_gap", "Final shortcut gap"),
    ]):
        for condition in conditions:
            means = []
            for sparsity in sweep["sparsities"]:
                values = [float(row[metric]) for row in all_summaries if row["condition"] == condition and float(row["sparsity"]) == float(sparsity)]
                means.append(sum(values) / len(values))
            axis.plot(sweep["sparsities"], means, marker="o", label=condition)
        axis.set(title=title, xlabel="Sparsity")
        axis.grid(alpha=0.25)
    axes[-1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    figure.savefig(output_dir / "sparsity_sweep.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved sparsity sweep to {output_dir}")


if __name__ == "__main__":
    main()
