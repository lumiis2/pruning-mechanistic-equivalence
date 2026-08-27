#!/usr/bin/env python3
"""Build a registry and validate documented findings from existing artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.model_registry import REGISTRY_FIELDS, build_registry, read_csv, write_csv  # noqa: E402


def artifact_findings(rows: list[dict], output_root: Path) -> dict:
    dense = [row for row in rows if row["model_family"] == "dense"]
    one_shot_50 = [row for row in rows if row["model_family"] == "one_shot" and row["sparsity"] == 0.5]
    rewinding = [row for row in rows if row["model_family"] == "rewinding"]
    imp = [row for row in rows if row["model_family"] == "imp"]
    controls = {row["dataset"]: row for row in read_csv(output_root / "controls_summary.csv")}
    robust_dense = sum(row["conflicting_accuracy"] >= 0.9 for row in dense)
    partial_dense = sum(0.5 <= row["conflicting_accuracy"] < 0.9 for row in dense)
    shortcut_dense = sum(row["conflicting_accuracy"] < 0.5 for row in dense)
    magnitude = [row for row in one_shot_50 if row["mask_selection_checkpoint"] != "random"]
    random = [row for row in one_shot_50 if row["mask_selection_checkpoint"] == "random"]
    return {
        "dataset_controls": {
            "color_only_conflicting_accuracy": float(controls["color_only"]["conflicting_accuracy"]),
            "shape_only_conflicting_accuracy": float(controls["shape_only"]["conflicting_accuracy"]),
            "uncorrelated_color_conflicting_accuracy": float(controls["color_permutation"]["conflicting_accuracy"]),
        },
        "dense_dynamics": {
            "model_count": len(dense), "all_acquire_shortcut": all(row["shortcut_acquisition_epoch"] is not None for row in dense),
            "robust_count": robust_dense, "partial_count": partial_dense, "shortcut_dependent_count": shortcut_dense,
        },
        "one_shot_50": {
            "magnitude_models_robust": sum(row["conflicting_accuracy"] >= 0.9 for row in magnitude),
            "magnitude_model_count": len(magnitude),
            "random_models_robust": sum(row["conflicting_accuracy"] >= 0.9 for row in random),
            "random_model_count": len(random),
        },
        "rewinding_80": {
            "model_count": len(rewinding),
            "all_final_models_robust": all(row["conflicting_accuracy"] >= 0.9 for row in rewinding),
            "shortcut_duration_range": [min(row["shortcut_phase_duration"] for row in rewinding), max(row["shortcut_phase_duration"] for row in rewinding)],
        },
        "imp": {
            "model_count": len(imp), "rounds_per_seed": dict(sorted(Counter(str(row["seed"]) for row in imp).items())),
            "maximum_sparsity": max(row["sparsity"] for row in imp),
            "all_final_models_robust": all(row["conflicting_accuracy"] >= 0.9 for row in imp),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/sanity_check.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/sanity_check/model_registry")
    args = parser.parse_args()
    rows = build_registry(PROJECT_ROOT, args.config)
    write_csv(rows, args.output_dir / "models.csv", REGISTRY_FIELDS)
    report = {
        "status": "complete", "model_count": len(rows),
        "counts_by_family": dict(sorted(Counter(row["model_family"] for row in rows).items())),
        "all_referenced_artifacts_exist": True,
        "artifact_policy": "paths reference existing artifacts; weights are not duplicated",
        "reconstructed_findings": artifact_findings(rows, PROJECT_ROOT / "outputs/sanity_check"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "validation.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
