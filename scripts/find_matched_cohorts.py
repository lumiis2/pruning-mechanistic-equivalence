#!/usr/bin/env python3
"""Identify behaviorally matched pairs and write a candidate-cohort report."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.model_registry import find_matched_pairs, write_csv  # noqa: E402


def load_registry(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric = ("seed", "sparsity", "overall_accuracy", "aligned_accuracy",
               "conflicting_accuracy", "worst_group_accuracy", "shortcut_gap")
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
        row["imp_round"] = int(row["imp_round"]) if row["imp_round"] else None
    return rows


def select_candidates(pairs: list[dict], registry: dict[str, dict], limit: int) -> list[dict]:
    """Favor close pairs while covering distinct experimental contrasts."""
    buckets: dict[str, list[dict]] = {"dense_sparse": [], "mask": [], "rewinding": [], "imp": [], "cross_seed": []}
    for pair in pairs:
        left, right = registry[pair["model_a"]], registry[pair["model_b"]]
        families = {left["model_family"], right["model_family"]}
        if "dense" in families and len(families) > 1:
            buckets["dense_sparse"].append(pair)
        if (left["model_family"] == right["model_family"] == "one_shot"
                and left["seed"] == right["seed"]
                and left["sparsity"] == right["sparsity"]
                and left["mask_selection_checkpoint"] != right["mask_selection_checkpoint"]):
            buckets["mask"].append(pair)
        if left["model_family"] == right["model_family"] == "rewinding" and left["rewinding_checkpoint"] != right["rewinding_checkpoint"]:
            buckets["rewinding"].append(pair)
        if "imp" in families and (left["imp_round"] != right["imp_round"] or len(families) > 1):
            buckets["imp"].append(pair)
        if left["seed"] != right["seed"]:
            buckets["cross_seed"].append(pair)
    selected, seen = [], set()
    while len(selected) < limit and any(buckets.values()):
        for bucket in buckets.values():
            while bucket and (bucket[0]["model_a"], bucket[0]["model_b"]) in seen:
                bucket.pop(0)
            if bucket and len(selected) < limit:
                pair = bucket.pop(0)
                seen.add((pair["model_a"], pair["model_b"]))
                selected.append(pair)
    return selected


def write_report(path: Path, selected: list[dict], tolerances: tuple[float, float, float]) -> None:
    lines = [
        "# Candidate behaviorally matched cohorts", "",
        "Models were selected automatically from saved artifacts; no new training was run.", "",
        "## Matching rule", "",
        f"- Overall accuracy difference <= {tolerances[0]:.3f}",
        f"- Worst-group accuracy difference <= {tolerances[1]:.3f}",
        f"- Shortcut-gap difference <= {tolerances[2]:.3f}", "",
        "## Strong candidate pairs", "",
        "| model A | model B | sparsity A/B | accuracy diff | WGA diff | gap diff | contrast |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for pair in selected:
        contrast = pair["structural_or_trajectory_difference"].replace("|", "/")
        lines.append(
            f"| {pair['model_a']} | {pair['model_b']} | {pair['sparsity_a']:.3f}/{pair['sparsity_b']:.3f} "
            f"| {pair['accuracy_difference']:.4f} | {pair['worst_group_difference']:.4f} "
            f"| {pair['shortcut_gap_difference']:.4f} | {contrast} |"
        )
    lines += ["", "## Proposed next implementation", "",
              "Freeze discovery and held-out model/condition sets, then implement balanced activation caching and linear shape/color probes at conv1, conv2, and hidden. In the same inference pass, save paired shape/color counterfactual logit, probability, and flip effects. Probe decodability must remain separate from causal sensitivity.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=PROJECT_ROOT / "outputs/sanity_check/model_registry/models.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/sanity_check/model_registry")
    parser.add_argument("--accuracy-tolerance", type=float, default=0.01)
    parser.add_argument("--worst-group-tolerance", type=float, default=0.02)
    parser.add_argument("--shortcut-gap-tolerance", type=float, default=0.02)
    parser.add_argument("--report-limit", type=int, default=15)
    args = parser.parse_args()
    rows = load_registry(args.registry)
    pairs = find_matched_pairs(rows, accuracy_tolerance=args.accuracy_tolerance,
                               worst_group_tolerance=args.worst_group_tolerance,
                               shortcut_gap_tolerance=args.shortcut_gap_tolerance)
    if not pairs:
        raise RuntimeError("no matched pairs found; change tolerances explicitly if justified")
    selected = select_candidates(pairs, {row["model_id"]: row for row in rows}, args.report_limit)
    write_csv(pairs, args.output_dir / "matched_pairs.csv")
    write_csv(selected, args.output_dir / "candidate_pairs.csv")
    write_report(args.output_dir / "candidate_cohorts.md", selected,
                 (args.accuracy_tolerance, args.worst_group_tolerance, args.shortcut_gap_tolerance))
    print(f"matched_pairs={len(pairs)} reported_candidates={len(selected)}")


if __name__ == "__main__":
    main()
