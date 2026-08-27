#!/usr/bin/env python3
"""Validate discovery smoke artifacts without reading held-out evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PROBE_METRICS = {
    "test_accuracy",
    "test_aligned_accuracy",
    "test_conflicting_accuracy",
    "test_worst_group_accuracy",
    "test_shortcut_gap",
    "test_group_accuracy",
}
CAUSAL_METRICS = {
    "target_probability_signed_change",
    "target_probability_absolute_change",
    "target_logit_margin_signed_change",
    "target_logit_margin_absolute_change",
    "prediction_flip_rate",
    "sample_count",
}


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"missing artifact: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require_finite(value: Any, location: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"expected finite number at {location}, got {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    root = args.output_dir.resolve()

    config = read_json(root / "config.json")
    if config.get("experiment_split") != "discovery":
        raise ValueError("smoke validation is restricted to discovery")
    if config.get("model_ids") != ["dense-s42"] or config.get("max_samples") != 64:
        raise ValueError("unexpected smoke model selection or sample count")
    if "original shape class" not in config.get("causal_effect_reference", ""):
        raise ValueError("causal-effect reference semantics are missing")

    model_dir = root / "dense-s42"
    for name in ("activations.pt", "probes.json", "causal_metrics.json", "summary.json", "status.json"):
        if not (model_dir / name).is_file():
            raise FileNotFoundError(f"missing artifact: {model_dir / name}")
    if read_json(model_dir / "status.json").get("status") != "complete":
        raise ValueError("model status is not complete")

    probes = read_json(model_dir / "probes.json")
    if set(probes) != {"conv1", "conv2", "hidden"}:
        raise ValueError("unexpected probe layers")
    for layer, targets in probes.items():
        if set(targets) != {"shape", "color"}:
            raise ValueError(f"unexpected probe targets at {layer}")
        for target, metrics in targets.items():
            if not PROBE_METRICS.issubset(metrics):
                raise ValueError(f"missing probe metrics at {layer}/{target}")
            if set(metrics["test_group_accuracy"]) != {"0", "1", "2", "3"}:
                raise ValueError(f"missing probe groups at {layer}/{target}")
            if (metrics["train_count"], metrics["validation_count"], metrics["test_count"]) != (36, 12, 16):
                raise ValueError(f"unexpected probe split counts at {layer}/{target}")
            for key in PROBE_METRICS - {"test_group_accuracy"}:
                require_finite(metrics[key], f"probes/{layer}/{target}/{key}")
            for group, value in metrics["test_group_accuracy"].items():
                require_finite(value, f"probes/{layer}/{target}/group/{group}")

    causal = read_json(model_dir / "causal_metrics.json")
    if len(causal) != 120:
        raise ValueError(f"expected 120 causal measurements, got {len(causal)}")
    for intervention, metrics in causal.items():
        if not CAUSAL_METRICS.issubset(metrics):
            raise ValueError(f"missing causal metrics at {intervention}")
        if metrics["sample_count"] != 64:
            raise ValueError(f"unexpected sample count at {intervention}")
        if set(metrics.get("group_metrics", {})) != {"0", "1", "2", "3"}:
            raise ValueError(f"missing causal groups at {intervention}")
        for key in CAUSAL_METRICS:
            require_finite(metrics[key], f"causal/{intervention}/{key}")
        for group, group_metrics in metrics["group_metrics"].items():
            if not CAUSAL_METRICS.issubset(group_metrics) or group_metrics["sample_count"] != 16:
                raise ValueError(f"invalid group metrics at {intervention}/{group}")
            for key in CAUSAL_METRICS:
                require_finite(group_metrics[key], f"causal/{intervention}/{group}/{key}")

    summaries = read_json(root / "all_model_summaries.json")
    if len(summaries) != 1 or summaries[0].get("model_id") != "dense-s42":
        raise ValueError("unexpected aggregate summary")
    print("MECHANISTIC DISCOVERY SMOKE ARTIFACTS VALIDATED")


if __name__ == "__main__":
    main()
