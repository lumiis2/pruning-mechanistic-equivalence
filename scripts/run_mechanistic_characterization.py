#!/usr/bin/env python3
"""Run balanced probes, recoverability, counterfactuals, and layer patching."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.mechanistic import (  # noqa: E402
    LAYERS, CounterfactualDataset, balanced_indices, collect_activations,
    counterfactual_and_patching_metrics, train_balanced_probe,
)
from src.data import create_splits  # noqa: E402
from src.models import SmallCNN  # noqa: E402
from src.utils import load_yaml, save_json, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/sanity_check.yaml")
    parser.add_argument("--mechanistic-config", type=Path, default=PROJECT_ROOT / "configs/mechanistic.yaml")
    parser.add_argument("--split", choices=("discovery", "evaluation"), default="discovery")
    parser.add_argument("--registry", type=Path, default=PROJECT_ROOT / "outputs/sanity_check/model_registry/models.csv")
    parser.add_argument("--candidate-pairs", type=Path, default=PROJECT_ROOT / "outputs/sanity_check/model_registry/candidate_pairs.csv")
    parser.add_argument("--model-ids", nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/mechanistic_characterization")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--probe-epochs", type=int, default=None)
    parser.add_argument("--spatial-grid", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--channel-patching", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def selected_ids(args: argparse.Namespace, mechanistic_config: dict) -> list[str]:
    if args.model_ids:
        return list(dict.fromkeys(args.model_ids))
    return list(mechanistic_config[args.split]["model_ids"])


def load_model(row: dict[str, str], hidden_dim: int, device: torch.device) -> SmallCNN:
    model = SmallCNN(hidden_dim=hidden_dim)
    checkpoint = torch.load(PROJECT_ROOT / row["checkpoint_path"], map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def main() -> None:
    args = parse_args()
    mechanistic_config = load_yaml(args.mechanistic_config)
    settings = mechanistic_config["characterization"]
    args.batch_size = args.batch_size or int(settings["batch_size"])
    args.probe_epochs = args.probe_epochs or int(settings["probe_epochs"])
    args.spatial_grid = args.spatial_grid or int(settings["spatial_grid"])
    if args.channel_patching is None:
        args.channel_patching = bool(settings["channel_patching"])
    if args.output_dir == PROJECT_ROOT / "outputs/mechanistic_characterization":
        args.output_dir = PROJECT_ROOT / settings["output_dir"] / args.split
    if args.batch_size < 1 or args.probe_epochs < 1 or args.spatial_grid < 1:
        raise ValueError("batch size, probe epochs, and spatial grid must be positive")
    config = load_yaml(args.config)
    registry = {row["model_id"]: row for row in read_rows(args.registry)}
    model_ids = selected_ids(args, mechanistic_config)
    if args.smoke_test:
        model_ids, args.max_samples, args.probe_epochs = model_ids[:1], 64, 3
        args.output_dir = args.output_dir / "smoke_test"
    unknown = sorted(set(model_ids) - set(registry))
    if unknown:
        raise ValueError(f"unknown model IDs: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "model_ids": model_ids, "batch_size": args.batch_size,
        "probe_epochs": args.probe_epochs, "spatial_grid": args.spatial_grid,
        "max_samples": args.max_samples, "dataset": "canonical balanced test split",
        "probe_split": "group-stratified 60/20/20", "seed_policy": "model seed",
        "experiment_split": args.split, "channel_patching": args.channel_patching,
        "causal_effect_reference": "probability and logit margin of the original shape class",
        "shape_swap_interpretation": (
            "A shape swap changes the true class. Signed effects are measured relative to the "
            "original class and are not counterfactual accuracy."
        ),
    }
    save_json(run_config, args.output_dir / "config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries = []
    for model_id in model_ids:
        row = registry[model_id]
        model_dir = args.output_dir / model_id
        status_path = model_dir / "status.json"
        if args.resume and status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "complete":
                print(f"{model_id}: already complete")
                summaries.append(json.loads((model_dir / "summary.json").read_text(encoding="utf-8")))
                continue
        model_dir.mkdir(parents=True, exist_ok=True)
        seed = int(float(row["seed"]))
        set_seed(seed)
        dataset = create_splits(seed=seed, data_config=config["data"], variant="original")["test_balanced"]
        indices = balanced_indices(dataset, args.max_samples, seed)
        paired = CounterfactualDataset(dataset, indices)
        loader = DataLoader(paired, batch_size=args.batch_size, shuffle=False, num_workers=0)
        model = load_model(row, int(config["model"]["hidden_dim"]), device)
        cache_path = model_dir / "activations.pt"
        if args.resume and cache_path.exists():
            activations = torch.load(cache_path, map_location="cpu")
        else:
            activations = collect_activations(model, loader, device, args.spatial_grid)
            torch.save(activations, cache_path)
        probes = {}
        for layer in LAYERS:
            probes[layer] = {}
            for target in ("shape", "color"):
                probes[layer][target] = train_balanced_probe(
                    activations[layer], activations[target], activations["group"],
                    seed=seed, epochs=args.probe_epochs,
                )
        save_json(probes, model_dir / "probes.json")
        causal = counterfactual_and_patching_metrics(
            model, loader, device, channel_patching=args.channel_patching
        )
        save_json(causal, model_dir / "causal_metrics.json")
        summary = {
            "model_id": model_id, "seed": seed, "sparsity": float(row["sparsity"]),
            "original_overall_accuracy": float(row["overall_accuracy"]),
            "original_worst_group_accuracy": float(row["worst_group_accuracy"]),
            "original_shortcut_gap": float(row["shortcut_gap"]),
            "balanced_hidden_head_accuracy": probes["hidden"]["shape"]["test_accuracy"],
            "balanced_hidden_head_aligned_accuracy": probes["hidden"]["shape"]["test_aligned_accuracy"],
            "balanced_hidden_head_conflicting_accuracy": probes["hidden"]["shape"]["test_conflicting_accuracy"],
            "balanced_hidden_head_worst_group_accuracy": probes["hidden"]["shape"]["test_worst_group_accuracy"],
            "balanced_hidden_head_shortcut_gap": probes["hidden"]["shape"]["test_shortcut_gap"],
            "shape_probe_accuracy": {layer: probes[layer]["shape"]["test_accuracy"] for layer in LAYERS},
            "color_probe_accuracy": {layer: probes[layer]["color"]["test_accuracy"] for layer in LAYERS},
            "causal_metrics": causal,
        }
        save_json(summary, model_dir / "summary.json")
        save_json({"status": "complete"}, status_path)
        summaries.append(summary)
        print(f"{model_id}: complete")
    save_json(summaries, args.output_dir / "all_model_summaries.json")
    print(f"completed {len(summaries)} models on {device}")


if __name__ == "__main__":
    main()
