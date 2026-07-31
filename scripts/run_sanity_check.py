#!/usr/bin/env python3
"""Run the complete dense-model Shape--Color sanity check once."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import create_splits, save_four_groups_figure  # noqa: E402
from src.models import SmallCNN  # noqa: E402
from src.training import (  # noqa: E402
    collect_predictions,
    evaluate_model,
    save_training_curves,
    train_model,
)
from src.utils import load_yaml, save_csv, save_json, save_yaml, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "sanity_check.yaml",
    )
    return parser.parse_args()


def run_experiment(
    name: str,
    config: dict,
    data_config: dict,
    variant: str,
    output_dir: Path,
    device: torch.device,
) -> dict:
    seed = int(config["seed"])
    training = config["training"]
    threshold = float(config["shortcut_gap_threshold"])
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    print(f"\nexperiment={name} variant={variant} device={device} seed={seed}")

    splits = create_splits(seed=seed, data_config=data_config, variant=variant)
    save_four_groups_figure(splits["test_balanced"], output_dir / "four_groups.png")

    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        splits["train_biased"], shuffle=True, generator=generator, **loader_kwargs
    )
    val_loader = DataLoader(splits["val_biased"], shuffle=False, **loader_kwargs)
    test_loader = DataLoader(splits["test_balanced"], shuffle=False, **loader_kwargs)

    model = SmallCNN(hidden_dim=int(config["model"]["hidden_dim"]))
    checkpoint_path = output_dir / "best_model.pt"
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=float(training["learning_rate"]),
        max_epochs=int(training["max_epochs"]),
        patience=int(training["patience"]),
        checkpoint_path=checkpoint_path,
        checkpoint_epochs=[int(epoch) for epoch in training["checkpoint_epochs"]],
    )
    save_csv(history, output_dir / "training_history.csv")
    save_training_curves(history, output_dir / "training_curves.png")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    val_metrics = evaluate_model(model, val_loader, device)
    test_metrics = evaluate_model(model, test_loader, device)
    predictions = collect_predictions(model, test_loader, device)
    save_csv(predictions, output_dir / "test_predictions.csv")

    gap = test_metrics["shortcut_gap"]
    passed = (
        gap is not None
        and test_metrics["aligned_accuracy"] > test_metrics["conflicting_accuracy"]
        and gap >= threshold
    )
    final_metrics = {
        **test_metrics,
        "sanity_check_passed": passed,
        "shortcut_gap_threshold": threshold,
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_loss": float(checkpoint["val_loss"]),
        "seed": seed,
        "device": str(device),
        "val_biased": val_metrics,
    }
    save_json(final_metrics, output_dir / "final_metrics.json")

    return final_metrics


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    configured_output_dir = Path(config["output_dir"])
    output_dir = configured_output_dir if configured_output_dir.is_absolute() else PROJECT_ROOT / configured_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, output_dir / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_metrics = run_experiment(
        name="original",
        config=config,
        data_config=config["data"],
        variant="original",
        output_dir=output_dir,
        device=device,
    )

    summary_rows = [
        {
            "dataset": "original",
            "information_available": "shape + biased color",
            "balanced_accuracy": original_metrics["overall_accuracy"],
            "aligned_accuracy": original_metrics["aligned_accuracy"],
            "conflicting_accuracy": original_metrics["conflicting_accuracy"],
            "shortcut_gap": original_metrics["shortcut_gap"],
        }
    ]
    information = {
        "color_only": "color only",
        "shape_only": "shape only",
        "color_permutation": "shape + uncorrelated color",
    }
    for control_name, control_config in config["controls"].items():
        control_data = copy.deepcopy(config["data"])
        if "train_alignment_probability" in control_config:
            control_data["train_alignment_probability"] = control_config["train_alignment_probability"]
        if "val_alignment_probability" in control_config:
            control_data["val_alignment_probability"] = control_config["val_alignment_probability"]
        metrics = run_experiment(
            name=control_name,
            config=config,
            data_config=control_data,
            variant=str(control_config["variant"]),
            output_dir=output_dir / "controls" / control_name,
            device=device,
        )
        summary_rows.append(
            {
                "dataset": control_name,
                "information_available": information[control_name],
                "balanced_accuracy": metrics["overall_accuracy"],
                "aligned_accuracy": metrics["aligned_accuracy"],
                "conflicting_accuracy": metrics["conflicting_accuracy"],
                "shortcut_gap": metrics["shortcut_gap"],
            }
        )
    save_csv(summary_rows, output_dir / "controls_summary.csv")

    print(f"\nSANITY CHECK {'PASSED' if original_metrics['sanity_check_passed'] else 'FAILED'}")
    print(f"aligned_accuracy: {original_metrics['aligned_accuracy']:.4f}")
    print(f"conflicting_accuracy: {original_metrics['conflicting_accuracy']:.4f}")
    print(f"shortcut_gap: {original_metrics['shortcut_gap']:.4f}")


if __name__ == "__main__":
    main()
