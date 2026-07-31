#!/usr/bin/env python3
"""Measure dense shortcut-to-robust learning dynamics across configured seeds."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import create_splits  # noqa: E402
from src.models import SmallCNN  # noqa: E402
from src.training import (  # noqa: E402
    evaluate_counterfactual_sensitivity,
    evaluate_model,
    select_functional_epochs,
    train_model,
)
from src.utils import load_yaml, save_csv, save_json, save_yaml, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "sanity_check.yaml"
    )
    return parser.parse_args()


def resolve_output_dir(configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    config = load_yaml(parse_args().config)
    dynamics_config = config["multi_seed"]
    thresholds = dynamics_config["functional_thresholds"]
    output_dir = resolve_output_dir(dynamics_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, output_dir / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = config["training"]
    all_history: list[dict] = []
    seed_summary: list[dict] = []

    for raw_seed in dynamics_config["seeds"]:
        seed = int(raw_seed)
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nseed={seed} device={device}")
        set_seed(seed)
        splits = create_splits(seed=seed, data_config=config["data"], variant="original")
        loader_kwargs = {
            "batch_size": int(training["batch_size"]),
            "num_workers": int(training["num_workers"]),
            "pin_memory": device.type == "cuda",
        }
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            splits["train_biased"], shuffle=True, generator=generator, **loader_kwargs
        )
        val_loader = DataLoader(splits["val_biased"], shuffle=False, **loader_kwargs)
        test_loader = DataLoader(splits["test_balanced"], shuffle=False, **loader_kwargs)
        sensitivity_evaluator = partial(
            evaluate_counterfactual_sensitivity,
            dataset=splits["test_balanced"],
            device=device,
            batch_size=int(training["batch_size"]),
            num_workers=int(training["num_workers"]),
        )

        model = SmallCNN(hidden_dim=int(config["model"]["hidden_dim"]))
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            learning_rate=float(training["learning_rate"]),
            max_epochs=int(training["max_epochs"]),
            patience=int(training["patience"]),
            checkpoint_path=seed_dir / "best_model.pt",
            checkpoint_epochs=[],
            sensitivity_evaluator=sensitivity_evaluator,
            epoch_checkpoint_dir=seed_dir / "epoch_states",
        )
        save_csv(history, seed_dir / "training_history.csv")
        all_history.extend({"seed": seed, **row} for row in history)

        functional_epochs = select_functional_epochs(
            history=history,
            shortcut_gap=float(thresholds["shortcut_gap"]),
            transition_gap=float(thresholds["transition_gap"]),
            robust_conflicting_accuracy=float(thresholds["robust_conflicting_accuracy"]),
        )
        functional_dir = seed_dir / "functional_checkpoints"
        functional_dir.mkdir(parents=True, exist_ok=True)
        rows_by_epoch = {int(row["epoch"]): row for row in history}
        for stage, epoch in functional_epochs.items():
            if epoch is None:
                continue
            checkpoint = torch.load(
                seed_dir / "epoch_states" / f"epoch_{epoch:02d}.pt", map_location="cpu"
            )
            checkpoint["functional_stage"] = stage
            checkpoint["metrics_at_selection"] = rows_by_epoch[epoch]
            torch.save(checkpoint, functional_dir / f"{stage}.pt")

        best_checkpoint = torch.load(seed_dir / "best_model.pt", map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        model.to(device)
        final_metrics = evaluate_model(model, test_loader, device)
        final_metrics.update(sensitivity_evaluator(model))
        final_metrics.update(
            {
                "seed": seed,
                "best_epoch": int(best_checkpoint["epoch"]),
                "functional_epochs": functional_epochs,
            }
        )
        save_json(final_metrics, seed_dir / "final_metrics.json")
        seed_summary.append(
            {
                "seed": seed,
                "shortcut_epoch": functional_epochs["shortcut"],
                "late_shortcut_epoch": functional_epochs["late_shortcut"],
                "transition_epoch": functional_epochs["transition"],
                "robust_epoch": functional_epochs["robust"],
                "best_epoch": int(best_checkpoint["epoch"]),
                "final_overall_accuracy": final_metrics["overall_accuracy"],
                "final_aligned_accuracy": final_metrics["aligned_accuracy"],
                "final_conflicting_accuracy": final_metrics["conflicting_accuracy"],
                "final_shortcut_gap": final_metrics["shortcut_gap"],
                "final_color_swap_sensitivity": final_metrics["color_swap_sensitivity"],
                "final_shape_swap_sensitivity": final_metrics["shape_swap_sensitivity"],
            }
        )

    save_csv(all_history, output_dir / "all_seeds_history.csv")
    save_csv(seed_summary, output_dir / "seed_summary.csv")
    print(f"\nSaved multi-seed results to {output_dir}")


if __name__ == "__main__":
    main()
