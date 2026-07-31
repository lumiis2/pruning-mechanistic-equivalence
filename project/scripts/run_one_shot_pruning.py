#!/usr/bin/env python3
"""Compare 50% one-shot masks selected at functional dense checkpoints."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import create_splits  # noqa: E402
from src.models import SmallCNN  # noqa: E402
from src.pruning import global_magnitude_masks, global_random_masks, mask_statistics  # noqa: E402
from src.training import evaluate_model, train_model  # noqa: E402
from src.utils import load_yaml, save_csv, save_json, save_yaml, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "sanity_check.yaml")
    parser.add_argument("--sparsity", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def trajectory_rows(seed: int, condition: str, history: list[dict]) -> list[dict]:
    return [
        {
            "seed": seed,
            "condition": condition,
            "epoch": row["epoch"],
            "loss": row["test_balanced_loss"],
            "overall_accuracy": row["test_balanced_overall_accuracy"],
            "aligned_accuracy": row["test_balanced_aligned_accuracy"],
            "conflicting_accuracy": row["test_balanced_conflicting_accuracy"],
            "worst_group_accuracy": row["test_balanced_worst_group_accuracy"],
            "shortcut_gap": row["test_balanced_shortcut_gap"],
        }
        for row in history
    ]


def summarize_trajectory(seed, condition, rows, escape_threshold, shortcut_phase_threshold):
    trained = [row for row in rows if int(row["epoch"]) > 0]
    acquisition_rows = [row for row in trained if float(row["shortcut_gap"]) > shortcut_phase_threshold]
    acquisition_epoch = int(acquisition_rows[0]["epoch"]) if acquisition_rows else None
    escape_rows = [
        row for row in trained
        if acquisition_epoch is not None
        and int(row["epoch"]) > acquisition_epoch
        and float(row["shortcut_gap"]) < escape_threshold
    ]
    final = trained[-1]
    return {
        "seed": seed,
        "condition": condition,
        "shortcut_acquisition_epoch": acquisition_epoch,
        "escape_epoch": int(escape_rows[0]["epoch"]) if escape_rows else None,
        "shortcut_phase_duration": sum(float(row["shortcut_gap"]) > shortcut_phase_threshold for row in trained),
        "shortcut_gap_area": sum(float(row["shortcut_gap"]) for row in trained),
        "final_epoch": int(final["epoch"]),
        "final_conflicting_accuracy": float(final["conflicting_accuracy"]),
        "final_worst_group_accuracy": float(final["worst_group_accuracy"]),
        "final_shortcut_gap": float(final["shortcut_gap"]),
    }


def save_gap_plot(rows: list[dict], output_path: Path) -> None:
    seeds = sorted({int(row["seed"]) for row in rows})
    conditions = ["dense", "random_mask", "shortcut_mask", "transition_mask", "robust_mask"]
    figure, axes = plt.subplots(1, len(seeds), figsize=(5 * len(seeds), 4), sharey=True, constrained_layout=True)
    if len(seeds) == 1:
        axes = [axes]
    for axis, seed in zip(axes, seeds):
        for condition in conditions:
            selected = [row for row in rows if int(row["seed"]) == seed and row["condition"] == condition]
            axis.plot([int(row["epoch"]) for row in selected], [float(row["shortcut_gap"]) for row in selected], label=condition)
        axis.axhline(0.2, color="black", linestyle="--", linewidth=0.8)
        axis.set(title=f"Seed {seed}", xlabel="Epoch", ylabel="Shortcut gap", ylim=(-0.05, 1.05))
        axis.grid(alpha=0.25)
    axes[-1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    pruning_config = config["one_shot_pruning"]
    multi_seed_dir = resolve_path(config["multi_seed"]["output_dir"])
    output_dir = args.output_dir or resolve_path(pruning_config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, output_dir / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = config["training"]
    sparsity = float(args.sparsity if args.sparsity is not None else pruning_config["sparsity"])
    all_rows, summaries = [], []

    for raw_seed in pruning_config["seeds"]:
        seed = int(raw_seed)
        print(f"\nseed={seed} device={device} sparsity={sparsity:.2f}")
        dense_seed_dir = multi_seed_dir / f"seed_{seed}"
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        with (dense_seed_dir / "training_history.csv").open() as handle:
            dense_history = list(csv.DictReader(handle))
        dense_rows = trajectory_rows(seed, "dense", dense_history)
        all_rows.extend(dense_rows)
        summaries.append(summarize_trajectory(seed, "dense", dense_rows, float(pruning_config["escape_gap_threshold"]), float(pruning_config["shortcut_phase_gap_threshold"])))

        set_seed(seed)
        splits = create_splits(seed=seed, data_config=config["data"], variant="original")
        loader_kwargs = {"batch_size": int(training["batch_size"]), "num_workers": int(training["num_workers"]), "pin_memory": device.type == "cuda"}
        val_loader = DataLoader(splits["val_biased"], shuffle=False, **loader_kwargs)
        test_loader = DataLoader(splits["test_balanced"], shuffle=False, **loader_kwargs)
        initialization = torch.load(dense_seed_dir / "epoch_states" / "epoch_00.pt", map_location="cpu")["model_state_dict"]

        initial_model = SmallCNN(hidden_dim=int(config["model"]["hidden_dim"]))
        initial_model.load_state_dict(initialization)
        masks_by_condition = {
            "random_mask": global_random_masks(initial_model, sparsity, seed + int(pruning_config["random_mask_seed_offset"]))
        }
        for condition, stage in {"shortcut_mask": "shortcut", "transition_mask": "transition", "robust_mask": "robust"}.items():
            source_path = dense_seed_dir / "functional_checkpoints" / f"{stage}.pt"
            if not source_path.exists():
                raise FileNotFoundError(f"seed {seed} has no required {stage} checkpoint")
            source_model = SmallCNN(hidden_dim=int(config["model"]["hidden_dim"]))
            source_model.load_state_dict(torch.load(source_path, map_location="cpu")["model_state_dict"])
            masks_by_condition[condition] = global_magnitude_masks(source_model, sparsity)

        for condition, masks in masks_by_condition.items():
            print(f"condition={condition}")
            condition_dir = seed_dir / condition
            condition_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"masks": masks, "statistics": mask_statistics(masks)}, condition_dir / "mask.pt")
            model = SmallCNN(hidden_dim=int(config["model"]["hidden_dim"]))
            model.load_state_dict(initialization)
            set_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(splits["train_biased"], shuffle=True, generator=generator, **loader_kwargs)
            history = train_model(
                model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                device=device, learning_rate=float(training["learning_rate"]), max_epochs=int(training["max_epochs"]),
                patience=int(training["patience"]), checkpoint_path=condition_dir / "best_model.pt",
                checkpoint_epochs=[], parameter_masks=masks,
            )
            save_csv(history, condition_dir / "training_history.csv")
            save_json(evaluate_model(model, test_loader, device), condition_dir / "final_metrics.json")
            torch.save({"epoch": int(history[-1]["epoch"]), "model_state_dict": model.state_dict()}, condition_dir / "final_model.pt")
            rows = trajectory_rows(seed, condition, history)
            all_rows.extend(rows)
            summaries.append(summarize_trajectory(seed, condition, rows, float(pruning_config["escape_gap_threshold"]), float(pruning_config["shortcut_phase_gap_threshold"])))

    save_csv(all_rows, output_dir / "all_trajectories.csv")
    save_csv(summaries, output_dir / "comparison_summary.csv")
    save_gap_plot(all_rows, output_dir / "shortcut_gap_curves.png")
    print(f"\nSaved one-shot pruning results to {output_dir}")


if __name__ == "__main__":
    main()
