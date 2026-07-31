#!/usr/bin/env python3
"""Fix a robust magnitude mask and vary only the weight rewinding state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import create_splits
from src.models import SmallCNN
from src.pruning import global_magnitude_masks, mask_statistics
from src.training import evaluate_model, train_model
from src.utils import save_csv, save_json, save_yaml, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "sanity_check.yaml")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def state_path(seed_dir: Path, state: str) -> Path:
    if state == "init":
        return seed_dir / "epoch_states" / "epoch_00.pt"
    return seed_dir / "functional_checkpoints" / f"{state}.pt"


def trajectory_rows(seed: int, state: str, history: list[dict]) -> list[dict]:
    return [{
        "seed": seed,
        "rewind_state": state,
        "epoch": row["epoch"],
        "overall_accuracy": row["test_balanced_overall_accuracy"],
        "aligned_accuracy": row["test_balanced_aligned_accuracy"],
        "conflicting_accuracy": row["test_balanced_conflicting_accuracy"],
        "worst_group_accuracy": row["test_balanced_worst_group_accuracy"],
        "shortcut_gap": row["test_balanced_shortcut_gap"],
    } for row in history]


def summarize(seed: int, state: str, rows: list[dict], escape: float, shortcut: float) -> dict:
    trained = [row for row in rows if int(row["epoch"]) > 0]
    acquisition = next((int(row["epoch"]) for row in trained if float(row["shortcut_gap"]) > shortcut), None)
    escape_epoch = next((int(row["epoch"]) for row in trained
                         if acquisition is not None and int(row["epoch"]) > acquisition
                         and float(row["shortcut_gap"]) < escape), None)
    final = trained[-1]
    return {
        "seed": seed,
        "rewind_state": state,
        "shortcut_acquisition_epoch": acquisition,
        "escape_epoch": escape_epoch,
        "shortcut_phase_duration": sum(float(row["shortcut_gap"]) > shortcut for row in trained),
        "shortcut_gap_area": sum(float(row["shortcut_gap"]) for row in trained),
        "final_epoch": int(final["epoch"]),
        "final_conflicting_accuracy": float(final["conflicting_accuracy"]),
        "final_worst_group_accuracy": float(final["worst_group_accuracy"]),
        "final_shortcut_gap": float(final["shortcut_gap"]),
    }


def save_plot(rows: list[dict], path: Path) -> None:
    seeds = sorted({int(row["seed"]) for row in rows})
    states = ["init", "shortcut", "transition", "robust"]
    figure, axes = plt.subplots(1, len(seeds), figsize=(5 * len(seeds), 4), sharey=True, constrained_layout=True)
    if len(seeds) == 1:
        axes = [axes]
    for axis, seed in zip(axes, seeds):
        for state in states:
            selected = [row for row in rows if int(row["seed"]) == seed and row["rewind_state"] == state]
            axis.plot([int(row["epoch"]) for row in selected],
                      [float(row["shortcut_gap"]) for row in selected], label=state)
        axis.axhline(0.2, color="black", linestyle="--", linewidth=0.8)
        axis.set(title=f"Seed {seed}", xlabel="Retraining epoch", ylabel="Shortcut gap", ylim=(-0.05, 1.05))
        axis.grid(alpha=0.25)
    axes[-1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    experiment = config["rewinding"]
    pruning = config["one_shot_pruning"]
    training = config["training"]
    dense_root = resolve_path(config["multi_seed"]["output_dir"])
    output_dir = resolve_path(experiment["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, output_dir / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows, summaries = [], []

    for raw_seed in experiment["seeds"]:
        seed = int(raw_seed)
        dense_seed_dir = dense_root / f"seed_{seed}"
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        mask_source = state_path(dense_seed_dir, str(experiment["mask_stage"]))
        source_model = SmallCNN(hidden_dim=int(config["model"]["hidden_dim"]))
        source_model.load_state_dict(torch.load(mask_source, map_location="cpu")["model_state_dict"])
        masks = global_magnitude_masks(source_model, float(experiment["sparsity"]))
        torch.save({"source": str(mask_source), "masks": masks, "statistics": mask_statistics(masks)}, seed_dir / "fixed_mask.pt")

        set_seed(seed)
        splits = create_splits(seed=seed, data_config=config["data"], variant="original")
        loader_kwargs = {"batch_size": int(training["batch_size"]), "num_workers": int(training["num_workers"]), "pin_memory": device.type == "cuda"}
        val_loader = DataLoader(splits["val_biased"], shuffle=False, **loader_kwargs)
        test_loader = DataLoader(splits["test_balanced"], shuffle=False, **loader_kwargs)

        for state in experiment["states"]:
            print(f"\nseed={seed} rewind_state={state} sparsity={float(experiment['sparsity']):.2f} device={device}")
            condition_dir = seed_dir / str(state)
            condition_dir.mkdir(parents=True, exist_ok=True)
            model = SmallCNN(hidden_dim=int(config["model"]["hidden_dim"]))
            checkpoint = torch.load(state_path(dense_seed_dir, str(state)), map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            set_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(splits["train_biased"], shuffle=True, generator=generator, **loader_kwargs)
            history = train_model(
                model, train_loader, val_loader, test_loader, device,
                float(training["learning_rate"]), int(training["max_epochs"]), int(training["patience"]),
                condition_dir / "best_model.pt", [], parameter_masks=masks,
            )
            save_csv(history, condition_dir / "training_history.csv")
            save_json(evaluate_model(model, test_loader, device), condition_dir / "final_metrics.json")
            rows = trajectory_rows(seed, str(state), history)
            all_rows.extend(rows)
            summaries.append(summarize(seed, str(state), rows, float(pruning["escape_gap_threshold"]), float(pruning["shortcut_phase_gap_threshold"])))

    save_csv(all_rows, output_dir / "all_trajectories.csv")
    save_csv(summaries, output_dir / "comparison_summary.csv")
    save_plot(all_rows, output_dir / "rewinding_curves.png")
    print(f"\nSaved rewinding experiment to {output_dir}")


if __name__ == "__main__":
    main()
