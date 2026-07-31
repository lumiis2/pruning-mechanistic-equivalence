#!/usr/bin/env python3
"""Run iterative magnitude pruning with fixed initialization rewinding."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import create_splits  # noqa: E402
from src.models import SmallCNN  # noqa: E402
from src.pruning import (  # noqa: E402
    iterative_global_magnitude_prune,
    layer_mask_statistics,
    mask_statistics,
    prunable_parameters,
)
from src.training import (  # noqa: E402
    evaluate_counterfactual_sensitivity,
    evaluate_model,
    train_model,
)
from src.utils import load_yaml, save_csv, save_json, save_yaml, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "sanity_check.yaml"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="skip rounds already marked complete (default: true)",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="run the first seed for two rounds and two epochs",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def dense_final_checkpoint(seed_dir: Path) -> Path:
    history = read_csv(seed_dir / "training_history.csv")
    final_epoch = int(history[-1]["epoch"])
    return seed_dir / "epoch_states" / f"epoch_{final_epoch:02d}.pt"


def all_one_masks(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: torch.ones_like(parameter, dtype=torch.bool, device="cpu")
        for name, parameter in prunable_parameters(model).items()
    }


def trajectory_rows(seed: int, round_index: int, sparsity: float, history: list[dict]) -> list[dict]:
    rows = []
    for row in history:
        result = {
            "seed": seed,
            "round": round_index,
            "sparsity": sparsity,
            "epoch": row["epoch"],
            "loss": row["test_balanced_loss"],
            "overall_accuracy": row["test_balanced_overall_accuracy"],
            "aligned_accuracy": row["test_balanced_aligned_accuracy"],
            "conflicting_accuracy": row["test_balanced_conflicting_accuracy"],
            "worst_group_accuracy": row["test_balanced_worst_group_accuracy"],
            "shortcut_gap": row["test_balanced_shortcut_gap"],
        }
        for key in ("color_swap_sensitivity", "shape_swap_sensitivity"):
            if key in row:
                result[key] = row[key]
        rows.append(result)
    return rows


def summarize_trajectory(
    seed: int,
    round_index: int,
    sparsity: float,
    rows: list[dict],
    escape_threshold: float,
    shortcut_threshold: float,
) -> dict:
    trained = [row for row in rows if int(row["epoch"]) > 0]
    acquisition = next(
        (int(row["epoch"]) for row in trained if float(row["shortcut_gap"]) > shortcut_threshold),
        None,
    )
    escape = next(
        (
            int(row["epoch"])
            for row in trained
            if acquisition is not None
            and int(row["epoch"]) > acquisition
            and float(row["shortcut_gap"]) < escape_threshold
        ),
        None,
    )
    final = trained[-1]
    summary = {
        "seed": seed,
        "round": round_index,
        "sparsity": sparsity,
        "shortcut_acquisition_epoch": acquisition,
        "escape_epoch": escape,
        "shortcut_phase_duration": sum(
            float(row["shortcut_gap"]) > shortcut_threshold for row in trained
        ),
        # This is the discrete sum specified in the experimental protocol.
        "shortcut_auc": sum(float(row["shortcut_gap"]) for row in trained),
        "final_epoch": int(final["epoch"]),
        "final_overall_accuracy": float(final["overall_accuracy"]),
        "final_conflicting_accuracy": float(final["conflicting_accuracy"]),
        "final_worst_group_accuracy": float(final["worst_group_accuracy"]),
        "final_shortcut_gap": float(final["shortcut_gap"]),
    }
    for key in ("color_swap_sensitivity", "shape_swap_sensitivity"):
        if key in final:
            summary[f"final_{key}"] = float(final[key])
    return summary


def save_curves(rows: list[dict], output_path: Path) -> None:
    seeds = sorted({int(row["seed"]) for row in rows})
    figure, axes = plt.subplots(
        1, len(seeds), figsize=(5 * len(seeds), 4), sharey=True, constrained_layout=True
    )
    if len(seeds) == 1:
        axes = [axes]
    for axis, seed in zip(axes, seeds):
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        for round_index in sorted({int(row["round"]) for row in seed_rows}):
            selected = [row for row in seed_rows if int(row["round"]) == round_index]
            sparsity = float(selected[0]["sparsity"])
            axis.plot(
                [int(row["epoch"]) for row in selected],
                [float(row["shortcut_gap"]) for row in selected],
                label=f"r{round_index} ({sparsity:.1%})",
            )
        axis.axhline(0.2, color="black", linestyle="--", linewidth=0.8)
        axis.set(
            title=f"Seed {seed}", xlabel="Retraining epoch",
            ylabel="Shortcut gap", ylim=(-0.05, 1.05),
        )
        axis.grid(alpha=0.25)
    axes[-1].legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def validate_config(experiment: dict) -> None:
    if experiment["rewind_state"] != "init":
        raise ValueError("this IMP experiment intentionally supports rewind_state=init only")
    if experiment["mask_selection"] != "final":
        raise ValueError("this IMP experiment intentionally supports mask_selection=final only")
    if int(experiment["rounds"]) < 1:
        raise ValueError("imp.rounds must be at least 1")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    experiment = dict(config["imp"])
    if args.rounds is not None:
        experiment["rounds"] = args.rounds
    if args.seeds is not None:
        experiment["seeds"] = args.seeds
    training = dict(config["training"])
    output_dir = args.output_dir or resolve_path(experiment["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if args.smoke_test:
        experiment["seeds"] = [int(experiment["seeds"][0])]
        experiment["rounds"] = min(2, int(experiment["rounds"]))
        training["max_epochs"] = 2
        training["patience"] = 2
        training["num_workers"] = 0
        if args.output_dir is None:
            output_dir = output_dir / "smoke_test"
    validate_config(experiment)

    output_dir.mkdir(parents=True, exist_ok=True)
    effective_config = dict(config)
    effective_config["imp"] = experiment
    effective_config["training"] = training
    save_yaml(effective_config, output_dir / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} output_dir={output_dir}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    dense_root = resolve_path(config["multi_seed"]["output_dir"])
    pruning_config = config["one_shot_pruning"]
    all_rows: list[dict] = []
    summaries: list[dict] = []

    for raw_seed in experiment["seeds"]:
        seed = int(raw_seed)
        dense_seed_dir = dense_root / f"seed_{seed}"
        initialization_path = dense_seed_dir / "epoch_states" / "epoch_00.pt"
        dense_source_path = dense_final_checkpoint(dense_seed_dir)
        if not initialization_path.exists() or not dense_source_path.exists():
            raise FileNotFoundError(
                f"missing dense checkpoints for seed {seed} under {dense_seed_dir}"
            )
        initialization = torch.load(initialization_path, map_location="cpu")[
            "model_state_dict"
        ]
        source_model = SmallCNN(hidden_dim=int(config["model"]["hidden_dim"]))
        source_model.load_state_dict(
            torch.load(dense_source_path, map_location="cpu")["model_state_dict"]
        )
        current_masks = all_one_masks(source_model)
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        set_seed(seed)
        splits = create_splits(seed=seed, data_config=config["data"], variant="original")
        loader_kwargs = {
            "batch_size": int(training["batch_size"]),
            "num_workers": int(training["num_workers"]),
            "pin_memory": device.type == "cuda",
        }
        val_loader = DataLoader(splits["val_biased"], shuffle=False, **loader_kwargs)
        test_loader = DataLoader(splits["test_balanced"], shuffle=False, **loader_kwargs)
        sensitivity_evaluator = None
        if bool(experiment["measure_counterfactual_sensitivity"]):
            sensitivity_evaluator = partial(
                evaluate_counterfactual_sensitivity,
                dataset=splits["test_balanced"],
                device=device,
                batch_size=int(training["batch_size"]),
                num_workers=int(training["num_workers"]),
            )

        for round_index in range(1, int(experiment["rounds"]) + 1):
            round_dir = seed_dir / f"round_{round_index:02d}"
            complete_path = round_dir / "status.json"
            if args.resume and complete_path.exists():
                with complete_path.open(encoding="utf-8") as handle:
                    status = json.load(handle)
                if status.get("status") == "complete":
                    print(f"seed={seed} round={round_index:02d} already complete; resuming")
                    mask_checkpoint = torch.load(round_dir / "mask.pt", map_location="cpu")
                    current_masks = mask_checkpoint["masks"]
                    source_model.load_state_dict(
                        torch.load(round_dir / "selection_model.pt", map_location="cpu")[
                            "model_state_dict"
                        ]
                    )
                    rows = read_csv(round_dir / "trajectory.csv")
                    with (round_dir / "summary.json").open(encoding="utf-8") as handle:
                        summary = json.load(handle)
                    all_rows.extend(rows)
                    summaries.append(summary)
                    continue

            round_dir.mkdir(parents=True, exist_ok=True)
            current_masks = iterative_global_magnitude_prune(
                source_model, current_masks, float(experiment["pruning_fraction"])
            )
            statistics = mask_statistics(current_masks)
            per_layer = layer_mask_statistics(current_masks)
            torch.save(
                {
                    "seed": seed,
                    "round": round_index,
                    "masks": current_masks,
                    "statistics": statistics,
                    "per_layer_statistics": per_layer,
                },
                round_dir / "mask.pt",
            )
            save_json(
                {"global": statistics, "per_layer": per_layer},
                round_dir / "mask_statistics.json",
            )
            save_json(
                {"status": "running", "seed": seed, "round": round_index},
                complete_path,
            )
            print(
                f"\nseed={seed} round={round_index:02d}/{experiment['rounds']} "
                f"sparsity={statistics['sparsity']:.4f}"
            )

            model = SmallCNN(hidden_dim=int(config["model"]["hidden_dim"]))
            model.load_state_dict(initialization)
            set_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                splits["train_biased"], shuffle=True, generator=generator, **loader_kwargs
            )
            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                learning_rate=float(training["learning_rate"]),
                max_epochs=int(training["max_epochs"]),
                patience=int(training["patience"]),
                checkpoint_path=round_dir / "best_model.pt",
                checkpoint_epochs=[],
                sensitivity_evaluator=sensitivity_evaluator,
                parameter_masks=current_masks,
            )
            save_csv(history, round_dir / "training_history.csv")
            rows = trajectory_rows(
                seed, round_index, float(statistics["sparsity"]), history
            )
            save_csv(rows, round_dir / "trajectory.csv")
            summary = summarize_trajectory(
                seed=seed,
                round_index=round_index,
                sparsity=float(statistics["sparsity"]),
                rows=rows,
                escape_threshold=float(pruning_config["escape_gap_threshold"]),
                shortcut_threshold=float(pruning_config["shortcut_phase_gap_threshold"]),
            )
            save_json(summary, round_dir / "summary.json")

            final_metrics = evaluate_model(model, test_loader, device)
            if sensitivity_evaluator is not None:
                final_metrics.update(sensitivity_evaluator(model))
            save_json(final_metrics, round_dir / "final_metrics.json")
            selection_checkpoint = {
                "seed": seed,
                "round": round_index,
                "epoch": int(history[-1]["epoch"]),
                "model_state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                },
                "mask_statistics": statistics,
            }
            torch.save(selection_checkpoint, round_dir / "selection_model.pt")

            source_model.load_state_dict(selection_checkpoint["model_state_dict"])
            all_rows.extend(rows)
            summaries.append(summary)
            save_json(
                {
                    "status": "complete",
                    "seed": seed,
                    "round": round_index,
                    "sparsity": statistics["sparsity"],
                },
                complete_path,
            )
            # Keep aggregate files useful even if Colab disconnects later.
            save_csv(all_rows, output_dir / "all_trajectories.csv")
            save_csv(summaries, output_dir / "round_summary.csv")

    save_csv(all_rows, output_dir / "all_trajectories.csv")
    save_csv(summaries, output_dir / "round_summary.csv")
    save_curves(all_rows, output_dir / "imp_shortcut_curves.png")
    print(f"\nIMP complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
