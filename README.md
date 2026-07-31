# Pruning, shortcuts, and mechanistic equivalence

This repository studies whether models with similar predictive performance use
the same internal mechanism, and whether mechanistic differences predict
responses to interventions or distribution shifts that are not visible in
current accuracy.

The controlled task is Shape–Color classification. Shape determines the label;
color is correlated with the label in training and acts as a shortcut. Because
shape and color can be changed independently, the project can track shortcut
acquisition, transition to robust prediction, and causal sensitivity to each
feature.

## Research direction

The current question is:

> How do pruning, weight rewinding, and optimization trajectory affect the
> transition from shortcut-based to robust prediction, and do the resulting
> mechanistic differences predict behavior under new interventions or shifts?

Final accuracy is not treated as evidence of mechanistic equivalence. The next
stage will compare behaviorally matched dense and sparse models, characterize
their internal organization, and test whether those measurements predict
held-out behavior better than accuracy, sparsity, and shortcut-gap metrics.

The full motivation and experimental design are in
[`research-proposal.md`](research-proposal.md).

## Preliminary findings

- The color-only, shape-only, and uncorrelated-color controls validate the
  dataset and evaluation pipeline.
- All five dense seeds acquire the color shortcut in the first epoch, but their
  transitions differ substantially. Three become robust, one remains partially
  dependent on color, and one remains strongly shortcut-dependent.
- At 50% one-shot sparsity, trained magnitude masks recover robust behavior,
  while random masks do not reliably escape the shortcut.
- Differences between checkpoint-selected masks become clear between 80% and
  90% sparsity.
- With a fixed robust mask at 80%, rewinding state changes the learning path;
  all tested conditions still reach a robust final solution.
- IMP preserves shortcut acquisition and later robust prediction through the
  tested range up to 89.3% sparsity. Higher-sparsity rounds extend this analysis
  to approximately 94.5%.

These results establish a set of models with different trajectories and sparse
structures. They do not yet establish whether those models use meaningfully
different causal mechanisms.

## Repository layout

```text
configs/       Experiment configuration
notebooks/     Ordered reports for each experiment
scripts/       Reproducible experiment entry points
src/           Data, models, training, evaluation, and pruning code
tests/         Dataset, model, metric, and pruning tests
outputs/       Generated artifacts; excluded from Git
```

The notebooks are indexed in [`notebooks/README.md`](notebooks/README.md). By
default, they read existing outputs and do not start training.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
PYTHONPATH=. pytest -q
```

All experiment parameters are defined in `configs/sanity_check.yaml`.

## Main commands

```bash
python scripts/run_sanity_check.py --config configs/sanity_check.yaml
python scripts/run_multiseed_dynamics.py --config configs/sanity_check.yaml
python scripts/run_one_shot_pruning.py --config configs/sanity_check.yaml
python scripts/run_sparsity_sweep.py --config configs/sanity_check.yaml
python scripts/run_rewinding.py --config configs/sanity_check.yaml
python scripts/run_imp.py --config configs/sanity_check.yaml
```

IMP supports resumption. Completed rounds are skipped, while an interrupted
round is recomputed deterministically. Generated checkpoints, masks, metrics,
predictions, and figures are written under `outputs/` and are not versioned.
