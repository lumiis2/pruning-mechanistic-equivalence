# Project context

## Central question

This project asks whether dense and sparse classifiers with matched predictive
performance rely on the same causal mechanism, and whether measured
mechanistic differences predict responses to interventions or distribution
shifts that ordinary accuracy does not reveal.

The work is an evaluation framework, not a proposal for a new pruning method.
Its central standard is prospective validity: a mechanistic explanation is
useful only if it transfers across models or predicts held-out behavior beyond
behavioral baselines such as accuracy, sparsity, shortcut gap, and trajectory
statistics.

## Controlled task

`ShapeColorDataset` generates deterministic 32x32 RGB images containing a
circle or square in red or blue. Shape is the label (`circle=0`, `square=1`),
while color is correlated with the label in biased train and validation data.
The balanced test set contains equal counts of:

| Group | Example | Label | Relationship |
| --- | --- | --- | --- |
| 0 | red circle | 0 | aligned |
| 1 | blue circle | 0 | conflicting |
| 2 | red square | 1 | conflicting |
| 3 | blue square | 1 | aligned |

Size and position are sampled independently of group. Circles and squares use
the same sampled continuous area. Counterfactual rendering swaps exactly one of
shape or color while holding size and position fixed. Dataset controls include
color-only, shape-only, and uncorrelated-color variants.

## Model and measurements

`SmallCNN` has two convolution/ReLU/max-pool stages followed by a hidden linear
layer and a two-class output. It can expose `conv1`, `conv2`, and `hidden`
activations for the planned mechanistic work.

Training uses Adam, cross-entropy, deterministic seeds, and early stopping on
biased-validation loss. The pipeline records epoch-zero behavior and every
trained epoch. Core measurements are overall, aligned, conflicting, per-group,
worst-group accuracy, and shortcut gap (`aligned - conflicting`). Optional
counterfactual metrics measure class-probability changes and prediction flips
under isolated color and shape swaps.

Behavioral thresholds define seed-specific shortcut, late-shortcut,
transition, and robust checkpoints. These semantic checkpoints make it possible
to compare equivalent phases across trajectories that evolve at different
rates.

## Pruning semantics

Prunable parameters are Conv2d and Linear weights; biases are excluded.
One-shot masks retain the globally largest magnitudes or an equal number of
seeded random weights. Iterative magnitude pruning removes a configured fraction
of the currently active weights globally, never allowing a pruned weight to
return. During masked retraining, gradients are masked and masks are reapplied
after every optimizer step.

The configured experiments are:

| Stage | Purpose | Main output dependency |
| --- | --- | --- |
| Dense sanity check | Validate shortcut learning and controls | None |
| Multi-seed dynamics | Characterize shortcut acquisition and escape | Dataset/model pipeline |
| One-shot pruning | Compare random and checkpoint-selected masks | Dense functional checkpoints |
| Sparsity sweep | Locate mask-selection differences at 50%, 80%, 90% | One-shot runs |
| Rewinding | Fix an 80% robust mask and vary weight state | Dense functional checkpoints |
| IMP | Repeatedly prune 20%, rewind to initialization, and retrain | Dense initialization/final states |

The canonical configuration currently uses seeds 42--46 for dense dynamics and
seeds 42, 45, and 46 for pruning experiments. IMP is configured for 13 rounds,
ending near 94.5% sparsity. Each completed IMP round writes a status marker,
mask, selection model, trajectory, summary, and final metrics; interrupted
rounds can be recomputed while completed rounds are resumed.

## Established evidence

The repository documentation and generated reports currently support these
preliminary conclusions:

- Dataset controls behave as intended.
- All five dense seeds first acquire the color shortcut, but later trajectories
  differ: three become robust, one remains partly color-dependent, and one
  remains strongly shortcut-dependent.
- At 50% one-shot sparsity, trained magnitude masks recover robust behavior in
  eligible seeds, whereas random masks do not reliably escape the shortcut.
- Checkpoint-selected mask differences become pronounced between 80% and 90%
  sparsity.
- With a fixed robust 80% mask, rewinding changes the learning path, although
  tested conditions converge to robust final behavior.
- IMP preserves shortcut acquisition followed by robust behavior through the
  tested range near 89.3%; later rounds extend the study toward 94.5%.

These are behavioral and structural findings. They do not yet demonstrate
mechanistic equivalence or contingency.

## Next scientific stage

The planned work is to form behaviorally matched cohorts across seeds, masks,
sparsities, and rewinding states, then characterize and causally test their
internal mechanisms. Candidate analyses include balanced linear probes,
counterfactual activation comparisons, activation replacement, channel or
subspace ablation, and cross-model intervention transfer.

Discovery models and conditions must be separated from held-out evaluation.
Mechanistic measurements should be judged by whether they improve prediction of
new shifts, failures, or intervention transfer beyond accuracy, sparsity,
shortcut gap, and learning-dynamics baselines. Probe decodability alone is not
evidence that a feature is causally used.

## Paths and commands

- `configs/sanity_check.yaml`: canonical experiment parameters.
- `src/data/`: deterministic generation, interventions, and data figures.
- `src/models/`: CNN and intermediate activation interface.
- `src/training/`: training, evaluation, functional checkpoints, and plots.
- `src/pruning.py`: global one-shot and iterative mask operations.
- `scripts/`: reproducible experiment entry points.
- `notebooks/`: ordered reports; numbered notebooks do not train by default.
- `outputs/`: generated, ignored artifacts and checkpoints.
- `tests/`: dataset, metrics, model, and pruning unit tests.

Typical setup and validation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
PYTHONPATH=. python -m pytest -q
```

Run experiment scripts in the order listed in `AGENTS.md`. Full training can be
expensive; use targeted tests or the IMP smoke test before launching long jobs.

## Remote development constraint

The remote `/home/luisa.lopes` is persistent QNAP-backed storage. It is suitable
for the repository but has failed for Codex Unix sockets and SQLite state.
Consequently, the remote shell exports:

```bash
CODEX_HOME=/run/user/836002271/codex-home
```

That path is local to viper08 and supports the required filesystem operations,
but it is runtime storage and may be cleared on reboot or full session cleanup.
The shell recreates it with mode `700`; authentication files should remain mode
`600`. Do not place credentials in this repository. The durable solution is a
persistent local/scratch path with normal Linux socket and SQLite support,
provided by the infrastructure administrator, and then changing `CODEX_HOME` to
that path.
