# Repository instructions

## Purpose

This repository studies whether behaviorally similar dense and pruned neural
networks use equivalent internal mechanisms. The controlled Shape--Color task
uses shape as the true label and correlated color as a shortcut.

Read `docs/PROJECT_CONTEXT.md` and `docs/SESSION_HANDOFF.md` before making
substantive changes. Treat `research-proposal.md` as the scientific
specification and `configs/sanity_check.yaml` as the canonical experiment
configuration.

## Environment and validation

- Use Python 3.10 or newer and install dependencies from `requirements.txt`.
- Run commands from the repository root.
- Set `PYTHONPATH=.` when running tests or importing `src` interactively.
- Run `PYTHONPATH=. python -m pytest -q` after changes to shared code.
- Prefer a targeted smoke test before launching a full experiment. IMP provides
  `python scripts/run_imp.py --smoke-test` for this purpose.
- Do not start long training jobs, GPU jobs, or Slurm submissions unless the
  user explicitly asks for them.

## Scientific invariants

- Preserve the dataset conventions in `src/data/dataset.py`: shape determines
  the label, color is the shortcut, and group is `2 * label + color`.
- Keep train and validation biased while the canonical test split remains
  exactly balanced across all four groups.
- Hold nuisance variables fixed in counterfactual color/shape interventions.
- Report aligned, conflicting, worst-group, and shortcut-gap metrics; overall
  accuracy alone is insufficient for this project.
- Keep functional checkpoints behavior-defined rather than tied to fixed epoch
  numbers.
- Pruning applies only to Conv2d and Linear weights. Biases are intentionally
  excluded, masks are global, and pruned weights must remain zero during
  retraining.
- Preserve seeded, deterministic behavior. Do not silently change seed,
  shuffling, pruning tie behavior, split construction, or early stopping.
- Avoid claims of mechanistic equivalence based only on accuracy, sparsity,
  probes, masks, or trajectories. The proposal requires causal and prospective
  validation on held-out models or conditions.

## Experiment workflow

The intended order is:

1. `scripts/run_sanity_check.py`
2. `scripts/run_multiseed_dynamics.py`
3. `scripts/run_one_shot_pruning.py`
4. `scripts/run_sparsity_sweep.py`
5. `scripts/run_rewinding.py`
6. `scripts/run_imp.py`

Later stages consume checkpoints or summaries produced by earlier stages. Keep
scripts reproducible from the YAML configuration and write generated artifacts
under `outputs/`. IMP is resumable: completed rounds are skipped and incomplete
rounds are recomputed deterministically.

## Repository hygiene

- Do not commit virtual environments, logs, generated outputs, checkpoints,
  credentials, or machine-local agent state.
- Preserve existing untracked work, especially `scripts/slurm/`, unless a task
  explicitly includes it.
- Do not edit notebook outputs merely by opening or executing notebooks.
- Keep numbered notebooks as short reports that read existing outputs by
  default; training must require an explicit `RUN_EXPERIMENT = True` choice.
- Update `docs/PROJECT_CONTEXT.md` when the scientific question, experiment
  graph, established findings, or next research stage changes materially.
- Update `docs/SESSION_HANDOFF.md` after a session changes project state,
  decisions, results, validation status, or next steps. Never record secrets.

## Remote Codex environment

The remote home is backed by QNAP storage, which does not reliably support the
Unix sockets and SQLite behavior required by Codex. `CODEX_HOME` therefore
points to `/run/user/836002271/codex-home` on viper08. Do not move the complete
Codex state back to `~/.codex`; doing so can leave the extension stuck during
startup. The runtime location is compatible but temporary, so authentication
and session history can disappear after a reboot or full session cleanup. A
persistent local filesystem path supplied by the infrastructure administrator
is the preferred long-term fix.
