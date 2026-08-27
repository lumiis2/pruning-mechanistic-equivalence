# Session handoff

## Purpose

This file preserves the minimum durable context needed to continue work when a
Codex chat or local CLI session is unavailable. Update it after sessions that
materially change project state, decisions, results, validation, or next steps.
Never record credentials, tokens, private keys, or other secrets here.

At the start of a new chat, read:

1. `AGENTS.md`
2. `docs/PROJECT_CONTEXT.md`
3. `docs/SESSION_HANDOFF.md`

## Current state

The repository implements the initial deterministic Shape--Color research
pipeline: dense baselines and controls, multi-seed learning dynamics, one-shot
pruning, a sparsity sweep, weight rewinding, and resumable iterative magnitude
pruning. The next scientific stage is mechanistic characterization and
prospective causal validation across behaviorally matched model cohorts.

## Current decisions

- Keep the repository under the persistent QNAP-backed home.
- Keep active Codex runtime state in
  `/run/user/836002271/codex-home` on viper08. The QNAP previously failed when
  Codex attempted to create its Unix socket and SQLite state there.
- Do not point the complete `CODEX_HOME` at this repository or `~/.codex` on
  the QNAP.
- The runtime path is compatible but temporary. A persistent local filesystem
  directory supplied by the administrator remains the ideal long-term home for
  literal CLI history and authentication state.
- Use this file for durable technical context. It complements chat sync and
  `codex resume`; it is not a raw transcript or credential backup.

## Changes from the continuity setup

- Added `AGENTS.md` with scientific invariants, experiment order, validation
  commands, repository hygiene, and remote Codex constraints.
- Added `docs/PROJECT_CONTEXT.md` with the research question, task, model,
  measurements, pruning semantics, preliminary findings, and planned work.
- Added this handoff file.
- An empty private `~/.codex` directory was created before the QNAP limitation
  was clarified. It is inactive because `CODEX_HOME` points to `/run/user/...`;
  no credential was copied into it.

## Validation and environment notes

- The remote Codex CLI was detected through the VS Code extension.
- ChatGPT login and `codex resume`, `codex resume --last`, and
  `codex resume --all` were available before the latest reconnection.
- Reinstalling VS Code and reconnecting preserved the current interface chat,
  indicating that this chat history is synchronized separately from local CLI
  runtime state.
- After the reinstall, running `codex` in the user's terminal initially returned
  `Command 'codex' not found`. The Ubuntu suggestion to install `snap codex`
  must not be followed because it is not the verified OpenAI installation path.
- The official CLI bundled with the current remote VS Code extension was found
  at
  `/home/luisa.lopes/.vscode-server/extensions/openai.chatgpt-26.810.52044-linux-x64/bin/linux-x86_64/codex`.
  Direct execution reported `codex-cli 0.148.0-alpha.9` and
  `Logged in using ChatGPT`, confirming that the binary and existing
  authentication state survived.
- The terminal failure was therefore a `PATH` issue, not loss of the CLI or
  login. Extension updates can change the versioned binary path.
- The Python tests could not run in the available environment: `.venv/bin/python`
  was absent and `.venv310` lacked `typing_extensions` while importing pytest.
- Pre-existing untracked `.venv310/`, `logs/`, and `scripts/slurm/` must remain
  untouched unless explicitly included in a task.

## CLI continuity test

Start a CLI session from the repository root:

```bash
cd /mnt/users/luisa.lopes/pruning-mechanistic-equivalence
echo "$CODEX_HOME"
export PATH="$HOME/.vscode-server/extensions/openai.chatgpt-26.810.52044-linux-x64/bin/linux-x86_64:$PATH"
command -v codex
codex login status
codex
```

The `export PATH=...` command repairs the current terminal only. For a stable
CLI independent of the extension's versioned directory, use the official
macOS/Linux installer:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

Do not use the Ubuntu `snap install codex` suggestion.

After a normal disconnect and reconnect:

```bash
cd /mnt/users/luisa.lopes/pruning-mechanistic-equivalence
codex resume --last
```

Use `codex resume` to choose a session for this repository and
`codex resume --all` to include sessions from other directories. CLI resumption
depends on the corresponding state still existing under `CODEX_HOME`.

## Next steps

- Validate the pending mechanistic-metric changes with the remote `.venv310`.
  The local machine does not need a separate experiment environment.
- Run `scripts/slurm/run_mechanistic_smoke_generic.sbatch`, which now performs
  the CUDA preflight, full unit-test suite, and discovery-only smoke test in the
  same Slurm allocation.
- The job finishes by running `scripts/validate_mechanistic_smoke.py`; it must
  validate the artifact schema, finite metrics, balanced counts, all four
  groups, and all 120 causal measurements before reporting success.
- Inspect the new smoke artifacts for balanced-head aligned/conflicting/worst-
  group/shortcut-gap metrics and signed, absolute, flip-rate, and per-group
  causal effects.
- If the smoke passes, document its job ID and artifact validation before
  preparing the full discovery run. Do not run or inspect evaluation.

## Pending local changes

The current local work adds the metrics required after smoke job `6627`:

- balanced probe aligned and conflicting accuracy, worst-group accuracy, and
  shortcut gap;
- signed and absolute target-class probability and logit-margin changes;
- causal metrics separated across all four Shape--Color groups;
- explicit shape-swap semantics relative to the original shape class;
- unit tests covering shortcut metrics and signed/grouped causal effects.

These changes compile syntactically but have not yet been executed with PyTorch
locally. Validation is intentionally deferred to the existing remote
`.venv310`, followed by the discovery smoke job. Replace this pending note with
the recorded validation result after the job completes.
