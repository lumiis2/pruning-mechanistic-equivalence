# Mechanistic discovery smoke validation

Date: 2026-08-27

## Scope

Slurm job `6627` validated the existing mechanistic-characterization pipeline
as an operational smoke test. It used only discovery model `dense-s42`, 64
balanced samples, and three probe epochs. No evaluation model or output was
accessed.

The smoke output is under:

```text
outputs/mechanistic_characterization_validation/smoke_test/
```

## Slurm result

- State: `COMPLETED`
- Exit code: `0:0`
- Elapsed time: 1 minute 8 seconds
- Node: `viper01`
- Allocated GPU: NVIDIA GeForce GTX 1080 Ti
- Effective PyTorch: `2.13.0+cu126` from `.venv310`
- CUDA matrix multiplication: passed
- Error log: empty

## Artifact validation

The smoke produced the expected configuration, activation cache, probes,
causal metrics, summary, aggregate summary, and completion marker.

Automated structural checks confirmed:

- experiment split was `discovery`;
- the only model was `dense-s42`;
- the sample limit was 64;
- all six layer/target probes were present;
- every probe used 36 train, 12 validation, and 16 test samples;
- all 120 causal measurements were finite and used 64 samples;
- no evaluation directory or artifact was created.

## Infrastructure finding

Jobs `6621` and `6624` failed immediately on `viper03`. PyTorch
`2.13.0+cu126` produced CUDA error 804, and the cluster PyTorch 2.5.1 module
reported CUDA unavailable. Job scripts must run a CUDA preflight before the
experiment. The scheduler policy rejected explicit `--nodelist` and
`--exclude`, so a failed preflight should terminate cleanly and the job can be
resubmitted.

Using the generic request `--gres=gpu:1`, as in the successful IMP job, placed
job `6627` on the working `viper01` node.

## Scientific status

This smoke validates execution only and must not be treated as a scientific
result. Before the full discovery run, the implementation still needs:

- aligned, conflicting, worst-group, and shortcut-gap metrics for the balanced
  hidden head;
- signed counterfactual effects in addition to absolute effects;
- counterfactual and patching results separated by the four groups;
- explicit interpretation of shape swaps, because changing shape changes the
  true class;
- unit tests for the new metrics.

The full discovery characterization and all evaluation runs remain pending.
