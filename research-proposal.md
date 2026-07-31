# Pruning, shortcuts, and the predictive validity of mechanistic explanations

## Motivation

Models trained with spurious correlations can follow different learning
trajectories while reaching similar final performance. In the Shape–Color task
used in this project, color is an easy but unreliable predictor and shape is the
label-defining feature. Dense and sparse networks usually learn color first and
may later transition to shape-based prediction. Seed, mask selection, sparsity,
and weight rewinding change the timing and outcome of that transition.

Different trajectories or internal organizations are not automatically
scientifically meaningful. If two models behave identically under every
relevant condition, identifying different neurons or pathways is descriptive
rather than predictive. A mechanistic account becomes stronger when it
generalizes across models or predicts behavior that was not used to construct
the account.

## Research question

> Can mechanistic differences between behaviorally equivalent models predict
> different responses to interventions or distribution shifts that are not
> revealed by their current accuracy?

The operational question is:

> How do pruning, weight rewinding, and optimization trajectory affect the
> internal transition from shortcut-based to robust prediction, and do the
> resulting mechanistic differences predict previously unmeasured behavior?

## Preliminary evidence

The completed experiments establish the behavioral setting:

1. Dataset controls behave as expected. Color alone yields perfect aligned and
   zero conflicting accuracy; shape alone and uncorrelated color yield robust
   classification.
2. All five dense seeds acquire the color shortcut in the first epoch. Their
   later behavior varies: three reach a robust solution, one remains partially
   color-dependent, and one remains strongly shortcut-dependent.
3. At 50% one-shot sparsity, magnitude masks selected from trained checkpoints
   escape the shortcut in all eligible seeds. Random masks do not.
4. Mask-selection effects become pronounced between 80% and 90% sparsity.
5. With a fixed robust mask at 80%, initialization and shortcut rewinding
   reproduce the shortcut-to-robust transition, whereas transition and robust
   states remain robust during retraining.
6. IMP preserves immediate shortcut acquisition and eventual robust prediction
   through 89.3% sparsity. Intermediate sparsity often shortens the shortcut
   phase; a weak degradation signal appears near the highest tested level.

These findings show that final performance can conceal substantial differences
in learning history and sparse structure. They do not yet show that internal
differences predict new behavior.

## Competing hypotheses

### Mechanistic convergence

Behaviorally robust models converge to a shared causal mechanism despite
different seeds, masks, and rewinding trajectories.

Predictions:

- causal components overlap across models;
- interventions discovered in one model transfer to others;
- mechanistic similarity increases after the robust transition;
- pruning removes redundant parameters while retaining a common causal core.

### Mechanistic contingency

Behaviorally robust models implement shape-based prediction through mechanisms
that depend on their training trajectory.

Predictions:

- causal explanations transfer poorly across seeds or masks;
- behaviorally matched models respond differently to new shifts;
- mechanistic similarity predicts these differences better than final
  performance or sparsity;
- conclusions from one trained model do not generalize reliably to the model
  class.

The hypotheses are not exhaustive. A mixed result is possible: some components
may form a stable shared core while others remain trajectory-specific.

## Experimental design

### 1. Learning dynamics

The first stage characterizes each trajectory using:

- shortcut acquisition epoch;
- shortcut escape epoch;
- duration of the shortcut phase;
- area under the shortcut-gap curve;
- final aligned, conflicting, and worst-group accuracy;
- sensitivity to isolated color and shape changes.

Functional checkpoints are defined by behavior rather than fixed epoch:
shortcut, late shortcut, transition, and robust. This avoids comparing models
at chronologically equal but functionally different points.

### 2. Behaviorally matched model cohorts

Models will be selected across seeds, mask-selection times, sparsity levels, and
rewinding states. Comparisons will prioritize pairs or groups with matched final
overall, conflicting, and worst-group accuracy but different trajectories or
sparse structures.

This matching step separates mechanistic questions from trivial performance
differences. Dense models and random-mask controls remain part of the analysis.

### 3. Mechanistic characterization

The analysis will measure where shape and color information is present and how
it affects the output.

- Linear probes will measure decodability of shape and color in intermediate
  activations using balanced data.
- Counterfactual input pairs will isolate sensitivity to shape and color.
- Activation replacement and ablation will test the causal contribution of
  layers, channels, or subspaces identified by the representational analysis.
- Cross-model comparison will quantify overlap and stability of the identified
  mechanisms.

Probe accuracy will not be interpreted as causal use. A feature may remain
represented while being ignored by the original classifier.

### 4. Prospective validation

Mechanisms will be identified on a discovery set of models and conditions, then
evaluated on held-out models or shifts. Candidate tests include:

- weaker, stronger, or reversed color–label correlation;
- partial corruption of shape;
- simultaneous changes in shortcut strength and shape quality;
- transfer of an ablation or activation intervention between models;
- additional pruning;
- balanced fine-tuning after mechanistic characterization.

The key comparison is predictive. A baseline using accuracy, sparsity,
shortcut gap, and trajectory metrics will be compared with a model that also
uses mechanistic similarity or causal-effect measurements. Mechanistic evidence
is informative only if it improves prediction on held-out behavior.

## Evaluation criteria

A mechanistic explanation will count as evidence beyond description if it
satisfies at least one criterion:

1. it identifies a causal mechanism stable across behaviorally equivalent
   models;
2. it predicts which matched models fail under a held-out shift;
3. it predicts whether an intervention transfers between models;
4. it explains held-out differences not captured by accuracy, sparsity,
   shortcut gap, or learning-dynamics metrics.

Results will be reported across seeds. Discovery and evaluation conditions will
be separated before fitting probes or choosing interventions to limit
selection bias.

## Falsification and null results

The central claim will be weakened if mechanistic measurements are unstable,
fail to transfer, or add no predictive value beyond behavioral baselines. Other
informative outcomes include:

- robust models converge to a small invariant causal mechanism;
- pruning changes optimization speed but not the final mechanism;
- representational differences exist without measurable behavioral
  consequences;
- trajectory metrics alone predict held-out behavior as well as mechanistic
  measurements;
- meaningful divergence appears only beyond a critical sparsity or shift
  severity.

## Expected contribution

The intended contribution is an evaluation framework rather than a new pruning
method. The project will provide:

1. a controlled testbed for shortcut-to-robust transitions;
2. matched dense and sparse model cohorts with distinct training histories;
3. a protocol combining representational and causal analysis;
4. prospective tests of explanation stability and intervention transfer;
5. evidence about when mechanistic similarity predicts behavior beyond standard
   performance metrics.

The broader claim under evaluation is that a faithful explanation of one model
should not be treated as a general discovery until it has been tested across
alternative implementations and against behavior not used to construct it.
