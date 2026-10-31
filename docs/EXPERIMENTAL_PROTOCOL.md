# Experimental protocol

## Common settings

- Splits are chronological and never shuffled: 70 percent train, 10 percent
  validation, 20 percent test by time step.
- Model selection uses the validation block only. The test block is touched once
  per configuration.
- Functional similarity edges are estimated from the training block only.
- Target standardisation statistics are computed on the training block only.
- Seeds are fixed in `configs/default.yaml`.

## Experiment A: multi-task fault, root cause and reconstruction

Baselines: logistic regression, random forest, histogram gradient boosting,
multilayer perceptron, and a random forest on one hop and two hop neighbour
aggregated features as a graph aware control. All use the same lagged telemetry
features. Training rows are subsampled at stride two for the CPU budget.

The DyHGT instance is trained with the full multi-task objective. The decision
threshold is selected on validation by maximising F1 and then applied unchanged
to the test block.

Root cause baselines: the most eigenvector central faulty node, and a random
faulty node. The random baseline is strong because most events involve few
simultaneous faulty nodes, and it is reported for that reason.

Reconstruction: one fixed artificial mask covering 20 percent of observed test
cells is shared by every method. Each masked cell is scored exactly once. Methods
that use future observations are marked as not causal.

## Experiment B: probabilistic travel time

Models: historical average, ridge autoregression, GRU, a reduced DCRNN, a
reduced STGCN, a temporal Transformer, and the DyHGT mixture density head.
Graph WaveNet was not reimplemented and is not reported.

Deterministic baselines receive a Gaussian predictive distribution whose per node
scale is fitted on validation residuals. Without that step CRPS would collapse
onto the absolute error and the comparison would favour the mixture model for
the wrong reason.

Conditions: normal test, active cascade steps, the test closure window, the
distribution shift block, and an independent missingness sweep.

## Experiment C: constrained decision policy

The environment is the graph linked transaction stream with a latent validity
label. Seven actions are available. Constraints on the equal opportunity gap,
mean decision latency and false accept rate are enforced by dual ascent on the
Lagrange multipliers after each episode.

Policies compared: legacy rule based, centralized utility maximising,
centralized constrained, decentralized constrained, decentralized constrained
without abstention, and decentralized utility maximising. The fairness budget is
swept to trace a trade off curve.

## Experiment D: ablation

Model side switches are retrained from scratch: static versus dynamic graph,
homogeneous versus heterogeneous relations, with and without reliability
conditioning, with and without the imputation head, with and without the
calibration term, single task versus multi-task, and with and without graph
message passing. Policy side switches are read from Experiment C.

## Experiment E: function approximation check on the decision layer

The tabular action value functions are replaced by a linear softmax policy
with a learned value baseline, trained by REINFORCE. The feature map combines
the six discretized state components with the hardware score, the upstream
fault probability, the node reliability index and the normalized fare, so
parameters are shared across states. Everything else is held fixed: the same
constrained process, the same seven actions, the same financial reward, the
same dual ascent, the same chronological split and the same five seeds.
Decentralized constrained, centralized constrained and decentralized
unconstrained variants are all trained. A cost sensitivity sweep then raises
the penalty for accepting an invalid record across five settings, with three
seeds each, and records the action mix the policy selects.

## Replication cohort

The whole pipeline is rerun unmodified on a second cohort built from the Kuala
Lumpur files of the v2.0 generator, with `DYHGT_CITY=kl` redirecting every
artifact to its own directory. The two cohorts share a network, because the
generator hardcodes one edge list for both cities, so the comparison tests
sensitivity to parameterisation rather than transfer to a new network. The two
cohorts also have different travel time scales, so only scale free quantities
and within-cohort changes are compared.

## Uncertainty reporting

Tables report the mean and the standard deviation over seeds. Where a single
seed was affordable this is stated in the table and no dispersion is claimed.

## Experiment G: training budget

Only the epoch budget varies. The fault detection model is retrained on the
revised stress corpus at 25 and at 45 epochs with every other setting held
fixed, including the architecture, the learning rate, the feature set, the
split and the threshold selection rule. The purpose is to separate an
architectural limit from an under-training artifact, so precision and recall
are reported alongside F1, since a degenerate model that predicts almost every
node step faulty is distinguishable from one that is still learning.

## Revised stress regime

A second corpus repairs the three generator properties that prevent the
proposed mechanisms from being tested in the regime they were built for. The
changes are implemented in `src/dyhgt/stress.py` and parameterised by the
`stress:` block of `configs/default.yaml`. Faults no longer clamp a telemetry
column. A faulty node keeps a signal between minus 95 and minus 75 dBm and a
plausible latency, and the fault appears as its reported travel time departing
from the neighbourhood implied value by a multiplicative factor drawn per
event. The shared temporal factor of the travel time matrix is shrunk by
bisection until the mean pairwise node correlation reaches the configured
target, with node specific autoregressive variation added so each series keeps
its diurnal shape. Latent transaction validity and the cost of a false accept
both depend on zone.

Topology, the split protocol, the seeds and every line of evaluation code are
unchanged, so the contrast between regimes isolates the generator rather than
the method. `DYHGT_REGIME=stress` redirects every artifact to
`data/processed_stress/` and `results/tables_stress/`.

## Encoder coupling in the decision layer

The upstream fault probability consumed by the decision layer is produced by
the trained encoder's own calibrated fault head, exported by
`export_fault_probability.py` as a node by step grid and read by
`run_experiment_c_policy.py --fault-source dyhgt`. The logistic estimator
fitted on training telemetry is retained as a deliberately decoupled control
and written to a separate result file, so the two sources are compared on the
same policy, the same seeds and the same transaction block. Neither source
sees the ground truth fault label.

## Early stopping and what it was found to do

Training uses early stopping on the validation objective with a patience of
three epochs. Experiment G exposed a problem with that rule. On the revised
stress corpus it halted at the same epoch whatever budget was set, so every
budget returned the identical model and the sweep measured nothing. With the
rule disabled the same ten epoch budget produced a materially better fault
head, which means the rule was ending training early rather than preventing
overfitting on that corpus.

`check_early_stopping_effect.py` tests the rule directly. It retrains the
multi-task fault head twice on the active regime, once with the published
patience of three and once with the rule disabled, holding the seed, the
budget, the features, the split and the threshold selection fixed. Results go
to `exp_h_early_stopping.csv`. On the primary benchmark the rule never fires
within the ten epoch budget, so the fault detection results of Section VI-A are
unaffected and stand as published. On the revised stress corpus it does fire,
so the encoder's fault head is retrained there with the rule disabled and the
reported score is the better of the two. Reporting the early stopped number
would have understated the method.

## Experiment I: external validation on METR-LA

Every other experiment runs on a generator the authors wrote. This one does
not. METR-LA is 207 loop detectors on the Los Angeles freeway network at five
minute resolution over four months of 2012, published with a road network
adjacency and widely used as a traffic forecasting benchmark.

What it can and cannot test is worth stating. The corpus has no fault labels,
no transactions and no gateway structure, so fault detection, the decision
layer and the fairness results cannot be examined on it. Probabilistic
forecasting and its behaviour under observation loss can, and that is the
claim the paper rests on.

The protocol follows Section V as closely as a different corpus allows.
Readings are aggregated to the fifteen minute grid used throughout the paper, a
zero is treated as a missing reading rather than a speed of zero, splits are
chronological at 70, 10 and 20 percent, model selection is on the validation
block, the test block is used once, deterministic baselines receive a Gaussian
predictive distribution fitted on validation residuals, and the metric code is
the same module the synthetic experiments use. Reliability conditioning, which
the synthetic corpus derives from telemetry, is computed here as the share of
genuine observations in each node's recent past, so the mechanism is exercised
on real dropout rather than injected dropout. Added missingness is applied to
the evaluation blocks only.

The corpus is not redistributed with this work. `run_experiment_i_external.py`
takes the directory holding the published files, and the adjacency is rebuilt
from the published sensor distance table with the thresholded Gaussian kernel
of Li et al. rather than by deserializing a pickle from the internet.
