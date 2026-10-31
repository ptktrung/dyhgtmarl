# Model card: DyHGT-MARL

## What it is

A dynamic heterogeneous graph Transformer with multi-task heads, coupled to a
constrained multi-agent decision policy. The encoder applies relation aware
attention across nodes and causal self attention across time. Attention logits
carry a learned edge bias for the relation snapshot active at that step and a
reliability term computed from sender and receiver reliability scores.

## Intended use

Research on synthetic benchmarks for urban IoT reliability, missing data repair,
probabilistic travel time forecasting and constrained decision making. It is a
methodology and benchmark contribution.

## Out of scope

Any operational deployment, revenue apportionment, fare adjudication, or
municipal decision making. One component has been evaluated on real data: the
forecasting head was rerun on METR-LA, a public corpus of 207 freeway loop
detectors, and Section VI-H of the manuscript reports the outcome. That test
supports the score and the calibration and withdraws the claim that the model
degrades more slowly than baselines under observation loss, which holds on the
synthetic corpus only. The fault head, the reconstruction head and the decision
layer have never been evaluated on real data.

## Architecture

- Model node set: 28 transit nodes plus 6 gateway nodes.
- Message passing relations: r1 travel connectivity, r3 communication
  dependency, r4 shared gateway association, r7 learned functional similarity.
- Relations r2, r5 and r6 of the schema are one to one or administrative and are
  consumed by the policy layer, not by message passing. The trained model
  therefore exercises four of the seven declared relations.
- Positional encoding: the first eight non trivial eigenvectors of the
  normalized Laplacian, with signs fixed deterministically by forcing the
  largest magnitude entry positive. This is a pragmatic substitute for a sign
  and basis invariant network and is a known limitation.
- Heads: fault logit, root cause score, masked reconstruction, three component
  Gaussian mixture for travel time, and a learned temperature for the fault
  probability.

## Training

Adam, learning rate 0.002, gradient norm clipping at 2.0, early stopping on the
validation objective with patience 3 or 4. Loss weights come from
`configs/default.yaml` and **were not tuned**. No search over loss weights was
run, so no claim of optimality is made.

Training was performed on a single CPU core without a GPU, which constrained the
epoch budget. Reported neural results should be read as a lower bound on what
the architecture can reach.

## Evaluation

Chronological splits, five model seeds for forecasting, three for the multi-task
model, two for the ablation. Conditions: normal test, active cascade, topology
change, distribution shift, and an independent missingness sweep at 0, 5, 10,
20, 30 and 40 percent.

## Function approximation check

The decision layer in the main results is a set of tabular action value
functions, one per zone. Experiment E repeats the same constrained process
with a linear softmax policy trained by REINFORCE with a learned value
baseline over a 22 dimensional feature map, holding the state information,
actions, reward, dual update, split and seeds fixed. The policy reached a
higher net revenue recovery, 93.80 percent against 88.96 percent, by accepting
every record and using none of the abstention machinery, and a cost
sensitivity sweep found the learned policy constant at every penalty setting
tested. The conclusion drawn is about the benchmark rather than the method:
the observable state carries little information about latent validity, so the
optimal policy is constant and only the cost ratio decides which constant.

## Known weaknesses

- On this benchmark, gradient boosting and a plain multilayer perceptron detect
  faults at least as well as the graph model, because fault injection overwrites
  telemetry directly and leaves a nearly separable signature.
- Non causal interpolation reconstructs masked cells better than the model,
  because it can use future observations. The comparison is reported with a
  causal flag so the two families are not confused.
- Calibration degrades sharply under distribution shift, which is the expected
  failure mode and is reported rather than hidden.
- Nominal per action latency values in the policy simulator are assumptions.
  Only wall clock decision time is a measurement, and only of this
  implementation on this machine.
- The decision layer is multi-agent only in the restricted sense of
  independent zone-level learners. There is no communication between agents,
  no joint value factorization and no opponent modelling.
- The function approximation check is linear. Nonlinear policies were not
  tested.
- On the revised stress corpus, where faults appear only as disagreement with
  the neighbourhood and the node series are decorrelated, every tabular
  baseline collapses and the graph encoder still does not win. An epoch budget
  sweep at 25 and 45 epochs shows no improvement, so under-training does not
  explain the result within the range tested. This is the clearest evidence
  that the fault detection head does not convert relational fault structure
  into accuracy at this scale.
- The fairness constraint binds only on the revised stress corpus, where latent
  validity depends on zone. On the primary benchmark and the replication cohort
  it never binds, so the dual ascent machinery is untested there.

## Coupling to the decision layer

The upstream fault probability consumed by the decision layer is the encoder's
own calibrated fault head, exported as a node by step grid. A logistic
estimator fitted on training telemetry is retained as a decoupled control.
Neither source sees the ground truth fault label. On the revised stress corpus
the coupled source improves net revenue recovery, the decision error rate and
the equal opportunity gap over the control, by modest margins on a single
corpus. The primary benchmark results in the manuscript use the control,
because that is what the first version of the experiment measured.
