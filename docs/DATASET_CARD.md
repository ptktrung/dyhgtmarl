# Dataset card: DyHGT-MARL integrated synthetic benchmark

## Summary

This benchmark harmonizes three independently generated synthetic files into one
time aligned, graph indexed corpus for studying fault detection, missing data
repair, probabilistic travel time forecasting and constrained decision making in
a mixed mobility urban IoT setting.

**Every value in this benchmark is synthetic.** Nothing here is measured
municipal traffic, official transport data, passenger records or field collected
telemetry. The node names are inspired by Ho Chi Minh City transport modes but
they do not correspond to real stops, corridors, vehicles or agencies.

## Source material and its exact dimensions

Measured by direct inspection of the supplied files:

| File | Shape | Temporal span |
|---|---|---|
| `hcmc_traffic_data.csv` | 720 rows by 28 columns | 720 hourly steps, 30 days, no timestamp column |
| `HCMC_IoT_Graph_Sensor_Data.csv` | 100,000 rows by 11 columns | 2025-01-01 06:00:07 to 2025-01-10 13:14:28, 9 days 7 hours |
| `hcmc_mixed_mobility_data.csv` | 10,000 rows by 8 columns | 2025-01-01 06:00:00 to 2025-01-01 19:53:15, about 14 hours |

Observed properties that matter for interpretation:

- The traffic matrix has 13 negative entries and 190 entries below 4 minutes.
  A negative travel time is not physically possible. Entries below one minute
  are clipped during alignment and the count of clipped cells is recorded in
  `data/processed/scenario_summary.json`.
- The traffic matrix has a mean pairwise node correlation of 0.909 and a mean
  lag one autocorrelation of 0.859. The series is therefore dominated by one
  shared temporal factor, which limits how much a spatial model can add.
- The supplied traffic file was not produced by the supplied
  `Generate_hcmc_traffic_data.py`. That script clips at 4.0 minutes and produces
  twin morning and evening peaks, and the supplied file has neither property.
- In the supplied IoT file, the probability of a fault given that the previous
  **row** was faulty is 0.221, against 0.029 otherwise, while consecutive rows
  usually belong to different nodes. The original cascade is therefore a row
  order artefact rather than propagation over a graph.
- The supplied transaction file contains no node identity and no label saying
  whether a flagged fare was genuinely owed. Its schema is also richer than the
  supplied `RL_Generate_hcmc_mixed_mobility_data.py` produces.

## What this benchmark adds

- A common node identity across the traffic and IoT files (`node_mapping.csv`).
- A 15 minute grid of 2880 steps over 30 days for 28 transit nodes, giving
  80,640 node step records.
- A heterogeneous graph of 69 nodes across five node types and seven relation
  types, with a snapshot index for the dynamic relations.
- Cascades that propagate over documented graph adjacency with an explicit
  exponential temporal decay, rather than over row order.
- Gateway correlated outages, peripheral MNAR dropout, edge closures, and a
  distribution shift block.
- A transaction stream whose latent validity label is known by construction, so
  that decision policies can be scored.

## Fields

Fields taken directly from the supplied files: `TrafficNodeID`, `IoTNodeID`,
`Zone`, `Mode`, `Fare`.

Fields synthesized by this work: `UnifiedNodeID`, `NodeType`, `GatewayID`,
`Timestamp`, `TravelTime_True`, `TravelTime_Observed`, `SignalStrength_dBm`,
`Latency_ms`, `Drift_Magnitude`, `HardwareScore`, `FaultType_True`,
`CascadeRootID`, `EventID`, `MissingMask`, `MissingnessMechanism`,
`ProvenanceID`.

Fields that are model outputs and are empty in the distributed benchmark:
`FaultProbability`, `RepairAction`, `RepairConfidence`, `ForecastMean`,
`ForecastLower95`, `ForecastUpper95`, `PolicyAction`, `PolicyConfidence`,
`EscalationRequired`. They are populated by
`scripts/attach_model_outputs.py` after the experiments have been run.

`TravelTime_True` is derived from the supplied hourly matrix by linear
interpolation onto the quarter hour grid plus a first order autoregressive
sub hourly component. It is not an independent ground truth.

## Splits

Chronological, never shuffled: steps 0 to 2014 for training, 2015 to 2302 for
validation, 2303 to 2879 for testing. The closure window at steps 2380 to 2500
and the distribution shift from step 2592 both fall inside the test block, so
robustness is evaluated on conditions the model has partly not seen.

## Limitations

1. Everything in this benchmark is synthetic, so its external validity is
   unknown. No claim about real HCMC operations can be supported by these
   results. The one external check that exists does not use this corpus: the
   forecasting comparison was rerun on METR-LA, which is described under
   "External corpus" below.
2. The traffic series covers only 30 days, which is short for seasonality and
   for evaluating distribution shift over a realistic horizon.
3. Cross node correlation of 0.909 means the spatial structure carries little
   independent signal, which suppresses the measurable advantage of graph models.
4. Fault types are injected by directly overwriting telemetry, which makes the
   detection task close to separable from raw features alone. This is visible in
   the results and is discussed in the manuscript.
5. The node mapping is an assumption made to join two files, not a survey of
   infrastructure.
6. The transaction stream has no real spatial ground truth, and validity is assigned
   by the simulator.

## Replication cohort: Kuala Lumpur and Klang Valley

A second cohort is built from the v2.0 generator's Kuala Lumpur files so that
the experiments can be rerun on a different parameterisation. The scope of
that comparison is narrower than the generator's own labelling suggests, and
the limits are recorded here.

**The two cohorts share a network.** The v2.0 generator uses one hardcoded
edge list for both cities. Verified against the shipped files, the two share
an identical 66 edge directed adjacency, identical node types, identical zone
assignment, identical sensor density tiers and byte-identical Laplacian
positional encodings. The traffic generator takes no city argument.

**What differs.** Modal split, 70.7, 16.7 and 12.6 percent against 40.7, 36.6
and 22.6 percent for motorcycles, cars and buses. Physical edge distance,
means of 4.36 km and 6.92 km. Coordinates, place names, fare currency and the
random seed, 42 against 142. Aggregate traffic statistics are close, with mean
volume 405.8 against 406.0 and mean travel time 6.050 against 6.049 minutes.

**Consequence.** This supports a claim of replication under a second
parameterisation. It does not support a claim of cross-network transfer or of
cross-city generalization, and the manuscript makes neither.

**Construction.** `scripts/build_kl_replication_inputs.py` converts the v2.0
files into the three inputs the pipeline reads. The travel time matrix is the
quarter hour table pivoted to 720 by 28, which contains no negative and no
sub-minute entries. The IoT table gains the four identity columns the v2.0
schema omits. The tap table has no hardware reliability field, so
`HardwareScore` is computed from the node's own telemetry with the same
composite index the benchmark uses elsewhere, namely 0.45 times normalized
signal plus 0.35 times inverse log latency plus 0.20 times inverse drift, with
an additional penalty on records flagged as an incomplete tap. Deriving it
instead from the reconciliation outcome was tried and rejected, because it
produced a near perfect proxy for the error flag and would have made the
decision problem trivially separable.

Run the cohort by setting `DYHGT_CITY=kl`, which redirects every processed
artifact and result table to its own directory.

## Revised stress regime

A third corpus repairs the three properties recorded under Limitations that
prevent the proposed mechanisms from being tested in the regime they were built
for. It is a diagnostic variant of the primary benchmark, not a second city.

**What changes.** Faults no longer clamp a telemetry column. A faulty node
reports a signal between minus 95 and minus 75 dBm and a plausible latency, and
the fault appears as its travel time departing from the value its graph
neighbours imply, so detection requires comparing a node with its
neighbourhood. The shared temporal factor of the travel time matrix is shrunk
until the mean pairwise node correlation reaches the configured target, with
node specific autoregressive variation added so each series keeps its diurnal
shape and overall level. Latent transaction validity and the cost of a false
accept both depend on zone, so a utility maximizing policy genuinely prefers to
treat the zones differently.

**What does not change.** Topology, node identity, the gateway assignment, the
scenario layer, the chronological splits, the seeds and every line of
evaluation code. The contrast between the two regimes therefore isolates the
generator rather than the method.

**Where it comes from.** `src/dyhgt/stress.py`, parameterised by the `stress:`
block of `configs/default.yaml`. Set `DYHGT_REGIME=stress` to redirect every
processed artifact and result table to `data/processed_stress/` and
`results/tables_stress/`.

**How to read results from it.** The two regimes are not a before and after
improvement. The stress regime is harder by construction, so absolute scores
fall for every model. Only the ordering between models and the behaviour of the
decision layer are comparable, and Table XIV of the manuscript reports both
regimes side by side for that purpose.

## External corpus: METR-LA

Section VI-H of the manuscript tests the forecasting claim on a corpus the
authors did not generate. It is **not redistributed here** and is not part of
this benchmark.

| Property | Value |
|---|---|
| Source | Released with the diffusion convolutional recurrent network of Li et al., ICLR 2018 |
| Sensors | 207 freeway loop detectors, Los Angeles |
| Span | Four months of 2012, five minute resolution |
| As used here | Aggregated to the 15 minute grid, 11,424 steps |
| Genuinely missing | 7.54 percent of readings, recorded as zeros in the source file |
| Mean pairwise node correlation | 0.2206, against 0.9345 for the harmonized synthetic corpus |
| Adjacency | Rebuilt from the published sensor distance table with a thresholded Gaussian kernel, mean degree 11.4 |
| Labels it does not carry | Fault labels, transactions, gateway structure, zone identity |

Two points of handling are worth recording. Zeros are treated as missing rather
than as observed zero speed, which is the convention the benchmark's own
publications use. The adjacency is rebuilt from the plain distance table rather
than loaded from the distributed pickle, because unpickling a downloaded file
executes whatever it contains, and the rebuilt matrix is what the experiment
uses.

Because the corpus carries no fault labels, no transactions and no zone
structure, it can test forecasting and calibration and nothing else. The fault
detection, reconstruction, decision layer and fairness results remain synthetic.

## Reproduction

```
python scripts/build_node_mapping.py
python scripts/build_dynamic_heterogeneous_graph.py
python scripts/inject_faults_and_missingness.py
python scripts/build_integrated_benchmark.py
python scripts/generate_mixed_mobility_transactions.py
```

Seeds and all generator parameters are in `configs/default.yaml`. The global
seed is 42 and the model seeds are 0 to 4.
