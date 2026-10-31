# Dynamic Heterogeneous Graph Transformers and Fairness-Constrained Selective Decision Learning for Resilient Urban IoT Mobility Systems

**Thai Ky Trung Pham** (1,2), **Quoc-Dai Luong Tran** (3), **Deshinta Arrova Dewi** (1)

(1) Faculty of Data Science and Information Technology, INTI International University, Persiaran Perdana BBN Putra Nilai, 71800 Nilai, Negeri Sembilan, Malaysia
(2) Department of Computer Science, Swinburne Vietnam, FPT University, A35 Bach Dang Street, Tan Hoa Son Ward, Ho Chi Minh City, Vietnam
(3) Faculty of Information Technology, Ton Duc Thang University, Ho Chi Minh City, Vietnam

Corresponding author: Quoc-Dai Luong Tran (tranquocdailuong@tdtu.edu.vn)

---

## Abstract

Urban mobility systems on dense sensor deployments fail structurally rather than in isolation. Separate models for fault detection, repair, forecasting and revenue decisions each see one facet, not the interaction. This paper presents DyHGT-MARL, a dynamic heterogeneous graph Transformer coupled to a fairness constrained decision layer, with a reproducible synthetic benchmark that makes the interaction measurable. The benchmark harmonizes three previously separate synthetic files into 80,640 node-step records over 2,880 quarter-hour steps in a heterogeneous graph of 69 nodes and seven relations, with cascades propagating over documented adjacency, correlated gateway outages, peripheral dropout missing not at random, two corridor closures and a distribution shift. Across five seeds on chronological splits the model attains coverage 0.952 against a nominal 0.95, calibration error 0.023, and on this corpus loses 4.2 percent of score at 40 percent missingness where a climatological baseline loses 95 percent. External validation on METR-LA, a public corpus the authors did not generate, reproduces the best score and the best calibration of seven models, and withdraws the degradation margin, where the model degrades fastest of the six that respond to their inputs. The decision layer raises net revenue recovery from 80.99 to 88.96 percent while closing a 0.152 equal opportunity gap between zones. Results that do not favour the system are reported in full. A multilayer perceptron detects faults far more accurately, calibration collapses under distribution shift, the fairness constraint never binds, a linear policy trained by policy gradient reaches 93.80 percent revenue by accepting everything, and the decision layer is a two-zone tabular contextual bandit rather than a multi-agent reinforcement learner. A revised generator repairs the three diagnosed flaws: faults appear only as disagreement with the neighbourhood, the node series are decorrelated, and latent validity depends on zone. The constraint then binds and halves the gap it is given, abstention becomes worthwhile, and the encoder still trails the perceptron, with an epoch sweep ruling out under-training.

**Keywords:** dynamic heterogeneous graphs, graph Transformers, spectral positional encoding, probabilistic forecasting, uncertainty calibration, constrained reinforcement learning, algorithmic fairness, urban IoT, synthetic benchmark, mixed-mobility networks

---

## I. Introduction

Reviews of smart city technology [1] and of smart city planning for urban resilience [2] describe a move toward instrumented municipal services, and proposed sixth generation network architectures assume the same density of connected devices [3]. Municipal mobility platforms rest on large Internet of Things (IoT) sensor estates: fare validators at bus doors, gantries at metro entrances, telemetry units on ferries, and the gateways carrying it back to a clearing system. The analytics above this estate are usually split into separate problems. One model watches for sensor faults, another imputes gaps, a third forecasts travel time, and a fourth, often a rule table rather than a model, decides whether a fare record is trustworthy enough to settle. Each has a mature literature and each is solved reasonably well in isolation.

The decomposition is convenient and it is also where the difficulty hides. The four problems are coupled through the topology of the network, and the coupling produces the failures that matter operationally. A gateway outage is at once a fault event, a block of correlated missing values, a degradation of forecast inputs for every node behind that gateway, and a reason for caution about every fare record originating there. A detector that treats the twelve affected validators as twelve independent alarms raises twelve tickets when one is warranted. An imputation model assuming values are missing at random will mis-impute a block that went missing because signal quality was poor, and a forecaster consuming those values inherits the error with no signal that it should widen its intervals.

This last point deserves plain statement. When an automated system rejects a fare record it makes a distributive decision. If the rejection rate is driven by sensor quality and sensor quality is correlated with district, the system transfers revenue away from operators serving peripheral areas. This is not a hypothetical concern introduced to motivate a fairness term but the direct consequence of a threshold rule applied to heteroskedastic data, and Section VI-D measures it at 15.2 percentage points for a legacy rule on this corpus.

This paper asks a single question. **How can a dynamic heterogeneous graph-learning system jointly represent topology change, sensor reliability, cascading faults, missing observations, probabilistic travel-time dynamics, and equitable decentralized control in a synthetic mixed-mobility urban IoT network?** The word jointly carries weight. There is no shortage of methods for each component. What is missing is a representation in which they share a substrate, and a corpus in which their interaction is present rather than assumed.

The work has three parts. The first is a benchmark, which harmonizes three independently generated synthetic files that share neither a node identity nor a clock. The second is an encoder, which combines relation-aware attention conditioned on sender and receiver reliability with causal temporal attention and six jointly trained heads. The third is a constrained decision layer, whose seven actions include deferral, because a system with no option to say that it does not know produces errors that are guaranteed to land somewhere. The contributions are as follows.

1. A reproducible integrated benchmark of 80,640 node-step records with cascades propagating over documented adjacency with temporal decay, correlated gateway outages, peripheral sensors that fail as a function of their own signal, corridor closures inside both the training and test blocks, and a closing distribution shift.
2. A dynamic heterogeneous graph Transformer with reliability conditioned relation aware attention and a six term multi-task objective, evaluated against classical, neural and graph aware baselines under chronological splits and multiple seeds.
3. A constrained decision layer with seven actions, explicit budgets and an abstention mechanism, scored against a latent validity label.
4. A critical evaluation reporting where the system loses, with each outcome traced to a property of the generators, a replication on a second parameterisation that confirms two findings and overturns two others, and a revised corpus in which the diagnosed generator flaws are repaired and the mechanisms retested.

The work is an integrated methodology and a reproducible synthetic benchmark. It is not a deployed system, only its forecasting component has been tested on real data, and no claim is made that it solves a real-city deployment problem.

---

## II. Related Work

### A. Spectral and geometric foundations

Graph methods are increasingly applied to the reliability and resilience of infrastructure systems [4]. Learning on non-Euclidean domains was consolidated by Bronstein et al. [5], with the spectral treatment of graph signals deriving from wavelet and Laplacian analysis [6], and graph convolutional networks [7] made the first-order approximation practical. These foundations supply the representation used here, but assume a graph that is given and fixed, which gateway outages and corridor closures violate.

### B. Graph Transformers and positional encoding

Applying self-attention to graphs requires injecting structure that attention does not otherwise see. Dwivedi and Bresson [8] generalized the Transformer to graphs using Laplacian eigenvectors as positional encodings. Kreuzer et al. [9] introduced learned spectral attention, Ying et al. [10] encoded structure through centrality, spatial and edge biases, and Rampasek et al. [11] combined local message passing with global attention, all on the self-attention formulation of Vaswani et al. [12]. The eigenvector approach is ambiguous up to sign and up to basis within a repeated eigenspace, and Lim et al. [13] gave architectures invariant to both. This work keeps the eigenvector encoding but resolves sign by convention, a limitation inherited from [14] and acknowledged in Section VIII.

### C. Heterogeneous graph learning

Urban knowledge graphs such as UrbanKG [15] show the same typed formulation applied to city scale entities. Typed relations were introduced to graph neural networks by Schlichtkrull et al. [16] with relation-specific transformations, extended with attention over metapaths by Wang et al. [17], and given a Transformer treatment by Hu et al. [18], whose type-dependent projections are closely related to the relation-aware attention used here. What these models do not carry is a notion that a node's observations may be untrustworthy at a particular moment, which is the reliability conditioning of Section IV.

### D. Dynamic and temporal graphs

Surveys of dynamic graph representation learning [19] separate discrete time snapshot models from continuous time event models. EvolveGCN [20] evolves the convolution weights, DySAT [21] applies structural then temporal attention, and temporal graph networks [22] maintain per-node memory updated by events. This benchmark uses the snapshot formulation, since its topology changes through documented closures rather than a stream of edge arrivals.

### E. Traffic forecasting on graphs

The dominant line of work models road networks as graphs and forecasts flow or speed. DCRNN [23] treats traffic as diffusion on a directed graph, STGCN [24] interleaves graph and temporal convolutions, and Graph WaveNet [25] learns an adaptive adjacency alongside the given one. T-GCN [26] couples graph convolution with the gated recurrence descending from long short-term memory [27], AGCRN [28] learns node-specific parameters, attention-based spatiotemporal networks [29] apply self-attention across both axes, and spatial-temporal Transformers [30] treat time steps as tokens. Two properties matter here. Almost all of it is deterministic point forecasting, which cannot express the widening uncertainty a partially observed network should report, and almost all of it evaluates on complete data, so the interaction between missingness and forecast quality is rarely measured.

### F. Probabilistic forecasting and calibration

Mixture density networks [31] remain the simplest way to emit a full conditional distribution, and models such as DeepAR [32] and normalizing flow forecasters [33] extend the idea. Proper scoring rules, particularly the continuous ranked probability score, supply the evaluation framework [34]. Calibration is separate from sharpness. Modern networks are systematically overconfident [35], deep ensembles improve both [36], and conformal prediction [37], [38] gives distribution free coverage under exchangeability, which is what the distribution shift block removes.

### G. Missing data

The taxonomy of missing completely at random (MCAR), missing at random and missing not at random (MNAR) is due to Rubin [39], with the full treatment in Little and Rubin [40]. Recurrent models with decay handle irregular gaps [41], bidirectional recurrent imputation [42] exploits both directions of the sequence, and graph-based imputation [43] uses spatial structure explicitly. Peripheral sensors that drop out because their own signal is weak are a textbook MNAR mechanism, which the benchmark generates deliberately rather than assuming away.

### H. Anomaly and fault detection

Classical density and isolation approaches [44], [45] and deep one-class methods [46] treat points independently, while graph deviation networks [47] and multivariate detection with graph attention [48] add structure to the problem. The gap identified by the source paper [14], that municipal monitoring treats sensors as isolated points and confuses congestion with cascading failure, remains the motivation, although Section VI shows it is easier to resolve on this corpus than expected.

### I. Multi-agent and constrained reinforcement learning

Centralized training with decentralized execution [49], counterfactual credit assignment [50] and value factorization [51] are the standard tools for multi-agent control, on the foundation of [52]. Constrained Markov decision processes [53] and their deep counterparts [54], [55] provide the machinery for enforcing budgets, which is what fairness, latency and risk become in Section IV.

### J. Fairness and deferral

Individual fairness [56], equalized odds and equality of opportunity [57], empirical risk minimization under fairness constraints [58] and reduction approaches [59] form the basis, with the textbook treatment in [60]. Most of that literature concerns one-shot classification, and sequential settings where the decision changes future data are less settled. Selective prediction, from the classical reject option [61] through selective neural networks [62], learning to defer [63] and consistent estimators for deferral [64], supplies the abstention mechanism. Combining a fairness budget with an explicit deferral action is the construction used here.

### K. Federated and edge learning

Federated averaging [65] and the survey of open problems [66] describe how such a system would be trained without centralizing municipal data, and secure operations frameworks [67] how it would be governed. Neither is evaluated here.

### L. The three source papers

Three conference papers by the present authors supply the foundations. The fault detection paper [14] introduced the Graph-Theoretic Spatiotemporal Transformer with Laplacian eigenvector positional encodings and reported a fault recall of 0.96 on a 100,000-record corpus. The travel time paper [68] applied the same encoder with a three-component mixture density output and reported mean absolute error (MAE) 4.1 minutes, continuous ranked probability score (CRPS) 4.02 and interval coverage of 97.3 percent on the 720 by 28 matrix. The data integrity paper [69] used a tabular Q-learning agent with a fairness term in the reward and reported 100.0 percent revenue recovery and a fairness score rising from 0.54 to 1.00 on 10,000 transactions.

**These figures are results of those individual papers on those individual corpora. They are not results of the integrated system described here, they are not comparable to the numbers in Section VI, and they are not reproduced by the code accompanying this paper.** The corpora, the splits and the metric definitions all differ. Two of the figures are also unattainable here by construction. The travel time paper's 97.3 percent coverage against a nominal 95 percent is over-coverage rather than the near-exact coverage obtained here, and 100.0 percent revenue recovery requires an environment in which no record can be invalid. Table I places the three papers alongside the wider literature.

What distinguishes the present work from its own predecessors is integration. Each source paper operated on one file with one task and a static graph. This paper puts the three files on one clock and one node identity, replaces a row-order cascade with propagation over documented adjacency, adds topology change and correlated outages that none of the generators produced, trains one encoder against six objectives, and scores the decision layer against a latent validity label that did not previously exist.

## III. Dataset Construction and Dynamic Heterogeneous Graph Formulation

### A. Provenance statement

All data used here are synthetic. They are inspired by the modal mix and core against periphery contrast of Ho Chi Minh City, but are not official traffic or transport data, not passenger or municipal records, and not field measurements. No real topology, timetable, fare table or sensor deployment was used. Every figure and table is produced by a script from a saved result file, with no value entered by hand.

### B. Exact dimensions of the source files

Dimensions were established by direct inspection rather than taken from documentation, and the inspection produced several facts that were not previously reported.

`hcmc_traffic_data.csv` contains 720 rows and 28 columns, one per node, with no timestamp column, covering 30 days if read as hourly. It holds 13 negative travel times, which are not realizable and so are generator artifacts rather than extreme observations, and a mean pairwise node correlation of 0.909. The supplied generator clips at 4.0 minutes and produces twin peaks, and the supplied matrix has neither property, so **the supplied matrix was not produced by the supplied script**. The regeneration script here reproduces the documented process, not the supplied file.

`HCMC_IoT_Graph_Sensor_Data.csv` contains 100,000 rows and 11 columns spanning 9 days and 7 hours rather than the 30 days of the traffic matrix, with an anomaly rate of 0.0359. The probability of a fault given a faulty preceding row is 0.221, against 0.029 otherwise, while consecutive rows usually belong to different nodes. The dependency is on row order rather than graph adjacency, so the original cascade is a proxy rather than a propagation process.

`hcmc_mixed_mobility_data.csv` contains 10,000 rows and 8 columns spanning roughly 14 hours of a single day, with an error rate of 0.0533 in the core zone and 0.1983 in the periphery. It carries no node identifier and no label indicating whether a flagged fare was genuinely owed, and its schema is richer than its own supplied generator produces.

Table II records these dimensions alongside the integrated benchmark and names, for each file, the limitation that most constrains its use.

### C. Common node identity

The traffic and IoT files use disjoint naming schemes and cannot be joined without an assumption. The assumption adopted is the mapping in Table 1 of the benchmark specification, written to `node_mapping.csv` by `scripts/build_node_mapping.py`:

| Traffic node range | IoT node range | Mode | Zone |
|---|---|---|---|
| `Node_0` to `Node_9` | `BusNode_01` to `BusNode_10` | Bus | Core |
| `Node_10` to `Node_14` | `MetroNode_01` to `MetroNode_05` | Metro | Core |
| `Node_15` to `Node_24` | `BusNode_11` to `BusNode_20` | Bus | Periphery |
| `Node_25` to `Node_27` | `FerryNode_01` to `FerryNode_03` | Ferry | Periphery |

**This mapping is a transparent synthetic-data harmonization assumption. It is not a factual map of Ho Chi Minh City transport infrastructure.** Some correspondence must be chosen if two independently generated files are to describe one network, and this one preserves the core-periphery split and the modal proportions both generators already encode. Every unified node receives an identifier `U00` through `U27` and a gateway assignment `GW_00` through `GW_05` by round robin, itself a labelled assumption.

### D. Time alignment

The three files have incompatible clocks, at 720 hourly steps, nine days of second-resolution events and fourteen hours of transactions, so a common quarter-hour grid of T = 2,880 steps over 30 days is adopted.

True travel time on this grid comes from the supplied hourly matrix by clipping sub-minute entries to a floor of one minute, interpolating between hourly anchors, and adding a first-order autoregressive sub-hourly component. The result, `TravelTime_True`, is not an independent ground truth. It is a deterministic function of the supplied matrix plus seeded noise, so any model is evaluated against a smoothed version of the source file.

The IoT and transaction streams are not resampled onto the grid, because their spans are too short to cover it and stretching nine days of telemetry to cover thirty would fabricate temporal structure. The scenario layer instead generates telemetry directly on the grid from the zone-conditional signal and latency distributions measured in the source IoT file, and the transaction generator draws from the measured transaction marginals, which keeps the assumption explicit.

### E. Dynamic heterogeneous graph

The benchmark is a time indexed heterogeneous graph

$$G_t = (V_t, E_t, R, X_t, A_t)$$

where $V_t$ is the node set at step $t$, $E_t$ the typed edge set, $R$ the set of relation types, $X_t \in \mathbb{R}^{|V_t| \times F}$ the node feature matrix, and $A_t$ the stack of relation adjacency matrices active at step $t$.

The node set contains 69 nodes across five types: 28 transit nodes, 28 sensor and validator nodes in one-to-one correspondence, 6 gateways, 4 clearing agents and 3 maintenance agents. Seven relation types are declared.

The travel relation **r1** is a ring with stochastic chords at offsets 2, 5 and 9 accepted with probability 0.58, reproducing the supplied generator so the topology stays traceable to its source, and yielding 77 undirected edges at mean degree 5.5. Its edges are removed during closures, and **r3 communication depends** is suppressed while a gateway is in outage, which is what makes the topology dynamic. The functional similarity relation **r7** is the top-three positive correlation of observed travel time, estimated on the training block only. Table III gives all seven with their temporal status, provenance and edge counts.

The topology is stored as a static edge list of 318 edges plus a snapshot index with five intervals recording which edges are suppressed when. Fig. 2 summarises the schema.

[[FIGURE:fig_02_heterogeneous_graph_schema]]

One scope statement matters. The trained encoder performs message passing over four of these relations, r1, r3, r4 and r7, on a model node set of the 28 transit nodes plus the 6 gateways. Relations r2, r5 and r6 are one-to-one or purely administrative and carry no information message passing could exploit, so they are consumed by the policy layer instead. The schema is therefore richer than what the network exercises, and Section VIII returns to this.

### F. Fault, cascade and missingness injection

The central mechanism that the source generators lack is propagation over graph structure. Here, a node $j$ that is adjacent to an active faulty node $i$ becomes faulty at the next step with hazard

$$p_{i \to j}(t) = p_0 \cdot \tilde{A}_{ij} \cdot \exp\!\left(-\frac{t - t_i^{0}}{\tau}\right)$$

where $\tilde{A}$ is the symmetrically normalized travel adjacency, $t_i^0$ the step at which node $i$ became faulty, $p_0 = 0.55$ and $\tau = 3$ steps. When several active nodes are adjacent to $j$, the maximum hazard is taken and the donor with the largest normalized edge weight is recorded as the propagation parent, so that `CascadeRootID` traces back to the originating node. During a closure window the suppressed edges contribute zero hazard, which is what makes the topology genuinely dynamic rather than decorative.

Cascade roots are scheduled fourteen each from the low, medium and high eigenvector-centrality strata. Spontaneous faults occur at a base probability of 0.0015 per node-step, raised by 0.010 when the signal falls below minus 90 dBm. Eight gateway outages knock out every sensor served by the affected gateway for 4 to 14 steps. Fault types are Signal Loss, which forces the signal to minus 120 dBm, High Latency, which adds 200 to 2,000 ms, Sensor Drift, which accumulates a monotone offset decaying at 0.985 per step, and Gateway Outage.

Missingness comes from four mechanisms, each labelled in `MissingnessMechanism`:

- **Gateway outage**, deterministic and correlated across all sensors on the affected gateway.
- **Peripheral MNAR**, where a peripheral sensor drops out with logistic probability in its own signal strength, intercept minus 4.2 and slope 0.055. Because the dropout depends on the unobserved value's own driver, this is missing not at random in the strict sense.
- **Topology closure**, where the endpoints of a closed corridor lose observations with probability 0.45 during the window.
- **Independent MCAR**, applied at evaluation time at rates of 0, 5, 10, 20, 30 and 40 percent. These are stored as six separate masks rather than six copies of the corpus.

The realized corpus has a fault rate of 0.0501 across 80,640 node-step records and 245 logged events, of which 42 are cascades, 195 spontaneous and 8 gateway outages. Structural missingness is 0.0348. Fig. 3 reports the full composition and its distribution over the 30 days, and the dataset card lists the per-type counts.

[[FIGURE:fig_03_dataset_and_scenario_statistics]]

Twelve scenario families are defined on top of these mechanisms, covering a clean baseline, the missingness sweep, each fault type, centrality-stratified cascades, gateway outages, peak-hour congestion, peripheral MNAR failure, topology change, distribution shift and targeted perturbation, with their realized sizes in Table V.

The mechanism was verified rather than assumed. The probability that a node newly becomes faulty given a faulty graph neighbour at the previous step is 0.0274, against 0.0046 otherwise, a hazard ratio of 5.9, so the row-order statistic of the source file is replaced by a structural dependency. Fig. 4 traces one cascade outward from its root across four successive snapshots.

[[FIGURE:fig_04_dynamic_cascade_snapshots]]

### G. Feature and label dictionary

Fields taken directly from the source files are `TrafficNodeID`, `IoTNodeID`, `Zone`, `Mode` and `Fare`. Fields synthesized by this work are `UnifiedNodeID`, `NodeType`, `GatewayID`, `Timestamp`, `TravelTime_True`, `TravelTime_Observed`, `SignalStrength_dBm`, `Latency_ms`, `Drift_Magnitude`, `HardwareScore`, `FaultType_True`, `CascadeRootID`, `EventID`, `MissingMask`, `MissingnessMechanism` and `ProvenanceID`. `HardwareScore` is a composite reliability index formed as a convex combination of normalized signal, inverse log latency and inverse drift with weights 0.45, 0.35 and 0.20.

Nine further fields are model outputs, listed in the dataset card, and are **left empty in the distributed benchmark**. They are populated only after the experiments have been run, so that a user who downloads the benchmark cannot mistake a model prediction for a data field.

### H. Splits

Splits are chronological and never shuffled: steps 0 to 2,014 for training, 2,015 to 2,302 for validation, and 2,303 to 2,879 for testing. The second closure window at steps 2,380 to 2,500 and the distribution shift from step 2,592 both fall inside the test block, so robustness is evaluated partly on conditions seen in training and partly on conditions not seen. The first closure window at steps 1,600 to 1,840 lies in the training block, which lets the model learn that closures happen at all.

### I. Reproducibility statement

All parameters live in `configs/default.yaml` with a global seed of 42 and model seeds 0 through 4. The delivered reproducibility checklist lists the build commands in order, and the dataset card records the written artifacts and the limitations in full.

### J. Dataset limitations

Five limitations bound every conclusion drawn from this corpus. It is synthetic, so external validity is unknown. The series spans 30 days, short for seasonality and shorter still for a distribution shift horizon. A mean pairwise node correlation of 0.909 leaves little spatial signal independent of a shared temporal factor. Faults are injected by overwriting telemetry, which leaves an almost separable signature. The transaction stream has no spatial ground truth, so its latent validity label is a modelling assumption. Section VI-G rebuilds the corpus to remove the third and fourth, Section VI-H tests the first for forecasting alone, and Section VIII returns to the rest.

---

## IV. DyHGT-MARL Methodology

### A. Why a static homogeneous graph is insufficient

Three properties of the benchmark defeat a static homogeneous formulation. The topology changes: during a closure window the corridor carrying the strongest correlation between two nodes no longer exists, so a model conditioned on the base adjacency keeps borrowing evidence across an edge that is not there. The relations are not interchangeable: a shared-gateway edge predicts simultaneous total failure while a travel edge predicts graded, delayed propagation, and one adjacency forces a single set of weights to represent both. Most important for a reliability setting, the informativeness of a neighbour varies over time. A neighbour reporting minus 120 dBm should contribute almost nothing at that step, and standard attention cannot express this because it conditions on feature content rather than on the trustworthiness of the sensor that produced it.

### B. Reliability-conditioned relation-aware attention

The encoder addresses all three with a single attention formulation, whose symbols are collected in Table IV. For relation $r$ at step $t$,

$$\alpha_{ij}^{(r,t)} = \mathrm{softmax}_j\left(\frac{(Q_i W_Q^r)(K_j W_K^r)^\top}{\sqrt{d}} + b_{ij}^{r,t} + \psi(q_i, q_j)\right)$$

Here $Q_i$ and $K_j$ are the hidden states of the receiving and sending nodes, $W_Q^r, W_K^r \in \mathbb{R}^{d \times d}$ are relation specific projections so each relation learns its own notion of similarity, and $d$ is the hidden dimension, 48 for forecasting and 40 for the multi-task configuration. $b_{ij}^{r,t}$ scales the weight of edge $(i,j)$ under relation $r$ **in the snapshot active at step $t$**, so a closure zeroes it, and non-edges are masked before the softmax. $\psi(q_i, q_j)$ is the reliability term,

$$\psi(q_i, q_j) = w_1^r q_i + w_2^r q_j + w_3^r |q_i - q_j|$$

where $q_i \in [0,1]$ is the reliability score of node $i$ at that step, computed from signal, latency and drift as described in Section III-G, and $w^r \in \mathbb{R}^3$ is learned per relation. The three components let the model discount an unreliable sender, modulate how much an unreliable receiver borrows from its neighbourhood, and treat reliability mismatch itself as a signal.

Relation outputs are concatenated and projected, with a residual connection and layer normalization closing the block. A causal temporal block then applies masked self-attention along the window axis per node, followed by a position-wise feed-forward network. Two such layers are stacked. Fig. 1 shows the resulting pipeline from harmonization through to the constrained policy.

[[FIGURE:fig_01_dyhgt_architecture]]

**This component is implemented and evaluated.** The reliability term is ablated in Section VI, and the ablation finds no measurable benefit on this corpus, which Section VII discusses.

### C. Spectral positional encoding

Node identity is injected through the first eight non-trivial eigenvectors of the normalized Laplacian $L = I - D^{-1/2} A_{\mathrm{sym}} D^{-1/2}$, with $A_{\mathrm{sym}} = \max(A, A^\top)$. Eigenvector sign is resolved by forcing the entry of largest magnitude positive. This is a deterministic convention, not an invariance property, and it is weaker than the sign and basis invariant networks of [13]. It is used because the source paper [14] used it, which keeps that comparison meaningful. Gateway nodes receive a zero spectral encoding plus a type indicator, since they do not lie in the transit graph.

### D. Multi-task heads and objective

Six heads read the encoder. A fault head emits a logit per node per step. A root-cause head emits a score per node at the final step, normalized over transit nodes. A reconstruction head predicts the standardized true travel time at every node-step, supervised only where an input value was artificially masked. A mixture density head emits three-component mixing weights, means and softplus scales for travel time at a horizon of four steps, one hour ahead. A calibration term and a stability term complete the objective:

$$L_{total} = \lambda_f L_{fault} + \lambda_r L_{root} + \lambda_m L_{mask} + \lambda_y L_{forecast} + \lambda_c L_{calibration} + \lambda_s L_{stability}$$

with $L_{fault}$ binary cross-entropy, $L_{root}$ cross-entropy over nodes restricted to steps where an event is active, $L_{mask}$ a smooth L1 loss on masked cells, $L_{forecast}$ the mixture negative log-likelihood

$$L_{forecast} = -\sum_{i=1}^{N}\log\left(\sum_{k=1}^{K}\pi_{i,k}\,\mathcal{N}(y_i \mid \mu_{i,k}, \sigma_{i,k}^2)\right)$$

$L_{calibration}$ the Brier score of the fault probability, and $L_{stability}$ the mean squared difference of consecutive-step fault logits, which discourages an alarm channel that flickers from step to step. The weights in (5) are $\lambda_f = 1.0$, $\lambda_r = 0.5$, $\lambda_m = 1.0$, $\lambda_y = 1.0$, $\lambda_c = 0.2$, $\lambda_s = 0.1$.

**No search over loss weights was performed. These values were fixed before the experiments, and no claim of optimality is made.** Section VI shows the objective trading fault accuracy against forecast quality, which is the kind of tension weight tuning would address and this study does not.

### E. Uncertainty and abstention

The mixture output gives a full predictive density, so the mean, variance and any quantile are available, with quantiles obtained by bisection on the mixture distribution function. Calibration is evaluated by the probability integral transform. For a correctly specified forecast the transformed values are uniform, so the expected calibration error is the mean absolute gap between nominal and empirical coverage across a grid of quantile levels. Coverage is reported at the nominal 95 percent level alongside mean interval width, so that coverage bought by uninformative width is visible.

A learned temperature is applied to the fault logit at inference, which is a single-parameter recalibration in the spirit of [35]. Conformal calibration was considered and is **not implemented**, since its exchangeability assumption is violated by the distribution-shift block. Section VIII treats it as the appropriate next step.

The selective prediction rule sits in the decision layer rather than in a separate threshold. An agent may choose `defer` or `escalate`, routing the record to human review at a modelled cost, which prices abstention rather than leaving it free so that review workload can be traded against error rate. Calibration must hold empirically for such a rule to mean anything, and Section VI reports that it fails on the distribution shift block, with calibration error rising from 0.023 to 0.195.

### F. Fairness-aware multi-agent decision layer

The decision process is a constrained contextual bandit, written below in the notation of a constrained Markov decision process because the constraint machinery is the same. The **state** of a transaction is the tuple (zone, error flag, hardware-score bucket, upstream fault-probability bucket, node-reliability bucket, corroboration signal), The upstream fault probability comes from one of two sources. Either the trained encoder's own calibrated fault head supplies it, which couples the two halves of the system at inference, or a logistic estimator fitted on training telemetry does, which serves as a deliberately decoupled control. Section VI-G compares the two, and the tables of Section VI-D use the control so that they are not confounded with the encoder's own fault accuracy. Neither source sees the ground truth fault label, and the corroboration signal takes a third value meaning not yet observed. The **action set** has seven elements: `accept`, `reject`, `impute`, `corroborate`, `defer`, `isolate` and `escalate`. The **transition** is terminal for every action except `corroborate`, which draws a second noisy observation of validity whose accuracy rises with the hardware score and returns the agent to an augmented state for one further decision.

The **reward** is the realized financial outcome: accepting a genuinely valid record earns the fare, accepting an invalid one costs twice the fare, imputation succeeds with probability $\min(0.35 + 0.65q, 0.99)$ and otherwise costs a quarter of the fare, deferral yields the correct outcome at a review cost of 2,500 VND, and escalation at 1.6 times that. The objective is

$$\max_{\pi} \ \mathbb{E}_{\pi}\left[\sum_t \gamma^t U_t\right] \quad \text{subject to} \quad C_{fair} \le \epsilon_f, \quad C_{latency} \le \epsilon_l, \quad C_{risk} \le \epsilon_r$$

with $\gamma = 0.85$. The constraints are enforced by dual ascent, with the Lagrange multipliers updated after each episode as $\lambda \leftarrow \max(0, \lambda + \eta (C - \epsilon))$ and the shaped per-decision reward subtracting the weighted constraint surrogates.

Fairness is deliberately **not** treated as a single number. Three measures are reported separately: the demographic parity gap, the maximum difference in acceptance rate between zones, the equal opportunity gap, the same difference among genuinely valid records, and the false positive rate gap, the same difference among genuinely invalid records. These can move in opposite directions, and a system that equalizes acceptance rates while accepting more invalid records in one zone has not become fairer in any useful sense. The constraint is placed on the equal opportunity gap, because the harm of interest is a genuinely valid fare being discarded.

Decentralization is implemented as one action-value table per zone with independent updates, against a centralized variant with a single table. Both are compared in Section VI.

The layer should be named accurately, because the name claims more than the implementation delivers. Every action except corroboration terminates the record, and nothing an agent does changes the state another agent sees or the distribution of later records. It is therefore a **two-zone tabular contextual bandit with one optional information gathering action**, not a multi-agent reinforcement learner in the sense of [49] to [51]: there is no communication, no joint value factorization, no opponent modelling and no cross-record credit assignment, and the discount factor applies only inside the two-step episode corroboration creates. The system name DyHGT-MARL is kept for continuity with the conference papers it builds on. Section VI-D reports that the centralized and decentralized variants agree within one standard deviation, which is what a bandit predicts.

---

## V. Experimental Protocol and Reproducibility

Splits are chronological throughout and never shuffled. Model selection, decision thresholds and the residual scales used to give deterministic baselines a predictive distribution are all computed on the validation block. The test block is used once per configuration. The functional-similarity relation is estimated on the training block only, and target standardization statistics are likewise training-only, so that neither the graph nor the normalization leaks test information.

Experiment A compares logistic regression, random forest, histogram gradient boosting, a multilayer perceptron, and a random forest on neighbour aggregated features as a graph aware control, all on identical lagged telemetry features, against the DyHGT fault head. The threshold is chosen on validation to maximize F1 then applied unchanged to test. Root cause attribution is compared against the most eigenvector central faulty node and a uniformly random one. Reconstruction uses one fixed mask over 20 percent of observed test cells, shared by every method and scored once per cell, with non-causal methods flagged.

Experiment B compares historical average, ridge autoregression, a GRU, a reduced DCRNN, a reduced STGCN, a temporal Transformer and the DyHGT mixture density head across five seeds. Reduced means a two-step diffusion convolution with one recurrent layer for DCRNN and two graph convolution layers with one recurrent layer for STGCN, both at the proposed model's hidden dimension of 48, so models are matched on width and depth rather than parameter count. Graph WaveNet was **not** reimplemented and is not reported, rather than approximated under its name. Deterministic baselines receive a Gaussian predictive distribution whose per-node scale is fitted on validation residuals, without which the continuous ranked probability score of a point forecast collapses onto its absolute error and the mixture model would appear to win for the wrong reason.

Experiment C evaluates six policies on the held-out transaction block across three seeds and sweeps the fairness budget over eight levels. Experiment D retrains eight model-side variants across two seeds and reads the policy-side switches from Experiment C, so one consistent set of policy numbers is reported. Experiment E repeats the constrained decision problem with a linear softmax policy trained by policy gradient with a learned value baseline across five seeds, then sweeps the penalty for accepting an invalid record across five settings. Experiment G varies the training budget alone, and Experiment I repeats the comparison of Experiment B on METR-LA under the same protocol. The delivered experimental protocol document records the remaining settings for each run.

All neural models are optimized with Adam [70] at a learning rate of 0.002, with gradient norm clipping at 2.0 and early stopping on the validation objective, and the function approximation policy of Experiment E uses stochastic gradient ascent with step size 0.05. Training ran on CPU under a fixed budget of ten to fourteen epochs with early stopping, which is short for the neural models and is why their results are reported as a lower bound rather than as a converged comparison. Seeds are fixed in NumPy and PyTorch, but bitwise reproduction across processors, BLAS implementations and library versions is neither guaranteed nor verified, so anyone rerunning this work should rerun every seed in their own environment rather than mixing numbers across machines.

## VI. Results

All values below are produced by the scripts in `scripts/tables/` from result files written by the experiment scripts. Dispersion is the standard deviation across model seeds, five for Experiment B, three for Experiments A and C, and two for Experiment D.

### A. Fault detection, root cause and detection delay

Table VI reports the detection results. The outcome is not the one this work's framing would predict, which is why it comes first.

| Model | Precision | Recall | F1 | PR-AUC | False alarms | Missed |
|---|---|---|---|---|---|---|
| Multilayer perceptron | 0.993 +/- 0.005 | 0.983 +/- 0.003 | **0.988 +/- 0.001** | 0.998 | 7 | 15 |
| Histogram gradient boosting | 0.982 +/- 0.006 | 0.983 +/- 0.003 | 0.982 +/- 0.002 | 0.998 | 16 | 16 |
| Logistic regression | 0.970 | 0.960 | 0.965 | 0.992 | 27 | 36 |
| Random forest | 0.967 +/- 0.016 | 0.850 +/- 0.019 | 0.905 +/- 0.004 | 0.979 | 26 | 135 |
| Random forest, graph features | 0.939 +/- 0.011 | 0.845 +/- 0.017 | 0.889 +/- 0.004 | 0.964 | 50 | 140 |
| DyHGT-MARL multi-task | 0.675 +/- 0.151 | 0.665 +/- 0.067 | 0.658 +/- 0.059 | 0.735 | 2,730 | 2,454 |

A multilayer perceptron on lagged telemetry misses fifteen of roughly 2,700 faulty node-steps. The graph-aware random forest is worse than the plain random forest, and the proposed model reaches F1 0.658 with more than two thousand false alarms.

Detection delay agrees. Across 53 test cascade events, gradient boosting detects all of them at a mean delay of 0.038 steps and DyHGT-MARL only 34, at 2.088 steps. Fig. 5 plots the scores and the recall against false-alarm trade off.

[[FIGURE:fig_05_fault_detection_performance]]

The explanation is in the generator, stated in Section III-F. Signal Loss sets the signal to exactly minus 120 dBm and High Latency adds at least 200 ms to a baseline of 20 to 45 ms. These are near-deterministic markers in a single feature, so a model that thresholds one column solves the task and topology adds nothing. Two factors compound it, since the multi-task model splits capacity across six objectives while the baselines optimize one, and it was trained for ten epochs under the fixed budget of Section V while the tabular baselines were fitted to convergence. The comparison is therefore not a clean architectural test, and Section VI-G repeats it on a corpus where the single column marker is removed.

Root-cause attribution is similarly uninformative. Over 350 event windows the model attains top-1 accuracy 0.377 and top-3 0.640, against 0.377 and 0.603 for the highest-centrality faulty node and 0.360 and 0.591 for a random one. The learned head is indistinguishable from picking a faulty node at random, because most events involve few simultaneously faulty nodes and the root is weakly determined by the observable state.

### B. Reconstruction and data integrity

Table VII reports masked-value reconstruction on a shared artificial mask of 3,003 test cells, by mean absolute error and root mean squared error (RMSE), with causal and non-causal methods distinguished.

| Method | MAE (min) | RMSE (min) | Causal |
|---|---|---|---|
| Linear interpolation | **1.288** | 1.768 | no |
| Last observation carried forward | 2.023 | 2.718 | yes |
| KNN imputer | 2.272 | 2.930 | no |
| Hour-of-week climatology | 2.765 | 3.564 | yes |
| DyHGT-MARL reconstruction head | 2.880 | 3.680 | yes |
| Graph diffusion | 3.092 | 3.954 | no |
| Global mean | 10.482 | 12.179 | yes |

Non-causal interpolation wins by a wide margin, which is why the causal flag exists. With a lag-one autocorrelation of 0.859 and a fifteen-minute step, a gap bracketed by two observations is nearly determined. What matters is that causal carry forward also beats the model, 2.023 against 2.880. By mechanism the model reconstructs MNAR cells at 2.822 and closure cells at 2.733 but gateway outage cells at 4.790, the correct ordering, because an outage removes a whole neighbourhood and with it the spatial evidence. Fig. 6 shows the ranking and one reconstructed series.

[[FIGURE:fig_06_reconstruction_example]]

### C. Probabilistic travel time

Table VIII reports the forecasting results, where the proposed model does win.

On the normal test block, DyHGT-MARL attains MAE 2.302, RMSE 3.055, mean absolute percentage error (MAPE) 13.64 percent, CRPS 1.650, prediction interval coverage probability (PICP) 0.952 against a nominal 0.95, and expected calibration error (ECE) 0.023, with MAE ranging from 2.270 to 2.355 across five seeds. Among the alternatives, historical average is closest at CRPS 1.705, with ridge autoregression worst at 2.802.

**A climatological baseline beats every neural baseline tested.** The series inherits a strong diurnal cycle from the supplied generator, which a per-node, per-quarter-hour average captures most of. The proposed model improves on that baseline by 3.2 percent in CRPS on clean data, which alone would be a weak result. Fig. 7 shows the predictive intervals widening as error grows.

[[FIGURE:fig_07_probabilistic_forecast]]

The case for the model is robustness, and it is substantial. Under the independent missingness sweep, CRPS moves as follows:

| Missing rate | 0% | 5% | 10% | 20% | 30% | 40% |
|---|---|---|---|---|---|---|
| DyHGT-MARL | 1.650 | 1.656 | 1.662 | 1.679 | 1.703 | **1.720** |
| Historical average | 1.705 | 1.811 | 1.945 | 2.338 | 2.800 | 3.330 |
| DCRNN | 2.287 | 2.406 | 2.577 | 2.951 | 3.431 | 3.951 |
| STGCN | 2.417 | 2.506 | 2.639 | 2.876 | 3.229 | 3.598 |
| GRU | 2.431 | 2.685 | 2.921 | 3.285 | 3.768 | 4.010 |
| Transformer | 2.791 | 3.057 | 3.304 | 3.713 | 4.177 | 4.604 |

The proposed model degrades by 4.2 percent between zero and forty percent missingness, against 95 percent for historical average and 65 percent for the Transformer and the GRU. At 40 percent missingness the model scores better than any baseline achieves on complete data. This is the substantive finding of Experiment B: the graph structure and the missingness channel buy not accuracy on clean data but the ability to keep functioning when observations disappear. Fig. 9 plots this degradation alongside the other stress conditions.

[[FIGURE:fig_09_dynamic_graph_robustness]]

Under active cascades the model holds at CRPS 1.689 and coverage 0.951, and under the test-block topology change at 1.622 and 0.950, while every baseline loses coverage, STGCN to 0.853 and ridge to 0.850.

Under distribution shift the model degrades and its calibration fails. MAE rises to 3.236, CRPS to 2.388, coverage falls to 0.873 and calibration error rises from 0.023 to 0.195. Historical average fails comparably at coverage 0.810 and error 0.202. Ridge autoregression is the best calibrated model on the shifted block, at error 0.032 and coverage 0.906, because its validation fitted Gaussian scale is wide enough to absorb the shift. A well-calibrated but inaccurate model can outperform a sharp one once the data generating process moves. Fig. 8 gives the reliability diagram and the transform histogram.

[[FIGURE:fig_08_uncertainty_calibration]]

### D. Policy and fairness

Table IX reports the decision-policy results on 11,929 held-out transactions with a latent validity rate of 0.980 and a flag rate of 0.142.

| Policy | Revenue (%) | Error rate | Abstention | Review (%) | Equal-opp. gap | Core TPR | Periphery TPR |
|---|---|---|---|---|---|---|---|
| Rule-based legacy | 80.99 | 0.1545 | 0.000 | 0.0 | **0.1523** | 0.9167 | 0.7644 |
| Centralized utility max | 85.89 +/- 3.54 | 0.0229 | 0.241 | 24.1 | 0.0008 | 0.9994 | 0.9997 |
| Decentralized utility max | 86.79 +/- 2.59 | 0.0331 | 0.181 | 18.1 | 0.0006 | 0.9997 | 0.9997 |
| Decentralized constrained | 88.96 +/- 1.76 | 0.0270 | 0.135 | 13.5 | 0.0013 | 0.9987 | 1.0000 |
| Centralized constrained | 89.04 +/- 0.65 | 0.0260 | 0.125 | 12.5 | 0.0006 | 0.9994 | 1.0000 |
| Decentralized, no abstention | **90.30 +/- 1.10** | 0.0419 | 0.000 | 0.0 | 0.0029 | 0.9952 | 0.9959 |

The legacy rule shows the pathology described in the introduction. It recovers 91.67 percent of genuinely valid core fares against 76.44 percent of peripheral ones, a gap of 0.152, and discards 14.03 percent of valid records. Every learned policy removes the gap and lifts net revenue by 4.9 to 9.3 points.

**The fairness constraint is not what removes the gap.** The unconstrained utility-maximizing policies achieve gaps of 0.0008 and 0.0006, statistically indistinguishable from the constrained variants. The fairness budget sweep in Table IX-b confirms this: across budgets from 0.30 down to 0.01 the achieved gap stays between 0.0001 and 0.0035 and revenue varies between 86.6 and 90.4 percent with no monotone trend. There is no Pareto frontier to trace because the constraint never binds. Fig. 10 shows the flat trade off curve.

[[FIGURE:fig_10_fairness_utility_pareto]]

The reason is a property of the simulator. Latent validity is conditioned on whether the node is faulty, not on which zone it is in. Peripheral nodes fail more often and are flagged more often, but a flagged peripheral record is no less likely to represent a genuine trip. Any policy learning from realized outcomes rather than from the flag discovers that the flag is a poor proxy for validity, and equity follows without being asked for. The legacy rule's bias is entirely an artifact of conditioning on an unequally distributed proxy.

This is a negative result about the experiment, not about fairness-constrained learning. A binding constraint requires that the utility-maximizing policy genuinely prefers to treat one group worse, which happens when validity rates, fare values or error costs differ across groups. Section VI-G makes exactly that change and reports the result.

**Function approximation check.** To test whether these conclusions depend on the table representation, a linear softmax policy with a learned value baseline was trained by policy gradient on the same constrained process, holding the state information, the seven actions, the financial reward, the dual update, the chronological split and the five seeds fixed. The policy maps a 22-dimensional feature vector, built from the six discretized state components together with the hardware score, the upstream fault probability, the node reliability index and the normalized fare, to a distribution over actions, so parameters are shared across states rather than stored per state.

The result does not support the decision layer as designed. Over five seeds the function approximation policy reached 93.80 percent revenue recovery against 88.96 percent for the tabular decentralized policy, and an equal opportunity gap of 0.0000 against 0.0013, by accepting every record. Abstention, corroboration and deferral were each selected in zero percent of decisions, and the error rate settled at 0.0211, the proportion of records that are invalid. All three variants produced identical policies, confirming through a second representation that the fairness budget does not bind and that centralization makes no difference here.

[[FIGURE:fig_13_function_approximation_policy]]

A gap of 0.0000 reflects a constant policy rather than equitable treatment, since accepting everything treats both zones identically only because it conditions on nothing. Table XII reports both representations side by side.

A cost sensitivity sweep locates the reason. Raising only the penalty for accepting an invalid record, the policy accepts everything at multipliers 2, 5 and 10, then defers everything at 20 and 40, recovering 69.38 percent with no decision errors and full review workload. No setting produced a mixed strategy. Because the observable state carries little information about latent validity, the optimal policy is constant and only the cost ratio decides which one. The selective machinery of Section IV-F therefore cannot be evaluated on this benchmark, which is what the revised corpus of Section VI-G is built to address.

Abstention behaves as expected. Removing it yields the highest tabular revenue, 90.30 percent, because review costs money, but raises the decision error rate from 0.0270 to 0.0419. Deferral buys a 36 percent error reduction for 13.5 percent review workload and 1.3 points of revenue, and the centralized and decentralized constrained policies agree within one standard deviation on every metric.

Measured wall-clock decision time is between 0.0076 and 0.0145 ms per decision for all policies, which is a measurement of this implementation on this machine and nothing more. The per-action latency values in Table X are simulator assumptions, not hardware benchmarks, and must not be read as deployment figures. Fig. 11 compares nominal against measured cost and the review workload.

[[FIGURE:fig_11_system_cost_latency]]

### E. Ablation

Table XI reports the model-side ablation over two seeds.

| Variant | F1 | Forecast CRPS | Forecast ECE | Reconstruction MAE |
|---|---|---|---|---|
| Full model | 0.649 | 2.126 | 0.060 | 2.909 |
| Static graph | 0.643 | 2.094 | 0.042 | 2.927 |
| Homogeneous relations | 0.659 | 2.109 | 0.102 | 2.847 |
| No reliability conditioning | 0.650 | 2.161 | 0.070 | 2.963 |
| No calibration term | 0.636 | 2.197 | 0.099 | 2.943 |
| No imputation head | 0.651 | 2.214 | 0.069 | **11.341** |
| No graph, temporal only | 0.657 | **2.665** | **0.214** | 3.246 |
| Single task, fault only | **0.779** | **7.416** | 0.073 | 11.889 |

Three components earn their place. Removing graph message passing raises CRPS from 2.126 to 2.665 and calibration error from 0.060 to 0.214, which is the clearest evidence that the spatial channel matters for probabilistic forecasting. Removing the imputation head raises reconstruction error to 11.341 and removing the calibration term raises calibration error to 0.099.

Three do not. Static graph, homogeneous relations and removal of reliability conditioning all produce differences within seed noise, and in two cases the ablated variant scores marginally better, so those components are not measurably contributing on this corpus.

The single-task ablation exposes a tension. Training the fault head alone raises F1 from 0.649 to 0.779 while forecast CRPS collapses from 2.126 to 7.416. Six objectives compete for a shared encoder of 40 hidden units, and the untuned weights of (5) resolve that competition in favour of forecasting. Fig. 12 gives the full matrix.

[[FIGURE:fig_12_ablation_matrix]]


### F. Replication on a second cohort

Every result above comes from one corpus. To test whether they survive a change
of parameterisation, the pipeline was rerun unmodified on a second cohort built
from the Kuala Lumpur and Klang Valley files of the v2.0 generator. The scope
needs stating precisely, because the generator's own labelling overstates it.
That generator uses one hardcoded edge list for both cities, so the cohorts
share an identical topology, adjacency and set of Laplacian positional
encodings, and its traffic process is not city dependent. What differs is the
modal split, 70.7, 16.7 and 12.6 percent against 40.7, 36.6 and 22.6 percent
for motorcycles, cars and buses, the edge distances, at means of 4.36 and 6.92
km, the fare scale and the seed. This is replication under a second
parameterisation, not transfer to an independently specified network, and no
cross-city generalization is claimed. The cohorts also differ in travel time
scale, at means of roughly 25.6 and 6.1 minutes, because the primary cohort
reads the original supplied matrix and the replication reads the v2.0 table, so
only scale free quantities and within-cohort changes are compared below.

Three findings replicate. Forecast accuracy is close, at MAPE 13.64
against 9.08 percent, and within each cohort the model remains the best
scoring forecaster. Calibration holds in both, at coverage 0.952 against
0.953 for a nominal 0.95 and calibration error 0.023 against 0.041. Most
importantly the robustness result reproduces: between zero and forty percent independent missingness the proposed model loses
+4.3 percent of score in the primary cohort and
+7.4 percent in the replication, while the climatological
baseline loses +95.3 and +39.2 percent.

The fairness finding replicates and strengthens. The legacy threshold rule
produces an equal opportunity gap of 0.1523 in the primary
cohort and 0.2098 in the replication, where the zone disparity
in sensor quality is sharper, and it recovers 80.99 against
75.05 percent of revenue. Learned policies close the gap in
both, to 0.0013 and 0.0074. The constraint
again fails to bind, since the unconstrained policy reaches
0.0006 and 0.0058 without being asked to.

Two further results reverse, and both are informative. Reconstruction is the
larger of the two. In the primary cohort the model ranks fifth of seven, behind
non-causal interpolation at 1.288 and causal carry forward at 2.023 minutes. In
the replication it ranks first of seven, at 1.025 minutes against
1.096 for the nearest baseline. The cause is visible in the inputs
rather than in the method. The primary cohort's series has a lag one
autocorrelation of 0.859, at which a gap bracketed by two observations is
nearly determined and interpolation is close to optimal, while the
replication's series has an autocorrelation of 0.732, at which temporal
interpolation degrades and the cross-node structure the encoder uses begins to
pay. The conclusion in Section VII-B should therefore be read as conditional on
the temporal persistence of the series rather than as a general statement about
the method.

The second reversal concerns the decision policy. Abstention is used in the
primary cohort, at 0.135 of decisions, and not at all in the replication.
Section VI-D concluded from the primary cohort that the learned policy is
constant at every cost setting, which is not true of the replication, where a
penalty multiplier of 40 produces a genuinely mixed policy. Both the value of
deferral and the constancy of the policy are therefore properties of the
parameterisation rather than of the method.

Table XIII sets the two cohorts side by side and Fig. 14 plots the three
comparisons. The replication supports the robustness and fairness results,
removes the claim that the learned policy is always constant, and shows that
the abstention machinery earns its place in one parameterisation and not in
the other.

[[FIGURE:fig_14_replication_cohort]]


### G. Revised generator and encoder coupling

Section VIII diagnoses three properties of the benchmark that prevent the
proposed mechanisms from being tested as intended. Because the generator is
ours, this subsection reports the repair. Faults no longer clamp a telemetry
column. A faulty node keeps a plausible signal and latency, and the fault shows
instead as its travel time departing from what its graph neighbours imply. The
shared temporal factor is shrunk, which moves the mean pairwise correlation of
the harmonized node series from 0.9345 to
0.5501. Latent validity and false accept cost both depend on
zone, which widens the core to periphery validity gap from 0.0010
to 0.1031. Topology, splits and evaluation code are unchanged, so the
contrast isolates the generator. Table XIV reports both regimes and Fig. 15
plots the changes.

**The fault injection flaw is confirmed and the graph hypothesis still fails.**
Removing the single column marker collapses every tabular baseline, the
multilayer perceptron from F1 0.988 to 0.459 and the rest
further still. The task is now what the architecture was designed for, and the
encoder does not win it, reaching 0.241. The remaining
explanation is the training budget. Experiment G varies the epoch budget alone
and does not support it. F1 is 0.221 at 10 epochs, 0.189 at 25 epochs, 0.218 at 45 epochs, while the validation objective it is trained on continues to fall from 0.1326 to 0.1231, so the model keeps improving on its own objective without converting that into discrimination. Within the range tested the limit is the architecture, not the schedule. Early stopping is disabled here. On this corpus it halts the fault head at 5 epochs and returns a degenerate model, F1 0.109 at precision 0.058, against 0.221 once the budget is allowed to bind.

**The fairness constraint now binds, and abstention earns its keep.** The
unconstrained policy opens an equal opportunity gap of 0.0298,
against 0.0006 in the primary benchmark, and dual ascent on the
budget reduces that gap to 0.0136, a fall of
54 percent. Demographic parity
stays wide at roughly 0.10, which is intended for the reason Section IV-F
gives. Deferral is selected in 0.374 of
decisions against 0.135 before, and removing it raises the
decision error rate from 0.0359 to 0.0868. The
cost sensitivity sweep also stops being degenerate, producing a mixed strategy
at a penalty multiplier of 5.

**Coupling the encoder to the decision layer shifts the operating point.**
Replacing the logistic surrogate with the encoder's calibrated fault
probability, with everything else fixed, raises net revenue recovery on the revised corpus from 75.12 to 79.45 percent. It also raises the decision error rate from 0.0359 to 0.0377 and widens the equal opportunity gap from 0.0136 to 0.0273. Abstention falls from 0.374 to 0.214, so the sharper signal buys revenue by deferring less often. On the primary benchmark the trade runs the other way, costing 1.23 points of revenue, lowering the error rate to 0.0258 and widening the gap to 0.0041. The coupled signal does not dominate the control on either corpus.
It moves where the dual ascent settles rather than improving every
objective at once, which is what a constrained policy should be expected to do,
and where faults are relational the policy converts the better fault estimate,
0.241 against 0.120, into revenue. The forecasting results survive. The proposed model keeps coverage at 0.950, remains the best scoring forecaster, and loses 3.1 percent of score between zero and forty percent missingness against 10.6 percent for the climatological baseline.

[[FIGURE:fig_15_stress_regime]]


### H. External validation on a public benchmark

Every result above comes from a generator the authors wrote, which Section VIII
records as the largest threat to this work. This subsection removes part of
that threat. The forecasting comparison is rerun on METR-LA, 207 loop detectors
on the Los Angeles freeway network at five-minute resolution over four months
of 2012, published with a road network adjacency and widely used as a traffic
forecasting benchmark [23]. The corpus was not made for this study and carries
real sensor dropout, 7.54 percent of readings, rather than injected dropout,
which is the property under test.

The scope of the test needs stating. METR-LA has no fault labels, no
transactions and no gateway structure, so the fault detection results of
Section VI-A and the decision layer results of Section VI-D cannot be examined
on it and remain synthetic. What can be examined is probabilistic forecasting
and its behaviour as observations disappear. The protocol is that of Section V,
with the same metric code, chronological splits, selection on validation and
the test block used once. Readings are aggregated to the fifteen-minute grid
the paper uses throughout, the adjacency is rebuilt from the published sensor
distance table with the thresholded Gaussian kernel of Li et al. [23], giving
mean degree 11.4, and three seeds are run.

One property of the public corpus is worth recording on its own. Its mean
pairwise node correlation is 0.22, against 0.93 for the harmonized synthetic
corpus and 0.55 for the revised regime of Section VI-G. The criticism that
section makes of the generator, that a shared temporal factor leaves the
spatial channel with little to carry, is therefore not a theoretical worry: the
synthetic corpus is four times as correlated as the real network it was built
to resemble.

**The probabilistic forecasting claim survives the move to real data.** The proposed model attains the best continuous ranked probability score of the seven models compared, 3.327 against 4.006 for the temporal Transformer. Its calibration is also the best of the seven, with 95 percent interval coverage 0.956 against a nominal 0.95 and expected calibration error 0.020, where the next best error is 0.082. Point accuracy and calibration part company on this corpus in a way the synthetic benchmark did not show. The model ranks fifth of seven on mean absolute error, at 4.769 against 4.333, and fifth on mean absolute percentage error, at 14.65 percent against 12.74 percent for the temporal Transformer. The mixture density head buys a predictive distribution that is accurate about its own width rather than a sharper conditional mean. On a corpus carrying real sensor noise, that trade becomes visible.

**The robustness claim, which Section VI-C states as the headline result, does not survive.** Between zero and 40 percent added observation loss the proposed model gives up 6.7 percent of its score, the largest loss of the six models that respond to their inputs at all, which give up between 2.6 and 5.8 percent. On the synthetic corpus the same comparison gave 4.2 percent against 65 to 95 percent, so that margin was a property of the corpus rather than of the method. The climatological baseline reads no observation at the prediction step, so its score cannot move when observations are removed, and it is excluded from this ranking. The slowest to degrade, the reduced STGCN at 2.6 percent, is also the weakest forecaster at every level, so part of the spread is a floor effect rather than resilience. What does hold is the ranking itself. The model has the best score at every level tested, 3.550 at 40 percent loss against 4.234 for the best baseline there, so it stays the model to prefer under observation loss. What fails is the mechanism. Reliability conditioning does not slow degradation on real data. It starts from a better score and then degrades at an ordinary rate. Table XV reports the comparison and the degradation of every model.

Two things follow, and they point in opposite directions. The forecasting and
calibration results now have support outside the authors' own generator, which
is the claim most exposed to the objection that the corpus is synthetic. The
degradation margin does not, so Section VII-A, Section IX and the abstract
report it as a property of the synthetic corpus rather than as a general
result. Fault detection, reconstruction, the decision layer and the fairness
results remain untested outside the generator, because no public corpus known
to the authors carries fault labels, fare transactions and zone structure
together. Building one is the obvious next piece of work.

---

## VII. Discussion

### A. What the evidence supports

The defensible claim is narrow, and the external test of Section VI-H narrows it further. What holds on both corpora is calibrated probabilistic forecasting. A dynamic heterogeneous graph Transformer with an explicit missingness channel and a mixture density output attains the best continuous ranked probability score and the best calibration of the models compared, on the synthetic benchmark and on METR-LA alike, and it keeps the best score at every missingness level on both. What does not hold is the degradation margin. On the synthetic corpus the model gives up 4.2 percent of its score at 40 percent missingness against 65 to 95 percent for the baselines, and on METR-LA it gives up 6.7 percent against 2.6 to 5.8 percent, which is no advantage at all. The margin is therefore a property of the synthetic corpus, where a mean pairwise node correlation of 0.909 leaves the baselines with little to fall back on once observations are removed, and not a property of reliability conditioning. The ablation still locates the source of the score itself, since removing graph message passing costs 25 percent in CRPS and quadruples the calibration error.

The second concerns the decision layer. Replacing a threshold rule with a policy that learns from realized outcomes raises net revenue recovery from 80.99 to 88.96 percent while eliminating a 15.2-point equal opportunity gap, and deferral trades review workload for a 36 percent reduction in decision error. Both replicate in the second cohort, although Section VI-F shows the deferral benefit does not.

### B. What the evidence does not support

Three claims the architecture would naturally invite are not supported.

Graph structure does not help fault detection, and this is now a tested claim rather than a conjecture. On the primary benchmark a multilayer perceptron beats the proposed model by 0.33 in F1, which the trivial fault injection could explain away. Section VI-G removes that explanation by rebuilding the generator so faults appear only as disagreement with the neighbourhood, which collapses every tabular baseline. The encoder still loses, and the epoch sweep rules out under-training within the range tested. On this corpus, at this scale, the architecture does not convert relational fault structure into detection accuracy. That is the most substantive negative result in the paper and it is a finding about the method, not the benchmark.

Heterogeneity and dynamic topology do not measurably help, since the static-graph and homogeneous-relation ablations sit within noise of the full model. Two explanations fit. A mean pairwise node correlation of 0.909 leaves one shared temporal factor explaining most of the variance and little for relation typing to resolve, and the encoder exercises only four of the seven declared relations, two of them largely redundant with the travel adjacency in this topology. A schema declared richer than the encoder uses is a design flaw of this study rather than a finding about heterogeneous graphs.

The fairness constraint is not binding. As Section VI-D sets out, the generator makes latent validity independent of zone, so utility maximization and equity coincide. The constrained and unconstrained policies are indistinguishable. The result still carries a useful message for practice, since where bias arises purely from conditioning on an unequally distributed proxy, replacing the proxy with outcome-based learning removes the bias without an explicit fairness mechanism. Evidence that the constraint machinery works comes instead from the revised corpus of Section VI-G, where the constraint does bind.

### C. Relation to the companion manuscripts

Two companion manuscripts by the present authors draw on the same source files, so the division is stated here. The fault detection study [14] uses the IoT telemetry file alone and the travel time study [68] the traffic matrix alone, each with a static graph, one objective and no decision layer. This work adds a harmonized corpus on one clock and one node identity, a cascade propagating over documented adjacency, a topology that changes inside both blocks, and a decision layer scored against a latent validity label. No number here is reused from either.

### D. Governance implications

If a system of this kind moved toward deployment, three properties would matter more than accuracy: a `ProvenanceID` on every record so an imputed settlement can be traced, abstention priced rather than free so the workload implied by a chosen error rate is visible to whoever staffs the review queue, and fairness reported as three gaps rather than one score. None of this substitutes for external audit on real data.

---

## VIII. Limitations and Threats to Validity

**Synthetic data for all but one experiment.** Section VI-H tests the forecasting comparison on METR-LA, a public corpus the authors did not generate. It supports the score and the calibration, withdraws the degradation margin, and is the only result here with outside support. Every other conclusion rests on a corpus that measures whether methods behave sensibly under mechanisms deliberately inserted, not whether those mechanisms resemble reality, and no hardware measurement or deployment study supports any number in this paper. The replication cohort of Section VI-F does not change this. Both cohorts come from the same generator and share a topology, so it tests sensitivity to parameterisation rather than external validity.

**Weaknesses inherited from the source generators.** The supplied traffic matrix contains unrealizable travel times, its hourly profile does not match its own documented generator, so its provenance is unresolved, and its mean pairwise node correlation of 0.909 leaves little independent spatial signal. The supplied IoT generator's cascade is a row-order dependency rather than propagation over a graph, which is the defect this work replaces, and the supplied transaction generator emits no node identity and no validity label.

**Weaknesses introduced by this work.** `TravelTime_True` is an interpolation of the supplied matrix plus seeded noise rather than an independent ground truth, the node mapping is an assumption, and latent transaction validity is a modelling choice. Fault injection in the primary benchmark overwrites telemetry directly and makes detection close to trivially separable, which invalidates that comparison as a test of graph methods. Section VI-G repairs this in a second corpus and reports what the repaired task shows.

**Scope of the implementation.** Four of seven declared relations participate in message passing, loss weights were fixed and never tuned, so the multi-task trade-off in Section VI-E is unresolved, and eigenvector sign is fixed by convention rather than by a sign-invariant architecture, inheriting the ambiguity discussed in [13]. Conformal calibration is discussed but not implemented, and Graph WaveNet is absent rather than approximated under its own name. The decision layer is tabular. Function approximation policies, including policy gradient training with learned reward signals of the kind used for conversational agents [71], are needed for continuous or high dimensional states, and only the linear case is examined, in Section VI-D.

**Statistical power.** Five seeds for forecasting, three for the multi-task model and two for the ablation are modest. Differences within roughly one standard deviation, which includes every heterogeneity and dynamic-topology comparison, should be treated as unresolved rather than as null results.

**Computational budget.** All results were obtained on a single CPU core with no GPU. The neural models were trained for ten to fourteen epochs with early stopping, while the tabular baselines were fit to convergence in seconds. The epoch sweep of Section VI-G finds no material gain between 10 and 45 epochs. That weakens the budget explanation without eliminating it, since no larger model under a longer schedule was tried.

**Reproducibility across environments.** Seeds are fixed but bitwise cross-platform determinism was not verified, and identical code with identical seeds has been observed to give different results on different machines. Any user should rerun all seeds in their own environment rather than combining numbers across environments.

**Non-binding constraint in the primary benchmark.** The fairness budget never binds in either cohort of the primary benchmark. Section VI-G shows this is a property of that corpus and not of the method, since making latent validity and false accept cost depend on zone produces a gap the dual update then halves. The constraint machinery is therefore tested, but only in a regime the authors constructed for the purpose. Section VI-D sharpens this for the primary cohort, where the observable state carries little information about latent validity and no cost setting favoured a mixed strategy, although Section VI-F shows the latter is cohort specific, since the replication does produce a mixed policy at a high penalty. Any claim about fairness constrained control on this corpus therefore depends on a revision in which validity rates, fare values or misclassification costs differ by zone. That same revision would give abstention and corroboration a regime in which they can pay for themselves.

**No external test of the other components.** No real dataset backs the fault detection, reconstruction, decision layer or fairness results. No public corpus known to the authors carries fault labels, fare transactions and zone structure together, so testing those components externally requires building such a corpus rather than locating one.

---

## IX. Conclusion

This paper set out to ask whether topology change, sensor reliability, cascading faults, missing observations, probabilistic travel-time dynamics and equitable decentralized control can be represented jointly rather than as six separate problems. The answer this study supports is partial and worth stating precisely.

The joint representation helps for probabilistic forecasting. On the synthetic corpus the model attains a calibration error of 0.023 and coverage of 0.952 against a nominal 0.95 on stationary test data, and degrades by 4.2 percent in continuous ranked probability score at 40 percent missingness where the strongest baseline degrades by 95 percent. Section VI-H retests this on METR-LA, a public corpus the authors did not generate, and the two halves of the result part company. The score and the calibration reproduce, at 3.327 and 0.020 against 4.006 and 0.082 for the best baseline on each, and the model keeps the best score at every missingness level. The degradation margin does not, at 6.7 percent against 2.6 to 5.8 percent for the baselines, so that margin is reported here as a property of the synthetic corpus rather than of the method. Ablation attributes this to graph message passing, whose removal costs 25 percent in score and quadruples the calibration error. The decision layer raises net revenue recovery from 80.99 to 88.96 percent while closing a 15.2-point equal opportunity gap, and prices abstention so that a 36 percent reduction in decision error can be weighed against 13.5 percent review workload.

A replication on a second parameterisation reproduces both results and withdraws the claim that the learned policy is constant at every cost setting, which holds only for the primary cohort. The joint representation does not help for fault detection, where a multilayer perceptron is better, nor for reconstruction, where causal carry forward is better, and neither relation typing nor dynamic topology nor reliability conditioning produces a measurable effect. Each outcome traces to a property of the source generators: telemetry-overwriting fault injection, a lag-one autocorrelation of 0.859, a mean pairwise node correlation of 0.909, and latent validity independent of zone.

Section VI-G acts on that diagnosis rather than deferring it. The generator is rebuilt so that faults appear only as disagreement with the neighbourhood, the node series are decorrelated, and latent validity depends on zone. Three things follow. The fairness constraint binds, abstention becomes worthwhile, and the learned policy becomes state dependent. What does not change is the fault detection result: the encoder stays behind a multilayer perceptron, with an epoch sweep ruling out under-training within the range tested. What remains is to narrow the schema to the relations the encoder uses or extend the encoder to use them all, search the loss weights, adopt sign-invariant spectral encodings, and implement conformal calibration with explicit treatment of the exchangeability violation that the distribution-shift block introduces. Validation on a public corpus is now partial rather than absent, and the part that is missing covers fault detection, reconstruction and the decision layer, so it remains the precondition for any operational claim, and none is made here.

---

---

## Author Contributions

T. K. T. Pham: conceptualization, data curation, software, formal analysis, investigation, visualization, writing of the original draft. Q.-D. L. Tran: methodology for the reinforcement learning decision layer, validation of the policy experiments, writing through review and editing, project administration for submission. D. A. Dewi: supervision, methodology review, writing through review and editing. All authors approved the final manuscript and agree to be accountable for the work.

## Acknowledgment

The authors used Claude, an AI assistant developed by Anthropic, in the preparation of this
work. Its use covered drafting and revision of the manuscript text, and the implementation of
the data generators, model and baseline code, experiment scripts, and the figure and table
scripts in the accompanying package. The research questions, experimental design, the
interpretation of every result and all conclusions are the authors' own. The authors verified
the code and the reported numbers against the saved result files and take full responsibility
for the content of this article. No AI system meets the criteria for authorship and none is
listed as an author.

## Funding

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

## Data and Code Availability

All data generated for this study are synthetic and are regenerated in full by the scripts accompanying the manuscript. The external validation of Section VI-H uses METR-LA, a public corpus released with the work introducing the diffusion convolutional recurrent network [23], which is not redistributed here. The accompanying code downloads it from its published location and rebuilds the adjacency from the published sensor distance table. The code package contains the three supplied source files, the generators that rebuild the integrated benchmark from a seed, the model and baseline implementations, the experiment scripts, and one script per figure and per table. Configuration and seeds are in a single file. The source code, benchmark generators and result files are available at https://github.com/ptktrung/dyhgt_marl. Queries about the code and data may be directed to T. K. T. Pham at trungptk@fe.edu.vn.

## Competing Interests

The authors declare no competing financial or non-financial interests.

## Ethics Statement

This study uses synthetic data throughout, with one exception. The external validation of Section VI-H uses METR-LA, a public corpus of freeway loop detector speeds that contains no personal data. No human participants, personal data, passenger records or municipal records were involved at any stage. The node names are inspired by the modal mix of a large city but do not correspond to real infrastructure, and no result reported here supports an operational claim about any real transport network.

## References


[1] D. D. Sanchez-Gallegos, D. E. Carrizales-Espinoza, C. Torres-Charles, and J. Carretero, "Smart cities: A systematic review of emerging technologies," *Smart Cities*, vol. 8, no. 5, art. 173, 2025. doi: 10.3390/smartcities8050173.

[2] A. I. Almulhim, "Building urban resilience through smart city planning: A systematic literature review," *Smart Cities*, vol. 8, no. 1, 22, 2025. doi: 10.3390/smartcities8010022.

[3] P. Mishra and G. Singh, "6G-IoT framework for sustainable smart city: Vision and challenges," *IEEE Consumer Electronics Magazine*, vol. 13, no. 5, pp. 93-103, 2024. doi: 10.1109/MCE.2023.3307225.

[4] T. Liu and F. Liu, "Graph neural networks for evaluating the reliability and resilience of infrastructure systems: A systematic review of models, applications, and future directions," *IEEE Access*, vol. 13, pp. 164883-164904, 2025. doi: 10.1109/ACCESS.2025.3611333.

[5] M. M. Bronstein, J. Bruna, Y. LeCun, A. Szlam, and P. Vandergheynst, "Geometric deep learning: Going beyond Euclidean data," *IEEE Signal Processing Magazine*, vol. 34, no. 4, pp. 18-42, 2017.

[6] D. K. Hammond, P. Vandergheynst, and R. Gribonval, "Wavelets on graphs via spectral graph theory," *Applied and Computational Harmonic Analysis*, vol. 30, no. 2, pp. 129-150, 2011.

[7] T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2017.

[8] V. P. Dwivedi and X. Bresson, "A generalization of Transformer networks to graphs," in *Proc. AAAI-21 Workshop on Deep Learning on Graphs: Methods and Applications*, 2021.

[9] D. Kreuzer, D. Beaini, W. L. Hamilton, V. Letourneau, and P. Tossou, "Rethinking graph Transformers with spectral attention," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 34, 2021.

[10] C. Ying, T. Cai, S. Luo, S. Zheng, G. Ke, D. He, Y. Shen, and T.-Y. Liu, "Do Transformers really perform badly for graph representation?," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 34, 2021.

[11] L. Rampasek, M. Galkin, V. P. Dwivedi, A. T. Luu, G. Wolf, and D. Beaini, "Recipe for a general, powerful, scalable graph Transformer," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 35, 2022.

[12] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.

[13] D. Lim, J. Robinson, L. Zhao, T. Smidt, S. Sra, H. Maron, and S. Jegelka, "Sign and basis invariant networks for spectral graph representation learning," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2023.

[14] T. K. T. Pham and D. A. Dewi, "A graph-theoretic approach to real-time IoT fault detection in mixed-mobility networks," *Proc. SPIE*, vol. 14309, 1430914, 2026. doi: 10.1117/12.3120488.

[15] Y. Liu, J. Ding, Y. Fu, and Y. Li, "UrbanKG: An urban knowledge graph system," *ACM Transactions on Intelligent Systems and Technology*, vol. 14, no. 4, art. 60, pp. 1-25, 2023. doi: 10.1145/3588577.

[16] M. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov, and M. Welling, "Modeling relational data with graph convolutional networks," in *Proc. European Semantic Web Conference (ESWC)*, 2018.

[17] X. Wang, H. Ji, C. Shi, B. Wang, Y. Ye, P. Cui, and P. S. Yu, "Heterogeneous graph attention network," in *Proc. The Web Conference (WWW)*, 2019.

[18] Z. Hu, Y. Dong, K. Wang, and Y. Sun, "Heterogeneous graph Transformer," in *Proc. The Web Conference (WWW)*, 2020.

[19] S. M. Kazemi, R. Goel, K. Jain, I. Kobyzev, A. Sethi, P. Forsyth, and P. Poupart, "Representation learning for dynamic graphs: A survey," *Journal of Machine Learning Research*, vol. 21, no. 70, pp. 1-73, 2020.

[20] A. Pareja, G. Domeniconi, J. Chen, T. Ma, T. Suzumura, H. Kanezashi, T. Kaler, T. B. Schardl, and C. E. Leiserson, "EvolveGCN: Evolving graph convolutional networks for dynamic graphs," in *Proc. AAAI Conference on Artificial Intelligence*, 2020.

[21] A. Sankar, Y. Wu, L. Gou, W. Zhang, and H. Yang, "DySAT: Deep neural representation learning on dynamic graphs via self-attention networks," in *Proc. ACM Int. Conf. Web Search and Data Mining (WSDM)*, 2020.

[22] E. Rossi, B. Chamberlain, F. Frasca, D. Eynard, F. Monti, and M. Bronstein, "Temporal graph networks for deep learning on dynamic graphs," in *Proc. ICML Workshop on Graph Representation Learning and Beyond*, 2020.

[23] Y. Li, R. Yu, C. Shahabi, and Y. Liu, "Diffusion convolutional recurrent neural network: Data-driven traffic forecasting," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2018.

[24] B. Yu, H. Yin, and Z. Zhu, "Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting," in *Proc. Int. Joint Conf. Artificial Intelligence (IJCAI)*, pp. 3634-3640, 2018. doi: 10.24963/ijcai.2018/505.

[25] Z. Wu, S. Pan, G. Long, J. Jiang, and C. Zhang, "Graph WaveNet for deep spatial-temporal graph modeling," in *Proc. Int. Joint Conf. Artificial Intelligence (IJCAI)*, pp. 1907-1913, 2019. doi: 10.24963/ijcai.2019/264.

[26] L. Zhao, Y. Song, C. Zhang, Y. Liu, P. Wang, T. Lin, M. Deng, and H. Li, "T-GCN: A temporal graph convolutional network for traffic prediction," *IEEE Transactions on Intelligent Transportation Systems*, vol. 21, no. 9, pp. 3848-3858, 2020. doi: 10.1109/TITS.2019.2935152.

[27] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735-1780, 1997. doi: 10.1162/neco.1997.9.8.1735.

[28] L. Bai, L. Yao, C. Li, X. Wang, and C. Wang, "Adaptive graph convolutional recurrent network for traffic forecasting," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, 2020.

[29] S. Guo, Y. Lin, H. Wan, X. Li, and G. Cong, "Learning dynamics and heterogeneity of spatial-temporal graph data for traffic forecasting," *IEEE Transactions on Knowledge and Data Engineering*, vol. 34, no. 11, pp. 5415-5428, 2022.

[30] M. Xu, W. Dai, C. Liu, X. Gao, W. Lin, G.-J. Qi, and H. Xiong, "Spatial-temporal Transformer networks for traffic flow forecasting," arXiv:2001.02908 [cs.LG], 2020.

[31] C. M. Bishop, "Mixture density networks," Technical Report NCRG/94/004, Aston University, 1994.

[32] D. Salinas, V. Flunkert, J. Gasthaus, and T. Januschowski, "DeepAR: Probabilistic forecasting with autoregressive recurrent networks," *International Journal of Forecasting*, vol. 36, no. 3, pp. 1181-1191, 2020. doi: 10.1016/j.ijforecast.2019.07.001.

[33] K. Rasul, A.-S. Sheikh, I. Schuster, U. Bergmann, and R. Vollgraf, "Multivariate probabilistic time series forecasting via conditioned normalizing flows," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2021.

[34] T. Gneiting and A. E. Raftery, "Strictly proper scoring rules, prediction, and estimation," *Journal of the American Statistical Association*, vol. 102, no. 477, pp. 359-378, 2007.

[35] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. Int. Conf. Machine Learning (ICML)*, 2017.

[36] B. Lakshminarayanan, A. Pritzel, and C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.

[37] V. Vovk, A. Gammerman, and G. Shafer, *Algorithmic Learning in a Random World*. New York: Springer, 2005.

[38] A. N. Angelopoulos and S. Bates, "Conformal prediction: A gentle introduction," *Foundations and Trends in Machine Learning*, vol. 16, no. 4, pp. 494-591, 2023.

[39] D. B. Rubin, "Inference and missing data," *Biometrika*, vol. 63, no. 3, pp. 581-592, 1976.

[40] R. J. A. Little and D. B. Rubin, *Statistical Analysis with Missing Data*, 3rd ed. Hoboken, NJ: Wiley, 2019.

[41] Z. Che, S. Purushotham, K. Cho, D. Sontag, and Y. Liu, "Recurrent neural networks for multivariate time series with missing values," *Scientific Reports*, vol. 8, 6085, 2018.

[42] W. Cao, D. Wang, J. Li, H. Zhou, L. Li, and Y. Li, "BRITS: Bidirectional recurrent imputation for time series," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 31, 2018.

[43] A. Cini, I. Marisca, and C. Alippi, "Filling the gaps: Multivariate time series imputation by graph neural networks," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2022.

[44] M. M. Breunig, H.-P. Kriegel, R. T. Ng, and J. Sander, "LOF: Identifying density-based local outliers," in *Proc. ACM SIGMOD Int. Conf. Management of Data*, 2000.

[45] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," in *Proc. IEEE Int. Conf. Data Mining (ICDM)*, 2008.

[46] L. Ruff, R. Vandermeulen, N. Goernitz, L. Deecke, S. A. Siddiqui, A. Binder, E. Mueller, and M. Kloft, "Deep one-class classification," in *Proc. Int. Conf. Machine Learning (ICML)*, 2018.

[47] A. Deng and B. Hooi, "Graph neural network-based anomaly detection in multivariate time series," in *Proc. AAAI Conference on Artificial Intelligence*, 2021.

[48] H. Zhao, Y. Wang, J. Duan, C. Huang, D. Cao, Y. Tong, B. Xu, J. Bai, J. Tong, and Q. Zhang, "Multivariate time-series anomaly detection via graph attention network," in *Proc. IEEE Int. Conf. Data Mining (ICDM)*, 2020.

[49] R. Lowe, Y. Wu, A. Tamar, J. Harb, P. Abbeel, and I. Mordatch, "Multi-agent actor-critic for mixed cooperative-competitive environments," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.

[50] J. Foerster, G. Farquhar, T. Afouras, N. Nardelli, and S. Whiteson, "Counterfactual multi-agent policy gradients," in *Proc. AAAI Conference on Artificial Intelligence*, 2018.

[51] T. Rashid, M. Samvelyan, C. Schroeder de Witt, G. Farquhar, J. Foerster, and S. Whiteson, "QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning," in *Proc. Int. Conf. Machine Learning (ICML)*, 2018.

[52] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. Cambridge, MA: MIT Press, 2018.

[53] E. Altman, *Constrained Markov Decision Processes*. Boca Raton, FL: Chapman and Hall/CRC, 1999.

[54] J. Achiam, D. Held, A. Tamar, and P. Abbeel, "Constrained policy optimization," in *Proc. Int. Conf. Machine Learning (ICML)*, 2017.

[55] C. Tessler, D. J. Mankowitz, and S. Mannor, "Reward constrained policy optimization," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2019.

[56] C. Dwork, M. Hardt, T. Pitassi, O. Reingold, and R. Zemel, "Fairness through awareness," in *Proc. Innovations in Theoretical Computer Science Conference (ITCS)*, 2012.

[57] M. Hardt, E. Price, and N. Srebro, "Equality of opportunity in supervised learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 29, 2016.

[58] M. Donini, L. Oneto, S. Ben-David, J. Shawe-Taylor, and M. Pontil, "Empirical risk minimization under fairness constraints," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 31, 2018.

[59] A. Agarwal, A. Beygelzimer, M. Dudik, J. Langford, and H. Wallach, "A reductions approach to fair classification," in *Proc. Int. Conf. Machine Learning (ICML)*, 2018.

[60] S. Barocas, M. Hardt, and A. Narayanan, *Fairness and Machine Learning: Limitations and Opportunities*. Cambridge, MA: MIT Press, 2023.

[61] C. K. Chow, "On optimum recognition error and reject tradeoff," *IEEE Transactions on Information Theory*, vol. 16, no. 1, pp. 41-46, 1970.

[62] Y. Geifman and R. El-Yaniv, "Selective classification for deep neural networks," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.

[63] D. Madras, T. Pitassi, and R. Zemel, "Predict responsibly: Improving fairness and accuracy by learning to defer," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 31, 2018.

[64] H. Mozannar and D. Sontag, "Consistent estimators for learning to defer to an expert," in *Proc. Int. Conf. Machine Learning (ICML)*, 2020.

[65] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, "Communication-efficient learning of deep networks from decentralized data," in *Proc. Int. Conf. Artificial Intelligence and Statistics (AISTATS)*, 2017.

[66] P. Kairouz et al., "Advances and open problems in federated learning," *Foundations and Trends in Machine Learning*, vol. 14, no. 1-2, pp. 1-210, 2021.

[67] X. Zhang, P. Zhao, J. Jaskolka, H. Li, and R. Lu, "SecMLOps: A comprehensive framework for integrating security throughout the machine learning operations lifecycle," *Empirical Software Engineering*, vol. 31, no. 3, 74, 2026. doi: 10.1007/s10664-025-10795-y.

[68] T. K. T. Pham and D. A. Dewi, "Graph-theoretic Transformer for stochastic travel time prediction in mixed-mobility urban networks," in *Proc. Int. Conf. Intelligent Systems, Blockchain, and Communication Technologies (ISBCom)*, Lecture Notes in Networks and Systems. Cham, Switzerland: Springer, 2026, in press.

[69] T. K. T. Pham and D. A. Dewi, "Decentralized data integrity in mixed-mobility sensor networks: A multi-agent reinforcement learning approach," *Proc. SPIE*, vol. 14309, 143090D, 2026. doi: 10.1117/12.3120466.

[70] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2015.

[71] Q.-D. L. Tran, A.-C. Le, and V.-N. Huynh, "Enhancing conversational model with deep reinforcement learning and adversarial learning," *IEEE Access*, vol. 11, pp. 75955-75970, 2023. doi: 10.1109/ACCESS.2023.3297652.


---

## Appendix A: Configuration

The complete configuration is `configs/default.yaml`. The principal values are as follows.

- Seeds. Global seed 42, model seeds 0 to 4.
- Grid. 720 hours at four steps per hour over 28 nodes.
- Graph. Laplacian encoding dimension 8, functional similarity degree 3.
- Cascades. Base probability 0.55, decay 3.0 steps, 14 roots per centrality stratum.
- Outages. 8 gateway outages of 4 to 14 steps.
- Missingness. MNAR intercept minus 4.2, slope 0.055.
- Topology. Closure windows at steps 1,600 to 1,840 and 2,380 to 2,500, 6 edges each.
- Shift. From step 2,592, scale 1.12, added noise 1.10.
- Splits. Chronological 0.70, 0.10, 0.20.
- Forecast model. Window 12 steps, horizon 4 steps, hidden dimension 48, 4 heads, 2 layers, 3 mixture components.
- Multi-task model. Hidden dimension 40, loss weights 1.0, 0.5, 1.0, 1.0, 0.2 and 0.1.
- Policy. Learning rate 0.12, discount 0.85, exploration 0.15, dual step size 0.6, review cost 2,500 VND, corroboration cost 400 VND, false accept multiplier 2.0.
- Function approximation policy. 22 features, 40 epochs, batch 6,000, policy step size 0.05, value step size 0.05.

## Appendix B: Artifacts

Figures: Fig. 1 through Fig. 15 appear in the body at the point of first citation. Each is saved as PDF and 300 dpi PNG by the corresponding script in `scripts/figures/`, with a missing-input guard that reports and exits cleanly rather than plotting placeholder data.

Tables: `table_01_literature_taxonomy` through `table_15_external_metr_la`, plus `table_08b_missingness_sweep` and `table_09b_fairness_pareto`, each written as CSV and Markdown by the corresponding script in `scripts/tables/` from a saved result file.

Result files: `exp_a_fault_detection.csv`, `exp_a_root_cause.csv`, `exp_a_reconstruction.csv`, `exp_a_detection_delay.csv`, `exp_b_forecast_conditions.csv`, `exp_b_missingness_sweep.csv`, `exp_c_policy.csv`, `exp_c_pareto.csv`, `exp_c_dual_variables.csv`, `exp_d_ablation_model.csv`, `exp_d_ablation_policy.csv`, `exp_e_policy_gradient.csv`, `exp_e_cost_sensitivity.csv`, `exp_h_early_stopping.csv` and `exp_i_external_metr_la.csv`. The replication cohort writes the same set under `results/tables_kl/`, and the revised regime of Section VI-G under `results/tables_stress/`.

Documentation: `docs/DATASET_CARD.md`, `docs/MODEL_CARD.md`, `docs/EXPERIMENTAL_PROTOCOL.md` and `docs/REPRODUCIBILITY_CHECKLIST.md`.
