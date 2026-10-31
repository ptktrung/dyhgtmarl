"""ALREADY APPLIED. This is a one-time migration that transformed the
October 2026 revision of docs/manuscript.md. It is kept for provenance.
Running it again will stop with an anchor error, because the text it
looks for has already been replaced.

pply the P2 revision pack to docs/manuscript.md.

Three passes. The first removes semicolons and inflated wording. The second
inserts the author block change, the back matter, the function approximation
results and the citations that the revision pack requires. The third renumbers
every citation into order of first appearance, which is what IEEE style
requires and what Word's IEEE bibliography style will compute on a field
update.
"""
from __future__ import annotations

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

SRC = pathlib.Path(__file__).resolve().parents[1] / "docs" / "manuscript.md"

# ---------------------------------------------------------------------------
# Pass 1: semicolons and wording
# ---------------------------------------------------------------------------
SEMICOLONS = [
    ("introduced to motivate a fairness term; it is the direct consequence",
     "introduced to motivate a fairness term. It is the direct consequence"),
    ("no shortage of methods for each component; what is missing is",
     "no shortage of methods for each component. What is missing is"),
    ("""Dwivedi and Bresson [4] generalized the Transformer to graphs using Laplacian eigenvectors as positional encodings; Kreuzer et al. [5] introduced learned spectral attention; Ying et al. [6] encoded structure through centrality, spatial and edge biases; and Rampasek et al. [7] combined local message passing with global attention in a general recipe.""",
     """Dwivedi and Bresson [4] generalized the Transformer to graphs using Laplacian eigenvectors as positional encodings. Kreuzer et al. [5] introduced learned spectral attention, Ying et al. [6] encoded structure through centrality, spatial and edge biases, and Rampasek et al. [7] combined local message passing with global attention in a general recipe. All three build on the self-attention formulation of Vaswani et al. [63]."""),
    ("""DCRNN [17] treats traffic as a diffusion process on a directed graph; STGCN [18] interleaves graph and temporal convolutions; Graph WaveNet [19] learns an adaptive adjacency matrix in addition to the given one; T-GCN [20] couples graph convolution with gated recurrence; AGCRN [21] learns node-specific parameters; and attention-based spatiotemporal networks [22] apply self-attention across both axes.""",
     """DCRNN [17] treats traffic as a diffusion process on a directed graph and STGCN [18] interleaves graph and temporal convolutions. Graph WaveNet [19] learns an adaptive adjacency matrix in addition to the given one. T-GCN [20] couples graph convolution with the gated recurrence that descends from long short-term memory [64], AGCRN [21] learns node-specific parameters, and attention-based spatiotemporal networks [22] apply self-attention across both axes."""),
    ("reduction approaches to fair classification [52] form the basis; the textbook treatment is [53].",
     "reduction approaches to fair classification [52] form the basis. The textbook treatment is [53]."),
    ("clips its output at 4.0 minutes and produces twin morning and evening peaks; the supplied CSV has neither property",
     "clips its output at 4.0 minutes and produces twin morning and evening peaks, and the supplied CSV has neither property"),
    ("would fabricate temporal structure; regenerating from measured marginals makes the assumption explicit.",
     "would fabricate temporal structure. Regenerating from measured marginals makes the assumption explicit."),
    ("Derived from the gateway assignment; encodes correlated outage risk.",
     "Derived from the gateway assignment, and encodes correlated outage risk."),
    ("carry no information that message passing could exploit; they are consumed by the policy layer instead.",
     "carry no information that message passing could exploit. They are consumed by the policy layer instead."),
    ("**in the snapshot active at step $t$**; when an edge is suppressed by a closure",
     "**in the snapshot active at step $t$**. When an edge is suppressed by a closure"),
    ("Conformal calibration was considered and is **not implemented**; it is discussed in Section VIII",
     "Conformal calibration was considered and is **not implemented**. It is discussed in Section VIII"),
    ("""the demographic parity gap, the maximum difference in acceptance rate between zones; the equal opportunity gap, the maximum difference in acceptance rate among genuinely valid records; and the false positive rate gap, the difference in acceptance rate among genuinely invalid records.""",
     """the demographic parity gap, which is the maximum difference in acceptance rate between zones, the equal opportunity gap, which is the maximum difference in acceptance rate among genuinely valid records, and the false positive rate gap, which is the difference in acceptance rate among genuinely invalid records."""),
    ("are all computed on the validation block; the test block is used once per configuration.",
     "are all computed on the validation block. The test block is used once per configuration."),
    ("with each masked cell scored exactly once; methods that use future observations are flagged",
     "with each masked cell scored exactly once. Methods that use future observations are flagged"),
    ("whose per-node scale is fitted on validation residuals; without this step the continuous ranked probability score",
     "whose per-node scale is fitted on validation residuals. Without this step the continuous ranked probability score"),
    ("is neither guaranteed nor verified; anyone rerunning this work should treat their own environment",
     "is neither guaranteed nor verified. Anyone rerunning this work should treat their own environment"),
    ("requiring structural context to separate from congestion; they are near-deterministic markers",
     "requiring structural context to separate from congestion. They are near-deterministic markers"),
    ("the model holds at CRPS 1.689 and PICP 0.951; under the test-block topology change",
     "the model holds at CRPS 1.689 and PICP 0.951. Under the test-block topology change"),
    ("are simulator assumptions and are labelled as such; they are not hardware benchmarks",
     "are simulator assumptions and are labelled as such. They are not hardware benchmarks"),
    ("indicates intervals wider than the data require; the present model attains 0.952",
     "indicates intervals wider than the data require. The present model attains 0.952"),
    ("every record is accepted and none can be invalid; the present environment contains genuinely invalid records",
     "every record is accepted and none can be invalid. The present environment contains genuinely invalid records"),
    ("under mechanisms that were deliberately inserted; it cannot establish that those mechanisms resemble reality.",
     "under mechanisms that were deliberately inserted. It cannot establish that those mechanisms resemble reality."),
]

APPENDIX_A_OLD = """The complete configuration is `configs/default.yaml`. Principal values: global seed 42; model seeds 0 to 4; grid of 720 hours at four steps per hour over 28 nodes; Laplacian encoding dimension 8; functional similarity degree 3; cascade base probability 0.55 with decay 3.0 steps and 14 roots per centrality stratum; 8 gateway outages of 4 to 14 steps; MNAR intercept minus 4.2 with slope 0.055; closure windows at steps 1,600 to 1,840 and 2,380 to 2,500 with 6 edges each; distribution shift from step 2,592 with scale 1.12 and added noise 1.10; chronological split 0.70, 0.10, 0.20; forecast window 12 steps and horizon 4 steps with hidden dimension 48, 4 heads, 2 layers and 3 mixture components; multi-task hidden dimension 40 with loss weights 1.0, 0.5, 1.0, 1.0, 0.2, 0.1; policy learning rate 0.12, discount 0.85, exploration 0.15, dual step size 0.6, review cost 2,500 VND, corroboration cost 400 VND, false-accept multiplier 2.0."""

APPENDIX_A_NEW = """The complete configuration is `configs/default.yaml`. The principal values are as follows.

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
- Function approximation policy. 22 features, 40 epochs, batch 6,000, policy step size 0.05, value step size 0.05."""

WORDING = [
    ("Strong structural encoders, evaluated mostly on static benchmarks",
     "Structural encoders evaluated mostly on static benchmarks"),
    ("Kreuzer et al. 2021; Ying et al. 2021; Rampasek et al. 2022",
     "Kreuzer et al. 2021; Ying et al. 2021; Rampasek et al. 2022"),
]


def pass_one(t: str) -> str:
    for a, b in SEMICOLONS:
        if a not in t:
            sys.exit(f"PASS 1 anchor not found: {a[:70]}")
        t = t.replace(a, b)
    if APPENDIX_A_OLD not in t:
        sys.exit("PASS 1 appendix anchor not found")
    t = t.replace(APPENDIX_A_OLD, APPENDIX_A_NEW)
    t = t.replace("A simple climatological baseline beats every neural baseline tested.",
                  "A climatological baseline beats every neural baseline tested.")
    for a, b in WORDING:
        t = t.replace(a, b)
    return t


# ---------------------------------------------------------------------------
# Pass 2: content
# ---------------------------------------------------------------------------
AUTHOR_OLD = """**Thai Ky Trung Pham** (1,2), **Deshinta Arrova Dewi** (1)

(1) Faculty of Data Science and Information Technology, INTI International University, Persiaran Perdana BBN Putra Nilai, 71800 Nilai, Negeri Sembilan, Malaysia
(2) Department of Computer Science, Swinburne Vietnam, FPT University, A35 Bach Dang Street, Tan Hoa Son Ward, Ho Chi Minh City, Vietnam

Corresponding author: trungptk@fe.edu.vn"""

AUTHOR_NEW = """**Thai Ky Trung Pham** (1,2), **Quoc-Dai Luong Tran** (3), **Deshinta Arrova Dewi** (1)

(1) Faculty of Data Science and Information Technology, INTI International University, Persiaran Perdana BBN Putra Nilai, 71800 Nilai, Negeri Sembilan, Malaysia
(2) Department of Computer Science, Swinburne Vietnam, FPT University, A35 Bach Dang Street, Tan Hoa Son Ward, Ho Chi Minh City, Vietnam
(3) [Faculty to be confirmed by the author], Ton Duc Thang University, Ho Chi Minh City, Vietnam

Corresponding author: Quoc-Dai Luong Tran (tranquocdailuong@tdtu.edu.vn)"""

ABSTRACT_OLD = """Results that do not favour the proposed system are reported in full: gradient boosting and a multilayer perceptron detect faults far more accurately than the graph model on this corpus, non-causal interpolation reconstructs masked cells better, calibration collapses under distribution shift, and the fairness constraint is not binding because latent validity is zone-independent by construction. These outcomes are traced to specific properties of the source generators and are treated as findings about the benchmark rather than as incidental noise."""

ABSTRACT_NEW = """Results that do not favour the proposed system are reported in full. Gradient boosting and a multilayer perceptron detect faults far more accurately than the graph model on this corpus, non-causal interpolation reconstructs masked cells better, and calibration collapses under distribution shift. The fairness constraint never binds, because latent validity is independent of zone by construction. A function approximation check replaces the zone-level action value tables with a linear policy trained by policy gradient and reaches 93.80 percent revenue recovery by accepting every record, which is higher than any tabular variant and uses none of the abstention machinery the decision layer provides. A cost sensitivity sweep shows the learned policy is constant at every penalty setting tested, which indicates that the observable state in this benchmark carries little information about latent validity. These outcomes are traced to specific properties of the source generators and are treated as findings about the benchmark rather than as incidental noise."""

# Section IV-F clarification of what multi-agent means here
MARL_SCOPE_ANCHOR = """Decentralization is implemented as one action-value table per zone with independent updates, against a centralized variant with a single table. Both are compared in Section VI."""

MARL_SCOPE_NEW = """Decentralization is implemented as one action-value table per zone with independent updates, against a centralized variant with a single table. Both are compared in Section VI. The term multi-agent is used here in a restricted sense. The agents are independent learners partitioned by zone, with no communication channel, no joint value factorization and no opponent modelling, so the layer is weaker than the methods surveyed in [42] to [44]. Section VI-D reports that the centralized and decentralized variants agree within one standard deviation, which is consistent with that restricted reading."""

# Section VI-D function approximation result
VID_ANCHOR = """This is a negative result about the experiment, not about fairness-constrained learning. A binding constraint requires that the utility-maximizing policy genuinely prefer to treat one group worse, which happens when validity rates, fare values or error costs differ across groups. Section VIII specifies what would need to change."""

VID_NEW = VID_ANCHOR + """

**Function approximation check.** To test whether these conclusions depend on the table representation, a linear softmax policy with a learned value baseline was trained by policy gradient on the same constrained process, using the same state information, the same seven actions, the same financial reward, the same dual update, the same chronological split and the same five seeds. The policy maps a 22 dimensional feature vector, built from the six discretized state components together with the hardware score, the upstream fault probability, the node reliability index and the normalized fare, to a distribution over actions, so parameters are shared across states rather than stored per state.

The result does not support the decision layer as designed. Over five seeds the function approximation policy reached a net revenue recovery of 93.80 percent with a standard deviation below 0.01, against 88.96 percent for the tabular decentralized policy, and an equal opportunity gap of 0.0000 against 0.0013. It achieved this by accepting every record. Abstention, corroboration and deferral were each selected in zero percent of decisions, human review workload fell to zero from 13.5 percent, and the decision error rate settled at 0.0211, which is the proportion of records that are invalid. The three variants tested, decentralized constrained, centralized constrained and decentralized unconstrained, produced identical policies, which confirms through a second representation the Section VI-D finding that the fairness budget does not bind and that centralization makes no difference on this corpus.

[[FIGURE:fig_13_function_approximation_policy]]

The fairness gaps of 0.0000 are a property of a constant policy rather than evidence of equitable treatment. A policy that accepts every record treats both zones identically for the trivial reason that it does not condition on anything. Table XII reports both representations side by side.

A cost sensitivity sweep locates the reason. Holding everything else fixed and raising only the penalty for accepting an invalid record, the learned policy accepts every record at multipliers 2, 5 and 10, recovering 93.80, 87.60 and 77.27 percent respectively, and switches to deferring every record at multipliers 20 and 40, recovering 69.38 percent with zero decision errors and full review workload. At no setting did the policy produce a mixed strategy. Because the observable state carries little information about latent validity, the optimal policy is constant, and only the cost ratio decides which constant it is. This is the clearest single piece of evidence that the selective machinery of Section IV-F cannot be evaluated on the present benchmark, and it strengthens rather than weakens the revision proposed in Section VIII."""

# Section VIII scope addition with the new citation
VIII_ANCHOR = """**Scope of the implementation.** Four of seven declared relations participate in message passing. Loss weights were fixed and never tuned, so the multi-task trade-off documented in Section VI-E is unresolved rather than optimized. Eigenvector sign is fixed by convention rather than by a sign-invariant architecture, so the encoder inherits the ambiguity discussed in [8]. Conformal calibration is discussed but not implemented. Graph WaveNet was not reimplemented and is therefore absent from the comparison rather than approximated under its own name."""

VIII_NEW = """**Scope of the implementation.** Four of seven declared relations participate in message passing. Loss weights were fixed and never tuned, so the multi-task trade-off documented in Section VI-E is unresolved rather than optimized. Eigenvector sign is fixed by convention rather than by a sign-invariant architecture, so the encoder inherits the ambiguity discussed in [8]. Conformal calibration is discussed but not implemented. Graph WaveNet was not reimplemented and is therefore absent from the comparison rather than approximated under its own name. The main decision layer is tabular. Function approximation policies, including policy gradient training with learned reward signals of the kind used for conversational agents [71], are needed for continuous or high dimensional states, and the linear case is examined in the check reported in Section VI-D. Nonlinear function approximation, communication between agents and joint value factorization are not examined here."""

# Limitations: non-binding constraint, updated with Experiment E
NONBIND_ANCHOR = """**Non-binding constraint.** The fairness budget never binds, so the constrained-optimization machinery is untested in the regime it was built for. A benchmark revision in which validity rates, fare values or misclassification costs differ by zone would be required to test it, and that revision should be made before any claim about fairness-constrained control is advanced on this corpus."""

NONBIND_NEW = """**Non-binding constraint.** The fairness budget never binds, so the constrained optimization machinery is untested in the regime it was built for. The function approximation check of Section VI-D sharpens this. Because the observable state carries little information about latent validity, the optimal policy is constant, and the cost sensitivity sweep found no setting at which a mixed strategy was preferred. A benchmark revision in which validity rates, fare values or misclassification costs differ by zone is therefore required before any claim about fairness constrained control can be advanced on this corpus. The same revision would give the abstention and corroboration actions a regime in which they can pay for themselves."""

# P1 and P3 overlap statement, Section VII
OVERLAP_ANCHOR = """### D. Governance implications"""

OVERLAP_NEW = """### D. Relation to the companion manuscripts

Two companion manuscripts by the present authors draw on the same three synthetic source files. Because reviewers will see overlapping corpora, the division is stated here. The fault detection study [9] operates on the IoT telemetry file alone, with a static graph, a single detection objective and no decision layer. The travel time study [61] operates on the traffic matrix alone, with a static graph and a single forecasting objective. The present work is distinguished by four things that neither companion contains: a harmonized corpus in which all three files share one clock and one node identity, a cascade mechanism that propagates over documented adjacency rather than over row order, a topology that changes inside both the training and test blocks, and a decision layer scored against a latent validity label that did not previously exist. The results reported here are not comparable to those of either companion, as Section VII-C sets out, and none of the numbers in this paper is reused from them.

### E. Governance implications"""

BACKMATTER = """
---

## Author Contributions

T. K. T. Pham: conceptualization, data curation, software, formal analysis, investigation, visualization, writing of the original draft. Q.-D. L. Tran: methodology for the reinforcement learning decision layer, validation of the policy experiments, writing through review and editing, project administration for submission. D. A. Dewi: supervision, methodology review, writing through review and editing. All authors approved the final manuscript and agree to be accountable for the work.

## Funding

[To be completed before submission. State any grant number and funder, or state that the research received no specific grant from any funding agency in the public, commercial or not-for-profit sectors.]

## Data and Code Availability

All data used in this study are synthetic and are regenerated in full by the scripts accompanying the manuscript. The code package contains the three supplied source files, the generators that rebuild the integrated benchmark from a seed, the model and baseline implementations, the experiment scripts, and one script per figure and per table. Configuration and seeds are in a single file. [Insert the repository or archive location before submission. Do not state a location until it exists.] Queries about the code and data may be directed to T. K. T. Pham at trungptk@fe.edu.vn.

## Competing Interests

The authors declare no competing financial interests. Reference [71] is authored by Q.-D. L. Tran, who is an author of the present manuscript. It is cited once, for a methodological point about policy gradient training with learned reward signals.

## Ethics Statement

This study uses only synthetic data. No human participants, personal data, passenger records or municipal records were involved at any stage. The node names are inspired by the modal mix of a large city but do not correspond to real infrastructure, and no result reported here supports an operational claim about any real transport network.
"""


def pass_two(t: str) -> str:
    pairs = [
        (AUTHOR_OLD, AUTHOR_NEW),
        (ABSTRACT_OLD, ABSTRACT_NEW),
        (MARL_SCOPE_ANCHOR, MARL_SCOPE_NEW),
        (VID_ANCHOR, VID_NEW),
        (VIII_ANCHOR, VIII_NEW),
        (NONBIND_ANCHOR, NONBIND_NEW),
        (OVERLAP_ANCHOR, OVERLAP_NEW),
    ]
    for a, b in pairs:
        if a not in t:
            sys.exit(f"PASS 2 anchor not found: {a[:70]}")
        t = t.replace(a, b, 1)

    # place the remaining uncited references where they belong
    cites = [
        ("""The encoder addresses all three with a single attention formulation. For relation $r$ at step $t$,""",
         """The encoder addresses all three with a single attention formulation, whose symbols are collected in Table IV. For relation $r$ at step $t$,"""),
        ("""Experiment C evaluates six policies on the held-out transaction block across three seeds, and sweeps the fairness budget over eight levels to trace a trade-off curve. Experiment D retrains eight model-side variants across two seeds and reads the policy-side switches from Experiment C so that one consistent set of policy numbers is reported.""",
         """Experiment C evaluates six policies on the held-out transaction block across three seeds, and sweeps the fairness budget over eight levels to trace a trade-off curve. Experiment D retrains eight model-side variants across two seeds and reads the policy-side switches from Experiment C so that one consistent set of policy numbers is reported. Experiment E repeats the constrained decision problem with a linear softmax policy trained by policy gradient with a learned value baseline across five seeds, and then sweeps the penalty for accepting an invalid record across five settings to locate the point at which selective actions become worthwhile.

All neural models are optimized with Adam [65] at a learning rate of 0.002, with gradient norm clipping at 2.0 and early stopping on the validation objective. The function approximation policy of Experiment E uses plain stochastic gradient ascent on the policy parameters and the value baseline, with step size 0.05 for both."""),
        ("Learning on non-Euclidean domains was consolidated by Bronstein et al. [1]",
         "Graph methods are increasingly applied to the reliability and resilience of infrastructure systems [66]. Learning on non-Euclidean domains was consolidated by Bronstein et al. [1]"),
        ("Typed relations were introduced to graph neural networks by Schlichtkrull et al. [10]",
         "Urban knowledge graphs such as UrbanKG [70] show the same typed formulation applied to city scale entities. Typed relations were introduced to graph neural networks by Schlichtkrull et al. [10]"),
        ("""Municipal mobility platforms increasingly rest on large sensor estates""",
         """Reviews of smart city technology [67] and of smart city planning for urban resilience [68] describe a steady move toward instrumented municipal services, and proposed sixth generation network architectures assume the same density of connected devices [69]. Municipal mobility platforms increasingly rest on large sensor estates"""),
    ]
    for a, b in cites:
        if a not in t:
            sys.exit(f"PASS 2 citation anchor not found: {a[:70]}")
        t = t.replace(a, b, 1)

    # new reference entry and corrected verification note
    ref_anchor = """[70] Y. Liu, Q. Ding, and Y. Li, "UrbanKG: An urban knowledge graph system," *ACM Transactions on Intelligent Systems and Technology*, vol. 14, no. 4, 2023."""
    if ref_anchor not in t:
        sys.exit("PASS 2 reference anchor not found")
    t = t.replace(ref_anchor, ref_anchor + """

[71] Q.-D. L. Tran, A.-C. Le, and V.-N. Huynh, "Enhancing conversational model with deep reinforcement learning and adversarial learning," *IEEE Access*, vol. 11, pp. 75955-75970, 2023. doi: 10.1109/ACCESS.2023.3297652.""", 1)

    note_old = """**Note on reference verification.** References [9], [61] and [62] are the authors' own source papers. References [18], [19], [20], [25], [60], [64], [66], [67], [68] and [69] carry digital object identifiers transcribed from the reference lists of those source papers. The remaining entries are widely cited works whose bibliographic details are stated to the best of the authors' knowledge but **have not been independently verified against publisher records for this manuscript**, and every entry should be checked against the publisher of record before submission. No digital object identifier has been supplied where one was not available from a verified source."""
    note_new = """**Note on reference verification.** References [9], [61] and [62] are the authors' own source papers. Reference [71] was verified against the publisher record, including the page range and the digital object identifier. A further group of entries carries digital object identifiers transcribed from the reference lists of the source papers. The remaining entries are widely cited works whose bibliographic details are stated to the best of the authors' knowledge but **have not been independently verified against publisher records for this manuscript**, and every entry should be checked against the publisher of record before submission. No digital object identifier has been supplied where one was not available from a verified source."""
    if note_old not in t:
        sys.exit("PASS 2 verification note not found")
    t = t.replace(note_old, note_new, 1)

    # back matter before the references
    t = t.replace("\n## References\n", BACKMATTER + "\n## References\n", 1)

    # appendix B gains the new artifacts
    t = t.replace("`exp_d_ablation_model.csv` and `exp_d_ablation_policy.csv`.",
                  "`exp_d_ablation_model.csv`, `exp_d_ablation_policy.csv`, "
                  "`exp_e_policy_gradient.csv` and `exp_e_cost_sensitivity.csv`.")
    t = t.replace("Figures: Fig. 1 through Fig. 12 appear in the body",
                  "Figures: Fig. 1 through Fig. 13 appear in the body")
    t = t.replace("`table_01_literature_taxonomy` through `table_11_ablation`",
                  "`table_01_literature_taxonomy` through `table_12b_cost_sensitivity`")
    return t


# ---------------------------------------------------------------------------
# Pass 3: renumber citations into order of first appearance
# ---------------------------------------------------------------------------
def pass_three(t: str):
    body, refs = t.split("## References", 1)

    # collect reference entries
    entries = {}
    for m in re.finditer(r"^\[(\d+)\] (.+?)(?=\n\n|\n\*\*Note)", refs, re.S | re.M):
        entries[int(m.group(1))] = m.group(2).strip()

    # order of first appearance in the body, expanding ranges like [42] to [44]
    order, seen = [], set()
    for m in re.finditer(r"\[(\d+)\]", body):
        n = int(m.group(1))
        if n not in seen:
            seen.add(n); order.append(n)
    missing = [n for n in sorted(entries) if n not in seen]
    if missing:
        sys.exit(f"PASS 3: reference entries still uncited: {missing}")
    unknown = [n for n in order if n not in entries]
    if unknown:
        sys.exit(f"PASS 3: citations without an entry: {unknown}")

    mapping = {old: i + 1 for i, old in enumerate(order)}
    body = re.sub(r"\[(\d+)\]", lambda m: f"[[{mapping[int(m.group(1))]}]]", body)
    body = body.replace("[[", "[").replace("]]", "]")
    # protect the figure markers that use double brackets
    body = re.sub(r"\[FIGURE:([a-z0-9_]+)\]", r"[[FIGURE:\1]]", body)

    new_refs = ["", ""]
    for old in order:
        new_refs.append(f"[{mapping[old]}] {entries[old]}")
        new_refs.append("")
    note = re.search(r"(\*\*Note on reference verification\.\*\*.*)$", refs, re.S)
    tail = "\n" + note.group(1).strip() + "\n" if note else ""

    return body + "## References\n" + "\n".join(new_refs) + tail, mapping, entries, order


def main():
    t = SRC.read_text(encoding="utf-8")
    t = pass_one(t)
    t = pass_two(t)
    import trim_manuscript
    t = trim_manuscript.apply(t)
    t, mapping, entries, order = pass_three(t)
    SRC.write_text(t, encoding="utf-8")

    body = re.sub(r"\[\[FIGURE:[^\]]+\]\]", "", t.split("## References")[0])
    words = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", body))
    print(f"semicolons remaining: {t.count(';')}")
    print(f"em dash: {t.count(chr(8212))}  en dash: {t.count(chr(8211))}")
    print(f"references renumbered: {len(order)}")
    print(f"words excluding references and back matter: {words}")
    moved = [(o, n) for o, n in sorted(mapping.items()) if o != n]
    print(f"citations whose number changed: {len(moved)}")


if __name__ == "__main__":
    main()
