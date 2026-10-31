"""Trim the manuscript body back inside the word budget after Section VI-H.

The external validation subsection adds roughly 1,100 words, which pushes the
body over the 11,000 word budget the submission works to and over the page
guidance IEEE Access states. This script recovers the words by compressing
passages that say something twice, not by removing findings. Every cut is an
exact-match pair so the edit is reviewable and rerunnable, and the script
refuses to write anything if a single anchor fails to match, which is what
stops a partial edit from silently losing a batch of changes.

What is deliberately not touched: any number, any verdict sentence, any
limitation, and any statement of what the evidence does not support. The
passages compressed here are the introduction's restatement of its own
contributions list, the overlap between the dataset limitations and Section
VIII, and the two paragraphs of Section VIII that both reported the external
validation status.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "manuscript.md"

# Idempotency sentinel: a phrase that exists only after the trim.
MARK = "The work has three parts: a benchmark harmonizing"

# (label, old, new). Order does not matter, every pair must match exactly once.
EDITS: list[tuple[str, str, str]] = [

    # --- I. Introduction -------------------------------------------------
    # The three "parts" paragraphs restate the numbered contributions that
    # follow them. One compact paragraph replaces the three.
    ("intro three parts",
     """The work has three parts. The first is a benchmark. Three synthetic files exist from prior conference work of the present authors: a 720 by 28 hourly travel time matrix, a 100,000 row IoT telemetry log, and a 10,000 row transaction log. They were generated independently, share neither a node identity nor a clock, and contain fault mechanisms that do not correspond to graph structure. Section III documents how they are harmonized, including properties the source files were not previously reported to have.

The second part is the model. DyHGT-MARL combines relation-aware attention, in which each relation has its own projections and each attention logit carries both an edge bias tied to the active graph snapshot and a term conditioned on sender and receiver reliability, with causal temporal attention and six jointly trained task heads covering fault classification, root-cause attribution, masked reconstruction, a three-component mixture density forecast, calibration and temporal stability.

The third part is the decision layer. A constrained multi-agent formulation gives each zone-level agent an action set of seven options that includes not only accept and reject but also impute, corroborate, defer to human review, isolate, and escalate. Fairness, latency and risk enter as explicit budgets enforced by dual ascent rather than as post hoc reporting. The inclusion of deferral is deliberate, since a system with no option to say that it does not know produces errors that are guaranteed to land somewhere.

The contributions are as follows.""",
     """The work has three parts: a benchmark harmonizing three independently generated synthetic files that share neither a node identity nor a clock, an encoder combining relation-aware attention conditioned on sender and receiver reliability with causal temporal attention and six jointly trained heads, and a constrained decision layer whose seven actions include deferral, because a system with no option to say that it does not know produces errors that are guaranteed to land somewhere. The contributions are as follows."""),

    ("intro closing claim",
     """The work is an integrated methodology and a reproducible synthetic benchmark. It is not a deployed system, it has not been validated on real data, and no claim is made that it solves a real-city deployment problem.""",
     """The work is an integrated methodology and a reproducible synthetic benchmark. It is not a deployed system, only its forecasting component has been tested on real data, and no claim is made that it solves a real-city deployment problem."""),

    # --- II-L ------------------------------------------------------------
    # The incomparability of the source-paper figures is stated twice, once
    # as a rule and once per paper. The rule is enough.
    ("source paper figures",
     """**These figures are results of those individual papers on those individual corpora. They are not results of the integrated system described here, they are not comparable to the numbers in Section VI, and they are not reproduced by the code accompanying this paper.** The corpora differ, the splits differ, the metrics are computed differently, the travel time paper's coverage of 97.3 percent against a nominal 95 percent indicates interval over-coverage rather than the near-exact coverage obtained here, and the data integrity paper's 100.0 percent revenue recovery is attainable only where no record can be invalid, whereas the present environment penalizes accepting invalid records and so has a ceiling below 100 percent by construction. Table I places the three papers alongside the wider literature and marks what each supplies and what each leaves open.""",
     """**These figures are results of those individual papers on those individual corpora. They are not results of the integrated system described here, they are not comparable to the numbers in Section VI, and they are not reproduced by the code accompanying this paper.** The corpora, the splits and the metric definitions all differ, and two of the figures are not attainable here by construction: the travel time paper's 97.3 percent coverage against a nominal 95 percent is over-coverage rather than the near-exact coverage obtained here, and 100.0 percent revenue recovery requires an environment in which no record can be invalid. Table I places the three papers alongside the wider literature."""),

    # --- III-J -----------------------------------------------------------
    # Section VIII states these limitations at length. Here a pointer serves.
    ("dataset limitations",
     """Five limitations bound every conclusion here. The corpus is synthetic, so external validity is unknown. The traffic series spans 30 days, short for seasonality and shorter still for a distribution shift horizon. The mean pairwise node correlation of 0.909 leaves little spatial signal independent of a shared temporal factor. Faults are injected by overwriting telemetry, which leaves an almost separable signature. The transaction stream has no spatial ground truth, so its latent validity label is a modelling assumption. Section VI-G rebuilds the corpus to remove the third and fourth of these, and Section VIII returns to the rest.""",
     """Five limitations bound every conclusion drawn from this corpus. It is synthetic, so external validity is unknown. The series spans 30 days, short for seasonality and shorter still for a distribution shift horizon. A mean pairwise node correlation of 0.909 leaves little spatial signal independent of a shared temporal factor. Faults are injected by overwriting telemetry, which leaves an almost separable signature. The transaction stream has no spatial ground truth, so its latent validity label is a modelling assumption. Section VI-G rebuilds the corpus to remove the third and fourth, Section VI-H tests the first for forecasting alone, and Section VIII returns to the rest."""),

    # --- V ---------------------------------------------------------------
    ("protocol experiment list",
     """Experiment C evaluates six policies on the held-out transaction block across three seeds and sweeps the fairness budget over eight levels. Experiment D retrains eight model-side variants across two seeds and reads the policy-side switches from Experiment C, so one consistent set of policy numbers is reported. Experiment E repeats the constrained decision problem with a linear softmax policy trained by policy gradient with a learned value baseline across five seeds, then sweeps the penalty for accepting an invalid record across five settings. Experiment G varies the training budget alone. The delivered experimental protocol document records the remaining settings for each run.""",
     """Experiment C evaluates six policies on the held-out transaction block across three seeds and sweeps the fairness budget over eight levels. Experiment D retrains eight model-side variants across two seeds and reads the policy-side switches from Experiment C, so one consistent set of policy numbers is reported. Experiment E repeats the constrained decision problem with a linear softmax policy trained by policy gradient with a learned value baseline across five seeds, then sweeps the penalty for accepting an invalid record across five settings. Experiment G varies the training budget alone, and Experiment I repeats the comparison of Experiment B on METR-LA under the same protocol. The delivered experimental protocol document records the remaining settings for each run."""),

    # --- VI-F ------------------------------------------------------------
    ("replication scope",
     """Every result above comes from one corpus. To test whether they survive a change
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
cross-city generalization is claimed. One measurement caution follows from it.
The cohorts have different travel time scales, at means of roughly 25.6 and 6.1
minutes, because the primary cohort reads the original supplied matrix and the
replication reads the v2.0 table, so only scale free quantities and
within-cohort changes are compared below.""",
     """Every result above comes from one corpus. To test whether they survive a change
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
only scale free quantities and within-cohort changes are compared below."""),

    # --- VI-H ------------------------------------------------------------
    ("external scope paragraph",
     """The scope of the test needs stating. METR-LA has no fault labels, no
transactions and no gateway structure, so the fault detection results of
Section VI-A and the decision layer results of Section VI-D cannot be examined
on it and remain synthetic. What can be examined is probabilistic forecasting
and its behaviour as observations disappear. The protocol follows Section V as
closely as a different corpus allows, with chronological splits, selection on
the validation block, the test block used once, a Gaussian predictive
distribution fitted on validation residuals for the deterministic baselines,
and the same metric code. Readings are aggregated to the fifteen minute grid
the paper uses throughout, the adjacency is rebuilt from the published sensor
distance table with the thresholded Gaussian kernel of Li et al. [23], giving
mean degree 11.4, and three seeds are run.""",
     """The scope of the test needs stating. METR-LA has no fault labels, no
transactions and no gateway structure, so the fault detection results of
Section VI-A and the decision layer results of Section VI-D cannot be examined
on it and remain synthetic. What can be examined is probabilistic forecasting
and its behaviour as observations disappear. The protocol is that of Section V,
with the same metric code, chronological splits, selection on validation and
the test block used once. Readings are aggregated to the fifteen minute grid
the paper uses throughout, the adjacency is rebuilt from the published sensor
distance table with the thresholded Gaussian kernel of Li et al. [23], giving
mean degree 11.4, and three seeds are run."""),

    ("external correlation paragraph",
     """One property of the public corpus is worth recording on its own. Its mean
pairwise node correlation is 0.22, against 0.93 for the harmonized synthetic
corpus. The criticism Section VI-G makes of the generator, that a shared
temporal factor leaves the spatial channel with little to carry, is therefore
not a theoretical worry. The synthetic corpus is four times as correlated as
the real network it was built to resemble, and the revised regime at 0.55 sits
between the two without reaching the real value.""",
     """One property of the public corpus is worth recording on its own. Its mean
pairwise node correlation is 0.22, against 0.93 for the harmonized synthetic
corpus and 0.55 for the revised regime of Section VI-G. The criticism that
section makes of the generator, that a shared temporal factor leaves the
spatial channel with little to carry, is therefore not a theoretical worry: the
synthetic corpus is four times as correlated as the real network it was built
to resemble."""),

    ("external closing paragraph",
     """Two things follow, and they point in opposite directions. The forecasting and
calibration results now have support outside the authors' own generator, which
is the claim most exposed to the objection that the corpus is synthetic. The
degradation margin does not, so Section VII-A and Section IX report it as a
property of the synthetic corpus and the abstract no longer offers it as a
general result. Fault detection, reconstruction, the decision layer and the
fairness results remain untested outside the generator, because no public
corpus known to the authors carries fault labels, fare transactions and zone
structure together. Building one is the obvious next piece of work.""",
     """Two things follow, and they point in opposite directions. The forecasting and
calibration results now have support outside the authors' own generator, which
is the claim most exposed to the objection that the corpus is synthetic. The
degradation margin does not, so Section VII-A, Section IX and the abstract
report it as a property of the synthetic corpus rather than as a general
result. Fault detection, reconstruction, the decision layer and the fairness
results remain untested outside the generator, because no public corpus known
to the authors carries fault labels, fare transactions and zone structure
together. Building one is the obvious next piece of work."""),

    # --- VIII ------------------------------------------------------------
    # Two paragraphs reported the external validation status. One does.
    ("limitations external merge",
     """**Synthetic data for all but one experiment.** No conclusion here transfers to any real network without validation on real data, which has been done only for the forecasting comparison of Section VI-H. The benchmark measures whether methods behave sensibly under mechanisms that were deliberately inserted, not whether those mechanisms resemble reality. The replication cohort of Section VI-F does not change this, since both cohorts come from the same generator and share a topology, so it tests sensitivity to parameterisation rather than external validity.""",
     """**Synthetic data for all but one experiment.** Section VI-H tests the forecasting comparison on METR-LA, a public corpus the authors did not generate. It supports the score and the calibration, withdraws the degradation margin, and is the only result here with outside support. Every other conclusion rests on a corpus that measures whether methods behave sensibly under mechanisms deliberately inserted, not whether those mechanisms resemble reality, and no hardware measurement or deployment study supports any number in this paper. The replication cohort of Section VI-F does not change this, since both cohorts come from the same generator and share a topology, so it tests sensitivity to parameterisation rather than external validity."""),

    ("limitations external duplicate",
     """

**Limited external validation.** Section VI-H tests the forecasting claim on METR-LA, a public corpus the authors did not generate, and that is the only result here with outside support. It supports the score and the calibration and withdraws the degradation margin. No real dataset backs the fault detection, reconstruction, decision layer or fairness results, and no hardware measurement or deployment study supports any number in this paper.""",
     """

**No external test of the other components.** No real dataset backs the fault detection, reconstruction, decision layer or fairness results. No public corpus known to the authors carries fault labels, fare transactions and zone structure together, so testing those components externally requires building such a corpus rather than locating one."""),

    ("non binding constraint",
     """**Non-binding constraint in the primary benchmark.** The fairness budget never binds in either cohort of the primary benchmark. Section VI-G shows this is a property of that corpus and not of the method, since making latent validity and false accept cost depend on zone produces a gap the dual update then halves. The constraint machinery is therefore tested, but only in a regime the authors constructed for the purpose. The function approximation check of Section VI-D sharpens this for the primary cohort, where the observable state carries little information about latent validity and the cost sensitivity sweep found no setting at which a mixed strategy was preferred. Section VI-F shows that the second half of that statement is cohort specific, since the replication does produce a mixed policy at a high penalty. A benchmark revision in which validity rates, fare values or misclassification costs differ by zone is therefore required before any claim about fairness constrained control can be advanced on this corpus. The same revision would give the abstention and corroboration actions a regime in which they can pay for themselves.""",
     """**Non-binding constraint in the primary benchmark.** The fairness budget never binds in either cohort of the primary benchmark. Section VI-G shows this is a property of that corpus and not of the method, since making latent validity and false accept cost depend on zone produces a gap the dual update then halves. The constraint machinery is therefore tested, but only in a regime the authors constructed for the purpose. Section VI-D sharpens this for the primary cohort, where the observable state carries little information about latent validity and no cost setting favoured a mixed strategy, although Section VI-F shows the latter is cohort specific, since the replication does produce a mixed policy at a high penalty. Any claim about fairness constrained control on this corpus therefore depends on a revision in which validity rates, fare values or misclassification costs differ by zone, which is also what would give the abstention and corroboration actions a regime in which they can pay for themselves."""),
]


def main() -> None:
    t = SRC.read_text(encoding="utf-8")
    if MARK in t:
        sys.exit("already trimmed")

    def words(x: str) -> int:
        body = re.sub(r"\$\$.*?\$\$", " ",
                      re.sub(r"\[\[FIGURE:[^\]]+\]\]", "",
                             x.split("## Author Contributions")[0]), flags=re.S)
        keep = "\n".join(l for l in body.split("\n") if not l.lstrip().startswith("|"))
        return len(re.sub(r"[#*`>]", " ", keep).split())

    before = words(t)
    bad = []
    for label, old, new in EDITS:
        n = t.count(old)
        if n != 1:
            bad.append(f"{label}: matched {n} times")
            continue
        t = t.replace(old, new, 1)
    if bad:
        print("anchors failed, nothing written:")
        for b in bad:
            print("  -", b)
        sys.exit(1)

    SRC.write_text(t, encoding="utf-8")
    after = words(t)
    print(f"body words {before} -> {after}  ({before - after} removed)")
    print("semicolons:", t.count(";"), " em dash:", t.count(chr(8212)))


if __name__ == "__main__":
    main()
