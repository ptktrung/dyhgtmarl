"""Final prose pass: correct counts, split overlong sentences, apply the triad.

Three kinds of change, and nothing else.

First, defects. Section VI-F announced three findings and then listed four, and
Section VII-B carried an ungrammatical clause. Both are the sort of thing a
reviewer notices before they reach the argument.

Second, sentence length. A handful of sentences ran past fifty-five words,
which is where a reader starts rereading. Each is split at a clause boundary.
No claim, number or qualification is dropped, so the split costs nothing except
the words that the joining conjunctions used.

Third, the rule of three, applied where the argument already has three parts
and the sentence hid them inside one clause. Making the three visible is a
presentation change. Nowhere is a fourth item dropped or a third invented to
reach the pattern, because a triad bought that way misrepresents the work.

Run after apply_author_review.py. `tools/check_language.py` reports what is
left.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "manuscript.md"

EDITS: list[tuple[str, str, str]] = [

    # ---- defects -------------------------------------------------------
    ("VII-B: ungrammatical clause",
     "The encoder still loses, and the epoch sweep ruling out under-training "
     "within the range tested.",
     "The encoder still loses, and the epoch sweep rules out under-training "
     "within the range tested."),

    # Four items followed the announcement of three. Forecast accuracy and the
    # within-cohort ranking are one finding about accuracy, so they are stated
    # as one, which makes the count true and the triad real.
    ("VI-F: count of replicated findings",
     "Three findings replicate. Relative forecast accuracy is close, at MAPE\n"
     "13.64 against 9.08 percent. Calibration holds in\n"
     "both, at coverage 0.952 against 0.953 for a nominal\n"
     "0.95 and calibration error 0.023 against 0.041. Within\n"
     "each cohort the model remains the best scoring forecaster. Most importantly "
     "the robustness\nresult reproduces:",
     "Three findings replicate. Forecast accuracy is close, at MAPE 13.64\n"
     "against 9.08 percent, and within each cohort the model remains the best\n"
     "scoring forecaster. Calibration holds in both, at coverage 0.952 against\n"
     "0.953 for a nominal 0.95 and calibration error 0.023 against 0.041. Most\n"
     "importantly the robustness result reproduces:"),

    # ---- sentence length, split at clause boundaries --------------------
    ("I: the three parts",
     "The work has three parts: a benchmark harmonizing three independently "
     "generated synthetic files that share neither a node identity nor a clock, "
     "an encoder combining relation-aware attention conditioned on sender and "
     "receiver reliability with causal temporal attention and six jointly trained "
     "heads, and a constrained decision layer whose seven actions include "
     "deferral, because a system with no option to say that it does not know "
     "produces errors that are guaranteed to land somewhere. The contributions "
     "are as follows.",
     "The work has three parts. The first is a benchmark, which harmonizes three "
     "independently generated synthetic files that share neither a node identity "
     "nor a clock. The second is an encoder, which combines relation-aware "
     "attention conditioned on sender and receiver reliability with causal "
     "temporal attention and six jointly trained heads. The third is a "
     "constrained decision layer, whose seven actions include deferral, because a "
     "system with no option to say that it does not know produces errors that are "
     "guaranteed to land somewhere. The contributions are as follows."),

    ("II-L: incomparability of the source figures",
     "The corpora, the splits and the metric definitions all differ, and two of "
     "the figures are not attainable here by construction: the travel time "
     "paper's 97.3 percent coverage against a nominal 95 percent is over-coverage "
     "rather than the near-exact coverage obtained here, and 100.0 percent "
     "revenue recovery requires an environment in which no record can be invalid.",
     "The corpora, the splits and the metric definitions all differ. Two of the "
     "figures are also unattainable here by construction. The travel time paper's "
     "97.3 percent coverage against a nominal 95 percent is over-coverage rather "
     "than the near-exact coverage obtained here, and 100.0 percent revenue "
     "recovery requires an environment in which no record can be invalid."),

    ("IV-F: the state tuple",
     "where the upstream fault probability is supplied either by the trained "
     "encoder's own calibrated fault head, which couples the two halves of the "
     "system at inference, or by a logistic estimator fitted on training "
     "telemetry, which serves as a deliberately decoupled control.",
     "The upstream fault probability comes from one of two sources. Either the "
     "trained encoder's own calibrated fault head supplies it, which couples the "
     "two halves of the system at inference, or a logistic estimator fitted on "
     "training telemetry does, which serves as a deliberately decoupled control."),

    ("VI-G: coupling result",
     "probability, with everything else fixed, raises net revenue recovery on the "
     "revised corpus from 75.12 to 79.45 percent while raising the decision error "
     "rate from 0.0359 to 0.0377 and widening the equal opportunity gap from "
     "0.0136 to 0.0273.",
     "probability, with everything else fixed, raises net revenue recovery on the "
     "revised corpus from 75.12 to 79.45 percent. It also raises the decision "
     "error rate from 0.0359 to 0.0377 and widens the equal opportunity gap from "
     "0.0136 to 0.0273."),

    ("VII-A: what holds on both corpora",
     "What holds on both corpora is calibrated probabilistic forecasting: a "
     "dynamic heterogeneous graph Transformer with an explicit missingness channel "
     "and a mixture density output attains the best continuous ranked probability "
     "score and the best calibration of the models compared, on the synthetic "
     "benchmark and on METR-LA, and keeps the best score at every missingness "
     "level on both.",
     "What holds on both corpora is calibrated probabilistic forecasting. A "
     "dynamic heterogeneous graph Transformer with an explicit missingness channel "
     "and a mixture density output attains the best continuous ranked probability "
     "score and the best calibration of the models compared, on the synthetic "
     "benchmark and on METR-LA alike, and it keeps the best score at every "
     "missingness level on both."),

    ("VI-H: the mixture density trade",
     "The mixture density head buys a predictive distribution that is accurate "
     "about its own width rather than a sharper conditional mean, and on a corpus "
     "carrying real sensor noise that trade is visible.",
     "The mixture density head buys a predictive distribution that is accurate "
     "about its own width rather than a sharper conditional mean. On a corpus "
     "carrying real sensor noise, that trade becomes visible."),

    ("VI-G: early stopping",
     "Early stopping is disabled here, because on this corpus it halts the fault "
     "head at 5 epochs and returns a degenerate model, F1 0.109 at precision "
     "0.058, against 0.221 once the budget is allowed to bind.",
     "Early stopping is disabled here. On this corpus it halts the fault head at "
     "5 epochs and returns a degenerate model, F1 0.109 at precision 0.058, "
     "against 0.221 once the budget is allowed to bind."),

    ("IX: the revised generator",
     "Rebuilding the generator so that faults appear only as disagreement with "
     "the neighbourhood, the node series are decorrelated and latent validity "
     "depends on zone makes the fairness constraint bind, makes abstention "
     "worthwhile, makes the learned policy state dependent, and still leaves the "
     "encoder behind a multilayer perceptron on fault detection, with an epoch "
     "sweep ruling out under-training within the range tested.",
     "The generator is rebuilt so that faults appear only as disagreement with "
     "the neighbourhood, the node series are decorrelated, and latent validity "
     "depends on zone. Three things follow. The fairness constraint binds, "
     "abstention becomes worthwhile, and the learned policy becomes state "
     "dependent. What does not change is the fault detection result: the encoder "
     "stays behind a multilayer perceptron, with an epoch sweep ruling out "
     "under-training within the range tested."),
]


def body_words(t: str) -> int:
    body = re.sub(r"\$\$.*?\$\$", " ",
                  re.sub(r"\[\[FIGURE:[^\]]+\]\]", "",
                         t.split("## Author Contributions")[0]), flags=re.S)
    keep = "\n".join(l for l in body.split("\n") if not l.lstrip().startswith("|"))
    return len(re.sub(r"[#*`>]", " ", keep).split())


def main() -> None:
    t = SRC.read_text(encoding="utf-8")
    before = body_words(t)
    failed, applied, already = [], 0, 0
    # Each edit decides for itself whether it has run, so the pass can be
    # extended and rerun without a single sentinel going stale.
    for label, old, new in EDITS:
        if new in t:
            already += 1
            continue
        n = t.count(old)
        if n != 1:
            failed.append(f"{label}: matched {n} times")
            continue
        t = t.replace(old, new, 1)
        applied += 1
    if failed:
        print("anchors failed, nothing written:")
        for f in failed:
            print("  -", f)
        sys.exit(1)

    SRC.write_text(t, encoding="utf-8")
    print(f"applied {applied} edits, {already} already in place")
    print(f"body words {before} -> {body_words(t)}")
    print("semicolons:", t.count(";"), " em dash:", t.count(chr(8212)))


if __name__ == "__main__":
    main()
