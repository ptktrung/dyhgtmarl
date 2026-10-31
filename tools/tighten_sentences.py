"""Split the sentences that run long enough to lose a reader.

`check_language.py` flags sentences above a conservative threshold. Most of
what it flags is ordinary academic prose carrying a list of results and reads
perfectly well, so rewriting all of it would be churn. This script changes the
handful that are genuinely hard to follow on one pass, and takes the chance to
group the items into threes where the material already falls that way.

Every change is an exact-match pair and the script writes nothing unless all
of them match, so a stale anchor cannot leave the manuscript half edited.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "manuscript.md"

SENTINEL = "A revised generator repairs the three diagnosed flaws"

EDITS: list[tuple[str, str, str]] = [

    ("abstract: revised generator",
     "A revised generator, in which faults appear only as disagreement with the "
     "neighbourhood, the node series are decorrelated and latent validity depends on "
     "zone, makes the constraint bind and halves the gap it is given, makes abstention "
     "worthwhile, and still leaves the encoder behind the perceptron, with an epoch "
     "sweep ruling out under-training.",
     "A revised generator repairs the three diagnosed flaws: faults appear only as "
     "disagreement with the neighbourhood, the node series are decorrelated, and latent "
     "validity depends on zone. The constraint then binds and halves the gap it is "
     "given, abstention becomes worthwhile, and the encoder still trails the "
     "perceptron, with an epoch sweep ruling out under-training."),

    ("limitations: replication cohort",
     "The replication cohort of Section VI-F does not change this, since both cohorts "
     "come from the same generator and share a topology, so it tests sensitivity to "
     "parameterisation rather than external validity.",
     "The replication cohort of Section VI-F does not change this. Both cohorts come "
     "from the same generator and share a topology, so it tests sensitivity to "
     "parameterisation rather than external validity."),

    ("limitations: fairness revision",
     "Any claim about fairness constrained control on this corpus therefore depends on "
     "a revision in which validity rates, fare values or misclassification costs differ "
     "by zone, which is also what would give the abstention and corroboration actions a "
     "regime in which they can pay for themselves.",
     "Any claim about fairness constrained control on this corpus therefore depends on "
     "a revision in which validity rates, fare values or misclassification costs differ "
     "by zone. That same revision would give abstention and corroboration a regime in "
     "which they can pay for themselves."),

    ("limitations: epoch sweep",
     "The epoch sweep of Section VI-G finds no material gain between 10 and 45 epochs, "
     "which weakens the budget explanation without eliminating it, since a larger model "
     "under a longer schedule was not tried.",
     "The epoch sweep of Section VI-G finds no material gain between 10 and 45 epochs. "
     "That weakens the budget explanation without eliminating it, since no larger model "
     "under a longer schedule was tried."),
]


def body_words(x: str) -> int:
    body = re.sub(r"\$\$.*?\$\$", " ",
                  re.sub(r"\[\[FIGURE:[^\]]+\]\]", "",
                         x.split("## Author Contributions")[0]), flags=re.S)
    keep = "\n".join(l for l in body.split("\n") if not l.lstrip().startswith("|"))
    return len(re.sub(r"[#*`>]", " ", keep).split())


def main() -> None:
    t = SRC.read_text(encoding="utf-8")
    if SENTINEL in t:
        sys.exit("already tightened")
    before = body_words(t)
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
    abstract = t.split("## Abstract")[1].split("**Keywords")[0]
    print(f"body words {before} -> {body_words(t)}")
    print(f"abstract words {len(abstract.split())}")
    print("semicolons:", t.count(";"), " em dash:", t.count(chr(8212)))


if __name__ == "__main__":
    main()
