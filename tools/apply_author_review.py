"""Apply the author's review pass to the manuscript source.

The author returned a reviewed copy of the IEEE Access build with 35 changed
passages. They fall into three groups and this script treats them differently.

Facts only the author holds are adopted as given: the third affiliation, the
funding declaration, the repository location, and the venue of reference 68.
Nothing here verifies them, because nothing here can.

Wording and punctuation corrections are adopted where they are right, chiefly
hyphenation of compound modifiers.

About twenty of the changed passages introduced grammatical errors, among them
"One model watch for sensor faults", "These models do not carry is a notion",
"the scope of the test needs starting", "probability of integral
transformation", "None of this substitute for external audit", and a stray
possessive in "the two cohorts' side by side". Those are not applied. The
manuscript already holds the correct wording, so rejecting them needs no edit,
and the delivery notes record each one so the author can see what was changed
back and why rather than finding it changed silently.

The run order is revise_manuscript_external.py, trim_manuscript.py, then this.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "manuscript.md"

SENTINEL = "Faculty of Information Technology, Ton Duc Thang University"

# Facts supplied by the author. Adopted as given.
AUTHOR_FACTS: list[tuple[str, str, str]] = [

    ("third affiliation",
     "(3) [Faculty to be confirmed by the author], Ton Duc Thang University, "
     "Ho Chi Minh City, Vietnam",
     "(3) Faculty of Information Technology, Ton Duc Thang University, "
     "Ho Chi Minh City, Vietnam"),

    ("funding statement",
     "[To be completed before submission. State any grant number and funder, or "
     "state that the research received no specific grant from any funding agency "
     "in the public, commercial or not-for-profit sectors.]",
     "This research received no specific grant from any funding agency in the "
     "public, commercial, or not-for-profit sectors."),

    # The author confirmed the repository is not public yet and will push before
    # submitting, so the URL is printed and the checklist carries the reminder.
    ("repository location",
     "[Insert the repository or archive location before submission. Do not state "
     "a location until it exists.] Queries about the code and data may be directed "
     "to T. K. T. Pham at trungptk@fe.edu.vn.",
     "The source code, benchmark generators and result files are available at "
     "https://github.com/ptktrung/dyhgt_marl. Queries about the code and data may "
     "be directed to T. K. T. Pham at trungptk@fe.edu.vn."),

    # The author's copy gave the volume as "Volume 1,2222", which is not a volume
    # number this script can resolve, and they asked for it to be omitted.
    ("reference 68 venue",
     '[68] T. K. T. Pham and D. A. Dewi, "Graph-theoretic Transformer for '
     'stochastic travel time prediction in mixed-mobility urban networks," in '
     '[Conference name and Springer book series to be supplied by the authors], '
     '2026, in press.',
     '[68] T. K. T. Pham and D. A. Dewi, "Graph-theoretic Transformer for '
     'stochastic travel time prediction in mixed-mobility urban networks," in '
     '*Proc. Int. Conf. Intelligent Systems, Blockchain, and Communication '
     'Technologies (ISBCom)*, Lecture Notes in Networks and Systems. Cham, '
     'Switzerland: Springer, 2026, in press.'),
]

# Wording the author corrected, where the correction is right. Compound
# modifiers before a noun take a hyphen, which the draft was inconsistent about.
WORDING: list[tuple[str, str, str]] = [
    ("hyphen: record corpus", "on a 100,000 record corpus", "on a 100,000-record corpus"),
    ("hyphen: mixture density", "with a three component mixture density output",
     "with a three-component mixture density output"),
    ("hyphen: diffusion convolution", "Reduced means a two step diffusion convolution",
     "Reduced means a two-step diffusion convolution"),
    ("hyphen: resolution", "at five minute resolution over four months",
     "at five-minute resolution over four months"),
    ("hyphen: feature vector", "maps a 22 dimensional feature vector",
     "maps a 22-dimensional feature vector"),
    ("comma before and", "These values were fixed before the experiments and no claim of "
     "optimality is made.",
     "These values were fixed before the experiments, and no claim of optimality is "
     "made."),
    ("subject agreement", "the utility-maximizing policy genuinely prefer to treat one "
     "group worse", "the utility-maximizing policy genuinely prefers to treat one group "
     "worse"),
    ("tighter: particularly", "Proper scoring rules, in particular the continuous ranked "
     "probability score", "Proper scoring rules, particularly the continuous ranked "
     "probability score"),
    ("tighter: carries weight", "The word jointly carries the weight.",
     "The word jointly carries weight."),
    ("tighter: and confuses", "treats sensors as isolated points and so confuses "
     "congestion with cascading failure",
     "treats sensors as isolated points and confuses congestion with cascading failure"),

    # Same hyphenation theme, found by tools/check_language.py rather than in the
    # author's copy, and corrected here so the whole pass lives in one place.
    ("hyphen: point gap", "a 15.2 point equal opportunity gap",
     "a 15.2-point equal opportunity gap"),
    # Distinct from the edit above: two edits sharing one replacement string
    # would make the second look already applied.
    ("hyphen: point gap, conclusion", "closing a 15.2 point equal-opportunity gap",
     "closing a 15.2-point equal opportunity gap"),
    ("hyphen: minute grid", "aggregated to the fifteen minute grid",
     "aggregated to the fifteen-minute grid"),
    ("hyphen: well calibrated", "A well calibrated but inaccurate model",
     "A well-calibrated but inaccurate model"),
]


def main() -> None:
    t = SRC.read_text(encoding="utf-8")
    failed, applied, already = [], 0, 0
    # Each edit decides for itself whether it has already run, so the pass is
    # rerunnable and the editorial chain can be replayed from any stage.
    for group in (AUTHOR_FACTS, WORDING):
        for label, old, new in group:
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
    print(f"applied {applied} edits from the author's review, "
          f"{already} already in place")
    for bad in ("[Faculty to be confirmed", "[To be completed", "[Insert the repository",
                "[Conference name"):
        if bad in t:
            print(f"  WARNING: placeholder still present: {bad}")
    body = re.sub(r"\$\$.*?\$\$", " ",
                  re.sub(r"\[\[FIGURE:[^\]]+\]\]", "",
                         t.split("## Author Contributions")[0]), flags=re.S)
    keep = "\n".join(l for l in body.split("\n") if not l.lstrip().startswith("|"))
    print("body words:", len(re.sub(r"[#*`>]", " ", keep).split()))
    print("semicolons:", t.count(";"), " em dash:", t.count(chr(8212)))


if __name__ == "__main__":
    main()
