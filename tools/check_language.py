"""Mechanical language checks over the manuscript source.

This does not try to judge prose. It finds the classes of error that are easy
to make across 12,000 words and easy for a reviewer to notice: repeated words,
stray spacing, compound modifiers left unhyphenated, bracket citations that
point outside the reference list, sentences long enough to lose the reader, and
wording of the kind that reads as machine written.

Run it after every editing pass. It reports and never rewrites, because an
automatic rewrite of prose is how meaning gets lost.
"""
from __future__ import annotations

import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "manuscript.md"

# Compound modifiers that take a hyphen when they sit before the noun. The
# right hand side is what the manuscript should read.
HYPHEN_PATTERNS = [
    (r"\b(\d+(?:\.\d+)?) "
     r"(dimensional|record|component|step|minute|hour|zone|column|row|point)\b",
     r"\1-\2"),
    (r"\b(two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty) "
     r"(step|minute|hour|component|dimensional|record|zone|fold)\b", r"\1-\2"),
    (r"\bnon (Euclidean|causal|binding|stationary|parametric)\b", r"non-\1"),
    (r"\btop (three|five|ten)\b", r"top-\1"),
    (r"\b(well|poorly|badly) (calibrated|specified|conditioned|posed)\b", r"\1-\2"),
    (r"\breal (time|world) (data|measurement|deployment|system)\b", r"real-\1 \2"),
    (r"\b(lag) (one|two) (autocorrelation)\b", r"\1-\2 \3"),
]

# Wording that signals generated text or inflates a claim.
BANNED = [
    "massive", "crucially", "groundbreaking", "revolutionary", "delve", "delves",
    "leverage", "leverages", "leveraging", "utilize", "utilizes", "utilizing",
    "seamless", "seamlessly", "cutting edge", "cutting-edge", "state of the art",
    "paradigm shift", "game changer", "robustly demonstrates", "it is important to note",
    "it should be noted", "in today's world", "in the realm of", "a myriad of",
    "plethora", "unprecedented", "pivotal", "paramount", "testament",
    "navigate the complexities", "shed light on", "underscore", "underscores",
    "intricate", "holistic", "synergy", "synergies", "tapestry",
]

# Terms that must stay spelled one way throughout.
CONSISTENCY = {
    "probability integral transform": ["probability of integral transform",
                                       "probability integral transformation",
                                       "probability of integral transformation"],
    "continuous ranked probability score": ["continuous rank probability score",
                                            "continuously ranked probability score"],
    "missing not at random": ["missing not-at-random"],
    "equal opportunity gap": ["equal-opportunity gap", "equality of opportunity gap"],
}


def body_of(t: str) -> str:
    return t.split("## References")[0]


def sentences(t: str) -> list[str]:
    flat = re.sub(r"\s+", " ", t)
    return [s.strip() for s in re.split(r"(?<=[.!?]) (?=[A-Z(])", flat) if s.strip()]


def main() -> int:
    t = SRC.read_text(encoding="utf-8")
    body = body_of(t)
    problems: list[str] = []

    def flag(kind: str, detail: str) -> None:
        problems.append(f"{kind}: {detail}")

    # --- repeated words, ignoring legitimate ones -----------------------
    for m in re.finditer(r"\b(\w+) \1\b", body, re.I):
        if m.group(1).lower() not in {"that", "had", "very"}:
            flag("repeated word", m.group(0))

    # --- spacing --------------------------------------------------------
    for pat, name in [(r"[a-z]  +[a-z]", "double space"),
                      (r" [,.;:]", "space before punctuation"),
                      (r"\([ ]", "space after open bracket"),
                      (r"[ ]\)", "space before close bracket")]:
        for m in re.finditer(pat, body):
            flag(name, repr(body[max(0, m.start() - 30):m.end() + 20]))

    # --- hyphenation ----------------------------------------------------
    for pat, want in HYPHEN_PATTERNS:
        for m in re.finditer(pat, body):
            flag("needs hyphen", f"{m.group(0)!r} -> {re.sub(pat, want, m.group(0))!r}")

    # --- banned wording -------------------------------------------------
    for word in BANNED:
        for m in re.finditer(rf"\b{re.escape(word)}\b", body, re.I):
            flag("wording", f"{word!r} near {body[max(0, m.start()-40):m.end()+30]!r}")

    # --- terminology consistency ----------------------------------------
    for right, wrongs in CONSISTENCY.items():
        for w in wrongs:
            if re.search(rf"\b{re.escape(w)}\b", body, re.I):
                flag("terminology", f"{w!r} should be {right!r}")

    # --- citations ------------------------------------------------------
    refs = {int(n) for n in re.findall(r"^\[(\d+)\]", t.split("## References")[1], re.M)}
    # Strip math before scanning citations, because interval notation such as
    # [0,1] is not a bracket citation.
    nomath = re.sub(r"\$\$.*?\$\$", " ", body, flags=re.S)
    nomath = re.sub(r"\$[^$\n]*\$", " ", nomath)
    cited: set[int] = set()
    for m in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", nomath):
        for n in re.findall(r"\d+", m.group(1)):
            cited.add(int(n))
    # "[49] to [51]" ranges cite everything between
    for m in re.finditer(r"\[(\d+)\] to \[(\d+)\]", nomath):
        cited |= set(range(int(m.group(1)), int(m.group(2)) + 1))
    for n in sorted(cited - refs):
        flag("citation", f"[{n}] cited but not in the reference list")
    for n in sorted(refs - cited):
        flag("citation", f"[{n}] in the reference list but never cited")
    if refs and sorted(refs) != list(range(1, max(refs) + 1)):
        flag("citation", "reference numbers are not contiguous from 1")

    # --- very long sentences --------------------------------------------
    prose = "\n".join(l for l in body.split("\n")
                      if not l.lstrip().startswith(("|", "$$", "#", "[[FIGURE")))
    long_ones = [(len(s.split()), s) for s in sentences(prose) if len(s.split()) > 55]
    for n, s in sorted(long_ones, reverse=True)[:12]:
        flag("long sentence", f"{n} words: {s[:110]}...")

    # --- report ---------------------------------------------------------
    by_kind = collections.Counter(p.split(":", 1)[0] for p in problems)
    for p in problems:
        print(p)
    print()
    if not problems:
        print("no language issues found")
        return 0
    print("summary:", dict(by_kind))
    print(f"{len(problems)} item(s) to review")
    return 0


if __name__ == "__main__":
    sys.exit(main())
