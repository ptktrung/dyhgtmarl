# tools

The editorial pipeline. Nothing here touches a result, and nothing in
`scripts/` knows these exist. The separation is deliberate: a reader checking a
number in the paper never needs to read this directory, and a change here cannot
alter a number.

`docs/manuscript.md` is the single source of the paper's text. The documents are
built from it, never edited in place, which is why a correction is made once in
the source and then rebuilt rather than applied to two .docx files by hand.

## Building the documents

```bash
python tools/build_journal_docx.py        # single column journal version
python tools/build_ieee_access_docx.py    # two column IEEE Access submission
python tools/build_reference_pdf.py       # LaTeX render, for checking equations
```

Both builders convert the markdown with pandoc, then rewrite what pandoc cannot
know about: bracketed citations become Word CITATION fields, the seventy-one
sources are written into the document so they appear under References, Manage
Sources, the reference list is wrapped in a BIBLIOGRAPHY field set to the IEEE
style, tables are inserted at the point of first citation with column widths
allocated by content, and displayed equations are numbered flush right. The
Access builder additionally applies the template's styles, the two column body
with full width floats, and the author biographies.

The two outputs are built from the same source and the same result files, so
their content cannot drift apart.

## Checking

```bash
python tools/check_language.py     # repeated words, spacing, hyphens, citations
python tools/audit_delivery.py     # 30 checks over the source and both builds
```

Both only report. `audit_delivery.py` exits non-zero on a failure, so it is
usable in a pre-submission check.

## The revision scripts

The manuscript was revised in stages, and each stage is a script rather than a
hand edit so the change is reviewable and repeatable. They run in this order and
each refuses to run twice:

| Order | Script | What it did |
|---|---|---|
| 1 | `apply_reviewer_fixes.py` | the two peer review reports |
| 2 | `revise_manuscript.py` | first consolidated revision |
| 3 | `revise_manuscript_kl.py` | inserted Section VI-F, the replication cohort |
| 4 | `trim_manuscript_kl.py` | recovered words for Section VI-F |
| 5 | `revise_manuscript_stress.py` | inserted Section VI-G, the revised regime |
| 6 | `revise_manuscript_external.py` | inserted Section VI-H, METR-LA |
| 7 | `trim_manuscript.py` | recovered words for Section VI-H |
| 8 | `apply_author_review.py` | the author's own review pass |
| 9 | `polish_prose.py` | corrected counts, split overlong sentences |

Each stage decides per edit whether it has already run, so the chain is
rerunnable and replaying stages 6 to 9 over the manuscript as it stood before
Section VI-H reproduces the current `docs/manuscript.md` byte for byte. A stage
whose anchor no longer matches writes nothing and says which edit failed, which
is what stops a half applied pass from losing a batch of changes silently.

The two that insert results, `revise_manuscript_stress.py` and
`revise_manuscript_external.py`, build their verdict sentences from the result
CSV rather than carrying fixed prose. If an experiment is rerun and its outcome
changes, those sentences change with it, including where the new outcome is less
favourable. That is the reason they are scripts.

## Packaging

```bash
python tools/package_delivery.py    # dyhgt_marl.zip
```
