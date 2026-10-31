# Pre-submission checklist

Produced in response to the two peer review reports. Items are grouped by who
has to act. The manuscript itself no longer carries any internal note, which
is why this file exists.

## 1. Facts only the authors hold

| Item | Where | Status |
|---|---|---|
| Dr Tran's faculty | Title page, affiliation 3 | Supplied by you: Faculty of Information Technology, Ton Duc Thang University. Adopted as given. |
| Dr Tran's written consent and ORCID | Submission system | Required before the name is used. |
| Dr Dewi's agreement to the change of corresponding author | Before submission | Required. |
| INTI policy on corresponding authorship for PhD-linked output | Before submission | Check. |
| Funding statement | Funding section | Placeholder present. Replace with the declaration or with the standard no-funding sentence. |
| Repository or archive location | Data and Code Availability | Placeholder present. Insert a Zenodo DOI or an anonymised link only once it exists. |
| Reference 68, the Springer proceedings paper | References | Conference name and series adopted from your copy. The volume you gave reads "Volume 1,2222", which is not a volume number, so the entry stays "in press". Supply the volume and page range when the proceedings appear. |

## 2. References: what was verified and what was not

The reviewers were right that the earlier bibliography contained errors. Two
were checked here against publisher records and both were wrong as written.

**Independently verified during this revision, and corrected:**

- Reference 1, Sanchez-Gallegos et al. The co-authors in the earlier draft
  were fabricated. The MDPI record gives D. D. Sanchez-Gallegos,
  D. E. Carrizales-Espinoza, C. Torres-Charles and J. Carretero. Corrected.
- Reference 3, Mishra and Singh. Volume, issue and pages were missing.
  OpenAlex gives vol. 13, no. 5, pp. 93 to 103. Corrected.
- Reference 15, UrbanKG. The earlier draft had the wrong initial for the
  second author and omitted the third. The record gives Y. Liu, J. Ding,
  Y. Fu and Y. Li, vol. 14, no. 4, art. 60, pp. 1 to 25. Corrected, DOI added.
- Reference 71, Tran et al. Verified in an earlier round, including the page
  range and the DOI.

**Corrected on reviewer authority, not independently checked here:**

- Reference 4, title subtitle restored.
- Reference 8, workshop name expanded.
- Reference 22, workshop name expanded.
- Reference 30, arXiv format normalised.

Please confirm these four against the publisher pages.

**Not verified:** the remaining entries. They are widely cited works and the
reviewers reported that all 71 correspond to real publications, but given that
two of the handful checked were wrong as written, every entry should be
confirmed against the publisher of record before submission. The quickest
route is to export the Word source list and run each DOI or title through
Crossref.

## 3. Fixed in this revision

- Title renamed. The decision layer is a contextual bandit, so the title no
  longer claims multi-agent learning and now reads "Dynamic Heterogeneous Graph
  Transformers and Fairness-Constrained Selective Decision Learning for
  Resilient Urban IoT Mobility Systems". The system acronym DyHGT-MARL is kept
  in the body for continuity with the three conference papers, and Section IV-F
  states in full what the layer is and is not. If a reviewer presses on the
  acronym, renaming it is a single search and replace in `docs/manuscript.md`.
- The internal note on reference verification is removed from the manuscript.
- Competing Interests no longer explains the citation to a co-author's paper,
  which is not a competing interest.
- The sandbox hardware sentence is replaced by a statement of the training
  budget and what it does and does not permit to be concluded.
- Figures and tables are renumbered into order of first citation. Nine figures
  and five tables moved.
- Table I now uses IEEE bracket numbers and no longer cites a work that is
  absent from the reference list.
- Table III counts are integers and include per relation edge counts.
- The IoT anomaly rate is corrected from 0.0668 to the measured 0.0359.
- Markdown heading markers no longer leak into the body text.
- Overlapping labels fixed in the figures the reviewers named.
- The Fig. 8 caption now states that the reliability curve is pooled across
  all four test conditions.
- The reduced DCRNN and STGCN baselines are now defined.
- The encoder is coupled to the decision layer, and the previous logistic
  surrogate is kept as a labelled control.
- A revised generator experiment is added as Section VI-G, with an epoch
  budget sweep that tests the under-training explanation directly. The three
  diagnosed generator flaws are repaired rather than deferred, which makes the
  fairness constraint bind, makes abstention worthwhile, makes the learned
  policy state dependent, and leaves the fault detection conclusion standing.
  Table XIV and Fig. 15 report both regimes, and the dataset card documents the
  corpus.
- The manuscript body is trimmed to stay inside the 11,000 word limit with
  Section VI-G added, and the abstract to the 250 to 320 word band. Section
  VII-C was removed as a duplicate of Section II-L, whose one unique point was
  merged into it.

- An early stopping defect was found and fixed while rerunning the budget
  sweep. The rule halted the encoder's fault head at the same epoch whatever
  budget was set, so the first version of that sweep reported three identical
  rows and measured nothing. With the rule disabled the sweep is informative
  and still finds no budget effect, which is now what Section VI-G reports.
  The rule never fires within the budget on the primary benchmark, so Table VI
  and Section VI-A are unaffected. It does fire on the revised corpus, so the
  encoder was retrained there without it and the better score is the one
  reported. `check_early_stopping_effect.py` and `exp_h_early_stopping.csv`
  record the paired comparison.
- The encoder coupling is reported on both corpora and is not a uniform win.
  It improves every measure on the revised corpus and is mixed on the primary
  benchmark, where the logistic control is already close to exact. Section
  VI-G gives the reason and Section IV-F states which source each table uses.

- External validation is added as Section VI-H and it changed two claims. The
  forecasting and calibration results now have support on METR-LA. The
  degradation margin does not, and the abstract, Section VII-A and Section IX
  were rewritten to report it as a property of the synthetic corpus. The
  verdict sentences are generated from the result file by
  `tools/revise_manuscript_external.py`, so they cannot drift from the data.
- The body is 11,895 words, above the 11,000 figure the earlier drafts worked
  to. That figure came from a venue with a word limit and IEEE Access has none.
  `tools/trim_manuscript.py` recovered 203 words from passages that said
  something twice, and the audit ceiling was raised to 12,000 as a stated
  decision. What binds at Access is the page count, which is the item below.

## 3b. For IEEE Access specifically

| Item | Status |
|---|---|
| Double column Access template | Done. `DyHGT_IEEE_Access_Submission.docx` is built from the supplied template by `tools/build_ieee_access_docx.py`. |
| Author biographies with photographs | Done, taken from the file you supplied. Check that each photograph sits with the right author. |
| Equations numbered flush right | Done, (1) to (7), using the template's own Equation style and tab stop. |
| AI use disclosed | Added as an Acknowledgment. Read it and adjust it to what you consider accurate before submitting, since it is your declaration and not mine. |
| Abbreviations defined at first use | Done. CRPS, MNAR, MCAR, MAPE, PICP, MAE and RMSE are now expanded at first use. |
| ORCID, public and populated | Yours to do. |
| Page count under 20 | Not confirmed, and it cannot be confirmed here. See the note below. |
| Source and PDF must match exactly | Export the PDF from Word yourself, from the file you submit. |

**On the page count.** Word reports 29 pages on the copy you returned, which
is the first trustworthy figure this project has had. The earlier estimate of
23 was a hand count and was wrong, and the earlier claim that the local
converter could not paginate at all was asserted rather than tested: it reports
30 for that same file, so it is accurate to about a page once Word has
normalised the document. Against a recommended maximum of 20, the paper is
roughly nine pages over.

Where the 29 pages go, approximately: 13 for the body text, 6 for the fifteen
figures and their captions, 5 for the eighteen tables, 2 for the 71 references,
and the remainder for the title block and the biographies. Prose alone cannot
close a nine page gap. Cutting a thousand words buys about one page, and this
revision has already taken the body as far as it goes without removing content
the reviewers asked for.

Three options, in order of how much they cost you.

1. **Move floats to supplementary material.** The notation table, the scenario
   table, the system cost table, the schema figure, the dataset composition
   figure and the cascade snapshot figure come to roughly three pages, and the
   guidance excludes supplementary material from the count. Going further, to
   around twenty pages, means moving most of the ablation and replication
   floats as well, which starts to hollow out Section VI. This is not a
   formatting change: those floats are renumbered S1 onward and every mention
   of them in the text follows, so it is left for you to approve. Say the word
   and it is one build.
2. **Submit at 29 pages.** The 20 page figure is a recommendation, Access
   charges no over length fee, and a paper carrying a full ablation, a
   replication cohort, a revised generator and an external benchmark has a
   reason to be long. The risk is a reviewer who treats length as padding.
3. **Ask the Editor in Chief** before submitting, which costs a few days and
   removes the uncertainty.

My own reading is that option 1 at three pages plus option 2 is the sensible
combination: move the six descriptive floats, submit at around 26, and say in
the cover letter why the remainder earns its space.

On the manuscript type, this paper fits the Negative Result category better
than the standard one. It reports that the encoder loses on fault detection and
on reconstruction, that the fairness constraint does not bind, and that the
degradation margin does not survive external validation. Submitted as a
standard paper, those read as flaws rather than as the finding.


## 4. Raised by the reviewers and not done, with reasons

- **GPU training for 50 to 100 epochs.** No GPU was available. Experiment G
  instead varies the epoch budget on CPU and finds fault detection F1 flat
  between 25 and 45 epochs, which addresses the underlying question even
  though it does not match the requested protocol. A GPU rerun on your RTX
  3060 would still strengthen it.
- **A public benchmark such as METR-LA or PEMS-BAY.** Done, and it changed a
  conclusion. Section VI-H reruns the forecasting comparison on METR-LA and
  Table XV reports it. The score and the calibration reproduce and the model
  holds the best score at every missingness level, but the degradation margin
  does not reproduce: 6.7 percent against 2.6 to 5.8 percent for the baselines,
  where the synthetic corpus gave 4.2 against 65 to 95. The abstract, Section
  VII-A and Section IX now report that margin as a property of the synthetic
  corpus. Read those three passages before submitting, because they are the
  ones a reviewer will test hardest and they no longer say what earlier drafts
  said. The gap is partial rather than closed: fault detection, reconstruction,
  the decision layer and the fairness results are still synthetic only, because
  that corpus carries no fault labels, no transactions and no zone structure.
- **Equations.** Both reviews reported the equations as broken. Here is what
  can and cannot be shown from this environment, because an earlier version of
  this note overstated the case.

  What is established: the markup in the .docx is standard Office Math Markup,
  the math namespace is declared on the document root, and the same LaTeX
  sources typeset correctly, which is what `DyHGT_MARL_Math_Reference.pdf`
  shows. Every displayed equation is now numbered, flush right, as IEEE
  requires, and the two places in the text that refer back to the multi-task
  objective cite it as (5).

  What is not established: how Word draws them. The LibreOffice build here
  drops Office Math Markup on import and renders the equations as blank gaps,
  and it does the same to a one line document produced by pandoc alone, so the
  behaviour is that converter's rather than this document's. It does mean that
  no PDF produced here from the .docx is evidence either way, which is why the
  converted PDF is no longer shipped.

  Please open the .docx in Word before submitting and check one equation. If
  they display, nothing more is needed. If they do not, say so and the build
  can embed them as vector images instead, which renders anywhere at the cost
  of no longer being editable in the equation editor.
