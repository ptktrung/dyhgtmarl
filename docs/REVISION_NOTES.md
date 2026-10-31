# Revision notes for P2

Applied from `P2_edit_pack_coauthor_and_citation.docx` on 24 October 2026.
This file lists what changed, and what still needs a fact that only the
authors can supply.

## Items that still need you

| Item | Where | What is needed |
|---|---|---|
| Dr Tran's faculty | Title page, affiliation (3) | The text reads `[Faculty to be confirmed by the author]`. The university and email come from the edit pack and the publisher record. The faculty was not verifiable from any source, so it was left marked. |
| Dr Tran's written consent | Before submission | The edit pack requires agreement before the name is added. The draft message is in Section 8 of the edit pack. |
| Dr Tran's ORCID | Submission system | Not in the manuscript, but most journals ask at submission. |
| Dr Dewi's agreement | Before submission | The edit pack requires it for a change of corresponding author. |
| INTI policy check | Before submission | Confirm whether INTI requires the candidate or supervisor to be corresponding author for PhD-linked output. |
| CRediT roles for Dr Tran | Author Contributions | The roles written in are the ones the edit pack proposed. Keep only those he actually performed. |
| Funding | Funding section | Marked `[To be completed before submission]`. |
| Repository location | Data and Code Availability | Marked `[Insert the repository or archive location before submission]`. Do not state one until it exists. |
| Reference verification | References | Reference [71] was verified against the publisher record. The rest were not. See the note at the end of the reference list. |

## What changed

**Author block.** Three authors, with Quoc-Dai Luong Tran as corresponding
author at Ton Duc Thang University.

**New back matter.** Author Contributions using CRediT roles, Funding, Data
and Code Availability, Competing Interests and an Ethics Statement. The
competing interests statement discloses that reference [71] is authored by a
co-author.

**Edit 3 was run, so its paragraph is real.** A linear softmax policy with a
learned value baseline was trained by policy gradient on the same constrained
process, state information, actions, reward, dual update, chronological split
and five seeds as Experiment C. Script: `scripts/run_experiment_e_policy_gradient.py`.
Results: `results/tables/exp_e_policy_gradient.csv` and
`exp_e_cost_sensitivity.csv`. Reported in Section VI-D, Table XII, Table XII b
and Fig. 13.

The outcome does not flatter the decision layer and is reported that way. The
function approximation policy reached 93.80 percent revenue recovery against
88.96 percent for the tabular policy, with an equal opportunity gap of 0.0000
against 0.0013, by accepting every record. Abstention, corroboration and
deferral were each chosen in zero percent of decisions. A cost sweep found the
learned policy constant at every penalty setting, switching from accept-all to
defer-all between multipliers 10 and 20 with no mixed strategy anywhere. This
supports the Section VIII argument that the benchmark, not the method, is what
needs revising next.

**Citation to [71].** One sentence at the end of Section VIII, Scope of the
implementation, exactly where the edit pack placed it. The reference was
verified against the DOAJ publisher record: IEEE Access, vol. 11,
pp. 75955-75970, 2023, doi 10.1109/ACCESS.2023.3297652. The edit pack listed
the end page and DOI as unconfirmed, and both are now confirmed.

**References [63] to [70] are now cited.** Each was placed where it belongs
rather than removed: Vaswani in Section II-B, long short-term memory in
Section II-E, Adam in Section V, infrastructure reliability in Section II-A,
UrbanKG in Section II-C, and the three smart city reviews in the opening
paragraph of Section I. Because IEEE numbers references by order of first
citation, the whole list was renumbered. Seventy of the seventy-one entries
changed number, so any note you kept against the old numbering needs remapping.

**MARL wording.** The title is unchanged, since it was fixed by the original
brief. Section IV-F now states plainly that multi-agent here means independent
zone-level learners with no communication, no joint value factorization and no
opponent modelling, and that this is weaker than the cited MARL literature.

**Abstract.** Now states the non-binding fairness constraint and the function
approximation result among the limitations, as the edit pack asked.

**Companion manuscripts.** New Section VII-D states how this work differs from
the two companion papers that use the same source files.

## Added 24 October 2026: replication cohort and bandit naming

**The decision layer is renamed.** Every action except corroboration terminates
the record, and nothing an agent does changes the state another agent sees or
the distribution of later records, so the layer is a two-zone tabular
contextual bandit with one optional information gathering action. Section IV-F
now says that plainly. The system name DyHGT-MARL and the title are unchanged,
since the title was fixed by the original brief.

**A replication cohort was added.** Section VI-F, Table XIII and Fig. 14. The
whole pipeline was rerun unmodified on the Kuala Lumpur files, with results in
`results/tables_kl/`.

### What the Kuala Lumpur comparison can and cannot claim

Verified against the shipped CSVs, not inferred from the documentation.

| Property | HCMC | KL | Same? |
|---|---|---|---|
| Directed edge list | 66 edges | 66 edges | identical |
| Adjacency, node types, zones, sensor tiers | - | - | identical |
| Laplacian positional encodings | - | - | byte identical |
| Modal split, moto/car/bus | 70.7/16.7/12.6 | 40.7/36.6/22.6 | differs |
| Mean physical edge distance | 4.36 km | 6.92 km | differs |
| Mean traffic volume | 405.8 | 406.0 | near identical |
| Random seed | 42 | 142 | differs |

`generate_nodes_and_edges` uses `HCMC_DIRECTED_EDGES` for both cities, and
`generate_traffic` takes no city argument. The two cohorts are therefore the
same network under a different fleet mix, a different spatial scale and a
different seed. The paper
calls this replication under a second parameterisation and explicitly declines
any cross-city generalization claim.

### Two discrepancies to fix in the thesis

These concern the thesis rather than the paper, and an examiner comparing
Section 5.7 against the artefacts would find them.

1. Section 3.4.2 states that "target adjacency A_(KL), Laplacian eigenvectors
   PE_(KL), phase conflict maps, and target safety-supervisor bounds" are
   recomputed for the target network. The shipped generator does not recompute
   them. The adjacency and the eigenvectors are identical to HCMC's.
2. The same section states a 40/50/10 modal split for KL. The generator's rule
   `car_s = (1 - moto_share) * 2 / 3` yields about 41/37/23 in the shipped
   file.

Either the generator should be changed to build a distinct KL topology and the
stated split, or the thesis text should be brought in line with what the
artefacts contain. The second is quicker and keeps the bounded framing that
Section 5.7 already uses.

### What the replication found

Five results replicate. Relative forecast accuracy, at MAPE 13.64 against 9.08
percent. Calibration, at coverage 0.952 against 0.953 and calibration error
0.023 against 0.041. Robustness to observation loss, a score change of +4.3
against +7.4 percent at 40 percent missingness, where climatology changes by
+95.3 and +39.2 percent. Fault detection, where a multilayer perceptron reaches
F1 0.988 and 0.985 against the proposed model's 0.658 and 0.691. And the legacy
rule's zone bias, which is larger in KL at a gap of 0.2098 against 0.1523.

Does not replicate, and both are now stated in the paper. Reconstruction
reverses: the model ranks fifth of seven in HCMC and first of seven in KL, at
1.025 minutes, because the KL series has a lag one autocorrelation of 0.732
against 0.859, so temporal interpolation loses its advantage. Abstention is
used in 13.5 percent of HCMC decisions and in none of the KL decisions, so the
value of the deferral mechanism is a property of the parameterisation.
Relatedly, the earlier claim that the learned policy is constant at every cost
setting holds only for HCMC, since KL produces a mixed policy at a penalty
multiplier of 40.

One measurement caution now stated in the paper: the two cohorts have travel
time means of roughly 25.6 and 6.1 minutes, because the primary cohort reads
the original supplied matrix and the replication reads the v2.0 table.
Absolute error is not comparable across cohorts, so only scale free quantities
and within-cohort changes are compared.

### Not done

The paper still runs on the older harmonised inputs for the primary cohort.
Migrating both cohorts to the v2.0 schema would be a larger change and would
overlap with the thesis, so it was not attempted. The thesis Section 5.7
transfer protocol remains the place where the KL material is developed in
full.

## Style pass

Semicolons were removed from the manuscript and from the table generators, so
the delivered document contains none. There are no em dashes or en dashes.
After Section VI-G was added, the body is 10,998 words excluding references,
back matter and table content, against a target of 8,500 to 11,000, and the
abstract is 320 against a target of 250 to 320. Fitting Section VI-G inside the
limit cost two passages that duplicated material held elsewhere. Section VII-C
repeated Section II-L on the source papers and was removed after its one unique
point, the ceiling implied by a reported 100 percent revenue recovery, was
merged into II-L. The scenario families subsection was folded into III-F, where
Table V is first cited. `tools/audit_delivery.py` rechecks all of this, along
with float ordering, citation fields and equation markup, and is run before
each delivery.

## Word features in the delivered file

Citations are Word CITATION fields, and the seventy-one sources are stored in
the document so they appear under References, Manage Sources. The bibliography
is a BIBLIOGRAPHY field inside a Bibliography content control set to the IEEE
style, with the formatted entries cached so the list reads correctly before any
field update. Equations are Office Math Markup, which Word treats as Standard
Equations and opens in the equation editor.

Two things to check when you first open the file in Word. Select all and press
F9 to update fields, then confirm the citation numbers still read 1 to 71 in
order. Open one equation to confirm it is editable rather than an image. These
were verified structurally here, but this environment has no copy of Word.

## Added 30 October 2026: external validation on METR-LA

Both reviews named the absence of a public benchmark as the largest weakness,
and the earlier reply to that point was that no external data was used. That is
no longer accurate. Experiment I reruns the forecasting comparison of Experiment
B on METR-LA, 207 freeway loop detectors released with the diffusion
convolutional recurrent network, across three seeds and five added missingness
rates. Section VI-H reports it and Table XV gives the numbers.

The result splits in two, and the paper now says so in both directions.

**What reproduced.** The proposed model has the best continuous ranked
probability score of the seven models compared, 3.327 against 4.006 for the
nearest baseline, and the best calibration by a wide margin, coverage 0.956
against a nominal 0.95 and calibration error 0.020 against 0.082 for the next
best. It holds the best score at every missingness level tested.

**What did not reproduce.** The degradation margin, which the earlier drafts
carried in the abstract and the conclusion as the headline finding. On the
synthetic corpus the model loses 4.2 percent of its score at 40 percent
missingness against 65 to 95 percent for the baselines. On METR-LA it loses 6.7
percent against 2.6 to 5.8 percent, so it degrades fastest of the six models
that respond to their inputs at all. The margin was a property of the synthetic
corpus, where a mean pairwise node correlation of 0.909 leaves the baselines
little to fall back on, and not a property of reliability conditioning. Section
VII-A, Section IX and the abstract were rewritten to say this. The claim that
survives is the score and the calibration, not the rate of decay.

**A third finding that was not visible before.** Point accuracy and calibration
part company on real data. The model ranks fifth of seven on both mean absolute
error and MAPE while ranking first on score and calibration. The mixture density
head buys an accurate predictive width rather than a sharper conditional mean,
which the synthetic corpus did not separate.

**One corpus property worth recording.** METR-LA has a mean pairwise node
correlation of 0.2206 against 0.9345 for the harmonized synthetic corpus and
0.5501 for the revised stress regime. The criticism Section VI-G makes of the
generator is therefore measured rather than hypothesised: the synthetic corpus
is four times as correlated as the real network it was built to resemble, and
even the revised regime does not reach the real value.

All of these sentences are generated from `exp_i_external_metr_la.csv` by
`tools/revise_manuscript_external.py`, including the two that cut against the
paper, so the prose cannot drift from the result file. Rerunning the experiment
with different data would rewrite the verdicts rather than leave them standing.

Scope that remains: METR-LA carries no fault labels, no transactions and no zone
structure, so fault detection, reconstruction, the decision layer and the
fairness results are still synthetic only. Section VIII says this under its own
heading rather than leaving it implied.

The body grew from 11,025 to 11,895 words. `tools/trim_manuscript.py` recovered
203 words from passages that said something twice, chiefly the introduction
restating its own contributions list and two paragraphs of Section VIII that both
reported the external validation status. The audit's word ceiling was raised from
11,000 to 12,000, which is a stated decision and not a quiet accommodation: the
11,000 figure came from a venue with a word limit, and IEEE Access has none. Its
constraint is a recommended maximum of 20 pages, which cannot be measured in this
environment and is listed in the submission checklist as a check to run in Word.

## Added 31 October 2026: the author's review pass and the float layout

The author returned a reviewed copy of the Access build. Three things came out
of it, and one of them was a defect this project had been measuring wrongly.

**The author's own corrections are adopted.** The third affiliation is now
Faculty of Information Technology, Ton Duc Thang University, the corresponding
address is tranquocdailuong@tdtu.edu.vn, and reference 68 carries the
conference name the author supplied. The volume the copy gave for reference 68
reads "Volume 1,2222", which is not a volume number, so the entry stays "in
press" and that one field is still the author's to fill.

**About twenty of the changed passages introduced grammatical errors** and are
not applied, among them "One model watch for sensor faults", "Each has mature
literature", "the scope of the test needs starting", "probability of integral
transformation", "None of this substitute", and "Three properties of the
benchmark defeated a static formulation". `tools/apply_author_review.py` lists
each one so the author can see what was changed back rather than finding it
reverted silently.

**The page count was being measured wrongly.** An earlier note in this file and
in the submission checklist said the local converter could not be trusted to
paginate the document. That was asserted, not tested. The author's copy had
been through Word, which stores its own page count, and it says 29 pages where
the local converter says 30 for the same file. The converter is therefore
accurate to about one page on a file Word has normalised, and
`scripts/page_count.py` now reports it. On a file straight from the build the
converter still disagrees badly, because it mishandles the continuous section
breaks that carry the full width floats, so the figure to trust is the one Word
reports after opening the file.

**Three whitespace defects were reported, below Table III, beside Table XIV and
below Table XV, and two separate causes were found.**

The first is that no caption was bound to its float. Word breaks a column
wherever it runs out of room, and with nothing holding a caption to its table
it sets the caption at the foot of one column and the table at the head of the
next, which leaves exactly the block of white space reported. Every table
caption and every figure now carries keepNext and keepLines, and the delivery
audit fails if any float is left unbound.

The second is that table grids disagreed with table cells. Pandoc writes a
tblGrid sized for its own default page, and the width pass was setting cell
widths without rewriting that grid. Under a fixed table layout the grid is what
a renderer lays out against, so five tables sized for a 3.33 inch column still
carried a 5.5 inch grid. Word absorbed the contradiction, which is why the
author saw whitespace rather than broken tables, and the local converter did
not, which is why it collapsed those tables into a column of one word lines.
Grid, table width and cell widths are now written from the same numbers, and
the audit checks that every table is sized either to the text block or to one
column with nothing in between.

**Language.** `tools/check_language.py` reports no remaining mechanical issues.
`tools/tighten_sentences.py` splits the four sentences that were genuinely hard
to follow on one pass, including the 53 word sentence that closed the abstract,
and groups the revised generator's three repairs as three. The body is 11,900
words and the abstract 318.
