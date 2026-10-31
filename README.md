# DyHGT-MARL

Code, data generators and experiments for *Dynamic Heterogeneous Graph
Transformers and Fairness-Constrained Selective Decision Learning for Resilient
Urban IoT Mobility Systems*.

A dynamic heterogeneous graph Transformer with reliability-conditioned
relation-aware attention, coupled to a fairness-constrained selective decision
layer, evaluated on a reproducible synthetic benchmark of a mixed-mobility
urban IoT network and, for forecasting, on the public METR-LA corpus.

## Data provenance, stated plainly

The benchmark is **synthetic**. It is inspired by the modal mix and the core
against periphery contrast of Ho Chi Minh City, and it is not official traffic
data, not official transport data, not passenger or municipal records, and not
field measurement. No result from it supports any operational claim about a
real city.

One experiment uses real data. Section VI-H reruns the forecasting comparison
on METR-LA, 207 freeway loop detectors published with the diffusion
convolutional recurrent network. That corpus is not redistributed here and the
code downloads it from its published location.

## Quick start

```bash
pip install -r requirements.txt    # or: conda env create -f environment.yml
make all                           # about 1 hour on one CPU core, no GPU
```

`make help` lists every stage. **[docs/REPRODUCTION.md](docs/REPRODUCTION.md)**
is the full guide: prerequisites, the commands one at a time, expected
runtimes, which script produces which table or figure, and troubleshooting.

```bash
make data           # build the integrated benchmark from the source files
make experiments    # Experiments A to E
make outputs        # every table and figure
make paper          # the journal and IEEE Access documents
make verify         # language report and 30 delivery checks
make kl             # the replication cohort of Section VI-F
make stress         # the revised stress regime of Section VI-G
make external DATA=~/metr-la   # the external validation of Section VI-H
```

## Layout

```
configs/default.yaml   every parameter and seed, one file
data/raw/              the three supplied source files and their generators
src/dyhgt/             graph, generators, models, metrics, stress regime
scripts/               the pipeline and the experiments
scripts/tables/        one script per numbered table
scripts/figures/       one script per numbered figure
results/               result CSVs and rendered figures
tools/                 manuscript assembly, auditing, packaging
docs/                  manuscript source, reproduction guide, dataset and model cards
```

`scripts/` is the research code and `tools/` the editorial pipeline that builds
the documents from `docs/manuscript.md`. Nothing in `tools/` can change a
result. Every number in every table and figure is written by a script from a
saved result file, with no value entered by hand.

## What the results say, including where the method loses

The substantive findings are in Section VI, and the ones that do not favour the
proposed model are reported in the same detail as the ones that do.

**Where it wins.** Probabilistic forecasting and calibration. On the synthetic
corpus it reaches coverage 0.952 against a nominal 0.95 at calibration error
0.023. On METR-LA it has the best continuous ranked probability score of the
seven models compared, 3.327 against 4.006, and the best calibration, 0.020
against 0.082, and it holds the best score at every missingness level. The
decision layer raises net revenue recovery from 80.99 to 88.96 percent while
closing a 15.2-point equal opportunity gap.

**Where it loses.** Fault detection, to a multilayer perceptron. Reconstruction,
to causal carry-forward. Point accuracy on real data, where it ranks fifth of
seven on both mean absolute error and MAPE, because the mixture density head
buys an accurate predictive width rather than a sharper conditional mean.
Neither relation typing nor dynamic topology nor reliability conditioning
produces a measurable effect on this corpus.

**Where a claim did not survive.** Robustness to observation loss. On the
synthetic corpus the model gives up 4.2 percent of its score at 40 percent
missingness against 65 to 95 percent for the baselines. On METR-LA it gives up
6.7 percent against 2.6 to 5.8 percent, the largest loss of the six models that
respond to their inputs. That margin is a property of the synthetic corpus,
where a mean pairwise node correlation of 0.909 leaves the baselines little to
fall back on, and not a property of reliability conditioning. The abstract,
Section VII-A and Section IX report it that way.

Each outcome is traced to a property of the source generators in Section VII,
and Section VI-G rebuilds the generator to remove two of those properties and
retests, which makes the fairness constraint bind and abstention worthwhile and
still leaves the encoder behind the perceptron on fault detection.

## Reproducibility

Seeds are fixed in `configs/default.yaml`. Splits are chronological and never
shuffled. The learned graph relation and all standardization statistics are
estimated on the training block only. Bitwise cross-platform determinism is
**not** guaranteed and was not verified, so rerun every seed in your own
environment rather than mixing numbers across machines. See
[docs/REPRODUCIBILITY_CHECKLIST.md](docs/REPRODUCIBILITY_CHECKLIST.md).

## Documents

| File | What it is |
|---|---|
| `DyHGT_IEEE_Access_Submission.docx` | the two column submission file |
| `DyHGT_MARL_..._Full_Paper.docx` | the single column journal version |
| `DyHGT_MARL_Math_Reference.pdf` | the same equations rendered through LaTeX |
| `docs/manuscript.md` | the source both documents are built from |

Equations in the .docx files are Office Math Markup, so Word opens them in the
equation editor. LibreOffice drops that markup on import and also mis-paginates
a document with many section breaks, so no PDF converted outside Word is
evidence about the equations or the page count. Open one equation in Word to
confirm, and read the reference PDF to check what each equation says.
