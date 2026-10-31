# Reproduction guide

Everything reported in the paper is produced by the code in this repository
from the three supplied source files and one public corpus. No number in any
table or figure was entered by hand. This guide is the short path from a clean
machine to the figures in the paper.

If you only want to check one number, skip to [Which script produces which
result](#which-script-produces-which-result) and run that script alone.

## 1. Prerequisites

| Requirement | Version | Note |
|---|---|---|
| Python | 3.12 or 3.13 | 3.12 is what `environment.yml` pins |
| CPU | any | no GPU is used anywhere in this work |
| Memory | 4 GB | the largest array is the harmonized benchmark |
| Disk | about 2 GB | includes trained checkpoints and rendered figures |
| pandoc | 3.x | only for the document builders, not for any result |
| LaTeX with `pdflatex` | any | only for the optional reference PDF |

The experiments need none of pandoc or LaTeX. They are listed because the
`paper` stage uses them.

## 2. Setup

Either route works. Pick one.

```bash
# pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# or conda
conda env create -f environment.yml
conda activate dyhgt-marl
```

Check it:

```bash
python -c "import numpy, pandas, torch, sklearn, networkx, h5py; print('ok')"
```

## 3. Run everything

```bash
make all          # or: python run_all.py
```

That runs four stages in order: build the benchmark, run the experiments,
regenerate every table and figure, build the documents. Expect about an hour on
one core, almost all of it in Experiment B.

Every stage is safe to rerun. Experiment scripts record each model and seed as
they finish and skip what is already in `results/`, so an interrupted run
continues from where it stopped when you issue the same command again. A failed
step stops the run rather than letting later steps consume a missing file.

## 4. Or run it one step at a time

`python run_all.py --list` prints every command with what it writes. To run a
single stage:

```bash
make data           # about 1 minute
make experiments    # about 55 minutes
make outputs        # about 3 minutes
make paper          # about 2 minutes
make verify         # seconds
```

The individual commands, in dependency order, are:

```bash
# Stage: data. Builds the integrated benchmark from the three source files.
python scripts/build_node_mapping.py
python scripts/build_dynamic_heterogeneous_graph.py
python scripts/inject_faults_and_missingness.py
python scripts/build_integrated_benchmark.py
python scripts/generate_mixed_mobility_transactions.py

# Stage: experiments. Each writes to results/tables/.
python scripts/run_experiment_a_faults.py --seeds 0 1 2 --epochs 10
python scripts/run_experiment_b_forecast.py --epochs 14
python scripts/run_experiment_c_policy.py --seeds 0 1 2 --episodes 8
python scripts/run_experiment_d_ablation.py --seeds 0 1 --epochs 5
python scripts/run_experiment_e_policy_gradient.py --epochs 40
python scripts/compute_detection_delay.py
python scripts/export_prediction_artifacts.py

# Stage: outputs.
python scripts/generate_all_tables.py
python scripts/generate_all_figures.py

# Stage: paper.
python tools/build_journal_docx.py
python tools/build_ieee_access_docx.py
python tools/build_reference_pdf.py
```

Measured fit times on one core, from Table X, give the reason Experiment B
dominates: the proposed model takes 273 s per seed against 23 to 108 s for the
neural baselines and under a second for the tabular ones.

## 5. The three secondary cohorts

Each writes to its own directories, so no cohort can overwrite another.

**Kuala Lumpur replication, Section VI-F.** A second parameterisation of the
same generator. Output goes to `data/processed_kl/` and `results/tables_kl/`.

```bash
make kl
```

**Revised stress regime, Section VI-G.** Faults appear only as disagreement
with the graph neighbourhood, the node series are decorrelated, and latent
validity and false accept cost depend on zone. Output goes to
`data/processed_stress/` and `results/tables_stress/`.

```bash
make stress
```

**External validation on METR-LA, Section VI-H.** This is the only step that
needs data from outside the repository, and that data is not redistributed
here. Download the three published files into an empty directory of their own:

```bash
mkdir -p ~/metr-la && cd ~/metr-la
curl -LO https://raw.githubusercontent.com/deepkashiwa20/DL-Traff-Graph/main/METRLA/metr-la.h5
curl -LO https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/distances_la_2012.csv
curl -LO https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_ids.txt
cd -

make external DATA=~/metr-la      # about 1 hour
```

Two choices in that step are deliberate and worth knowing about. The adjacency
is rebuilt from the published sensor distance table rather than loaded from the
distributed pickle, because unpickling a downloaded file executes whatever it
contains. Zeros in the speed matrix are treated as missing rather than as
observed zero speed, which is the convention the benchmark's own publications
use.

## 6. Which script produces which result

| Paper object | Script |
|---|---|
| Table I to Table XV | `scripts/tables/table_NN_*.py`, one file per table |
| Fig. 1 to Fig. 15 | `scripts/figures/fig_NN_*.py`, one file per figure |
| Section VI-A, fault detection and root cause | `scripts/run_experiment_a_faults.py` |
| Section VI-B, reconstruction | `scripts/run_experiment_a_faults.py` |
| Section VI-C, forecasting and the missingness sweep | `scripts/run_experiment_b_forecast.py` |
| Section VI-D, policy and fairness | `scripts/run_experiment_c_policy.py` |
| Section VI-D, function approximation policy | `scripts/run_experiment_e_policy_gradient.py` |
| Section VI-E, ablation | `scripts/run_experiment_d_ablation.py` |
| Section VI-F, replication cohort | `make kl` |
| Section VI-G, revised regime and epoch budget | `make stress` |
| Section VI-G, early stopping comparison | `scripts/check_early_stopping_effect.py` |
| Section VI-H, METR-LA | `scripts/run_experiment_i_external.py` |

Each table and figure script checks its inputs and exits with
`SKIPPED: required input not found` rather than plotting placeholder data, so a
partial run produces fewer outputs instead of wrong ones.

## 7. What the archive ships and what it rebuilds

Shipped: the three raw source files, all code, every result CSV, every rendered
figure, and the built documents.

Not shipped: `data/processed*`, which holds the harmonized arrays and trained
checkpoints. They are large and the `data` and `experiments` stages rebuild
every one of them from the raw files and `configs/default.yaml`.

## 8. Determinism

Seeds are fixed in `configs/default.yaml`, with a global seed of 42 and model
seeds 0 to 4. Splits are chronological and never shuffled. The
functional-similarity relation and all standardization statistics are estimated
on the training block only.

Bitwise reproduction across different CPUs, BLAS builds and library versions is
**not** guaranteed and was not verified. Identical code with identical seeds has
given different results on different machines. Rerun every seed in your own
environment rather than combining numbers across machines.

## 9. Verifying a run

```bash
make verify
```

`tools/check_language.py` reports repeated words, spacing, hyphenation,
citations that do not resolve and overlong sentences. `tools/audit_delivery.py`
runs 30 checks over the manuscript and the built documents, covering the style
constraints, the word and abstract budgets, float ordering, equation markup,
citation fields and the two column layout invariants. Both report and neither
rewrites anything.

## 10. Repository layout

```
configs/default.yaml      every parameter and seed, one file
data/raw/                 the three supplied source files and their generators
data/processed/           built arrays and checkpoints, not shipped
src/dyhgt/                graph, generators, models, metrics, stress regime
scripts/                  the pipeline: build the benchmark, run the experiments
scripts/tables/           one script per numbered table
scripts/figures/          one script per numbered figure
results/tables/           result CSVs and manuscript-ready tables
results/figures/          rendered figures, PDF and 300 dpi PNG
tools/                    manuscript assembly, auditing and packaging
docs/                     manuscript source, this guide, dataset and model cards
run_all.py, Makefile      entry points
```

`scripts/` holds the research code and `tools/` the editorial pipeline that
turns `docs/manuscript.md` into the submitted documents. The split matters for
a reader: nothing in `tools/` can change a result, and nothing in `scripts/`
knows about the manuscript. `tools/README.md` describes that pipeline.

## 11. Troubleshooting

**`SKIPPED: required input not found`** from a table or figure script. The
experiment that writes that input has not been run. `python run_all.py --list`
names which one.

**`FileNotFoundError` on `data/processed/...`** Run `make data` first.

**A different number from the paper.** Check that you ran every seed the paper
uses, that `DYHGT_CITY` and `DYHGT_REGIME` are unset for primary cohort
results, and section 8 above on cross-environment determinism.

**Equations look blank in a PDF you converted yourself.** LibreOffice drops
Office Math Markup on import, and it also mis-paginates a document with many
section breaks. Open the .docx in Word, or read
`DyHGT_MARL_Math_Reference.pdf`, which renders the same equations through
LaTeX.

**`pandoc: command not found`** Only the `paper` stage needs it. Every result
in the paper is reproducible without it.
