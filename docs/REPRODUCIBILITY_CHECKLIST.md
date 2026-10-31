# Reproducibility checklist

| Item | Status |
|---|---|
| All raw inputs included | yes, under `data/raw/` |
| Regeneration scripts for every dataset | yes, `scripts/generate_*.py` |
| Single configuration file with all parameters | yes, `configs/default.yaml` |
| Global and per model seeds fixed | yes, seed 42 and model seeds 0 to 4 |
| Chronological splits, no shuffling | yes |
| Validation used for model selection only | yes |
| Graph structure estimated on the training block only | yes, functional similarity edges |
| Every numeric table produced by a script from a saved result file | yes, `scripts/tables/` |
| Every figure produced by a script from a saved result file | yes, `scripts/figures/` |
| Figures skip cleanly when inputs are missing | yes, `need()` guard |
| Multiple seeds with reported dispersion | yes, 5 for forecasting and the policy gradient check, 3 for multi-task and policy, 2 for ablation |
| Hardware reported | single CPU core, no GPU, 3 GB memory |
| Deterministic reproduction across machines | not verified, see note below |
| External validation on real data | no, and none is claimed |
| Replication on a second parameterisation | yes, the Kuala Lumpur cohort, same topology |
| Diagnosed benchmark flaws repaired and retested | yes, the revised stress regime, `DYHGT_REGIME=stress` |
| Training budget varied to rule out under-training | yes, Experiment G at 25 and 45 epochs |
| Encoder coupled to the decision layer at inference | yes, `--fault-source dyhgt`, with the logistic surrogate kept as a control |
| Loss weights tuned | no, stated in the model card |
| Pretrained checkpoints saved | yes, under `data/processed/` and `results/` |

## Note on cross environment determinism

Seeds are fixed in NumPy and PyTorch, but exact bitwise reproduction across
different CPUs, BLAS builds and library versions is not guaranteed and was not
verified. Anyone rerunning this work should treat their own environment as
canonical and rerun all seeds rather than mixing numbers across machines.

## Commands

```
pip install -r requirements.txt
python scripts/build_node_mapping.py
python scripts/build_dynamic_heterogeneous_graph.py
python scripts/inject_faults_and_missingness.py
python scripts/build_integrated_benchmark.py
python scripts/generate_mixed_mobility_transactions.py
python scripts/run_experiment_a_faults.py
python scripts/run_experiment_b_forecast.py
python scripts/run_experiment_c_policy.py
python scripts/run_experiment_d_ablation.py
python scripts/run_experiment_e_policy_gradient.py
python scripts/export_prediction_artifacts.py
python scripts/generate_all_tables.py
python scripts/generate_all_figures.py
python tools/build_journal_docx.py
```

The revised stress regime reruns the same pipeline with `DYHGT_REGIME=stress`
set, which redirects every artifact to its own directory. The README gives the
command sequence, including Experiment G and the coupled decision layer run.

## External validation

Experiment I is the only run that touches data the authors did not generate, so
it is the only one with a prerequisite this repository cannot satisfy on its
own. METR-LA is not redistributed here. Download the three published files into
an empty directory of their own and pass that directory:

```
python scripts/run_experiment_i_external.py --data <that directory> --seeds 0 1 2 --epochs 8
python scripts/tables/table_15_external_metr_la.py
```

Two properties of this step are deliberate. The adjacency is rebuilt from the
published sensor distance table rather than read from the distributed pickle,
because unpickling a downloaded file executes whatever it contains. The download
directory holds no Python, so nothing in it can be imported by a script run
against it. The run writes `results/tables/exp_i_external_metr_la.csv`, which is
shipped, so Table XV can be rechecked without repeating the download.
