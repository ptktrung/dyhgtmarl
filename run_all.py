#!/usr/bin/env python3
"""Run the whole study, or any one stage of it, in the right order.

    python run_all.py --list                 show the stages and what each writes
    python run_all.py                        run every stage for the primary cohort
    python run_all.py --stage data           run one stage
    python run_all.py --stage experiments    run one stage
    python run_all.py --cohort kl            the Kuala Lumpur replication
    python run_all.py --cohort stress        the revised stress regime
    python run_all.py --dry-run              print the commands and stop

Each stage is a list of commands run in order. A command that fails stops the
run and the exit status says which one, because a pipeline that carries on
after a failed step produces results nobody can trust. Nothing here hides
output: every command's stdout goes to the terminal.

Most steps skip work already recorded in `results/`, so an interrupted run can
be restarted with the same command. The exception is the external validation,
which needs a download and is therefore not part of the default run. See
docs/REPRODUCTION.md for that one and for expected runtimes.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
PY = sys.executable


def s(path: str, *args: str) -> list[str]:
    return [PY, path, *args]


# (stage, [(command, what it writes)])
STAGES: dict[str, list[tuple[list[str], str]]] = {
    "data": [
        (s("scripts/build_node_mapping.py"),
         "data/processed/node_mapping.csv, node and relation metadata"),
        (s("scripts/build_dynamic_heterogeneous_graph.py"),
         "adjacency, typed edge list, graph snapshots"),
        (s("scripts/inject_faults_and_missingness.py"),
         "cascades, outages, MNAR and MCAR masks, closures"),
        (s("scripts/build_integrated_benchmark.py"),
         "data/processed/integrated_benchmark.csv and splits.csv"),
        (s("scripts/generate_mixed_mobility_transactions.py"),
         "the transaction stream with its latent validity label"),
    ],
    "experiments": [
        (s("scripts/run_experiment_a_faults.py", "--seeds", "0", "1", "2",
           "--epochs", "10"),
         "exp_a_fault_detection.csv, exp_a_root_cause.csv, exp_a_reconstruction.csv"),
        (s("scripts/run_experiment_b_forecast.py", "--epochs", "14"),
         "exp_b_forecast_conditions.csv, exp_b_missingness_sweep.csv"),
        (s("scripts/run_experiment_c_policy.py", "--seeds", "0", "1", "2",
           "--episodes", "8"),
         "exp_c_policy.csv, exp_c_pareto.csv, exp_c_dual_variables.csv"),
        (s("scripts/run_experiment_d_ablation.py", "--seeds", "0", "1",
           "--epochs", "5"),
         "exp_d_ablation_model.csv, exp_d_ablation_policy.csv"),
        (s("scripts/run_experiment_e_policy_gradient.py", "--epochs", "40"),
         "exp_e_policy_gradient.csv, exp_e_cost_sensitivity.csv"),
        (s("scripts/compute_detection_delay.py"),
         "exp_a_detection_delay.csv"),
        (s("scripts/export_prediction_artifacts.py"),
         "saved predictions used by the forecast and calibration figures"),
    ],
    "outputs": [
        (s("scripts/generate_all_tables.py"),
         "results/tables/table_*.csv and .md"),
        (s("scripts/generate_all_figures.py"),
         "results/figures/*.pdf and *.png at 300 dpi"),
    ],
    "paper": [
        (s("tools/build_journal_docx.py"),
         "the single column journal .docx"),
        (s("tools/build_ieee_access_docx.py"),
         "the two column IEEE Access submission .docx"),
        (s("tools/build_reference_pdf.py"),
         "the LaTeX reference PDF, for checking the equations"),
    ],
    "verify": [
        (s("tools/check_language.py"),
         "language report, nothing written"),
        (s("tools/audit_delivery.py"),
         "delivery audit, nothing written"),
    ],
}

# Stages that only make sense for the primary cohort.
PRIMARY_ONLY = {"paper", "verify"}

# Extra stages for the two secondary cohorts.
COHORT_EXTRA: dict[str, list[tuple[list[str], str]]] = {
    "stress": [
        (s("scripts/run_experiment_g_budget.py", "--epochs", "25", "45"),
         "exp_g_budget.csv, the epoch budget sweep"),
        (s("scripts/export_fault_probability.py", "--retrain", "--patience", "0"),
         "the encoder's calibrated fault probability for the coupled policy"),
        (s("scripts/run_experiment_c_policy.py", "--seeds", "0", "1", "2",
           "--episodes", "8", "--fault-source", "dyhgt"),
         "exp_c_policy_dyhgt.csv, the coupled decision layer"),
    ],
}

COHORT_ENV = {"primary": {}, "kl": {"DYHGT_CITY": "kl"},
              "stress": {"DYHGT_REGIME": "stress"}}

COHORT_PREP = {
    "kl": [(s("scripts/build_kl_replication_inputs.py"),
            "data/raw/kl_*.csv, the replication inputs")],
}


def plan(stage: str | None, cohort: str) -> list[tuple[list[str], str]]:
    names = [stage] if stage else list(STAGES)
    out: list[tuple[list[str], str]] = []
    out += COHORT_PREP.get(cohort, [])
    for n in names:
        if n not in STAGES:
            sys.exit(f"unknown stage {n!r}, choose from {', '.join(STAGES)}")
        if cohort != "primary" and n in PRIMARY_ONLY:
            continue
        out += STAGES[n]
        if n == "experiments":
            out += COHORT_EXTRA.get(cohort, [])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", help="data, experiments, outputs, paper or verify")
    ap.add_argument("--cohort", default="primary", choices=sorted(COHORT_ENV),
                    help="primary (default), kl, or stress")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    ap.add_argument("--list", action="store_true", help="describe the stages and exit")
    a = ap.parse_args()

    if a.list:
        for name, cmds in STAGES.items():
            print(f"\n{name}")
            for cmd, writes in cmds:
                print(f"  {' '.join(cmd[1:]):68s} -> {writes}")
        print("\nSecondary cohorts add:")
        for coh, cmds in COHORT_EXTRA.items():
            for cmd, writes in cmds:
                print(f"  [{coh}] {' '.join(cmd[1:]):61s} -> {writes}")
        print("\nThe external validation of Section VI-H needs a download and is "
              "run on its own.\nSee docs/REPRODUCTION.md.")
        return 0

    env = dict(os.environ)
    env.update(COHORT_ENV[a.cohort])
    steps = plan(a.stage, a.cohort)

    print(f"cohort {a.cohort}, {len(steps)} step(s)")
    if a.cohort != "primary":
        print("writing to the cohort's own directories, "
              f"env {COHORT_ENV[a.cohort]}")

    started = time.time()
    for i, (cmd, writes) in enumerate(steps, 1):
        label = " ".join(cmd[1:])
        print(f"\n[{i}/{len(steps)}] {label}\n    writes {writes}", flush=True)
        if a.dry_run:
            continue
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT, env=env)
        if r.returncode != 0:
            print(f"\nFAILED at step {i}: {label} (exit {r.returncode})")
            print("Nothing after this step was run. Fix the cause and rerun the "
                  "same command: completed work is skipped.")
            return r.returncode
        print(f"    done in {time.time() - t0:.0f}s", flush=True)

    if not a.dry_run:
        print(f"\nall {len(steps)} step(s) finished in "
              f"{(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
