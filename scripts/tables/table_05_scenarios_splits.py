"""Table V: scenario families, their parameters and the data splits."""
from _common import PROC, emit, need
import json
import numpy as np
import pandas as pd
import yaml


def main():
    need(PROC / "scenario_summary.json", PROC / "splits.csv", PROC / "event_log.csv")
    ss = json.load(open(PROC / "scenario_summary.json"))
    cfg = yaml.safe_load(open(PROC.parents[1] / "configs" / "default.yaml"))
    sc = cfg["scenarios"]
    ev = pd.read_csv(PROC / "event_log.csv")
    splits = pd.read_csv(PROC / "splits.csv")
    mech = ss["missing_mechanism_counts"]
    rows = [
        ["S1 clean baseline", "all steps outside any injected event window",
         f"{ss['fault_type_counts']['None']} nominal node steps"],
        ["S2 independent missingness", f"rates {sc['missing_rates']}",
         "six MCAR masks stored separately, applied at evaluation"],
        ["S3 signal loss", "signal forced to -120 dBm", f"{ss['fault_type_counts']['Signal_Loss']} node steps"],
        ["S4 latency spike", "200 to 2000 ms added", f"{ss['fault_type_counts']['High_Latency']} node steps"],
        ["S5 monotonic drift", f"accumulating, recovery factor {sc['drift_recovery']}",
         f"{ss['fault_type_counts']['Sensor_Drift']} node steps"],
        ["S6 graph cascades",
         f"p0 = {sc['cascade_base_prob']}, decay tau = {sc['cascade_decay_steps']} steps, "
         f"{sc['n_cascades_per_stratum']} roots per centrality stratum",
         f"{int((ev.EventType == 'cascade').sum())} cascade events"],
        ["S7 gateway outages", f"{sc['n_gateway_outages']} scheduled outages",
         f"{mech['gateway_outage']} node steps missing"],
        ["S8 peak hour congestion", "inherited from the supplied traffic matrix",
         "present in every day of the series"],
        ["S9 peripheral MNAR", f"logistic in own signal, slope {sc['mnar_slope']}",
         f"{mech['MNAR_signal']} node steps missing"],
        ["S10 topology change", f"windows {ss['closure_windows']}, {sc['n_closed_edges']} edges each",
         f"{mech['topology_closure']} node steps missing"],
        ["S11 distribution shift", f"scale {sc['shift_scale']}, extra noise {sc['shift_noise']}",
         f"from step {ss['distribution_shift_start_step']} to the end"],
        ["S12 targeted perturbation", "independent masks applied to selected nodes at evaluation",
         "used in the robustness sweep"],
    ]
    df = pd.DataFrame(rows, columns=["Scenario", "Parameters", "Realised size"])
    sp = splits.Split.value_counts()
    df = pd.concat([df, pd.DataFrame([
        ["Split train", "chronological, steps 0 to 2014", f"{int(sp['train'])} steps"],
        ["Split validation", "chronological, steps 2015 to 2302", f"{int(sp['val'])} steps"],
        ["Split test", "chronological, steps 2303 to 2879", f"{int(sp['test'])} steps"],
    ], columns=df.columns)], ignore_index=True)
    emit(df, "table_05_scenarios_splits",
         "Table V. Scenario definitions, generator parameters and chronological splits.")


if __name__ == "__main__":
    main()
