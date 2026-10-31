"""Scenario layer: cascades, outages, drift, MNAR and MCAR missingness."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from dyhgt.datagen import align_travel_time, simulate_scenarios, independent_missing_masks
from dyhgt.graph import build_node_mapping, build_topology
from dyhgt.utils import CITY, PROC, RAW, REGIME, ensure_dirs, load_config, raw_file, save_json, require
from dyhgt import stress as stressmod

def main():
    cfg = load_config(); ensure_dirs()
    traffic = pd.read_csv(require(raw_file("traffic"),
                                  "Place the cohort traffic CSV in data/raw.")).values
    travel_q, clipped = align_travel_time(traffic, seed=11)
    mapping = build_node_mapping()
    adj = build_topology(cfg["grid"]["n_nodes"], cfg["graph"]["topology_seed"])

    stress_info = {}
    if REGIME == "stress":
        travel_q, c_before, c_after = stressmod.decorrelate(
            travel_q, adj, cfg["stress"]["target_pairwise_correlation"], seed=cfg["seed"])
        stress_info = {"corr_before": c_before, "corr_after": c_after}
        print(f"[stress] pairwise correlation {c_before:.4f} -> {c_after:.4f}")

    out = simulate_scenarios(travel_q, mapping, adj, cfg, seed=cfg["seed"])

    if REGIME == "stress":
        obs, sig, lat, dri = stressmod.relational_faults(
            out["travel_true"], out["fault"], adj, out["signal"], out["latency"],
            out["drift"], cfg, seed=cfg["seed"])
        out["signal"], out["latency"], out["drift"] = sig, lat, dri
        keep_missing = out["miss"]
        obs = obs.copy()
        obs[keep_missing] = np.nan
        out["travel_observed"] = obs
        stress_info["relational_faults"] = True

    np.save(PROC / "travel_true.npy", out["travel_true"])
    np.save(PROC / "travel_observed.npy", out["travel_observed"])
    np.save(PROC / "fault_true.npy", out["fault"])
    np.save(PROC / "root_id.npy", out["root_id"])
    np.save(PROC / "event_id.npy", out["event_id"])
    np.save(PROC / "drift.npy", out["drift"])
    np.save(PROC / "signal.npy", out["signal"])
    np.save(PROC / "latency.npy", out["latency"])
    np.save(PROC / "missing_mask.npy", out["miss"])
    np.save(PROC / "missing_mechanism.npy", out["mech"])
    out["events"].to_csv(PROC / "event_log.csv", index=False)

    masks = independent_missing_masks(out["travel_true"].shape,
                                      cfg["scenarios"]["missing_rates"])
    for r, m in masks.items():
        np.save(PROC / f"mcar_mask_{int(r*100):02d}.npy", m)

    save_json({
        "clipped_subminute_entries_in_supplied_traffic": clipped,
        "fault_rate": float((out["fault"] > 0).mean()),
        "fault_type_counts": {k: int(v) for k, v in
                              zip(["None", "Signal_Loss", "High_Latency", "Sensor_Drift", "Gateway_Outage"],
                                  np.bincount(out["fault"].ravel(), minlength=5))},
        "missing_rate_structural": float(out["miss"].mean()),
        "missing_mechanism_counts": {k: int(v) for k, v in
                                     zip(["observed", "MCAR", "MNAR_signal", "gateway_outage", "topology_closure"],
                                         np.bincount(out["mech"].ravel(), minlength=5))},
        "n_events": int(len(out["events"])),
        "closure_windows": [list(w) for w in out["closure_windows"]],
        "distribution_shift_start_step": int(out["shift_start"]),
        "regime": REGIME, **{f"stress_{k}": v for k, v in stress_info.items()},
    }, PROC / "scenario_summary.json")
    print(f"[{CITY}] faults injected. fault rate", round(float((out["fault"] > 0).mean()), 5),
          "structural missing", round(float(out["miss"].mean()), 5),
          "events", len(out["events"]))

if __name__ == "__main__":
    main()
