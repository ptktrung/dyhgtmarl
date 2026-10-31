"""Assemble the time aligned integrated benchmark table and the split index."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from dyhgt.datagen import FAULT_TYPES, START, STEP_MINUTES
from dyhgt.graph import build_node_mapping
from dyhgt.utils import PROC, ensure_dirs, load_config, save_json, require

MECHS = ["observed", "MCAR", "MNAR_signal", "gateway_outage", "topology_closure"]

def main():
    cfg = load_config(); ensure_dirs()
    require(PROC / "travel_true.npy", "Run inject_faults_and_missingness.py first.")
    tt = np.load(PROC / "travel_true.npy"); to = np.load(PROC / "travel_observed.npy")
    fault = np.load(PROC / "fault_true.npy"); root = np.load(PROC / "root_id.npy")
    eid = np.load(PROC / "event_id.npy"); drift = np.load(PROC / "drift.npy")
    sig = np.load(PROC / "signal.npy"); lat = np.load(PROC / "latency.npy")
    miss = np.load(PROC / "missing_mask.npy"); mech = np.load(PROC / "missing_mechanism.npy")
    mapping = build_node_mapping()
    T, N = tt.shape

    # HardwareScore is a synthesized node-step reliability index in [0, 1]
    s_term = np.clip((sig + 100.0) / 40.0, 0, 1)
    l_term = np.clip(1.0 - (lat - 5.0) / 300.0, 0, 1)
    d_term = np.clip(1.0 - drift / 12.0, 0, 1)
    hw = np.clip(0.45 * s_term + 0.35 * l_term + 0.20 * d_term, 0.0, 1.0)

    ts = pd.to_datetime(START) + pd.to_timedelta(np.arange(T) * STEP_MINUTES, unit="m")
    step_idx = np.repeat(np.arange(T), N)
    node_idx = np.tile(np.arange(N), T)

    df = pd.DataFrame({
        "Step": step_idx,
        "Timestamp": np.repeat(ts.values, N),
        "UnifiedNodeID": mapping.UnifiedNodeID.values[node_idx],
        "TrafficNodeID": mapping.TrafficNodeID.values[node_idx],
        "IoTNodeID": mapping.IoTNodeID.values[node_idx],
        "NodeType": "transit",
        "GatewayID": mapping.GatewayID.values[node_idx],
        "Zone": mapping.Zone.values[node_idx],
        "Mode": mapping.Mode.values[node_idx],
        "TravelTime_True": np.round(tt.ravel(), 4),
        "TravelTime_Observed": np.round(to.ravel(), 4),
        "SignalStrength_dBm": np.round(sig.ravel(), 3),
        "Latency_ms": np.round(lat.ravel(), 3),
        "Drift_Magnitude": np.round(drift.ravel(), 4),
        "HardwareScore": np.round(hw.ravel(), 4),
        "FaultType_True": np.array(FAULT_TYPES)[fault.ravel()],
        "CascadeRootID": np.where(root.ravel() >= 0,
                                  mapping.UnifiedNodeID.values[np.clip(root.ravel(), 0, N - 1)], ""),
        "EventID": eid.ravel(),
        "MissingMask": miss.ravel().astype(int),
        "MissingnessMechanism": np.array(MECHS)[mech.ravel()],
    })
    df["ProvenanceID"] = ("PRV-" + df["UnifiedNodeID"] + "-" + df["Step"].astype(str).str.zfill(5))
    # model output columns, filled by scripts/attach_model_outputs.py after the
    # experiments are run. They are empty in the distributed benchmark.
    for c in ["FaultProbability", "RepairAction", "RepairConfidence", "ForecastMean",
              "ForecastLower95", "ForecastUpper95", "PolicyAction", "PolicyConfidence",
              "EscalationRequired"]:
        df[c] = pd.NA

    df.to_csv(PROC / "integrated_benchmark.csv", index=False)

    tr = int(cfg["splits"]["train_frac"] * T); va = tr + int(cfg["splits"]["val_frac"] * T)
    splits = pd.DataFrame({"Step": np.arange(T),
                           "Split": np.where(np.arange(T) < tr, "train",
                                             np.where(np.arange(T) < va, "val", "test"))})
    splits.to_csv(PROC / "splits.csv", index=False)
    save_json({"rows": int(len(df)), "steps": int(T), "nodes": int(N),
               "train_steps": int(tr), "val_steps": int(va - tr), "test_steps": int(T - va),
               "columns": list(df.columns),
               "model_output_columns_left_empty": ["FaultProbability", "RepairAction",
                    "RepairConfidence", "ForecastMean", "ForecastLower95", "ForecastUpper95",
                    "PolicyAction", "PolicyConfidence", "EscalationRequired"]},
              PROC / "benchmark_summary.json")
    print("integrated benchmark", df.shape, "train/val/test steps", tr, va - tr, T - va)

if __name__ == "__main__":
    main()
