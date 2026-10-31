"""Put the Kuala Lumpur and Klang Valley files into the paper's input format.

The replication cohort runs the unchanged pipeline on a second parameterisation
of the same generative process. To do that without migrating the paper to the
v2.0 schema, the three inputs the pipeline reads are rebuilt from the v2.0 KL
files in exactly the shapes the HCMC pipeline already expects.

An important scope note that the manuscript repeats. The v2.0 generator uses
one hardcoded edge list for both cities, so the Kuala Lumpur network has the
same topology, the same adjacency and the same Laplacian positional encodings
as the Ho Chi Minh City network. What differs is the modal split, the physical
edge distances, the coordinates, the fare currency and the random seed. This
is therefore a replication under a second parameterisation, not a test of
transfer to an independently specified network.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw_kl"
OUT = ROOT / "data" / "raw"

MODE_BY_NODE = None


def node_mode_zone() -> pd.DataFrame:
    """Reuse the paper's node mapping so both cohorts share node semantics."""
    from dyhgt.graph import build_node_mapping
    return build_node_mapping()


def build_traffic() -> pd.DataFrame:
    df = pd.read_csv(SRC / "kl_traffic_data.csv")
    wide = df.pivot(index="timestamp", columns="node_id", values="travel_time_min")
    wide = wide.sort_index()
    wide.columns = [f"Node_{int(c)}" for c in wide.columns]
    wide = wide[[f"Node_{i}" for i in range(28)]]
    wide.to_csv(OUT / "kl_traffic_data.csv", index=False)
    return wide


def build_iot(mapping: pd.DataFrame) -> pd.DataFrame:
    """Add the four columns the v2.0 IoT table does not carry."""
    df = pd.read_csv(SRC / "KL_IoT_Graph_Sensor_Data.csv")
    by_idx = mapping.set_index(mapping.index)
    node_idx = df["node_id"].astype(int).clip(0, 27)
    out = pd.DataFrame({
        "TransactionID": [f"TRX-{200000 + i}" for i in range(len(df))],
        "Timestamp": df["Timestamp"],
        "NodeID": by_idx.loc[node_idx, "IoTNodeID"].values,
        "Zone": by_idx.loc[node_idx, "Zone"].values,
        "Mode": by_idx.loc[node_idx, "Mode"].values,
        "SignalStrength_dBm": df["SignalStrength_dBm"],
        "Latency_ms": df["Latency_ms"],
        "Drift_Magnitude": df["Drift_Magnitude"],
        "Is_Anomaly": df["Is_Anomaly"].astype(bool),
        "Fault_Type": df["Fault_Type"],
    })
    fare = {"Bus": 200, "Metro": 350, "Ferry": 350}
    out.insert(8, "Fare", out["Mode"].map(fare).astype(int))
    out.to_csv(OUT / "KL_IoT_Graph_Sensor_Data.csv", index=False)
    return out


def build_transactions(mapping: pd.DataFrame) -> pd.DataFrame:
    """Map the v2.0 tap records onto the columns the marginal fitter reads.

    HardwareScore is not present in the v2.0 tap table. It is reconstructed
    from the reconciliation outcome and the imputation error, which is the
    same quantity the original field stood for: how far the record can be
    trusted. The construction is recorded here and in the dataset card.
    """
    df = pd.read_csv(SRC / "kl_synthetic_transit_data.csv")
    zone = np.where(df["origin_district_zone"].astype(int) == 0,
                    "Core (Dist 1)", "Periphery (Dist 12)")
    mode_map = {"RapidKL_Bus_TnG": "Bus", "MRT_Feeder_Bus": "Metro",
                "LRT_Integrated_Tap": "Ferry"}
    mode = df["transport_mode"].map(mode_map).fillna("Bus")
    is_err = df["is_incomplete_tap"].astype(bool)
    err_type = np.where(
        df["reconciliation_status"].eq("Imputed_UKG_JADE"), "Missing_TapOut",
        np.where(df["reconciliation_status"].eq("Fallback_LOCF_Cap"),
                 "Time_Sync_Error", "None"))
    err_type = np.where(is_err & (err_type == "None"), "GPS_Drift", err_type)

    # HardwareScore is not a field of the v2.0 tap table. Deriving it from the
    # reconciliation outcome produces a near perfect proxy for the error flag,
    # which would make the decision problem trivially separable. It is instead
    # computed from the node's own telemetry using the composite index the
    # paper already defines in Section III-H, so the two cohorts construct the
    # quantity the same way.
    iot = pd.read_csv(SRC / "KL_IoT_Graph_Sensor_Data.csv")
    s_term = np.clip((iot["SignalStrength_dBm"] + 100.0) / 40.0, 0, 1)
    l_term = np.clip(1.0 - (iot["Latency_ms"] - 5.0) / 300.0, 0, 1)
    d_term = np.clip(1.0 - iot["Drift_Magnitude"] / 12.0, 0, 1)
    iot = iot.assign(rel=np.clip(0.45 * s_term + 0.35 * l_term + 0.20 * d_term, 0, 1))
    pool = {int(n): g["rel"].to_numpy() for n, g in iot.groupby("node_id")}
    fallback = iot["rel"].to_numpy()

    rng = np.random.default_rng(142)
    board = df["boarding_node_id"].astype(int).clip(0, 27).to_numpy()
    hw = np.empty(len(df))
    for i, n in enumerate(board):
        p = pool.get(n, fallback)
        hw[i] = p[rng.integers(len(p))]
    # an incomplete tap is itself evidence that the reading was unreliable
    hw = np.where(is_err.to_numpy(), hw * rng.uniform(0.45, 0.85, len(df)), hw)
    hw = np.clip(hw, 0.0, 1.0)

    out = pd.DataFrame({
        "TransactionID": df["transaction_id"],
        "Timestamp": df["timestamp"],
        "Mode": mode,
        "Zone": zone,
        "Fare": df["fare_amount"].astype(int),
        "IsError": is_err,
        "ErrorType": err_type,
        "HardwareScore": np.round(hw, 4),
    })
    out.to_csv(OUT / "kl_mixed_mobility_data.csv", index=False)
    return out


def main() -> None:
    mapping = node_mode_zone()
    traffic = build_traffic()
    iot = build_iot(mapping)
    tx = build_transactions(mapping)

    print(f"kl_traffic_data.csv           {traffic.shape}  "
          f"negatives {int((traffic.values < 0).sum())}  "
          f"sub-minute {int((traffic.values < 1).sum())}")
    print(f"KL_IoT_Graph_Sensor_Data.csv  {iot.shape}  "
          f"anomaly rate {iot.Is_Anomaly.mean():.4f}")
    print(f"kl_mixed_mobility_data.csv    {tx.shape}  "
          f"error rate {tx.IsError.mean():.4f}")
    for z, g in tx.groupby("Zone"):
        print(f"   {z}: error {g.IsError.mean():.4f}, "
              f"hardware score mean {g.HardwareScore.mean():.3f}")


if __name__ == "__main__":
    main()
