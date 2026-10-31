"""Table II: source datasets and the new integrated benchmark."""
from _common import PROC, RAW, emit, need
import json
import numpy as np
import pandas as pd


def main():
    need(RAW / "hcmc_traffic_data.csv", RAW / "HCMC_IoT_Graph_Sensor_Data.csv",
         RAW / "hcmc_mixed_mobility_data.csv", PROC / "benchmark_summary.json",
         PROC / "scenario_summary.json")
    tr = pd.read_csv(RAW / "hcmc_traffic_data.csv")
    iot = pd.read_csv(RAW / "HCMC_IoT_Graph_Sensor_Data.csv", parse_dates=["Timestamp"])
    tx = pd.read_csv(RAW / "hcmc_mixed_mobility_data.csv", parse_dates=["Timestamp"])
    bs = json.load(open(PROC / "benchmark_summary.json"))
    ss = json.load(open(PROC / "scenario_summary.json"))
    span_iot = iot.Timestamp.max() - iot.Timestamp.min()
    span_tx = tx.Timestamp.max() - tx.Timestamp.min()
    rows = [
        {"Dataset": "hcmc_traffic_data.csv (supplied)", "Rows": len(tr),
         "Columns": tr.shape[1], "Temporal span": "720 hourly steps, 30 days",
         "Positive or fault rate": "not applicable",
         "Known issue": f"{int((tr.values < 1.0).sum())} entries below one minute, "
                        f"{int((tr.values < 0).sum())} of them negative. Mean pairwise node "
                        f"correlation {np.corrcoef(tr.values.T)[np.triu_indices(tr.shape[1],1)].mean():.3f}"},
        {"Dataset": "HCMC_IoT_Graph_Sensor_Data.csv (supplied)", "Rows": len(iot),
         "Columns": iot.shape[1], "Temporal span": f"{span_iot.days} days {span_iot.seconds//3600} hours",
         "Positive or fault rate": f"{iot.Is_Anomaly.mean():.4f}",
         "Known issue": "cascade state is carried by the preceding row rather than by a graph neighbour"},
        {"Dataset": "hcmc_mixed_mobility_data.csv (supplied)", "Rows": len(tx),
         "Columns": tx.shape[1], "Temporal span": f"{span_tx.seconds//3600} hours",
         "Positive or fault rate": f"{tx.IsError.mean():.4f}",
         "Known issue": "no node identity and no label for whether a flagged fare was genuinely owed"},
        {"Dataset": "integrated_benchmark.csv (this work)", "Rows": bs["rows"],
         "Columns": len(bs["columns"]),
         "Temporal span": f"{bs['steps']} steps of 15 minutes, 30 days, {bs['nodes']} nodes",
         "Positive or fault rate": f"{ss['fault_rate']:.4f}",
         "Known issue": "fully synthetic. Travel time is interpolated from the supplied hourly matrix"},
        {"Dataset": "transactions.csv (this work)",
         "Rows": len(pd.read_csv(PROC / "transactions.csv")), "Columns": 14,
         "Temporal span": "aligned to the 2880 step grid",
         "Positive or fault rate": "flag rate and latent validity rate given in the dataset card",
         "Known issue": "latent validity is a modelling assumption, not an observed label"},
    ]
    emit(pd.DataFrame(rows), "table_02_source_datasets",
         "Table II. Source datasets and the integrated benchmark, with the limitation "
         "that matters most for each.")


if __name__ == "__main__":
    main()
