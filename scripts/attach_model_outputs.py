"""Populate the model output columns of the integrated benchmark.

The distributed benchmark leaves these columns empty so that a prediction can
never be mistaken for a data field. This script fills them from the exported
prediction artifacts, and it fills only the rows those artifacts cover.
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from dyhgt.data import N_TRANSIT
from dyhgt.graph import build_node_mapping
from dyhgt.utils import PROC, require


def main():
    require(PROC / "integrated_benchmark.csv", "Run build_integrated_benchmark.py first.")
    df = pd.read_csv(PROC / "integrated_benchmark.csv")
    mapping = build_node_mapping()
    order = {n: i for i, n in enumerate(mapping.UnifiedNodeID)}
    node_idx = df.UnifiedNodeID.map(order).to_numpy()
    step = df.Step.to_numpy()
    filled = []

    fe = PROC / "forecast_example.npz"
    if fe.exists():
        z = np.load(fe)
        for col, key in [("ForecastMean", "mean"), ("ForecastLower95", "lo"),
                         ("ForecastUpper95", "hi")]:
            grid = np.full((df.Step.max() + 1, N_TRANSIT), np.nan)
            grid[z["steps"]] = z[key]
            df[col] = np.round(grid[step, node_idx], 4)
        filled += ["ForecastMean", "ForecastLower95", "ForecastUpper95"]

    re_ = PROC / "reconstruction_example.npz"
    if re_.exists():
        z = np.load(re_)
        grid = np.full((df.Step.max() + 1, N_TRANSIT), np.nan)
        grid[z["steps"]] = z["fault_prob"]
        df["FaultProbability"] = np.round(grid[step, node_idx], 5)
        rg = np.full((df.Step.max() + 1, N_TRANSIT), np.nan)
        rg[z["steps"]] = z["recon"]
        df["RepairConfidence"] = np.where(np.isfinite(rg[step, node_idx]), 1.0 - df["MissingMask"], np.nan)
        df["RepairAction"] = np.where(df["MissingMask"] == 1, "impute", "none")
        filled += ["FaultProbability", "RepairAction", "RepairConfidence"]

    out = PROC / "integrated_benchmark_with_model_outputs.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out.name}; populated columns: {filled or 'none, no artifacts found'}")
    print("Columns left empty remain empty by design.")


if __name__ == "__main__":
    main()
