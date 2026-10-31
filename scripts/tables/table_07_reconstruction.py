"""Table VII: reconstruction and data integrity results."""
from _common import TAB, emit, need
import pandas as pd


def main():
    need(TAB / "exp_a_reconstruction.csv")
    d = pd.read_csv(TAB / "exp_a_reconstruction.csv")
    g = d.groupby("model").agg(mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
                               rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
                               n_masked=("n_masked", "max")).reset_index()
    if "causal" in d:
        g = g.merge(d.groupby("model").causal.first().reset_index(), on="model", how="left")
    g["MAE minutes"] = [f"{m:.3f}" + (f" +/- {s:.3f}" if pd.notna(s) else "")
                        for m, s in zip(g.mae_mean, g.mae_sd)]
    g["RMSE minutes"] = [f"{m:.3f}" + (f" +/- {s:.3f}" if pd.notna(s) else "")
                         for m, s in zip(g.rmse_mean, g.rmse_sd)]
    out = g[["model", "MAE minutes", "RMSE minutes", "n_masked"] +
            (["causal"] if "causal" in g else [])].sort_values("MAE minutes")
    out = out.rename(columns={"model": "Method", "n_masked": "Masked cells scored",
                              "causal": "Causal, uses no future observation"})
    emit(out, "table_07_reconstruction",
         "Table VII. Reconstruction of masked travel time observations on the test block. "
         "Methods that use future observations are smoothers and are marked as not causal.")


if __name__ == "__main__":
    main()
