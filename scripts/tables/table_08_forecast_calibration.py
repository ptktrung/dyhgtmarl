"""Table VIII: travel time accuracy and uncertainty calibration."""
from _common import TAB, agg, emit, need
import pandas as pd


def main():
    need(TAB / "exp_b_forecast_conditions.csv")
    d = pd.read_csv(TAB / "exp_b_forecast_conditions.csv")
    cols = ["mae", "rmse", "mape", "crps", "picp95", "ece"]
    parts = []
    for cond in ["normal_test", "cascade_active", "topology_change", "distribution_shift"]:
        sub = d[d.condition == cond]
        if sub.empty:
            continue
        t = agg(sub, "model", cols)
        t.insert(1, "condition", cond)
        parts.append(t)
    emit(pd.concat(parts, ignore_index=True), "table_08_forecast_calibration",
         "Table VIII. Travel time forecasting and uncertainty calibration by evaluation "
         "condition. Mean and standard deviation over five model seeds. Deterministic "
         "baselines are given a Gaussian predictive distribution fitted on validation "
         "residuals so that CRPS, PICP and ECE are comparable.")
    if (TAB / "exp_b_missingness_sweep.csv").exists():
        s = pd.read_csv(TAB / "exp_b_missingness_sweep.csv")
        t = agg(s, ["model", "missing_rate"], ["mae", "crps", "picp95", "ece"])
        emit(t, "table_08b_missingness_sweep",
             "Table VIII b. Independent missingness sweep on the normal test block.")


if __name__ == "__main__":
    main()
