"""Table XIII: Kuala Lumpur replication cohort against the primary cohort."""
from _common import TAB, emit, need
import pathlib
import pandas as pd

TAB_KL = pathlib.Path(str(TAB) + "_kl")


def pair(metric_rows):
    return pd.DataFrame(metric_rows)


def main():
    need(TAB / "exp_b_forecast_conditions.csv", TAB_KL / "exp_b_forecast_conditions.csv")
    rows = []

    for label, path in (("Ho Chi Minh City", TAB), ("Kuala Lumpur", TAB_KL)):
        b = pd.read_csv(path / "exp_b_forecast_conditions.csv")
        n = b[(b.model == "dyhgt") & (b.condition == "normal_test")]
        # The two cohorts have different travel time scales, with cohort means
        # of roughly 25.6 and 6.1 minutes, so absolute error is not comparable
        # across them. Scale free quantities are given first and the absolute
        # ones are marked.
        for m, nice in (("mape", "Forecast MAPE, percent (scale free)"),
                        ("picp95", "Interval coverage at 95 percent (scale free)"),
                        ("ece", "Calibration error (scale free)"),
                        ("mae", "Forecast MAE, minutes (within cohort only)"),
                        ("crps", "Forecast CRPS (within cohort only)")):
            rows.append({"Quantity": nice, "Cohort": label,
                         "Value": f"{n[m].mean():.3f} +/- {n[m].std():.3f}"})
        s = pd.read_csv(path / "exp_b_missingness_sweep.csv")
        d = s[s.model == "dyhgt"].groupby("missing_rate").crps.mean()
        ha = s[s.model == "ha"].groupby("missing_rate").crps.mean()
        rows.append({"Quantity": "CRPS change, 0 to 40 percent missing, proposed model",
                     "Cohort": label,
                     "Value": f"{(d.iloc[-1] / d.iloc[0] - 1) * 100:+.1f} percent"})
        rows.append({"Quantity": "CRPS change, 0 to 40 percent missing, climatology",
                     "Cohort": label,
                     "Value": f"{(ha.iloc[-1] / ha.iloc[0] - 1) * 100:+.1f} percent"})

        if (path / "exp_a_fault_detection.csv").exists():
            a = pd.read_csv(path / "exp_a_fault_detection.csv")
            for mdl, nice in (("mlp", "Fault F1, multilayer perceptron"),
                              ("dyhgt_marl", "Fault F1, proposed model")):
                g = a[a.model == mdl]
                if not g.empty:
                    rows.append({"Quantity": nice, "Cohort": label,
                                 "Value": f"{g.f1.mean():.3f} +/- {g.f1.std():.3f}"})

        c = pd.read_csv(path / "exp_c_policy.csv")
        leg = c[c.policy == "rule_based_legacy"]
        dec = c[c.policy == "decentralized_constrained"]
        rows.append({"Quantity": "Legacy rule, equal opportunity gap", "Cohort": label,
                     "Value": f"{leg.equal_opportunity_gap.mean():.4f}"})
        rows.append({"Quantity": "Learned policy, equal opportunity gap", "Cohort": label,
                     "Value": f"{dec.equal_opportunity_gap.mean():.4f}"})
        rows.append({"Quantity": "Legacy rule, revenue recovered percent", "Cohort": label,
                     "Value": f"{leg.revenue_recovered_pct.mean():.2f}"})
        rows.append({"Quantity": "Learned policy, revenue recovered percent", "Cohort": label,
                     "Value": f"{dec.revenue_recovered_pct.mean():.2f}"})
        rows.append({"Quantity": "Learned policy, abstention rate", "Cohort": label,
                     "Value": f"{dec.abstention_rate.mean():.3f}"})

        e = path / "exp_e_cost_sensitivity.csv"
        if e.exists():
            s2 = pd.read_csv(e).groupby("false_accept_multiplier").frac_accept.mean()
            mixed = [m for m, v in s2.items() if 0.02 < v < 0.98]
            rows.append({"Quantity": "Penalty settings producing a mixed policy",
                         "Cohort": label,
                         "Value": "none" if not mixed else
                                  ", ".join(f"{m:g}" for m in mixed)})

    df = pd.DataFrame(rows).pivot(index="Quantity", columns="Cohort",
                                  values="Value").reset_index()
    emit(df, "table_13_replication_cohort",
         "Table XIII. Primary cohort against the Kuala Lumpur replication cohort. Both "
         "cohorts use the same network topology and the same scenario layer, and differ "
         "in modal split, physical edge distance, fare scale and random seed.")


if __name__ == "__main__":
    main()
