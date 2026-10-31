"""Table X: system cost and end to end summary."""
from _common import TAB, emit, need
import pandas as pd


def main():
    need(TAB / "exp_c_policy.csv")
    rows = []
    c = pd.read_csv(TAB / "exp_c_policy.csv")
    for pol, g in c.groupby("policy"):
        rows.append({"Component": f"policy: {pol}",
                     "Cost measure": "milliseconds per decision",
                     "Nominal model value": round(g.nominal_latency_ms.mean(), 3),
                     "Measured on this machine": round(g.measured_wall_ms_per_decision.mean(), 4),
                     "Note": "nominal values are simulator assumptions"})
    for f, lab in [("exp_a_fault_detection.csv", "fault model"),
                   ("exp_b_forecast_conditions.csv", "forecast model")]:
        p = TAB / f
        if p.exists():
            d = pd.read_csv(p)
            if "train_seconds" in d:
                for m, g in d.groupby("model"):
                    rows.append({"Component": f"{lab}: {m}",
                                 "Cost measure": "seconds to fit one seed",
                                 "Nominal model value": "",
                                 "Measured on this machine": round(g.train_seconds.mean(), 1),
                                 "Note": "single CPU core, no GPU"})
    emit(pd.DataFrame(rows), "table_10_system_cost",
         "Table X. Computational cost of each component. Training times were measured on a "
         "single CPU core and are not a hardware benchmark of a deployed system.")


if __name__ == "__main__":
    main()
