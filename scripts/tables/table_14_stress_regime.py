"""Table XIV: primary benchmark against the revised stress regime."""
from _common import PROC, TAB, emit, need
import json
import pathlib
import pandas as pd

TAB_S = pathlib.Path(str(TAB) + "_stress")
PROC_S = pathlib.Path(str(PROC) + "_stress")


def main():
    need(TAB / "exp_a_fault_detection.csv", TAB_S / "exp_a_fault_detection.csv",
         PROC / "transactions.csv", PROC_S / "transactions.csv")
    rows = []

    def add(q, a, b):
        rows.append({"Quantity": q, "Primary benchmark": a, "Stress regime": b})

    # generator properties first, because they are what the regime changes
    sj = PROC_S / "scenario_summary.json"
    if sj.exists():
        s = json.loads(sj.read_text())
        if "stress_corr_before" in s:
            add("Mean pairwise node correlation, harmonized series",
                f"{s['stress_corr_before']:.4f}", f"{s['stress_corr_after']:.4f}")

    tp = pd.read_csv(PROC / "transactions.csv")
    ts = pd.read_csv(PROC_S / "transactions.csv")

    def vgap(d):
        c = d.loc[d.Zone.str.startswith("Core"), "TrueValid"].mean()
        p = d.loc[~d.Zone.str.startswith("Core"), "TrueValid"].mean()
        return c, p

    cp, pp = vgap(tp)
    cs, ps = vgap(ts)
    add("Latent validity, core zone", f"{cp:.4f}", f"{cs:.4f}")
    add("Latent validity, peripheral zone", f"{pp:.4f}", f"{ps:.4f}")
    add("Latent validity gap between zones", f"{cp - pp:+.4f}", f"{cs - ps:+.4f}")

    # fault detection
    pa = pd.read_csv(TAB / "exp_a_fault_detection.csv").groupby("model").f1.mean()
    sa = pd.read_csv(TAB_S / "exp_a_fault_detection.csv").groupby("model").f1.mean()
    for m, nice in (("mlp", "Fault F1, multilayer perceptron"),
                    ("hist_gradient_boosting", "Fault F1, gradient boosting"),
                    ("random_forest_graph_features", "Fault F1, graph feature forest"),
                    ("dyhgt_marl", "Fault F1, proposed encoder")):
        if m in pa.index or m in sa.index:
            add(nice, f"{pa.get(m, float('nan')):.3f}", f"{sa.get(m, float('nan')):.3f}")

    # forecasting on the undisturbed test block
    def crps(tab):
        f = tab / "exp_b_forecast_conditions.csv"
        if not f.exists():
            return None
        d = pd.read_csv(f)
        d = d[d.condition == "normal_test"].groupby("model").crps.mean()
        return d

    bp, bs = crps(TAB), crps(TAB_S)
    if bp is not None and bs is not None:
        for m, nice in (("dyhgt", "Forecast CRPS, proposed encoder"),
                        ("ha", "Forecast CRPS, climatological baseline"),
                        ("stgcn", "Forecast CRPS, spatio temporal baseline"),
                        ("ridge", "Forecast CRPS, linear baseline")):
            if m in bp.index or m in bs.index:
                add(nice, f"{bp.get(m, float('nan')):.3f}", f"{bs.get(m, float('nan')):.3f}")

    # decision layer
    if (TAB_S / "exp_c_policy.csv").exists():
        pc = pd.read_csv(TAB / "exp_c_policy.csv")
        sc = pd.read_csv(TAB_S / "exp_c_policy.csv")
        for pol, fld, nice in (
            ("decentralized_utility_max", "equal_opportunity_gap",
             "Unconstrained policy, equal opportunity gap"),
            ("decentralized_constrained", "equal_opportunity_gap",
             "Constrained policy, equal opportunity gap"),
            ("decentralized_constrained", "abstention_rate",
             "Constrained policy, abstention rate"),
            ("decentralized_constrained", "revenue_recovered_pct",
             "Constrained policy, revenue recovered percent"),
        ):
            fmt = "{:.2f}" if fld.endswith("_pct") else "{:.4f}"
            add(nice,
                fmt.format(pc[pc.policy == pol][fld].mean()),
                fmt.format(sc[sc.policy == pol][fld].mean()))

    # decision layer with the encoder's own fault head in place of the surrogate
    cf, cfp = TAB_S / "exp_c_policy_dyhgt.csv", TAB / "exp_c_policy_dyhgt.csv"
    if cf.exists():
        def coupled(path):
            if not path.exists():
                return None
            g = pd.read_csv(path)
            return g[g.policy == "decentralized_constrained"]
        cs, cpr = coupled(cf), coupled(cfp)
        for fld, nice in (("revenue_recovered_pct", "Coupled fault head, revenue recovered percent"),
                          ("decision_error_rate", "Coupled fault head, decision error rate"),
                          ("equal_opportunity_gap", "Coupled fault head, equal opportunity gap")):
            fmt = "{:.2f}" if fld.endswith("_pct") else "{:.4f}"
            rows.append({"Quantity": nice,
                         "Primary benchmark": fmt.format(cpr[fld].mean()) if cpr is not None else "not run",
                         "Stress regime": fmt.format(cs[fld].mean())})

    emit(pd.DataFrame(rows), "table_14_stress_regime",
         "Table XIV. The primary benchmark against the revised stress regime, in which "
         "faults show as neighbourhood disagreement rather than telemetry clamping, the "
         "node series are decorrelated, and latent validity and false accept cost differ "
         "by zone. Topology, splits and evaluation code are unchanged.")


if __name__ == "__main__":
    main()
