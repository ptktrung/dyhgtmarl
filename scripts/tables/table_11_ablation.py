"""Table XI: ablation results."""
from _common import TAB, agg, emit, need
import pandas as pd


def main():
    need(TAB / "exp_d_ablation_model.csv")
    d = pd.read_csv(TAB / "exp_d_ablation_model.csv")
    cols = [c for c in ["f1", "pr_auc", "root_top1", "reconstruction_mae", "forecast_mae",
                        "forecast_crps", "forecast_picp95", "forecast_ece"] if c in d]
    t = agg(d, "variant", cols)
    parts = [t]
    p = TAB / "exp_d_ablation_policy.csv"
    if p.exists():
        c = pd.read_csv(p)
        t2 = agg(c, "variant", ["revenue_recovered_pct", "equal_opportunity_gap",
                                "review_workload_pct", "false_accept_rate"])
        parts.append(t2)
    emit(pd.concat(parts, ignore_index=True), "table_11_ablation",
         "Table XI. Ablation of model side and policy side design choices. Model rows are "
         "retrained from scratch; policy rows are taken from Experiment C.")


if __name__ == "__main__":
    main()
