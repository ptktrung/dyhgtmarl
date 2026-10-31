"""Table IX: policy utility, workload and fairness results."""
from _common import TAB, agg, emit, need
import pandas as pd


def main():
    need(TAB / "exp_c_policy.csv")
    d = pd.read_csv(TAB / "exp_c_policy.csv")
    cols = ["revenue_recovered_pct", "decision_error_rate", "false_accept_rate",
            "valid_rejection_rate", "abstention_rate", "review_workload_pct",
            "demographic_parity_gap", "equal_opportunity_gap"]
    t = agg(d, "policy", cols)
    zone_cols = [c for c in d.columns if c.startswith("tpr_")]
    t = t.merge(agg(d, "policy", zone_cols), on="policy")
    emit(t, "table_09_policy_fairness",
         "Table IX. Decision policy outcomes on the held out transaction block. Mean and "
         "standard deviation over seeds. Revenue is net of review and corroboration costs.")
    if (TAB / "exp_c_pareto.csv").exists():
        p = pd.read_csv(TAB / "exp_c_pareto.csv")
        emit(agg(p, "fairness_budget", ["revenue_recovered_pct", "equal_opportunity_gap",
                                        "review_workload_pct", "abstention_rate"]),
             "table_09b_fairness_pareto",
             "Table IX b. Fairness budget sweep for the constrained decentralized policy.")


if __name__ == "__main__":
    main()
