"""Table XII: function approximation check on the decision layer."""
from _common import TAB, agg, emit, need
import pandas as pd


def main():
    need(TAB / "exp_e_policy_gradient.csv", TAB / "exp_c_policy.csv")
    e = pd.read_csv(TAB / "exp_e_policy_gradient.csv")
    c = pd.read_csv(TAB / "exp_c_policy.csv")
    keep = ["decentralized_constrained", "centralized_constrained",
            "decentralized_utility_max"]
    c = c[c.policy.isin(keep)].copy()
    c["policy"] = "tabular_" + c["policy"]
    cols = ["revenue_recovered_pct", "decision_error_rate", "false_accept_rate",
            "abstention_rate", "review_workload_pct", "equal_opportunity_gap",
            "demographic_parity_gap"]
    both = pd.concat([c[["policy", "seed"] + cols], e[["policy", "seed"] + cols]],
                     ignore_index=True)
    t = agg(both, "policy", cols, nd=4)
    n = both.groupby("policy").seed.count().rename("n_seeds").reset_index()
    t = t.merge(n, on="policy")
    emit(t, "table_12_function_approximation",
         "Table XII. Tabular action value policies against linear softmax policies "
         "trained by REINFORCE with a learned value baseline, on the same constrained "
         "process, state information, reward, dual ascent, split and seeds.")

    s = TAB / "exp_e_cost_sensitivity.csv"
    if s.exists():
        d = pd.read_csv(s)
        g = d.groupby("false_accept_multiplier").agg(
            accept=("frac_accept", "mean"), defer=("frac_defer", "mean"),
            abstention=("abstention_rate", "mean"),
            revenue=("revenue_recovered_pct", "mean"),
            error=("decision_error_rate", "mean"),
            review=("review_workload_pct", "mean")).round(4).reset_index()
        g.columns = ["False accept multiplier", "Fraction accept", "Fraction defer",
                     "Abstention rate", "Revenue recovered percent",
                     "Decision error rate", "Review workload percent"]
        emit(g, "table_12b_cost_sensitivity",
             "Table XII b. Action mix chosen by the function approximation policy as the "
             "penalty for accepting an invalid record rises. The policy is constant at "
             "every setting tested, and the switch between the two constants occurs "
             "between multipliers 10 and 20.")


if __name__ == "__main__":
    main()
