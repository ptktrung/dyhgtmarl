"""Figure 10: fairness and utility trade off for the constrained policy."""
from _common import PALETTE, TAB, need, save
import matplotlib.pyplot as plt
import pandas as pd


def main():
    need(TAB / "exp_c_pareto.csv", TAB / "exp_c_policy.csv")
    p = pd.read_csv(TAB / "exp_c_pareto.csv")
    base = pd.read_csv(TAB / "exp_c_policy.csv")
    g = p.groupby("fairness_budget").agg(
        rev=("revenue_recovered_pct", "mean"), rev_sd=("revenue_recovered_pct", "std"),
        gap=("equal_opportunity_gap", "mean"), gap_sd=("equal_opportunity_gap", "std"),
        review=("review_workload_pct", "mean")).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    ax = axes[0]
    ax.errorbar(g.gap, g.rev, xerr=g.gap_sd.fillna(0), yerr=g.rev_sd.fillna(0),
                marker="o", markersize=4, linewidth=1.1, capsize=2, color=PALETTE[0],
                label="constrained decentralized policy")
    # the budgets cluster because the constraint never binds, so labels are
    # staggered vertically instead of overprinting one another
    for k, (_, r) in enumerate(g.sort_values("gap").iterrows()):
        ax.annotate(f"eps={r.fairness_budget:g}", (r.gap, r.rev), fontsize=6.2,
                    xytext=(5, 9 - 7 * (k % 4)), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    for i, lab in enumerate(["rule_based_legacy", "centralized_utility_max"]):
        sub = base[base.policy == lab]
        if not sub.empty:
            ax.scatter(sub.equal_opportunity_gap.mean(), sub.revenue_recovered_pct.mean(),
                       marker="s", s=48, color=PALETTE[i + 1], label=lab.replace("_", " "))
    ax.set_xlabel("equal opportunity gap between zones")
    ax.set_ylabel("net revenue recovered, percent of potential")
    ax.legend(fontsize=7.5); ax.set_title("Fairness and utility trade off")

    ax = axes[1]
    ax.plot(g.fairness_budget, g.gap, marker="o", markersize=3.4, color=PALETTE[1],
            label="achieved gap")
    ax.plot(g.fairness_budget, g.fairness_budget, "--", color="#90a4ae", linewidth=1,
            label="budget")
    ax2 = ax.twinx()
    ax2.plot(g.fairness_budget, g.review, marker="^", markersize=3.4, color=PALETTE[2],
             label="review workload")
    ax2.set_ylabel("human review workload, percent"); ax2.grid(False)
    ax.set_xlabel("fairness budget epsilon_f"); ax.set_ylabel("equal opportunity gap")
    ax.legend(fontsize=7.5, loc="upper left"); ax2.legend(fontsize=7.5, loc="lower right")
    ax.set_title("Constraint satisfaction and its workload cost")
    fig.tight_layout()
    save(fig, "fig_10_fairness_utility_pareto")


if __name__ == "__main__":
    main()
