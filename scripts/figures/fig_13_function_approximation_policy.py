"""Figure 13: function approximation policy against the tabular policy."""
from _common import PALETTE, TAB, need, save
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    need(TAB / "exp_e_policy_gradient.csv", TAB / "exp_c_policy.csv",
         TAB / "exp_e_cost_sensitivity.csv")
    e = pd.read_csv(TAB / "exp_e_policy_gradient.csv")
    c = pd.read_csv(TAB / "exp_c_policy.csv")
    s = pd.read_csv(TAB / "exp_e_cost_sensitivity.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))

    ax = axes[0]
    pairs = [("decentralized_constrained", "pg_decentralized_constrained", "decentralized\nconstrained"),
             ("centralized_constrained", "pg_centralized_constrained", "centralized\nconstrained"),
             ("decentralized_utility_max", "pg_decentralized_utility_max", "decentralized\nutility max")]
    xs = np.arange(len(pairs))
    tab_m = [c[c.policy == a].revenue_recovered_pct.mean() for a, _, _ in pairs]
    tab_s = [c[c.policy == a].revenue_recovered_pct.std() for a, _, _ in pairs]
    pg_m = [e[e.policy == b].revenue_recovered_pct.mean() for _, b, _ in pairs]
    pg_s = [e[e.policy == b].revenue_recovered_pct.std() for _, b, _ in pairs]
    ax.bar(xs - 0.2, tab_m, yerr=np.nan_to_num(tab_s), width=0.4, color=PALETTE[0],
           capsize=3, label="tabular action value")
    ax.bar(xs + 0.2, pg_m, yerr=np.nan_to_num(pg_s), width=0.4, color=PALETTE[1],
           capsize=3, label="linear policy gradient")
    ax.set_xticks(xs); ax.set_xticklabels([p[2] for p in pairs], fontsize=8)
    ax.set_ylim(80, 97)
    ax.set_ylabel("net revenue recovered, percent")
    ax.legend(fontsize=7.5)
    ax.set_title("Representation changes the optimum reached,\nnot the ordering between variants",
                 fontsize=9)

    ax = axes[1]
    g = s.groupby("false_accept_multiplier").agg(
        acc=("frac_accept", "mean"), abst=("abstention_rate", "mean"),
        rev=("revenue_recovered_pct", "mean")).reset_index()
    ax.plot(g.false_accept_multiplier, g.acc, marker="o", markersize=4,
            color=PALETTE[2], linewidth=1.2, label="fraction accepted")
    ax.plot(g.false_accept_multiplier, g.abst, marker="s", markersize=4,
            color=PALETTE[1], linewidth=1.2, label="fraction abstained")
    ax.set_xscale("log")
    ax.set_xticks(g.false_accept_multiplier)
    ax.set_xticklabels([f"{v:g}" for v in g.false_accept_multiplier])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("penalty multiplier for accepting an invalid record")
    ax.set_ylabel("fraction of decisions")
    ax2 = ax.twinx()
    ax2.plot(g.false_accept_multiplier, g.rev, marker="^", markersize=4,
             color=PALETTE[5], linewidth=1.0, linestyle="--", label="revenue")
    ax2.set_ylabel("net revenue recovered, percent"); ax2.grid(False)
    ax.legend(fontsize=7.5, loc="center left")
    ax2.legend(fontsize=7.5, loc="center right")
    ax.set_title("The learned policy is constant at every cost setting,\n"
                 "switching between the two constants near a multiplier of 15",
                 fontsize=9)
    fig.tight_layout()
    save(fig, "fig_13_function_approximation_policy")


if __name__ == "__main__":
    main()
