"""Figure 14: replication of the headline results on the second cohort."""
from _common import PALETTE, TAB, need, save
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TAB_KL = pathlib.Path(str(TAB) + "_kl")


def main():
    need(TAB / "exp_b_missingness_sweep.csv", TAB_KL / "exp_b_missingness_sweep.csv",
         TAB / "exp_c_policy.csv", TAB_KL / "exp_c_policy.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))

    ax = axes[0]
    for (label, path, ls) in (("Ho Chi Minh City", TAB, "-"), ("Kuala Lumpur", TAB_KL, "--")):
        s = pd.read_csv(path / "exp_b_missingness_sweep.csv")
        for i, (m, nice) in enumerate((("dyhgt", "proposed"), ("ha", "climatology"))):
            g = s[s.model == m].groupby("missing_rate").crps.mean()
            ax.plot(g.index * 100, g.values, ls, marker="o" if i == 0 else "s",
                    markersize=3.4, linewidth=1.2, color=PALETTE[i],
                    label=f"{nice}, {label}")
    ax.set_xlabel("independent missingness rate, percent")
    ax.set_ylabel("CRPS")
    ax.legend(fontsize=6.8)
    ax.set_title("Robustness to observation loss replicates", fontsize=9)

    ax = axes[1]
    labels, hc, kl = [], [], []
    for pol, nice in (("rule_based_legacy", "legacy rule"),
                      ("decentralized_constrained", "learned,\nconstrained"),
                      ("decentralized_utility_max", "learned,\nunconstrained")):
        labels.append(nice)
        hc.append(pd.read_csv(TAB / "exp_c_policy.csv").query("policy==@pol")
                  .equal_opportunity_gap.mean())
        kl.append(pd.read_csv(TAB_KL / "exp_c_policy.csv").query("policy==@pol")
                  .equal_opportunity_gap.mean())
    xs = np.arange(len(labels))
    ax.bar(xs - 0.2, hc, width=0.4, color=PALETTE[0], label="Ho Chi Minh City")
    ax.bar(xs + 0.2, kl, width=0.4, color=PALETTE[1], label="Kuala Lumpur")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("equal opportunity gap between zones")
    ax.legend(fontsize=7.5)
    ax.set_title("The legacy rule's zone gap is larger\nin the second cohort", fontsize=9)

    ax = axes[2]
    for (label, path, ls) in (("Ho Chi Minh City", TAB, "-"), ("Kuala Lumpur", TAB_KL, "--")):
        p = path / "exp_e_cost_sensitivity.csv"
        if not p.exists():
            continue
        g = pd.read_csv(p).groupby("false_accept_multiplier").frac_accept.mean()
        ax.plot(g.index, g.values, ls, marker="o", markersize=4, linewidth=1.2,
                label=label)
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 40])
    ax.set_xticklabels(["2", "5", "10", "20", "40"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("penalty multiplier for accepting an invalid record")
    ax.set_ylabel("fraction of records accepted")
    ax.legend(fontsize=7.5)
    ax.set_title("The switching point differs,\nand only the second cohort mixes",
                 fontsize=9)
    fig.tight_layout()
    save(fig, "fig_14_replication_cohort")


if __name__ == "__main__":
    main()
