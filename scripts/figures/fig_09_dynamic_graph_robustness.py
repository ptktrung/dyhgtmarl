"""Figure 9: robustness across stress conditions and the missingness sweep."""
from _common import PALETTE, TAB, need, save
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    need(TAB / "exp_b_forecast_conditions.csv", TAB / "exp_b_missingness_sweep.csv")
    d = pd.read_csv(TAB / "exp_b_forecast_conditions.csv")
    s = pd.read_csv(TAB / "exp_b_missingness_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))

    ax = axes[0]
    conds = ["normal_test", "cascade_active", "topology_change", "distribution_shift"]
    models = sorted(d.model.unique())
    xs = np.arange(len(conds)); w = 0.8 / len(models)
    for i, m in enumerate(models):
        sub = d[d.model == m].groupby("condition").mae
        mean = [sub.mean().get(c, np.nan) for c in conds]
        sd = [sub.std().get(c, 0.0) for c in conds]
        ax.bar(xs + (i - (len(models) - 1) / 2) * w, mean, yerr=np.nan_to_num(sd), width=w,
               label=m, color=PALETTE[i % len(PALETTE)], capsize=1.6)
    ax.set_xticks(xs); ax.set_xticklabels([c.replace("_", "\n") for c in conds], fontsize=8)
    ax.set_ylabel("MAE, minutes"); ax.legend(fontsize=7, ncol=2)
    ax.set_title("Point error by evaluation condition")

    ax = axes[1]
    for i, m in enumerate(sorted(s.model.unique())):
        sub = s[s.model == m].groupby("missing_rate").crps
        ax.errorbar(sub.mean().index * 100, sub.mean().values,
                    yerr=np.nan_to_num(sub.std().values), marker="o", markersize=3,
                    linewidth=1.1, capsize=2, color=PALETTE[i % len(PALETTE)], label=m)
    ax.set_xlabel("independent missingness rate, percent"); ax.set_ylabel("CRPS")
    ax.legend(fontsize=7, ncol=2); ax.set_title("Degradation under independent missingness")
    fig.tight_layout()
    save(fig, "fig_09_dynamic_graph_robustness")


if __name__ == "__main__":
    main()
