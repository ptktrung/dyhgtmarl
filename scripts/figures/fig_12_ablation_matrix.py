"""Figure 12: ablation of model and policy design choices."""
from _common import PALETTE, TAB, need, save
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    need(TAB / "exp_d_ablation_model.csv")
    d = pd.read_csv(TAB / "exp_d_ablation_model.csv")
    cols = [c for c in ["f1", "pr_auc", "root_top1", "reconstruction_mae",
                        "forecast_mae", "forecast_crps", "forecast_ece"] if c in d]
    g = d.groupby("variant")[cols].mean()
    if "full_model" in g.index:
        rel = g.divide(g.loc["full_model"]).sub(1.0) * 100.0
        rel = rel.drop(index="full_model")
    else:
        rel = g
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    im = ax.imshow(rel.values, cmap="RdBu_r", vmin=-np.nanmax(np.abs(rel.values)),
                   vmax=np.nanmax(np.abs(rel.values)), aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, fontsize=7.5, ha="right")
    ax.set_yticks(range(len(rel))); ax.set_yticklabels([i.replace("_", " ") for i in rel.index],
                                                       fontsize=7.5)
    for i in range(rel.shape[0]):
        for j in range(rel.shape[1]):
            v = rel.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=6.6)
    ax.grid(False)
    ax.set_title("Percent change relative to the full model.\n"
                 "Error columns: positive is worse. Score columns: negative is worse.",
                 fontsize=8.5)
    fig.colorbar(im, ax=ax, fraction=0.035, label="percent change")

    ax = axes[1]
    p = TAB / "exp_d_ablation_policy.csv"
    if p.exists():
        c = pd.read_csv(p).groupby("variant").agg(
            rev=("revenue_recovered_pct", "mean"), gap=("equal_opportunity_gap", "mean"))
        xs = np.arange(len(c))
        ax.bar(xs - 0.2, c.rev, width=0.4, color=PALETTE[0], label="revenue recovered, percent")
        ax2 = ax.twinx()
        ax2.bar(xs + 0.2, c.gap, width=0.4, color=PALETTE[1], label="equal opportunity gap")
        ax2.set_ylabel("gap"); ax2.grid(False)
        ax.set_xticks(xs); ax.set_xticklabels([i.replace("_", "\n") for i in c.index], fontsize=6.6)
        ax.set_ylabel("percent"); ax.legend(fontsize=7, loc="upper left")
        ax2.legend(fontsize=7, loc="lower right")
        ax.set_title("Policy side switches")
    else:
        ax.axis("off"); ax.text(0.5, 0.5, "policy ablation file not found", ha="center")
    fig.tight_layout()
    save(fig, "fig_12_ablation_matrix")


if __name__ == "__main__":
    main()
