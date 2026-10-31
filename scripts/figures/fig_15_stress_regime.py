"""Figure 15: what the revised generator changes."""
from _common import PALETTE, PROC, TAB, need, save
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TAB_S = pathlib.Path(str(TAB) + "_stress")
PROC_S = pathlib.Path(str(PROC) + "_stress")


def main():
    need(TAB / "exp_a_fault_detection.csv", TAB_S / "exp_a_fault_detection.csv",
         PROC / "travel_true.npy", PROC_S / "travel_true.npy")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))

    ax = axes[0]
    pa = pd.read_csv(TAB / "exp_a_fault_detection.csv").groupby("model").f1.mean()
    sa = pd.read_csv(TAB_S / "exp_a_fault_detection.csv").groupby("model").f1.mean()
    models = [m for m in ["logistic_regression", "random_forest",
                          "random_forest_graph_features", "hist_gradient_boosting",
                          "mlp", "dyhgt_marl"] if m in pa.index and m in sa.index]
    xs = np.arange(len(models))
    ax.bar(xs - 0.2, [pa[m] for m in models], width=0.4, color=PALETTE[0],
           label="primary benchmark")
    ax.bar(xs + 0.2, [sa[m] for m in models], width=0.4, color=PALETTE[1],
           label="stress regime")
    ax.set_xticks(xs)
    ax.set_xticklabels([m.replace("_", " ") for m in models], fontsize=6.4,
                       rotation=35, ha="right")
    ax.set_ylabel("fault detection F1")
    ax.legend(fontsize=7.5)
    ax.set_title("Fault detection under relational faults", fontsize=9)

    ax = axes[1]
    for lab, p, col in (("primary", PROC / "travel_true.npy", PALETTE[0]),
                        ("stress", PROC_S / "travel_true.npy", PALETTE[1])):
        v = np.load(p)
        c = np.corrcoef(v.T)[np.triu_indices(v.shape[1], 1)]
        ax.hist(c, bins=40, alpha=0.6, color=col, label=f"{lab}, mean {c.mean():.3f}")
    ax.set_xlabel("pairwise node correlation")
    ax.set_ylabel("node pairs")
    ax.legend(fontsize=7.5)
    ax.set_title("Pairwise node correlation before and after\ndecorrelation", fontsize=9)

    ax = axes[2]
    if (TAB_S / "exp_c_policy.csv").exists():
        pc = pd.read_csv(TAB / "exp_c_policy.csv")
        sc = pd.read_csv(TAB_S / "exp_c_policy.csv")
        pols = ["decentralized_utility_max", "decentralized_constrained"]
        xs = np.arange(len(pols))
        ax.bar(xs - 0.2, [pc[pc.policy == p].equal_opportunity_gap.mean() for p in pols],
               width=0.4, color=PALETTE[0], label="primary benchmark")
        ax.bar(xs + 0.2, [sc[sc.policy == p].equal_opportunity_gap.mean() for p in pols],
               width=0.4, color=PALETTE[1], label="stress regime")
        ax.set_xticks(xs)
        ax.set_xticklabels(["unconstrained", "constrained"], fontsize=8)
        ax.set_ylabel("equal opportunity gap between zones")
        ax.legend(fontsize=7.5)
        ax.set_title("Equal opportunity gap under zone dependent\nvalidity", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_15_stress_regime")


if __name__ == "__main__":
    main()
