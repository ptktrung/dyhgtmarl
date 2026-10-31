"""Figure 5: fault detection performance across models, mean over seeds."""
from _common import PALETTE, TAB, need, save
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    need(TAB / "exp_a_fault_detection.csv")
    d = pd.read_csv(TAB / "exp_a_fault_detection.csv")
    g = d.groupby("model").agg(["mean", "std"])
    order = g[("f1", "mean")].sort_values().index.tolist()
    metrics = ["precision", "recall", "f1", "pr_auc"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    ax = axes[0]
    w = 0.2
    xs = np.arange(len(order))
    for i, m in enumerate(metrics):
        ax.bar(xs + (i - 1.5) * w, g.loc[order, (m, "mean")],
               yerr=g.loc[order, (m, "std")].fillna(0), width=w, label=m,
               color=PALETTE[i], capsize=2)
    ax.set_xticks(xs); ax.set_xticklabels([o.replace("_", "\n") for o in order], fontsize=7.5)
    ax.set_ylabel("score"); ax.legend(ncol=4, fontsize=7.5)
    ax.set_title("Detection scores, mean and standard deviation over seeds")

    ax = axes[1]
    ax.scatter(g.loc[order, ("false_alarm_rate", "mean")], g.loc[order, ("recall", "mean")],
               s=46, color=PALETTE[0])
    # labels are placed alternately above and below and nudged apart, because
    # several models sit at almost the same operating point
    pts = sorted(order, key=lambda o: g.loc[o, ("false_alarm_rate", "mean")])
    for k, o in enumerate(pts):
        dx, dy = (6, 6) if k % 2 == 0 else (6, -11)
        ax.annotate(o.replace("_", " "),
                    (g.loc[o, ("false_alarm_rate", "mean")], g.loc[o, ("recall", "mean")]),
                    fontsize=6.4, xytext=(dx, dy), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.set_xlabel("false alarm rate on nominal node steps"); ax.set_ylabel("fault recall")
    ax.set_title("Operating trade off")
    fig.tight_layout()
    save(fig, "fig_05_fault_detection_performance")


if __name__ == "__main__":
    main()
