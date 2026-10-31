"""Figure 11: decision cost, workload and training cost of the components."""
from _common import PALETTE, TAB, need, save
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    need(TAB / "exp_c_policy.csv")
    c = pd.read_csv(TAB / "exp_c_policy.csv")
    g = c.groupby("policy").agg(nom=("nominal_latency_ms", "mean"),
                                wall=("measured_wall_ms_per_decision", "mean"),
                                review=("review_workload_pct", "mean"),
                                abst=("abstention_rate", "mean")).sort_values("nom")
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    xs = np.arange(len(g))
    ax = axes[0]
    ax.bar(xs - 0.2, g.nom, width=0.4, color=PALETTE[0], label="nominal model cost")
    ax.bar(xs + 0.2, g.wall, width=0.4, color=PALETTE[1], label="measured wall clock")
    ax.set_xticks(xs)
    ax.set_xticklabels([i.replace("_", " ") for i in g.index], fontsize=6.4,
                       rotation=35, ha="right")
    ax.set_ylabel("milliseconds per decision"); ax.legend(fontsize=7)
    ax.set_title("Decision cost. Nominal values are simulator assumptions.", fontsize=8.5)

    ax = axes[1]
    ax.bar(xs, g.review, color=PALETTE[2])
    ax.set_xticks(xs)
    ax.set_xticklabels([i.replace("_", " ") for i in g.index], fontsize=6.4,
                       rotation=35, ha="right")
    ax.set_ylabel("percent of transactions"); ax.set_title("Human review workload")

    ax = axes[2]
    frames = []
    for f, lab in [("exp_a_fault_detection.csv", "fault models"),
                   ("exp_b_forecast_conditions.csv", "forecast models")]:
        p = TAB / f
        if p.exists():
            d = pd.read_csv(p)
            if "train_seconds" in d:
                frames.append(d.groupby("model").train_seconds.mean().rename(lab))
    if frames:
        t = pd.concat(frames, axis=1).fillna(0).sum(axis=1).sort_values()
        ax.barh([i.replace("_", " ") for i in t.index], t.values, color=PALETTE[3])
        ax.set_xlabel("mean fit time per seed, seconds, single CPU core")
        ax.set_title("Training cost of each component")
    fig.tight_layout()
    save(fig, "fig_11_system_cost_latency")


if __name__ == "__main__":
    main()
