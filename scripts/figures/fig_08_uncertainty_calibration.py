"""Figure 8: calibration of the predictive distribution."""
from _common import PALETTE, PROC, TAB, need, save
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    need(PROC / "forecast_example.npz")
    z = np.load(PROC / "forecast_example.npz")
    pit = z["pit"].ravel()
    levels = np.linspace(0.02, 0.98, 33)
    emp = np.array([(pit <= q).mean() for q in levels])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color="#90a4ae", linewidth=1, label="perfect calibration")
    ax.plot(levels, emp, color=PALETTE[0], linewidth=1.3, marker="o", markersize=2.6,
            label="DyHGT-MARL mixture")
    ax.set_xlabel("nominal quantile level"); ax.set_ylabel("empirical coverage")
    ax.set_title("Reliability diagram, pooled over all four test conditions\n"
                 f"including distribution shift, ECE = {np.mean(np.abs(emp - levels)):.3f}",
                 fontsize=8.5)
    ax.legend(fontsize=7.5)
    ax = axes[1]
    ax.hist(pit, bins=24, color=PALETTE[2], alpha=0.85, density=True)
    ax.axhline(1.0, color="#90a4ae", linestyle="--", linewidth=1)
    ax.set_xlabel("probability integral transform value"); ax.set_ylabel("density")
    ax.set_title("A calibrated forecast gives a flat histogram")
    if (TAB / "exp_b_forecast_conditions.csv").exists():
        d = pd.read_csv(TAB / "exp_b_forecast_conditions.csv")
        sub = d[d.model == "dyhgt"].groupby("condition")[["picp95", "ece"]].mean()
        txt = "  ".join(f"{i}: PICP {r.picp95:.3f}, ECE {r.ece:.3f}" for i, r in sub.iterrows())
        fig.text(0.5, -0.03, txt, ha="center", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_08_uncertainty_calibration")


if __name__ == "__main__":
    main()
