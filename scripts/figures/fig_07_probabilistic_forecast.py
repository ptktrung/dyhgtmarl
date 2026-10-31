"""Figure 7: probabilistic travel time forecast with 95 percent bounds."""
from _common import PALETTE, PROC, need, save
import matplotlib.pyplot as plt
import numpy as np


def main():
    need(PROC / "forecast_example.npz")
    z = np.load(PROC / "forecast_example.npz")
    err = np.abs(z["y"] - z["mean"]).mean(axis=0)
    node = int(np.argmax(z["y"].std(axis=0)))
    sl = slice(0, min(200, len(z["steps"])))
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.0), sharex=True)
    ax = axes[0]
    ax.fill_between(z["steps"][sl], z["lo"][sl, node], z["hi"][sl, node],
                    color=PALETTE[0], alpha=0.20, label="95 percent interval")
    ax.plot(z["steps"][sl], z["y"][sl, node], color="#263238", linewidth=1.1, label="true")
    ax.plot(z["steps"][sl], z["mean"][sl, node], color=PALETTE[1], linewidth=1.0,
            label="mixture mean")
    ax.set_ylabel("travel time, minutes"); ax.legend(fontsize=7.5, ncol=3)
    ax.set_title(f"Most volatile test node, index {node}")
    ax = axes[1]
    width = z["hi"][sl] - z["lo"][sl]
    ax.plot(z["steps"][sl], width.mean(axis=1), color=PALETTE[2], linewidth=1.0,
            label="mean interval width across nodes")
    ax.plot(z["steps"][sl], np.abs(z["y"] - z["mean"])[sl].mean(axis=1), color=PALETTE[3],
            linewidth=1.0, label="mean absolute error across nodes")
    ax.set_xlabel("step"); ax.set_ylabel("minutes"); ax.legend(fontsize=7.5)
    ax.set_title("Interval width tracks error over the test block")
    fig.tight_layout()
    save(fig, "fig_07_probabilistic_forecast")


if __name__ == "__main__":
    main()
