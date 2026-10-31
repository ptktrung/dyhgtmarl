"""Figure 6: reconstruction error by method and one reconstructed series."""
from _common import PALETTE, PROC, TAB, need, save
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    need(TAB / "exp_a_reconstruction.csv")
    d = pd.read_csv(TAB / "exp_a_reconstruction.csv")
    g = d.groupby("model").mae.mean().sort_values()
    has_example = (PROC / "reconstruction_example.npz").exists()
    fig, axes = plt.subplots(1, 2 if has_example else 1, figsize=(11.2 if has_example else 6, 4.0),
                             squeeze=False)
    ax = axes[0][0]
    cols = [PALETTE[1] if "dyhgt" in k else PALETTE[0] for k in g.index]
    ax.barh([k.replace("_", " ") for k in g.index], g.values, color=cols)
    ax.set_xlabel("reconstruction MAE, minutes")
    ax.set_title("Masked value reconstruction, lower is better")
    for i, v in enumerate(g.values):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=7.5)

    if has_example:
        z = np.load(PROC / "reconstruction_example.npz")
        node = int(np.argmax(z["mask"].sum(axis=0)))
        sl = slice(0, min(220, len(z["steps"])))
        ax = axes[0][1]
        ax.plot(z["steps"][sl], z["truth"][sl, node], color="#263238", linewidth=1.1, label="true")
        ax.plot(z["steps"][sl], z["recon"][sl, node], color=PALETTE[1], linewidth=1.0,
                label="DyHGT reconstruction")
        m = z["mask"][sl, node].astype(bool)
        ax.scatter(z["steps"][sl][m], z["truth"][sl][m, node], s=13, color=PALETTE[3],
                   zorder=3, label="masked cells")
        ax.set_xlabel("step"); ax.set_ylabel("travel time, minutes")
        ax.set_title(f"Node index {node}, test block"); ax.legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_06_reconstruction_example")


if __name__ == "__main__":
    main()
