"""Figure 1: schematic of the DyHGT-MARL pipeline. Schematic, no numeric data."""
from _common import PALETTE, save
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, x, y, w, h, text, color, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                linewidth=1.1, edgecolor=color, facecolor=color + "22"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=0.9, color="#444444"))


def main():
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 5.4); ax.axis("off"); ax.grid(False)
    box(ax, 0.1, 3.9, 2.0, 1.0, "Sources\ntraffic matrix\nIoT telemetry\ntransactions", PALETTE[5])
    box(ax, 0.1, 2.5, 2.0, 1.0, "Harmonization\nnode mapping\ntime alignment", PALETTE[5])
    box(ax, 0.1, 1.1, 2.0, 1.0, "Scenario layer\ncascades, outages\nMNAR, closures", PALETTE[5])
    box(ax, 2.5, 2.3, 1.8, 2.2, "Dynamic\nheterogeneous\ngraph\nG_t = (V_t, E_t,\nR, X_t, A_t)", PALETTE[0])
    box(ax, 4.6, 3.5, 2.2, 1.0, "Relation aware\nattention with\nreliability term", PALETTE[0])
    box(ax, 4.6, 2.2, 2.2, 1.0, "Causal temporal\nTransformer\nblocks", PALETTE[0])
    for i, (lab, yy) in enumerate([("fault", 4.55), ("root cause", 3.95), ("reconstruction", 3.35),
                                   ("travel time MDN", 2.75), ("calibration", 2.15)]):
        box(ax, 7.2, yy - 0.26, 1.7, 0.52, lab, PALETTE[2], fs=7.5)
    box(ax, 4.6, 0.6, 4.3, 1.2,
        "Fairness constrained multi agent policy\naccept, reject, impute, corroborate,\ndefer, isolate, escalate", PALETTE[1])
    arrow(ax, (1.1, 3.9), (1.1, 3.5)); arrow(ax, (1.1, 2.5), (1.1, 2.1))
    arrow(ax, (2.1, 3.4), (2.5, 3.4)); arrow(ax, (4.3, 3.7), (4.6, 4.0))
    arrow(ax, (4.3, 3.0), (4.6, 2.7)); arrow(ax, (5.7, 3.5), (5.7, 3.2))
    for yy in (4.55, 3.95, 3.35, 2.75, 2.15):
        arrow(ax, (6.8, 3.0), (7.2, yy))
    arrow(ax, (8.0, 1.89), (8.0, 1.8)); arrow(ax, (5.7, 2.2), (5.7, 1.8))
    ax.set_title("DyHGT-MARL: data harmonization, dynamic heterogeneous encoder, "
                 "multi-task heads, constrained policy", fontsize=9.5)
    save(fig, "fig_01_dyhgt_architecture")


if __name__ == "__main__":
    main()
