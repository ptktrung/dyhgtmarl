"""Figure 2: node types and relation types of the benchmark schema."""
from _common import PALETTE, PROC, need, save
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main():
    need(PROC / "node_metadata.csv", PROC / "relation_metadata.csv")
    nodes = pd.read_csv(PROC / "node_metadata.csv")
    rels = pd.read_csv(PROC / "relation_metadata.csv")
    counts = nodes.NodeType.value_counts().to_dict()
    pos = {"transit": (5.0, 3.4), "sensor_validator": (1.5, 3.4), "gateway": (1.5, 1.4),
           "clearing_agent": (5.0, 0.6), "maintenance_agent": (8.4, 3.4)}
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 4.6); ax.axis("off"); ax.grid(False)
    for i, (k, (x, y)) in enumerate(pos.items()):
        c = PALETTE[i % len(PALETTE)]
        ax.add_patch(FancyBboxPatch((x - 0.85, y - 0.34), 1.7, 0.68,
                                    boxstyle="round,pad=0.02", edgecolor=c, facecolor=c + "22"))
        ax.text(x, y, f"{k}\nn = {counts.get(k, 0)}", ha="center", va="center", fontsize=8)
    edges = [("sensor_validator", "transit", "r2 senses"),
             ("sensor_validator", "gateway", "r3 comm depends"),
             ("sensor_validator", "clearing_agent", "r5 provenance"),
             ("maintenance_agent", "transit", "r6 authority"),
             ("transit", "transit", "r1 travel, r7 similarity"),
             ("gateway", "transit", "r4 shared gateway")]
    for a, b, lab in edges:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        if a == b:
            ax.text(x1, y1 + 0.55, lab, ha="center", fontsize=7.2, color="#333333")
            continue
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                     linewidth=0.8, color="#666666",
                                     connectionstyle="arc3,rad=0.12"))
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.16, lab, ha="center", fontsize=7.2, color="#333333")
    txt = "  |  ".join(f"{r.RelationID.split('_')[0]}: {r.Temporal}/{r.Provenance}"
                       for r in rels.itertuples())
    ax.text(5.2, 0.05, txt, ha="center", fontsize=7)
    ax.set_title("Heterogeneous schema: five node types and seven relation types", fontsize=9.5)
    save(fig, "fig_02_heterogeneous_graph_schema")


if __name__ == "__main__":
    main()
