"""Figure 4: snapshots of one cascade propagating over the travel graph."""
from _common import PALETTE, PROC, need, save
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx


def main():
    need(PROC / "fault_true.npy", PROC / "travel_adjacency.npy", PROC / "event_log.csv")
    fault = np.load(PROC / "fault_true.npy") > 0
    adj = np.load(PROC / "travel_adjacency.npy")
    ev = pd.read_csv(PROC / "event_log.csv")
    casc = ev[ev.EventType == "cascade"]
    if casc.empty:
        print("SKIPPED: no cascade events in the event log"); return
    spread = [(int(r.StartStep), int(r.RootIndex),
               fault[int(r.StartStep):int(r.StartStep) + 6].sum())
              for r in casc.itertuples()]
    t0, root, _ = max(spread, key=lambda z: z[2])

    g = nx.from_numpy_array(adj)
    pos = nx.spring_layout(g, seed=3, weight="weight")
    offsets = [0, 1, 3, 5]
    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.4))
    for ax, d in zip(axes, offsets):
        t = min(t0 + d, fault.shape[0] - 1)
        colors = [PALETTE[1] if fault[t, i] else "#cfd8dc" for i in range(adj.shape[0])]
        colors[root] = PALETTE[3]
        nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.25, width=0.7)
        nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors, node_size=95,
                               edgecolors="#37474f", linewidths=0.4)
        ax.set_title(f"step t0 + {d}\nfaulty nodes: {int(fault[t].sum())}", fontsize=8.5)
        ax.axis("off"); ax.grid(False)
    fig.suptitle(f"Cascade from root node index {root} at step {t0}. "
                 "Purple is the root, red is faulty.", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig_04_dynamic_cascade_snapshots")


if __name__ == "__main__":
    main()
