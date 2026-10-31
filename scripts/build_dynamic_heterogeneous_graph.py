"""Build the time indexed heterogeneous graph G_t = (V_t, E_t, R, X_t, A_t).

Static relations are written once as edge lists. Dynamic relations are written
as a snapshot index that records, for every change point, which edges are
suppressed. Snapshots are stored sparsely because only a small number of steps
differ from the base topology.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from dyhgt.graph import (build_node_mapping, build_topology, closure_edges,
                         laplacian_pe, eigen_centrality, node_metadata, relation_metadata)
from dyhgt.utils import PROC, ensure_dirs, load_config, save_json

def main():
    cfg = load_config(); ensure_dirs()
    mapping = build_node_mapping()
    nodes = node_metadata()
    adj = build_topology(cfg["grid"]["n_nodes"], cfg["graph"]["topology_seed"])
    pd.DataFrame(adj, index=mapping.UnifiedNodeID, columns=mapping.UnifiedNodeID).to_csv(
        PROC / "travel_adjacency.csv")
    np.save(PROC / "travel_adjacency.npy", adj)
    np.save(PROC / "laplacian_pe.npy", laplacian_pe(adj, cfg["graph"]["laplacian_k"]))

    edges = []
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j] > 0:
                edges.append(("r1_travel_connects", mapping.UnifiedNodeID[i],
                              mapping.UnifiedNodeID[j], float(adj[i, j])))
    for r in mapping.itertuples():
        sid = f"S{r.UnifiedNodeID[1:]}"
        edges.append(("r2_senses", sid, r.UnifiedNodeID, 1.0))
        edges.append(("r3_comm_depends", sid, r.GatewayID, 1.0))
        edges.append(("r5_provenance", sid,
                      f"CA_{hash(r.Zone) % 2 + (0 if r.Zone == 'Core' else 2):02d}", 1.0))
    for g in sorted(mapping.GatewayID.unique()):
        served = [f"S{u[1:]}" for u in mapping.loc[mapping.GatewayID == g, "UnifiedNodeID"]]
        for a_ in range(len(served)):
            for b_ in range(a_ + 1, len(served)):
                edges.append(("r4_shares_gateway", served[a_], served[b_], 1.0))
    maint = nodes[nodes.NodeType == "maintenance_agent"].UnifiedNodeID.tolist()
    for i, r in enumerate(mapping.itertuples()):
        edges.append(("r6_operational_authority", maint[i % len(maint)], r.UnifiedNodeID, 1.0))
    pd.DataFrame(edges, columns=["RelationID", "Source", "Target", "Weight"]).to_csv(
        PROC / "edge_list_static.csv", index=False)

    T = cfg["grid"]["hours"] * cfg["grid"]["steps_per_hour"]
    windows = [tuple(w) for w in cfg["scenarios"]["closure_windows"]]
    snaps, cursor = [], 0
    for wi, (lo, hi) in enumerate(windows):
        closed = closure_edges(adj, cfg["scenarios"]["n_closed_edges"], seed=7 + wi)
        snaps.append({"StartStep": cursor, "EndStep": lo, "SuppressedEdges": ""})
        snaps.append({"StartStep": lo, "EndStep": hi,
                      "SuppressedEdges": ";".join(f"{mapping.UnifiedNodeID[a]}-{mapping.UnifiedNodeID[b]}"
                                                  for a, b in closed)})
        cursor = hi
    snaps.append({"StartStep": cursor, "EndStep": T, "SuppressedEdges": ""})
    pd.DataFrame(snaps).to_csv(PROC / "graph_snapshots.csv", index=False)
    relation_metadata().to_csv(PROC / "relation_metadata.csv", index=False)
    save_json({"n_transit": int(len(mapping)), "n_nodes_total": int(len(nodes)),
               "n_travel_edges_directed": int((adj > 0).sum()),
               "n_static_edges_all_relations": int(len(edges)),
               "eigen_centrality": {mapping.UnifiedNodeID[i]: float(c)
                                    for i, c in enumerate(eigen_centrality(adj))}},
              PROC / "graph_summary.json")
    print("graph built:", len(nodes), "nodes,", len(edges), "static edges,",
          len(snaps), "snapshot intervals")

if __name__ == "__main__":
    main()
