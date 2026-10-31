"""Node identity, transport topology, and the dynamic heterogeneous graph.

The travel topology reproduces the ring plus chord construction used by the
supplied generator Generate_hcmc_traffic_data.py so that the integrated
benchmark stays traceable to the original source material. Every other
relation type introduced here is a documented synthetic harmonization
assumption and is labelled as such in relation_metadata().
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_NODES = 28
N_GATEWAYS = 6
N_CLEARING = 4
N_MAINT = 3

# Node mapping fixed by the benchmark specification. It is a transparent
# harmonization assumption between two independently generated synthetic
# files. It is not a map of physical HCMC infrastructure.
MAPPING_BLOCKS = [
    (0, 9, "BusNode_{:02d}", 1, "Bus", "Core"),
    (10, 14, "MetroNode_{:02d}", 1, "Metro", "Core"),
    (15, 24, "BusNode_{:02d}", 11, "Bus", "Periphery"),
    (25, 27, "FerryNode_{:02d}", 1, "Ferry", "Periphery"),
]


def build_node_mapping() -> pd.DataFrame:
    rows = []
    for lo, hi, template, start, mode, zone in MAPPING_BLOCKS:
        for offset, idx in enumerate(range(lo, hi + 1)):
            rows.append(
                {
                    "UnifiedNodeID": f"U{idx:02d}",
                    "TrafficNodeID": f"Node_{idx}",
                    "IoTNodeID": template.format(start + offset),
                    "NodeType": "transit",
                    "Mode": mode,
                    "Zone": zone,
                }
            )
    df = pd.DataFrame(rows)
    df["GatewayID"] = [f"GW_{i % N_GATEWAYS:02d}" for i in range(len(df))]
    return df


def build_topology(n_nodes: int = N_NODES, seed: int = 42) -> np.ndarray:
    """Ring plus stochastic chords, identical in form to the supplied script."""
    rng = np.random.default_rng(seed)
    a = np.zeros((n_nodes, n_nodes), dtype=float)
    for i in range(n_nodes):
        a[i, (i + 1) % n_nodes] = 1.0
        a[(i + 1) % n_nodes, i] = 1.0
    for i in range(n_nodes):
        for offset in (2, 5, 9):
            j = (i + offset) % n_nodes
            if rng.random() < 0.58:
                w = rng.uniform(0.35, 0.90)
                a[i, j] = max(a[i, j], w)
                a[j, i] = max(a[j, i], w)
    np.fill_diagonal(a, 0.0)
    return a


def normalized_adjacency(a: np.ndarray) -> np.ndarray:
    a_hat = a + np.eye(a.shape[0])
    deg = a_hat.sum(axis=1)
    d = np.diag(1.0 / np.sqrt(np.clip(deg, 1e-12, None)))
    return d @ a_hat @ d


def laplacian_pe(a: np.ndarray, k: int = 8) -> np.ndarray:
    """Normalized-Laplacian eigenvector positional encoding, sign fixed.

    Eigenvector sign is resolved deterministically by forcing the entry of
    largest magnitude to be positive. This is a pragmatic substitute for a
    sign-invariant network and is reported as such.
    """
    n = a.shape[0]
    a_sym = np.maximum(a, a.T)
    deg = np.clip(a_sym.sum(axis=1), 1e-12, None)
    d_inv = np.diag(1.0 / np.sqrt(deg))
    lap = np.eye(n) - d_inv @ a_sym @ d_inv
    vals, vecs = np.linalg.eigh(lap)
    order = np.argsort(vals)
    vecs = vecs[:, order][:, 1 : k + 1]
    for j in range(vecs.shape[1]):
        col = vecs[:, j]
        if col[np.argmax(np.abs(col))] < 0:
            vecs[:, j] = -col
    return vecs


def eigen_centrality(a: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(np.maximum(a, a.T))
    v = np.abs(vecs[:, int(np.argmax(vals))])
    return v / (v.sum() + 1e-12)


def centrality_strata(a: np.ndarray) -> dict:
    c = eigen_centrality(a)
    order = np.argsort(c)
    third = len(order) // 3
    return {
        "low": order[:third].tolist(),
        "medium": order[third : 2 * third].tolist(),
        "high": order[2 * third :].tolist(),
    }


def closure_edges(a: np.ndarray, n_edges: int = 6, seed: int = 7):
    """Edges removed during the topology-change scenario window."""
    rng = np.random.default_rng(seed)
    idx = np.argwhere(np.triu(a, 1) > 0)
    pick = rng.choice(len(idx), size=min(n_edges, len(idx)), replace=False)
    return [tuple(int(x) for x in idx[p]) for p in pick]


def node_metadata() -> pd.DataFrame:
    """All node types of the heterogeneous graph, not only transit nodes."""
    mapping = build_node_mapping()
    rows = [
        {
            "UnifiedNodeID": r.UnifiedNodeID,
            "NodeType": "transit",
            "Mode": r.Mode,
            "Zone": r.Zone,
            "GatewayID": r.GatewayID,
            "ParentNodeID": "",
        }
        for r in mapping.itertuples()
    ]
    for r in mapping.itertuples():
        rows.append(
            {
                "UnifiedNodeID": f"S{r.UnifiedNodeID[1:]}",
                "NodeType": "sensor_validator",
                "Mode": r.Mode,
                "Zone": r.Zone,
                "GatewayID": r.GatewayID,
                "ParentNodeID": r.UnifiedNodeID,
            }
        )
    for g in range(N_GATEWAYS):
        served = mapping[mapping["GatewayID"] == f"GW_{g:02d}"]
        zone = "Core" if (served["Zone"] == "Core").mean() >= 0.5 else "Periphery"
        rows.append(
            {
                "UnifiedNodeID": f"GW_{g:02d}",
                "NodeType": "gateway",
                "Mode": "Mixed",
                "Zone": zone,
                "GatewayID": f"GW_{g:02d}",
                "ParentNodeID": "",
            }
        )
    for c in range(N_CLEARING):
        rows.append(
            {
                "UnifiedNodeID": f"CA_{c:02d}",
                "NodeType": "clearing_agent",
                "Mode": "Mixed",
                "Zone": "Core" if c < 2 else "Periphery",
                "GatewayID": "",
                "ParentNodeID": "",
            }
        )
    for m in range(N_MAINT):
        rows.append(
            {
                "UnifiedNodeID": f"MA_{m:02d}",
                "NodeType": "maintenance_agent",
                "Mode": "Mixed",
                "Zone": "Core" if m == 0 else "Periphery",
                "GatewayID": "",
                "ParentNodeID": "",
            }
        )
    return pd.DataFrame(rows)


def relation_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("r1_travel_connects", "transit", "transit", "directed",
             "dynamic", "simulated",
             "Ring plus chord topology from the supplied traffic generator. Edges are removed during the closure window."),
            ("r2_senses", "sensor_validator", "transit", "directed",
             "static", "simulated",
             "One validator per transit node, fixed by the node mapping assumption."),
            ("r3_comm_depends", "sensor_validator", "gateway", "directed",
             "dynamic", "simulated",
             "Backhaul dependency. The edge is suppressed while the gateway is in an outage state."),
            ("r4_shares_gateway", "sensor_validator", "sensor_validator", "undirected",
             "static", "inferred",
             "Derived from the gateway assignment, encodes correlated outage risk."),
            ("r5_provenance", "sensor_validator", "clearing_agent", "directed",
             "static", "simulated",
             "Each record carries a ProvenanceID naming the validator and the clearing agent that settled it."),
            ("r6_operational_authority", "maintenance_agent", "transit", "directed",
             "static", "simulated",
             "Which maintenance agent may isolate or repair a node."),
            ("r7_functional_similarity", "transit", "transit", "undirected",
             "dynamic", "inferred",
             "Top-k correlation of observed travel time on the training split only, recomputed per split to avoid leakage."),
        ],
        columns=[
            "RelationID", "SourceType", "TargetType", "Direction",
            "Temporal", "Provenance", "Definition",
        ],
    )
