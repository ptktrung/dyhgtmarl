"""Tensor assembly for the integrated benchmark.

The model node set is the 28 transit nodes followed by the 6 gateway nodes.
Gateway features are the mean of the features of the sensors they serve, which
is how a real backhaul aggregate would look to a monitoring system.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .graph import build_node_mapping, build_topology, closure_edges, laplacian_pe, normalized_adjacency
from .utils import PROC

N_TRANSIT = 28
N_GW = 6
N_MODEL = N_TRANSIT + N_GW
RELATIONS = ["r1_travel_connects", "r3_comm_depends", "r4_shares_gateway", "r7_functional_similarity"]


def load_arrays():
    a = {k: np.load(PROC / f"{k}.npy") for k in
         ["travel_true", "travel_observed", "fault_true", "root_id", "event_id",
          "drift", "signal", "latency", "missing_mask", "missing_mechanism"]}
    a["mapping"] = build_node_mapping()
    a["splits"] = pd.read_csv(PROC / "splits.csv")
    a["events"] = pd.read_csv(PROC / "event_log.csv")
    return a


def split_bounds(splits: pd.DataFrame):
    tr = int((splits.Split == "train").sum())
    va = tr + int((splits.Split == "val").sum())
    return tr, va, len(splits)


def build_relation_adjacency(travel_observed: np.ndarray, train_end: int, cfg: dict):
    """Return the base stack and the closure stack, both (n_rel, 34, 34).

    Functional similarity is estimated only on the training split so that no
    test information enters the graph.
    """
    mapping = build_node_mapping()
    adj_t = build_topology(N_TRANSIT, cfg["graph"]["topology_seed"])
    gw_ids = sorted(mapping.GatewayID.unique())

    def empty():
        return np.zeros((len(RELATIONS), N_MODEL, N_MODEL))

    base = empty()
    base[0, :N_TRANSIT, :N_TRANSIT] = normalized_adjacency(adj_t)
    for i, g in enumerate(mapping.GatewayID.values):
        gi = N_TRANSIT + gw_ids.index(g)
        base[1, i, gi] = 1.0
        base[1, gi, i] = 1.0
    for g in gw_ids:
        idx = np.where(mapping.GatewayID.values == g)[0]
        for a_ in idx:
            for b_ in idx:
                if a_ != b_:
                    base[2, a_, b_] = 1.0
    series = np.nan_to_num(travel_observed[:train_end], nan=np.nanmean(travel_observed))
    corr = np.corrcoef(series.T)
    np.fill_diagonal(corr, -np.inf)
    k = cfg["graph"]["functional_similarity_k"]
    for i in range(N_TRANSIT):
        for j in np.argsort(-corr[i])[:k]:
            base[3, i, j] = float(max(corr[i, j], 0.0))
            base[3, j, i] = float(max(corr[i, j], 0.0))
    for r in range(len(RELATIONS)):
        np.fill_diagonal(base[r], np.maximum(np.diag(base[r]), 1e-6))

    stacks = [base]
    windows = [tuple(w) for w in cfg["scenarios"]["closure_windows"]]
    for wi, _ in enumerate(windows):
        c = base.copy()
        for (u, v) in closure_edges(adj_t, cfg["scenarios"]["n_closed_edges"], seed=7 + wi):
            c[0, u, v] = 0.0
            c[0, v, u] = 0.0
        stacks.append(c)
    return np.stack(stacks), windows


def adj_id_for_steps(steps: np.ndarray, windows) -> np.ndarray:
    """Index 0 is the base topology, index w + 1 is closure window w."""
    out = np.zeros(len(steps), dtype=np.int64)
    for wi, (lo, hi) in enumerate(windows):
        out[(steps >= lo) & (steps < hi)] = wi + 1
    return out


def pe_matrix(cfg: dict) -> np.ndarray:
    adj_t = build_topology(N_TRANSIT, cfg["graph"]["topology_seed"])
    pe_t = laplacian_pe(adj_t, cfg["graph"]["laplacian_k"])
    pe = np.zeros((N_MODEL, pe_t.shape[1] + 1))
    pe[:N_TRANSIT, :-1] = pe_t
    pe[N_TRANSIT:, -1] = 1.0
    return pe


def node_type_index() -> np.ndarray:
    return np.concatenate([np.zeros(N_TRANSIT, dtype=np.int64), np.ones(N_GW, dtype=np.int64)])


def build_features(arrays: dict, extra_mask: np.ndarray | None = None):
    """Observation level features. No ground truth label is used as an input."""
    obs = arrays["travel_observed"].copy()
    miss = arrays["missing_mask"].copy()
    if extra_mask is not None:
        obs = obs.copy()
        obs[extra_mask] = np.nan
        miss = miss | extra_mask
    nan = np.isnan(obs)
    ref = np.nanmean(arrays["travel_observed"][: int(0.7 * len(obs))])
    filled = np.where(nan, ref, obs)
    T, N = obs.shape
    step = np.arange(T)
    hour = (step % 96) / 96.0 * 2 * np.pi
    feats = [
        (filled - ref) / 10.0,
        nan.astype(float),
        (arrays["signal"] + 70.0) / 20.0,
        np.log1p(arrays["latency"]) / 5.0,
        arrays["drift"] / 5.0,
        np.repeat(np.sin(hour)[:, None], N, axis=1),
        np.repeat(np.cos(hour)[:, None], N, axis=1),
        np.repeat(((step % 672) // 96 >= 5).astype(float)[:, None], N, axis=1),
    ]
    x = np.stack(feats, axis=-1).astype(np.float32)
    rel = np.clip(0.45 * np.clip((arrays["signal"] + 100.0) / 40.0, 0, 1)
                  + 0.35 * np.clip(1.0 - (arrays["latency"] - 5.0) / 300.0, 0, 1)
                  + 0.20 * np.clip(1.0 - arrays["drift"] / 12.0, 0, 1), 0, 1).astype(np.float32)
    return x, rel, filled.astype(np.float32), nan


def to_model_nodes(x: np.ndarray, mapping: pd.DataFrame) -> np.ndarray:
    """Append gateway nodes carrying the mean feature of their served sensors."""
    gw_ids = sorted(mapping.GatewayID.unique())
    T, _, F = x.shape
    out = np.zeros((T, N_MODEL, F), dtype=np.float32)
    out[:, :N_TRANSIT] = x
    for i, g in enumerate(gw_ids):
        idx = np.where(mapping.GatewayID.values == g)[0]
        out[:, N_TRANSIT + i] = x[:, idx].mean(axis=1)
    return out


def make_windows(T: int, window: int, horizon: int, lo: int, hi: int):
    """Window end indices whose target step lies inside [lo, hi)."""
    starts = np.arange(max(lo - window - horizon + 1, 0), hi - window - horizon + 1)
    starts = starts[(starts + window + horizon - 1 >= lo) & (starts + window + horizon - 1 < hi)]
    return starts
