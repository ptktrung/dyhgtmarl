"""Stress regime: the benchmark revision that Section VIII asks for.

The primary benchmark has three properties that stop the proposed mechanisms
from being tested in the regime they were designed for. Faults are injected by
clamping a single telemetry column, so detection needs no structure. Node
series are correlated at 0.909, so spatial edges carry little independent
information. Latent transaction validity is independent of zone, so the
fairness constraint never binds.

This module builds a second corpus in which all three are repaired, while
holding the topology, the split protocol and the evaluation code fixed, so the
contrast isolates the generator rather than the method.

1. Spatial decorrelation. The shared temporal factor is partly removed from the
   travel time matrix and replaced with node specific variation, lowering the
   mean pairwise correlation into the target band.
2. Relational faults. A faulty node's telemetry stays inside its normal
   operating range. The fault shows up as the node's travel time drifting away
   from what its graph neighbours imply, which is a disagreement no single
   column reveals.
3. Zone dependent validity. Latent validity and the cost of a false accept
   differ between core and peripheral zones, so a utility maximizing policy
   genuinely prefers to treat the zones differently and the fairness budget has
   something to bind against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .graph import normalized_adjacency


def decorrelate(travel: np.ndarray, adjacency: np.ndarray, target: float = 0.55,
                seed: int = 42) -> tuple[np.ndarray, float, float]:
    """Shrink the shared temporal factor until pairwise correlation hits target.

    The first principal component of the node series carries the city wide
    demand cycle. Scaling it down while adding node specific autoregressive
    variation preserves the diurnal shape of each series and the overall level,
    and leaves the spatial structure to carry information it did not carry
    before.
    """
    rng = np.random.default_rng(seed)
    T, N = travel.shape
    mu = travel.mean(axis=0, keepdims=True)
    X = travel - mu
    before = float(np.corrcoef(travel.T)[np.triu_indices(N, 1)].mean())

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    shared = np.outer(U[:, 0] * S[0], Vt[0])
    rest = X - shared

    # node specific autoregressive component, smooth enough to stay physical
    innov = rng.normal(0.0, 1.0, size=(T, N))
    local = np.zeros_like(innov)
    for t in range(1, T):
        local[t] = 0.80 * local[t - 1] + innov[t]
    local *= (X.std(axis=0, keepdims=True) / np.maximum(local.std(axis=0, keepdims=True), 1e-9))

    lo, hi = 0.0, 1.0
    for _ in range(40):
        a = 0.5 * (lo + hi)
        cand = mu + a * shared + rest + (1.0 - a) * 0.9 * local
        c = float(np.corrcoef(cand.T)[np.triu_indices(N, 1)].mean())
        if c > target:
            hi = a
        else:
            lo = a
    out = np.clip(mu + a * shared + rest + (1.0 - a) * 0.9 * local, 1.0, None)
    after = float(np.corrcoef(out.T)[np.triu_indices(N, 1)].mean())
    return out, before, after


def relational_faults(travel: np.ndarray, fault: np.ndarray, adjacency: np.ndarray,
                      signal: np.ndarray, latency: np.ndarray, drift: np.ndarray,
                      cfg: dict, seed: int = 42):
    """Replace telemetry clamping with neighbourhood disagreement.

    A faulty node keeps a plausible signal, between minus 75 and minus 95 dBm,
    and a plausible latency. What changes is that its reported travel time
    departs from the value its graph neighbours imply, by a multiplicative
    factor drawn per event. Detecting it requires comparing the node with its
    neighbourhood, which is the hypothesis the architecture was built for.
    """
    rng = np.random.default_rng(seed + 991)
    st = cfg["stress"]
    prop = normalized_adjacency(adjacency)
    T, N = travel.shape
    faulty = fault > 0

    # telemetry stays in the normal operating band for every node
    sig = np.clip(rng.normal(-85.0, 4.5, size=signal.shape), -95.0, -75.0)
    lat = np.clip(rng.normal(latency.mean(), latency.std() * 0.35, size=latency.shape),
                  8.0, 120.0)
    dri = np.zeros_like(drift)

    neighbour_implied = travel @ prop.T
    bias = np.zeros((T, N))
    sign = rng.choice([-1.0, 1.0], size=(T, N))
    mag = rng.uniform(*st["relational_bias_range"], size=(T, N))
    bias[faulty] = (sign * mag)[faulty]
    observed = travel.copy()
    observed[faulty] = (travel[faulty]
                        + bias[faulty] * np.abs(neighbour_implied - travel)[faulty]
                        + bias[faulty] * st["relational_offset_min"] * travel[faulty])
    observed = np.clip(observed, 0.5, None)

    # a weak, deliberately insufficient telemetry trace, so a single column
    # model retains some signal but cannot solve the task
    leak = rng.random((T, N)) < st["telemetry_leak_prob"]
    sig[faulty & leak] -= rng.uniform(4.0, 9.0, size=int((faulty & leak).sum()))
    return observed, sig, lat, dri


def zone_dependent_validity(n: int, zone: np.ndarray, node_fault: np.ndarray,
                            cfg: dict, rng) -> tuple[np.ndarray, np.ndarray]:
    """Latent validity and false accept cost both depend on zone.

    In the primary benchmark validity is conditionally independent of zone, so
    utility maximization and equity coincide and the constraint never binds.
    Here peripheral records are genuinely less often valid and a peripheral
    false accept is cheaper to the operator, which is exactly the configuration
    in which an unconstrained policy prefers to under-serve one zone.
    """
    st = cfg["stress"]
    peri = zone == "Periphery"
    base = np.where(peri, st["valid_rate_periphery"], st["valid_rate_core"])
    p_valid = np.where(node_fault, base - st["fault_validity_drop"], base)
    true_valid = rng.random(n) < np.clip(p_valid, 0.01, 0.999)
    fa_mult = np.where(peri, st["false_accept_mult_periphery"],
                       st["false_accept_mult_core"])
    return true_valid, fa_mult


def summarize(travel_before: np.ndarray, travel_after: np.ndarray,
              corr_before: float, corr_after: float, tx: pd.DataFrame) -> dict:
    return {
        "mean_pairwise_correlation_before": round(corr_before, 4),
        "mean_pairwise_correlation_after": round(corr_after, 4),
        "lag1_autocorr_after": round(float(np.mean(
            [pd.Series(travel_after[:, i]).autocorr(1) for i in range(travel_after.shape[1])])), 4),
        "validity_rate_core": round(float(tx.loc[tx.Zone == "Core", "TrueValid"].mean()), 4),
        "validity_rate_periphery": round(float(tx.loc[tx.Zone == "Periphery", "TrueValid"].mean()), 4),
        "validity_gap": round(float(tx.loc[tx.Zone == "Core", "TrueValid"].mean()
                                    - tx.loc[tx.Zone == "Periphery", "TrueValid"].mean()), 4),
    }
