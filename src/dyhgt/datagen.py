"""Generators for the integrated DyHGT-MARL benchmark.

Three source-style generators are provided so that every raw file used by the
study can be rebuilt from a seed. A fourth set of routines adds the scenario
layers that the source generators do not provide: true graph-neighbour cascade
propagation, gateway-correlated outages, MNAR peripheral failures, edge
closures, and a distribution-shift window.

All quantities are synthetic. Nothing here is measured municipal data.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .graph import (
    build_node_mapping,
    build_topology,
    centrality_strata,
    closure_edges,
    normalized_adjacency,
)

STEPS_PER_HOUR = 4
STEP_MINUTES = 60 // STEPS_PER_HOUR
START = datetime(2025, 1, 1, 0, 0, 0)

FAULT_TYPES = ["None", "Signal_Loss", "High_Latency", "Sensor_Drift", "Gateway_Outage"]


# ----------------------------------------------------------------------------
# Source-style generators (reproductions of the supplied scripts, seeded)
# ----------------------------------------------------------------------------
def generate_traffic_matrix(hours: int = 720, n_nodes: int = 28, seed: int = 42):
    """Reproduction of the supplied Generate_hcmc_traffic_data.py process.

    The supplied hcmc_traffic_data.csv was not produced by this exact script
    (its hourly profile and its 13 negative entries do not match the script),
    so this routine is provided for reproducibility of the pipeline and not as
    a claim that it recovers the supplied file.
    """
    rng = np.random.default_rng(seed)
    adjacency = build_topology(n_nodes, seed)
    prop = normalized_adjacency(adjacency)
    t = np.arange(hours)
    hour = t % 24
    day = t // 24
    morning = np.exp(-0.5 * ((hour - 8.0) / 1.55) ** 2)
    evening = np.exp(-0.5 * ((hour - 17.3) / 1.85) ** 2)
    daily = 5.8 * morning + 7.4 * evening
    weekly = 1.2 * np.sin(2 * np.pi * day / 7.0)
    weekend = np.where(day % 7 >= 5, -2.2, 0.0)
    base = rng.uniform(17.0, 27.0, n_nodes)
    vol = rng.uniform(0.7, 2.2, n_nodes)
    susc = rng.uniform(0.45, 1.25, n_nodes)
    trend = rng.normal(0.0, 0.008, n_nodes)
    traffic = np.zeros((hours, n_nodes))
    latent = rng.normal(0.0, 0.4, n_nodes)
    events = []
    for s in range(hours):
        peak = morning[s] + evening[s]
        noise = rng.normal(0.0, vol * (1.0 + 0.75 * peak), n_nodes)
        shock = rng.normal(0.0, 0.65 + 1.35 * peak)
        spike = np.zeros(n_nodes)
        if peak > 0.35 and rng.random() < 0.22:
            root = int(rng.integers(0, n_nodes))
            amp = rng.exponential(5.0) + 4.0
            spike = amp * np.linalg.matrix_power(prop, int(rng.integers(1, 4)))[root] * susc
            events.append({"hour": s, "event_type": "peak_hour_bottleneck",
                           "root_node": f"Node_{root}", "amplitude_minutes": round(float(amp), 3)})
        latent = 0.62 * latent + 0.28 * (prop @ latent) + noise + spike
        traffic[s] = np.clip(base + daily[s] + weekly[s] + weekend[s] + shock + latent + trend * s, 4.0, None)
    cols = [f"Node_{i}" for i in range(n_nodes)]
    return pd.DataFrame(traffic, columns=cols), adjacency, pd.DataFrame(events)


def generate_iot_source_style(num_rows: int = 100000, seed: int = 42) -> pd.DataFrame:
    """Reproduction of the supplied IoT generator, with its row-order cascade.

    This is kept deliberately faithful, including the limitation that the
    cascade state is carried by the previous row rather than by a graph
    neighbour. It is used only as a comparison point for the integrated
    benchmark, which replaces that mechanism.
    """
    rng = np.random.default_rng(seed)
    mapping = build_node_mapping()
    core = mapping[mapping.Zone == "Core"].IoTNodeID.tolist()
    peri = mapping[mapping.Zone == "Periphery"].IoTNodeID.tolist()
    rows = []
    now = START + timedelta(hours=6)
    prev_fault = False
    for i in range(num_rows):
        now = now + timedelta(seconds=int(rng.integers(1, 16)))
        zone = "Core" if rng.random() < 0.6 else "Periphery"
        node = core[int(rng.integers(len(core)))] if zone == "Core" else peri[int(rng.integers(len(peri)))]
        mode = "Metro" if "Metro" in node else ("Ferry" if "Ferry" in node else "Bus")
        signal = (-60.0 if zone == "Core" else -85.0) + rng.normal(0, 5)
        latency = max(5.0, rng.normal(20.0 if mode == "Metro" else 45.0, 10.0))
        drift = 0.0
        p = 0.02 + (0.15 if signal < -90 else 0.0) + (0.20 if prev_fault else 0.0)
        is_err, ftype = False, "None"
        if rng.random() < p:
            is_err, prev_fault = True, True
            r = rng.random()
            if r < 1 / 3:
                ftype, signal = "Signal_Loss", -120.0
            elif r < 2 / 3:
                ftype = "High_Latency"
                latency += float(rng.integers(200, 2001))
            else:
                ftype = "Sensor_Drift"
                drift = float(rng.uniform(5.0, 50.0))
        else:
            prev_fault = False
        fare = 12000 if mode == "Metro" else (15000 if mode == "Ferry" else 7000)
        rows.append([f"TRX-{100000 + i}", now, node, zone, mode, round(signal, 2),
                     round(latency, 2), round(drift, 4), fare, is_err, ftype])
    return pd.DataFrame(rows, columns=["TransactionID", "Timestamp", "NodeID", "Zone", "Mode",
                                       "SignalStrength_dBm", "Latency_ms", "Drift_Magnitude",
                                       "Fare", "Is_Anomaly", "Fault_Type"])


def fit_transaction_marginals(tx: pd.DataFrame) -> dict:
    """Method-of-moments summary of the supplied transaction file."""
    zmap = {"Core (Dist 1)": "Core", "Periphery (Dist 12)": "Periphery"}
    tx = tx.copy()
    tx["ZoneShort"] = tx["Zone"].map(zmap).fillna(tx["Zone"])
    out = {"zone_p": tx["ZoneShort"].value_counts(normalize=True).to_dict(),
           "mode_p": tx["Mode"].value_counts(normalize=True).to_dict(),
           "fare": tx.groupby("Mode")["Fare"].median().to_dict(),
           "error_rate_by_zone": tx.groupby("ZoneShort")["IsError"].mean().to_dict(),
           "error_type_p": tx.loc[tx.IsError, "ErrorType"].value_counts(normalize=True).to_dict(),
           "hw": {}}
    for z, g in tx.groupby("ZoneShort"):
        for flag in (False, True):
            s = g.loc[g.IsError == flag, "HardwareScore"].clip(1e-3, 1 - 1e-3)
            m, v = float(s.mean()), float(s.var())
            k = max(m * (1 - m) / max(v, 1e-6) - 1, 1e-3)
            out["hw"][f"{z}|{int(flag)}"] = {"alpha": m * k, "beta": (1 - m) * k,
                                             "mean": m, "var": v, "n": int(len(s))}
    return out


# ----------------------------------------------------------------------------
# Integrated benchmark: time alignment
# ----------------------------------------------------------------------------
def align_travel_time(traffic_hourly: np.ndarray, seed: int = 11):
    """Lift the 720 by 28 hourly matrix onto the 2880 step quarter-hour grid.

    Values below one minute in the supplied file are physically impossible for
    a travel time, so they are clipped and the number of clipped entries is
    returned for the dataset card.
    """
    rng = np.random.default_rng(seed)
    clipped = int((traffic_hourly < 1.0).sum())
    hourly = np.clip(traffic_hourly, 1.0, None)
    hours, n = hourly.shape
    t_h = np.arange(hours)
    t_q = np.arange(hours * STEPS_PER_HOUR) / STEPS_PER_HOUR
    out = np.stack([np.interp(t_q, t_h, hourly[:, i]) for i in range(n)], axis=1)
    micro = np.zeros_like(out)
    e = rng.normal(0, 0.22, size=out.shape)
    for t in range(1, out.shape[0]):
        micro[t] = 0.55 * micro[t - 1] + e[t]
    return np.clip(out + micro, 0.8, None), clipped


# ----------------------------------------------------------------------------
# Scenario layer: faults, cascades, outages, missingness
# ----------------------------------------------------------------------------
def simulate_scenarios(travel_true: np.ndarray, mapping: pd.DataFrame, adjacency: np.ndarray,
                       cfg: dict, seed: int = 42):
    """Generate fault, cascade, outage, drift and missingness layers.

    Cascade propagation uses the documented travel adjacency and an explicit
    temporal decay. A neighbour j of an active node i becomes faulty at the
    next step with probability p0 * A_norm[i, j] * exp(-dt / tau). No part of
    the mechanism depends on row order.
    """
    rng = np.random.default_rng(seed)
    T, N = travel_true.shape
    prop = normalized_adjacency(adjacency)
    strata = centrality_strata(adjacency)
    zone = mapping["Zone"].values
    is_peri = (zone == "Periphery")
    gateway = mapping["GatewayID"].values
    gw_ids = sorted(set(gateway))

    sc = cfg["scenarios"]
    tau = float(sc["cascade_decay_steps"])
    p0 = float(sc["cascade_base_prob"])

    fault = np.zeros((T, N), dtype=np.int8)          # index into FAULT_TYPES
    active = np.zeros(N, dtype=bool)
    remaining = np.zeros(N, dtype=int)
    start_step = np.zeros(N, dtype=int)
    root_id = np.full((T, N), -1, dtype=np.int16)
    event_id = np.full((T, N), -1, dtype=np.int32)
    drift = np.zeros((T, N))
    drift_state = np.zeros(N)
    gw_down = np.zeros((T, len(gw_ids)), dtype=bool)
    events = []

    # signal baseline, zone dependent, matching the supplied source statistics
    signal = np.where(is_peri, -85.0, -60.0)[None, :] + rng.normal(0, 5.0, size=(T, N))
    base_latency = np.where(mapping["Mode"].values == "Metro", 20.0, 45.0)
    latency = np.maximum(5.0, rng.normal(base_latency[None, :], 10.0, size=(T, N)))

    # scheduled cascade roots, stratified by eigenvector centrality
    n_casc = int(sc["n_cascades_per_stratum"])
    schedule = []
    for name in ("low", "medium", "high"):
        for _ in range(n_casc):
            schedule.append((int(rng.integers(20, T - 40)), int(rng.choice(strata[name])), name))
    schedule.sort()
    sched_idx = 0

    # gateway outages
    n_out = int(sc["n_gateway_outages"])
    outages = sorted((int(rng.integers(20, T - 40)), int(rng.integers(0, len(gw_ids))),
                      int(rng.integers(*sc["gateway_outage_len_steps"]))) for _ in range(n_out))

    # topology closure windows, one inside the training period and one inside
    # the test period so the effect can be both learned and evaluated
    windows = [tuple(w) for w in sc["closure_windows"]]
    closed_by_window = {wi: closure_edges(adjacency, sc["n_closed_edges"], seed=7 + wi)
                        for wi in range(len(windows))}

    ev_counter = 0
    for t in range(T):
        # launch scheduled cascades
        while sched_idx < len(schedule) and schedule[sched_idx][0] == t:
            _, root, stratum = schedule[sched_idx]
            active[root] = True
            remaining[root] = int(rng.integers(*sc["fault_len_steps"]))
            start_step[root] = t
            root_id[t, root] = root
            ev_counter += 1
            event_id[t, root] = ev_counter
            events.append({"EventID": ev_counter, "StartStep": t, "EventType": "cascade",
                           "RootUnifiedNodeID": mapping.UnifiedNodeID.iloc[root],
                           "RootIndex": root, "CentralityStratum": stratum})
            sched_idx += 1

        # gateway outages
        for (t0, g, length) in outages:
            if t0 <= t < t0 + length:
                gw_down[t, g] = True
                if t == t0:
                    ev_counter += 1
                    events.append({"EventID": ev_counter, "StartStep": t, "EventType": "gateway_outage",
                                   "RootUnifiedNodeID": gw_ids[g], "RootIndex": -1,
                                   "CentralityStratum": "na"})

        # independent hazard from weak signal, no graph term
        weak = signal[t] < -90
        spontaneous = rng.random(N) < (float(sc["base_fault_prob"]) + weak * float(sc["weak_signal_bonus"]))
        for i in np.where(spontaneous & ~active)[0]:
            active[i] = True
            remaining[i] = int(rng.integers(*sc["fault_len_steps"]))
            start_step[i] = t
            root_id[t, i] = i
            ev_counter += 1
            event_id[t, i] = ev_counter
            events.append({"EventID": ev_counter, "StartStep": t, "EventType": "spontaneous",
                           "RootUnifiedNodeID": mapping.UnifiedNodeID.iloc[i], "RootIndex": int(i),
                           "CentralityStratum": "na"})

        # true graph-neighbour propagation with temporal decay
        if active.any():
            src = np.where(active)[0]
            age = np.array([t - start_step[i] for i in src], dtype=float)
            decay = np.exp(-age / tau)
            adj_eff = prop.copy()
            for wi, (lo_w, hi_w) in enumerate(windows):
                if lo_w <= t < hi_w:
                    for (u, v) in closed_by_window[wi]:
                        adj_eff[u, v] = 0.0
                        adj_eff[v, u] = 0.0
            hazard = np.zeros(N)
            for k, i in enumerate(src):
                hazard = np.maximum(hazard, p0 * adj_eff[i] * decay[k])
            hazard[active] = 0.0
            hit = np.where(rng.random(N) < hazard)[0]
            for j in hit:
                donor = src[int(np.argmax([adj_eff[i, j] for i in src]))]
                active[j] = True
                remaining[j] = int(rng.integers(*sc["fault_len_steps"]))
                start_step[j] = t
                r = root_id[:t + 1, donor]
                r = r[r >= 0]
                root_id[t, j] = int(r[-1]) if len(r) else int(donor)
                e = event_id[:t + 1, donor]
                e = e[e >= 0]
                event_id[t, j] = int(e[-1]) if len(e) else -1

        # materialise fault types and telemetry effects
        for i in np.where(active)[0]:
            if root_id[t, i] < 0:
                prev = root_id[:t, i]
                prev = prev[prev >= 0]
                root_id[t, i] = prev[-1] if len(prev) else i
                prev_e = event_id[:t, i]
                prev_e = prev_e[prev_e >= 0]
                event_id[t, i] = prev_e[-1] if len(prev_e) else -1
            r = rng.random()
            if r < 0.34:
                fault[t, i] = 1
                signal[t, i] = -120.0
            elif r < 0.67:
                fault[t, i] = 2
                latency[t, i] += float(rng.integers(200, 2001))
            else:
                fault[t, i] = 3
                drift_state[i] += float(rng.uniform(0.4, 3.0))
            remaining[i] -= 1
            if remaining[i] <= 0:
                active[i] = False
        drift_state *= float(sc["drift_recovery"])
        drift[t] = drift_state

        # gateway outage overrides node state for served sensors
        for g, gid in enumerate(gw_ids):
            if gw_down[t, g]:
                served = np.where(gateway == gid)[0]
                fault[t, served] = 4
                signal[t, served] = -120.0

    # ---- missingness -------------------------------------------------------
    mech = np.zeros((T, N), dtype=np.int8)   # 0 observed 1 MCAR 2 MNAR 3 outage 4 closure
    miss = np.zeros((T, N), dtype=bool)
    # gateway outage is a deterministic, correlated missingness source
    for g, gid in enumerate(gw_ids):
        served = np.where(gateway == gid)[0]
        rowsd = np.where(gw_down[:, g])[0]
        if len(rowsd):
            miss[np.ix_(rowsd, served)] = True
            mech[np.ix_(rowsd, served)] = 3
    # MNAR: peripheral sensors drop out as a function of their own signal
    logits = float(sc["mnar_intercept"]) + float(sc["mnar_slope"]) * (-signal - 85.0)
    p_mnar = 1.0 / (1.0 + np.exp(-logits))
    p_mnar[:, ~is_peri] = 0.0
    hit = (rng.random((T, N)) < p_mnar) & ~miss
    miss[hit] = True
    mech[hit] = 2
    # closure windows remove observations on the closed corridor endpoints
    for wi, (lo_w, hi_w) in enumerate(windows):
        for (u, v) in closed_by_window[wi]:
            for node in (u, v):
                w = np.arange(lo_w, hi_w)
                sel = w[rng.random(len(w)) < float(sc["closure_missing_prob"])]
                newly = sel[~miss[sel, node]]
                miss[newly, node] = True
                mech[newly, node] = 4

    # ---- distribution shift on the final block ----------------------------
    shift_start = int(sc["shift_start_frac"] * T)
    travel_shift = travel_true.copy()
    travel_shift[shift_start:] *= float(sc["shift_scale"])
    travel_shift[shift_start:] += rng.normal(0, float(sc["shift_noise"]),
                                             size=travel_shift[shift_start:].shape)

    # ---- observed travel time ---------------------------------------------
    observed = travel_shift + drift * float(sc["drift_to_minutes"])
    observed = observed + np.where(fault == 2, rng.normal(3.0, 1.5, size=observed.shape), 0.0)
    observed[fault == 1] = np.nan
    observed[miss] = np.nan

    return {
        "fault": fault, "root_id": root_id, "event_id": event_id, "drift": drift,
        "signal": signal, "latency": latency, "gw_down": gw_down, "gw_ids": gw_ids,
        "miss": miss, "mech": mech, "travel_true": travel_shift,
        "travel_observed": observed, "events": pd.DataFrame(events),
        "closed_edges": closed_by_window, "closure_windows": windows,
        "shift_start": shift_start,
    }


def independent_missing_masks(shape, rates, seed: int = 202):
    """MCAR masks used by the missingness sweep, one per rate."""
    rng = np.random.default_rng(seed)
    return {r: (rng.random(shape) < r) for r in rates}


# ----------------------------------------------------------------------------
# Transaction environment with known latent validity
# ----------------------------------------------------------------------------
def generate_transactions(n: int, marginals: dict, mapping: pd.DataFrame,
                          fault: np.ndarray, drift: np.ndarray, seed: int = 42,
                          cfg: dict | None = None, regime: str = "primary") -> pd.DataFrame:
    """Graph-linked transaction stream whose true validity is known.

    The supplied transaction file records an error flag but no label for
    whether a flagged fare was genuinely owed, and no node identity. Both are
    required to score a decision policy, so they are generated here. Zone,
    mode, fare and hardware-score distributions are taken from the supplied
    file; the latent validity label is a modelling assumption of this study.
    """
    rng = np.random.default_rng(seed)
    T, N = fault.shape
    zones = list(marginals["zone_p"])
    zp = np.array([marginals["zone_p"][z] for z in zones], dtype=float)
    zp = zp / zp.sum()
    idx_by_zone = {z: mapping.index[mapping.Zone == z].to_numpy() for z in zones}
    fare_by_mode = {k: float(v) for k, v in marginals["fare"].items()}

    step = rng.integers(0, T, size=n)
    zone = rng.choice(zones, size=n, p=zp)
    node = np.array([int(rng.choice(idx_by_zone[z])) for z in zone])
    mode = mapping["Mode"].values[node]
    fare = np.array([fare_by_mode.get(m, 7000.0) for m in mode])

    node_fault = fault[step, node] > 0
    node_drift = drift[step, node]

    # latent validity. In the primary regime this is independent of zone, which
    # is why the fairness constraint never binds. The stress regime makes it
    # zone dependent so that an unconstrained policy prefers to under-serve one
    # zone and the budget has something to bind against.
    short_zone = np.where(np.char.startswith(zone.astype(str), "Core"), "Core", "Periphery")
    fa_mult = np.full(n, float(cfg["policy"]["false_accept_multiplier"])
                      if cfg else 2.0)
    if regime == "stress" and cfg is not None:
        from .stress import zone_dependent_validity
        true_valid, fa_mult = zone_dependent_validity(n, short_zone, node_fault, cfg, rng)
    else:
        p_valid = np.where(node_fault, 0.88, 0.985)
        true_valid = rng.random(n) < p_valid

    # hardware score sampled from the zone and flag conditional Beta fits
    flag_p = np.clip(np.array([marginals["error_rate_by_zone"][z] for z in zone])
                     + 0.35 * node_fault + 0.01 * np.clip(node_drift, 0, 6), 0, 0.95)
    flagged = rng.random(n) < flag_p
    hw = np.empty(n)
    for z in zones:
        for f in (0, 1):
            key = f"{z}|{f}"
            sel = (zone == z) & (flagged == bool(f))
            if sel.sum():
                a, b = marginals["hw"][key]["alpha"], marginals["hw"][key]["beta"]
                hw[sel] = rng.beta(a, b, size=int(sel.sum()))
    etypes = list(marginals["error_type_p"])
    ep = np.array([marginals["error_type_p"][e] for e in etypes], dtype=float)
    ep = ep / ep.sum()
    err_type = np.where(flagged, rng.choice(etypes, size=n, p=ep), "None")

    return pd.DataFrame({
        "TransactionID": [f"TRX-{900000 + i}" for i in range(n)],
        "Step": step,
        "UnifiedNodeID": mapping.UnifiedNodeID.values[node],
        "NodeIndex": node,
        "Zone": zone,
        "Mode": mode,
        "Fare": fare,
        "IsError": flagged,
        "ErrorType": err_type,
        "HardwareScore": np.round(hw, 4),
        "NodeFault_True": node_fault,
        "Drift_Magnitude": np.round(node_drift, 4),
        "TrueValid": true_valid,
        "FalseAcceptMultiplier": np.round(fa_mult, 3),
        "ProvenanceID": [f"PRV-{mapping.UnifiedNodeID.values[node[i]]}-{step[i]:05d}" for i in range(n)],
    })
