"""Experiment B: probabilistic travel time forecasting.

Chronological splits, five model seeds, and four evaluation conditions:
normal test, independent missingness sweep, active cascade steps, and the
topology change window. Deterministic baselines receive a Gaussian predictive
distribution whose scale is fitted on validation residuals so that CRPS, PICP
and the calibration error are comparable across all models.
"""
from __future__ import annotations

import argparse
import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dyhgt import metrics as M
from dyhgt.data import (N_MODEL, N_TRANSIT, adj_id_for_steps, build_features,
                        build_relation_adjacency, load_arrays, make_windows,
                        node_type_index, pe_matrix, split_bounds, to_model_nodes)
from dyhgt.models import (DCRNNLite, DyHGT, GRUBaseline, STGCNLite, TransformerBaseline,
                          mdn_moments, mdn_nll)
from dyhgt.utils import TAB, ensure_dirs, load_config, set_seed


def batches(idx, bs, rng=None):
    if rng is not None:
        idx = idx[rng.permutation(len(idx))]
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


def gather(x, starts, window):
    return np.stack([x[s:s + window] for s in starts])


class Runner:
    def __init__(self, cfg, extra_mask=None):
        self.cfg = cfg
        self.arr = load_arrays()
        self.tr_end, self.va_end, self.T = split_bounds(self.arr["splits"])
        self.x, self.rel, self.filled, self.nanmask = build_features(self.arr, extra_mask)
        self.mapping = self.arr["mapping"]
        self.xm = to_model_nodes(self.x, self.mapping)
        self.relm = to_model_nodes(self.rel[..., None], self.mapping)[..., 0]
        self.adj_stack, self.windows = build_relation_adjacency(
            self.arr["travel_observed"], self.tr_end, cfg)
        self.pe = pe_matrix(cfg)
        self.ntype = torch.tensor(node_type_index())
        self.W = cfg["forecast"]["window"]
        self.H = cfg["forecast"]["horizon"]
        self.y = self.arr["travel_true"]          # ground truth target
        self.obs = self.filled                    # model input series
        # Target standardisation on the training block only. Without it the
        # mixture heads start far from the data scale and do not converge in a
        # useful number of CPU epochs.
        blk = self.y[:self.tr_end, :N_TRANSIT]
        self.y_mu = float(blk.mean())
        self.y_sd = float(blk.std() + 1e-6)

    def target_norm(self, starts):
        y, t = self.target(starts)
        return ((y - self.y_mu) / self.y_sd).astype(np.float32), t

    def window_sets(self):
        w, h = self.W, self.H
        return (make_windows(self.T, w, h, 0, self.tr_end),
                make_windows(self.T, w, h, self.tr_end, self.va_end),
                make_windows(self.T, w, h, self.va_end, self.T))

    def target(self, starts):
        t = starts + self.W + self.H - 1
        return self.y[t][:, :N_TRANSIT].astype(np.float32), t

    def series_input(self, starts):
        return torch.tensor(gather(self.obs, starts, self.W), dtype=torch.float32)

    def hetero_input(self, starts):
        xb = torch.tensor(gather(self.xm, starts, self.W))
        qb = torch.tensor(gather(self.relm, starts, self.W))
        aid = adj_id_for_steps(starts + self.W - 1, self.windows)
        return xb, qb, aid


def train_dyhgt(run: Runner, seed: int, use_graph=True, use_reliability=True,
                heterogeneous=True, epochs=None):
    cfg = run.cfg
    set_seed(seed)
    fc = cfg["forecast"]
    model = DyHGT(n_feat=run.xm.shape[-1], n_nodes=N_MODEL,
                  n_rel=run.adj_stack.shape[1], pe=run.pe, d=fc["d_model"],
                  heads=fc["n_heads"], layers=fc["n_layers"], k_mix=fc["mdn_components"],
                  use_graph=use_graph, use_reliability=use_reliability,
                  heterogeneous=heterogeneous)
    opt = torch.optim.Adam(model.parameters(), lr=fc["lr"])
    adj = torch.tensor(run.adj_stack, dtype=torch.float32)
    if not heterogeneous:
        adj = adj[:, :1].clone()
        adj[:, 0] = torch.tensor(np.clip(run.adj_stack.sum(axis=1), 0, 1), dtype=torch.float32)
    tr, va, _ = run.window_sets()
    rng = np.random.default_rng(seed)
    epochs = epochs or fc["epochs"]
    best, best_state, patience = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for bidx in batches(tr, fc["batch_size"], rng):
            xb, qb, aid = run.hetero_input(bidx)
            yb = torch.tensor(run.target_norm(bidx)[0])
            opt.zero_grad()
            loss = 0.0
            for a in np.unique(aid):
                sel = aid == a
                out = model(xb[sel], adj[a], qb[sel], run.ntype)
                pi, mu, sg = out["pi"][:, :N_TRANSIT], out["mu"][:, :N_TRANSIT], out["sigma"][:, :N_TRANSIT]
                l = mdn_nll(pi, mu, sg, yb[sel])
                mean, _ = mdn_moments(pi, mu, sg)
                l = l + fc.get("lambda_stability", 0.1) * (mean[:, 1:] - mean[:, :-1]).pow(2).mean()
                loss = loss + l * sel.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        vm = evaluate_dyhgt(run, model, adj, va)
        if vm["crps"] < best - 1e-4:
            best, best_state, patience = vm["crps"], {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 4:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, adj


@torch.no_grad()
def predict_dyhgt(run: Runner, model, adj, starts):
    model.eval()
    pis, mus, sgs = [], [], []
    for bidx in batches(starts, 128):
        xb, qb, aid = run.hetero_input(bidx)
        p = np.zeros((len(bidx), N_TRANSIT, model.k_mix), dtype=np.float32)
        m = np.zeros_like(p); s = np.zeros_like(p)
        for a in np.unique(aid):
            sel = aid == a
            out = model(xb[sel], adj[a], qb[sel], run.ntype)
            p[sel] = out["pi"][:, :N_TRANSIT].numpy()
            m[sel] = out["mu"][:, :N_TRANSIT].numpy()
            s[sel] = out["sigma"][:, :N_TRANSIT].numpy()
        pis.append(p); mus.append(m); sgs.append(s)
    pi, mu, sg = np.concatenate(pis), np.concatenate(mus), np.concatenate(sgs)
    return pi, mu * run.y_sd + run.y_mu, sg * run.y_sd


def evaluate_dyhgt(run, model, adj, starts):
    pi, mu, sg = predict_dyhgt(run, model, adj, starts)
    y, _ = run.target(starts)
    mean = (pi * mu).sum(-1)
    return {"crps": M.crps_mixture(y, pi, mu, sg), "mae": M.mae(y, mean)}


def train_point_model(run: Runner, kind: str, seed: int, epochs: int = 20):
    set_seed(seed)
    fc = run.cfg["forecast"]
    adj_n = run.adj_stack[0, 0, :N_TRANSIT, :N_TRANSIT]
    if kind == "gru":
        model = GRUBaseline(N_TRANSIT, d=fc["d_model"])
    elif kind == "transformer":
        model = TransformerBaseline(N_TRANSIT, d=fc["d_model"], heads=fc["n_heads"],
                                    layers=fc["n_layers"])
    elif kind == "stgcn":
        model = STGCNLite(N_TRANSIT, adj_n, d=fc["d_model"])
    elif kind == "dcrnn":
        model = DCRNNLite(N_TRANSIT, adj_n, d=fc["d_model"])
    else:
        raise ValueError(kind)
    opt = torch.optim.Adam(model.parameters(), lr=fc["lr"])
    lossf = nn.L1Loss()
    node_idx = torch.arange(N_TRANSIT)
    tr, va, _ = run.window_sets()
    rng = np.random.default_rng(seed)
    best, best_state, patience = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for bidx in batches(tr, fc["batch_size"], rng):
            xb = run.series_input(bidx)[:, :, :N_TRANSIT]
            yb = torch.tensor(run.target_norm(bidx)[0])
            opt.zero_grad()
            loss = lossf(model(xb, node_idx), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        with torch.no_grad():
            model.eval()
            pv = model(run.series_input(va)[:, :, :N_TRANSIT], node_idx).numpy()
        v = M.mae(run.target_norm(va)[0], pv)
        if v < best - 1e-4:
            best, best_state, patience = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 4:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_point(model, run, starts):
    model.eval()
    node_idx = torch.arange(N_TRANSIT)
    out = []
    for bidx in batches(starts, 256):
        out.append(model(run.series_input(bidx)[:, :, :N_TRANSIT], node_idx).numpy())
    return np.concatenate(out) * run.y_sd + run.y_mu


def historical_average(run: Runner, starts):
    period = 96
    obs = run.obs[:run.tr_end, :N_TRANSIT]
    prof = np.stack([obs[np.arange(run.tr_end) % period == k].mean(axis=0) for k in range(period)])
    t = starts + run.W + run.H - 1
    return prof[t % period]


def ridge_ar(run: Runner, tr, starts, lam=1.0):
    """Per node ridge autoregression on the observed window."""
    Xtr = gather(run.obs[:, :N_TRANSIT], tr, run.W)
    Ytr = run.target(tr)[0]
    Xte = gather(run.obs[:, :N_TRANSIT], starts, run.W)
    pred = np.zeros((len(starts), N_TRANSIT))
    for n in range(N_TRANSIT):
        A = np.hstack([Xtr[:, :, n], np.ones((len(tr), 1))])
        b = Ytr[:, n]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ b)
        pred[:, n] = np.hstack([Xte[:, :, n], np.ones((len(starts), 1))]) @ w
    return pred


def gaussian_wrap(y_val, p_val, p_test):
    """Per node residual scale, the standard way to score a point model."""
    sigma = np.maximum(np.std(y_val - p_val, axis=0, keepdims=True), 0.2)
    return np.repeat(sigma, len(p_test), axis=0)


def condition_masks(run: Runner, starts):
    t = starts + run.W + run.H - 1
    ev = run.arr["events"]
    fault = run.arr["fault_true"] > 0
    cascade_any = fault[t].any(axis=1)
    lo, hi = run.cfg["scenarios"]["closure_windows"][1]
    shift = int(run.cfg["scenarios"]["shift_start_frac"] * run.T)
    return {
        "normal_test": (t >= 0) & ~((t >= lo) & (t < hi)) & (t < shift),
        "cascade_active": cascade_any & (t < shift),
        "topology_change": (t >= lo) & (t < hi),
        "distribution_shift": t >= shift,
    }


def score_all(y, mean, sigma, pi=None, mu=None, sg=None):
    if pi is not None:
        crps = M.crps_mixture(y, pi, mu, sg)
        lo = M.mixture_quantile(0.025, pi, mu, sg)
        hi = M.mixture_quantile(0.975, pi, mu, sg)
        pit = M.pit_mixture(y, pi, mu, sg)
    else:
        crps = M.crps_gaussian(y, mean, sigma)
        lo, hi = mean - 1.959964 * sigma, mean + 1.959964 * sigma
        pit = M.pit_gaussian(y, mean, sigma)
    return {"mae": M.mae(y, mean), "rmse": M.rmse(y, mean), "mape": M.mape(y, mean),
            "crps": crps, "picp95": M.picp(y, lo, hi), "interval_width": M.interval_width(lo, hi),
            "ece": M.calibration_error(pit.ravel())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--models", nargs="*", default=["ha", "ridge", "gru", "stgcn", "dcrnn",
                                                    "transformer", "dyhgt"])
    a = ap.parse_args()
    cfg = load_config(); ensure_dirs()
    seeds = a.seeds if a.seeds else cfg["model_seeds"]
    run = Runner(cfg)
    tr, va, te = run.window_sets()
    y_val = run.target(va)[0]
    y_te = run.target(te)[0]
    conds = condition_masks(run, te)
    rates = cfg["scenarios"]["missing_rates"]

    proc = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed"
    runners = {0.0: run}
    for r in rates:
        if r > 0:
            runners[r] = Runner(cfg, extra_mask=np.load(proc / f"mcar_mask_{int(r*100):02d}.npy"))
    print("runners cached for missingness rates", list(runners), flush=True)

    cond_path, sweep_path = TAB / "exp_b_forecast_conditions.csv", TAB / "exp_b_missingness_sweep.csv"
    done = set()
    rows, sweep_rows = [], []
    if cond_path.exists():
        prev = pd.read_csv(cond_path)
        rows = prev.to_dict("records")
        done = set(zip(prev.model, prev.seed))
    if sweep_path.exists():
        sweep_rows = pd.read_csv(sweep_path).to_dict("records")
    print("already complete:", sorted(done), flush=True)
    for name in a.models:
        for seed in seeds:
            if (name, seed) in done:
                continue
            t0 = time.time()
            if name == "ha":
                pv, pt = historical_average(run, va), historical_average(run, te)
                sig_te = gaussian_wrap(y_val, pv, pt)
                pack = dict(mean=pt, sigma=sig_te)
            elif name == "ridge":
                pv, pt = ridge_ar(run, tr, va), ridge_ar(run, tr, te)
                pack = dict(mean=pt, sigma=gaussian_wrap(y_val, pv, pt))
            elif name == "dyhgt":
                model, adj = train_dyhgt(run, seed, epochs=a.epochs)
                pi, mu, sg = predict_dyhgt(run, model, adj, te)
                mean, sd = (pi * mu).sum(-1), None
                pack = dict(mean=mean, sigma=None, pi=pi, mu=mu, sg=sg)
                torch.save(model.state_dict(), TAB.parent / f"dyhgt_forecast_seed{seed}.pt")
            else:
                model = train_point_model(run, name, seed)
                pv, pt = predict_point(model, run, va), predict_point(model, run, te)
                pack = dict(mean=pt, sigma=gaussian_wrap(y_val, pv, pt))
            elapsed = time.time() - t0

            for cond, m in conds.items():
                if m.sum() < 10:
                    continue
                kw = {}
                if pack.get("pi") is not None:
                    kw = dict(pi=pack["pi"][m], mu=pack["mu"][m], sg=pack["sg"][m])
                    s = score_all(y_te[m], pack["mean"][m], None, **kw)
                else:
                    s = score_all(y_te[m], pack["mean"][m], pack["sigma"][m])
                s.update(model=name, seed=seed, condition=cond, n_windows=int(m.sum()),
                         train_seconds=round(elapsed, 2))
                rows.append(s)

            # independent missingness sweep, evaluated on the normal test block
            if name in ("ha", "ridge", "gru", "stgcn", "dcrnn", "transformer", "dyhgt"):
                for r in rates:
                    run_r = runners[r]
                    tr_r, va_r, te_r = run_r.window_sets()
                    if name == "ha":
                        p = historical_average(run_r, te_r)
                        pvr = historical_average(run_r, va_r)
                        sg_ = gaussian_wrap(y_val, pvr, p); pk = dict(mean=p, sigma=sg_)
                    elif name == "ridge":
                        p = ridge_ar(run_r, tr_r, te_r)
                        pvr = ridge_ar(run_r, tr_r, va_r)
                        pk = dict(mean=p, sigma=gaussian_wrap(y_val, pvr, p))
                    elif name == "dyhgt":
                        pi, mu, sg = predict_dyhgt(run_r, model, adj, te_r)
                        pk = dict(mean=(pi * mu).sum(-1), pi=pi, mu=mu, sg=sg)
                    else:
                        p = predict_point(model, run_r, te_r)
                        pvr = predict_point(model, run_r, va_r)
                        pk = dict(mean=p, sigma=gaussian_wrap(y_val, pvr, p))
                    mnorm = condition_masks(run_r, te_r)["normal_test"]
                    if pk.get("pi") is not None:
                        s = score_all(y_te[mnorm], pk["mean"][mnorm], None,
                                      pi=pk["pi"][mnorm], mu=pk["mu"][mnorm], sg=pk["sg"][mnorm])
                    else:
                        s = score_all(y_te[mnorm], pk["mean"][mnorm], pk["sigma"][mnorm])
                    s.update(model=name, seed=seed, missing_rate=r)
                    sweep_rows.append(s)
            pd.DataFrame(rows).to_csv(cond_path, index=False)
            pd.DataFrame(sweep_rows).to_csv(sweep_path, index=False)
            print(f"  {name} seed {seed} done in {elapsed:.1f}s", flush=True)

    pd.DataFrame(rows).to_csv(TAB / "exp_b_forecast_conditions.csv", index=False)
    pd.DataFrame(sweep_rows).to_csv(TAB / "exp_b_missingness_sweep.csv", index=False)
    print("wrote exp_b_forecast_conditions.csv and exp_b_missingness_sweep.csv")


if __name__ == "__main__":
    main()
