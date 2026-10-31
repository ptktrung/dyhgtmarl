"""Experiment I: external validation of the robustness claim on METR-LA.

Every other experiment in this paper runs on synthetic data, which bounds what
can be concluded. This one does not. It takes the single claim the paper rests
on, that the proposed encoder degrades more slowly than its baselines as
observations disappear, and tests it on a public benchmark that the authors did
not generate.

METR-LA is 207 loop detectors on the Los Angeles freeway network at five minute
resolution over four months in 2012, with a road network adjacency published
alongside it. It carries real sensor dropout rather than injected dropout,
which is the property under test. It carries no fault labels, no transactions
and no gateway structure, so only the forecasting claim can be examined here
and the fault detection and decision layer results stay synthetic.

The protocol matches Section V as closely as a different corpus allows. Splits
are chronological, model selection is on the validation block, the test block
is used once, deterministic baselines receive a Gaussian predictive
distribution whose scale is fitted on validation residuals, and the same
metrics are computed by the same code.

Inputs are the published files, not a redistribution of them. Pass the
directory holding metr-la.h5, distances_la_2012.csv and graph_sensor_ids.txt.
The adjacency is rebuilt from the distance table with the thresholded Gaussian
kernel of Li et al., so no pickle from the internet is deserialized.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dyhgt import metrics as M
from dyhgt.graph import laplacian_pe, normalized_adjacency
from dyhgt.models import (DCRNNLite, DyHGT, GRUBaseline, STGCNLite,
                          TransformerBaseline, mdn_moments, mdn_nll)
from dyhgt.utils import TAB, ensure_dirs, set_seed

W, H = 12, 4           # three hour window, one hour ahead, as in Section V
RESAMPLE = 3           # five minute steps to a fifteen minute grid
D_HIDDEN = 48


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_metr_la(root: pathlib.Path):
    import h5py

    with h5py.File(root / "metr-la.h5", "r") as f:
        g = f["df"]
        values = np.array(g["block0_values"], dtype=np.float64)
        ids = [s.decode() if isinstance(s, bytes) else str(s)
               for s in np.array(g["block0_items"])]

    # a zero is a missing reading in this corpus, not a speed of zero
    miss = values <= 0.0
    T = (values.shape[0] // RESAMPLE) * RESAMPLE
    values, miss = values[:T], miss[:T]
    v = values.reshape(-1, RESAMPLE, values.shape[1])
    m = miss.reshape(-1, RESAMPLE, miss.shape[1])
    with np.errstate(invalid="ignore"):
        agg = np.where(m, np.nan, v)
        series = np.nanmean(agg, axis=1)
    missing = np.isnan(series)

    sensor_ids = (root / "graph_sensor_ids.txt").read_text().strip().split(",")
    dist = pd.read_csv(root / "distances_la_2012.csv", dtype={"from": str, "to": str})
    index = {s: i for i, s in enumerate(sensor_ids)}
    n = len(sensor_ids)
    d = np.full((n, n), np.inf)
    np.fill_diagonal(d, 0.0)
    for a, b, c in zip(dist["from"], dist["to"], dist["cost"]):
        if a in index and b in index:
            d[index[a], index[b]] = c
    finite = d[np.isfinite(d) & (d > 0)]
    sigma = finite.std()
    adj = np.exp(-(d / sigma) ** 2)
    adj[adj < 0.1] = 0.0                      # the standard sparsity threshold
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0.0)

    if ids != sensor_ids:
        order = [ids.index(s) for s in sensor_ids if s in ids]
        series, missing = series[:, order], missing[:, order]
    return series.astype(np.float32), missing, adj.astype(np.float32)


class ExternalRunner:
    """The minimum of the Section V protocol that a different corpus supports."""

    def __init__(self, series, missing, adj, extra_mask=None, stride=6):
        self.T, self.N = series.shape
        self.tr_end = int(0.7 * self.T)
        self.va_end = int(0.8 * self.T)

        obs = series.copy()
        miss = missing.copy()
        if extra_mask is not None:
            obs[extra_mask] = np.nan
            miss = miss | extra_mask
        nan = np.isnan(obs) | miss

        ref = float(np.nanmean(series[:self.tr_end][~missing[:self.tr_end]]))
        filled = np.where(nan, ref, obs)
        # carry the last observation forward so a gap does not read as the mean
        for t in range(1, self.T):
            gap = nan[t]
            filled[t, gap] = filled[t - 1, gap]

        step = np.arange(self.T)
        tod = (step % 96) / 96.0 * 2 * np.pi
        feats = [
            (filled - ref) / 10.0,
            nan.astype(np.float32),
            np.repeat(np.sin(tod)[:, None], self.N, axis=1),
            np.repeat(np.cos(tod)[:, None], self.N, axis=1),
            np.repeat(((step % 672) // 96 >= 5).astype(np.float32)[:, None], self.N, axis=1),
        ]
        self.x = np.stack(feats, axis=-1).astype(np.float32)
        self.obs = filled.astype(np.float32)
        self.y = np.where(missing, np.nan, series).astype(np.float32)
        self.nan = nan

        # reliability is the share of genuine observations in the recent past,
        # which is the same quantity the synthetic corpus derives from telemetry
        k = 12
        c = np.cumsum(np.vstack([np.zeros((1, self.N)), (~nan).astype(np.float32)]), axis=0)
        rel = (c[k:] - c[:-k]) / k
        self.rel = np.vstack([np.repeat(rel[:1], k, axis=0), rel])[:self.T].astype(np.float32)

        self.adj = torch.tensor(normalized_adjacency(adj), dtype=torch.float32)[None]
        self.pe = laplacian_pe(adj, k=8)
        self.ntype = torch.zeros(self.N, dtype=torch.long)
        self.stride = stride

        blk = self.y[:self.tr_end]
        self.y_mu = float(np.nanmean(blk))
        self.y_sd = float(np.nanstd(blk) + 1e-6)

    def windows(self, lo, hi, stride=None):
        s = np.arange(lo, hi - W - H + 1, stride or self.stride)
        # a target that was never observed cannot be scored
        t = s + W + H - 1
        return s[~np.isnan(self.y[t]).all(axis=1)]

    def sets(self):
        return (self.windows(0, self.tr_end),
                self.windows(self.tr_end, self.va_end, 3),
                self.windows(self.va_end, self.T, 3))

    def target(self, starts):
        return self.y[starts + W + H - 1]

    def series_input(self, starts):
        """Standardized on the training block, as the deterministic baselines
        need the same scaling the mixture head gets or they do not leave their
        initialization in a CPU sized budget."""
        w = np.stack([self.obs[s:s + W] for s in starts])
        return torch.tensor((w - self.y_mu) / self.y_sd)

    def hetero_input(self, starts):
        xb = torch.tensor(np.stack([self.x[s:s + W] for s in starts]))
        qb = torch.tensor(np.stack([self.rel[s:s + W] for s in starts]))
        return xb, qb


def batches(idx, bs, rng=None):
    if rng is not None:
        idx = idx[rng.permutation(len(idx))]
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def train_dyhgt(run, seed, epochs, bs=32):
    set_seed(seed)
    model = DyHGT(n_feat=run.x.shape[-1], n_nodes=run.N, n_rel=1, pe=run.pe,
                  d=D_HIDDEN, n_types=1)
    opt = torch.optim.Adam(model.parameters(), lr=0.002)
    tr, va, _ = run.sets()
    rng = np.random.default_rng(seed)
    best, best_state, patience = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for sel in batches(tr, bs, rng):
            xb, qb = run.hetero_input(sel)
            y = run.target(sel)
            keep = ~np.isnan(y)
            yt = torch.tensor(np.nan_to_num((y - run.y_mu) / run.y_sd))
            kt = torch.tensor(keep)
            opt.zero_grad()
            out = model(xb, run.adj, qb, run.ntype)
            loss = mdn_nll(out["pi"][kt], out["mu"][kt], out["sigma"][kt], yt[kt])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        vl = eval_nll(run, model, va, bs)
        if vl < best - 1e-4:
            best, best_state, patience = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 3:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def eval_nll(run, model, starts, bs=32):
    model.eval()
    tot, n = 0.0, 0
    for sel in batches(starts, bs):
        xb, qb = run.hetero_input(sel)
        y = run.target(sel)
        keep = ~np.isnan(y)
        yt = torch.tensor(np.nan_to_num((y - run.y_mu) / run.y_sd))
        kt = torch.tensor(keep)
        out = model(xb, run.adj, qb, run.ntype)
        tot += float(mdn_nll(out["pi"][kt], out["mu"][kt], out["sigma"][kt], yt[kt])) * int(kt.sum())
        n += int(kt.sum())
    return tot / max(n, 1)


@torch.no_grad()
def predict_dyhgt(run, model, starts, bs=32):
    model.eval()
    P, Mu, Sg = [], [], []
    for sel in batches(starts, bs):
        xb, qb = run.hetero_input(sel)
        out = model(xb, run.adj, qb, run.ntype)
        P.append(out["pi"].numpy())
        Mu.append(out["mu"].numpy() * run.y_sd + run.y_mu)
        Sg.append(out["sigma"].numpy() * run.y_sd)
    return np.concatenate(P), np.concatenate(Mu), np.concatenate(Sg)


def train_point(run, kind, seed, epochs, bs=32):
    set_seed(seed)
    cls = {"gru": GRUBaseline, "dcrnn": DCRNNLite, "stgcn": STGCNLite,
           "transformer": TransformerBaseline}[kind]
    model = (cls(run.N, run.adj[0], d=D_HIDDEN) if needs_adj(kind)
             else cls(run.N, d=D_HIDDEN))
    opt = torch.optim.Adam(model.parameters(), lr=0.002)
    tr, va, _ = run.sets()
    rng = np.random.default_rng(seed)
    best, best_state, patience = np.inf, None, 0
    nidx = torch.arange(run.N)
    for ep in range(epochs):
        model.train()
        for sel in batches(tr, bs, rng):
            xb = run.series_input(sel)
            y = run.target(sel)
            keep = torch.tensor(~np.isnan(y))
            yt = torch.tensor(np.nan_to_num((y - run.y_mu) / run.y_sd))
            opt.zero_grad()
            p = model(xb, nidx)
            loss = nn.functional.smooth_l1_loss(p[keep], yt[keep])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        vl = point_val(run, model, kind, va, bs)
        if vl < best - 1e-4:
            best, best_state, patience = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 3:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def needs_adj(kind):
    return kind in ("dcrnn", "stgcn")


@torch.no_grad()
def point_val(run, model, kind, starts, bs=32):
    model.eval()
    tot, n = 0.0, 0
    nidx = torch.arange(run.N)
    for sel in batches(starts, bs):
        xb = run.series_input(sel)
        y = run.target(sel)
        keep = ~np.isnan(y)
        p = model(xb, nidx).numpy() * run.y_sd + run.y_mu
        tot += float(np.abs(p[keep] - y[keep]).sum())
        n += int(keep.sum())
    return tot / max(n, 1)


@torch.no_grad()
def predict_point(run, model, kind, starts, bs=32):
    model.eval()
    nidx = torch.arange(run.N)
    out = []
    for sel in batches(starts, bs):
        out.append(model(run.series_input(sel), nidx).numpy() * run.y_sd + run.y_mu)
    return np.concatenate(out)


def historical_average(run, starts):
    """Per node, per quarter hour mean over every training step, not only the
    steps that happen to be a window target."""
    tr_steps = np.arange(run.tr_end)
    table = np.zeros((96, run.N))
    counts = np.zeros((96, run.N))
    y = run.y[:run.tr_end]
    ok = ~np.isnan(y)
    for t_ in tr_steps:
        slot = t_ % 96
        table[slot] += np.where(ok[t_], np.nan_to_num(y[t_]), 0.0)
        counts[slot] += ok[t_]
    table = np.where(counts > 0, table / np.maximum(counts, 1), np.nanmean(y[ok]))
    return table[(starts + W + H - 1) % 96]


def ridge_ar(run, starts, lam=10.0):
    """Pooled autoregression on a node's own lags.

    A ridge on the flattened window across all nodes would carry 12 times 207
    coefficients against roughly a thousand training windows, which is not
    identified. Pooling the lags across nodes keeps the baseline honest and is
    the usual form of a linear autoregressive control.
    """
    tr = run.sets()[0]
    X = np.concatenate([run.obs[s:s + W].T for s in tr])          # (n_win*N, W)
    Y = np.concatenate([np.nan_to_num(run.target(np.array([s])))[0] for s in tr])
    keep = ~np.isnan(np.concatenate([run.target(np.array([s]))[0] for s in tr]))
    X, Y = X[keep], Y[keep]
    X = np.hstack([X, np.ones((len(X), 1))])
    A = X.T @ X + lam * np.eye(X.shape[1])
    beta = np.linalg.solve(A, X.T @ Y)
    Xt = np.concatenate([run.obs[s:s + W].T for s in starts])
    Xt = np.hstack([Xt, np.ones((len(Xt), 1))])
    return (Xt @ beta).reshape(len(starts), run.N)


def gaussian_scale(y_val, p_val):
    r = (y_val - p_val)[~np.isnan(y_val)]
    return float(np.sqrt(np.mean(r ** 2)) + 1e-6)


def score(y, mean, sigma, pi=None, mu=None, sg=None):
    keep = ~np.isnan(y)
    y1 = y[keep]
    if pi is None:
        m1, s1 = mean[keep], np.full(keep.sum(), sigma)
        crps = M.crps_gaussian(y1, m1, s1)
        pit = M.pit_gaussian(y1, m1, s1)
        lo, hi = m1 - 1.96 * s1, m1 + 1.96 * s1
    else:
        p1, m1, s1 = pi[keep], mu[keep], sg[keep]
        crps = M.crps_mixture(y1, p1, m1, s1)
        pit = M.pit_mixture(y1, p1, m1, s1)
        lo = M.mixture_quantile(0.025, p1, m1, s1)
        hi = M.mixture_quantile(0.975, p1, m1, s1)
        m1 = mean[keep]
    return {"mae": M.mae(y1, m1), "rmse": M.rmse(y1, m1), "mape": M.mape(y1, m1),
            "crps": float(crps), "picp95": M.picp(y1, lo, hi),
            "ece": M.calibration_error(pit), "n": int(keep.sum())}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory holding the METR-LA files")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--missing", type=float, nargs="*", default=[0.0, 0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--probe", action="store_true", help="time one epoch and stop")
    a = ap.parse_args()
    ensure_dirs()
    root = pathlib.Path(a.data)

    series, missing, adj = load_metr_la(root)
    print(f"METR-LA: {series.shape[0]} steps of 15 minutes, {series.shape[1]} sensors, "
          f"{missing.mean() * 100:.2f} percent missing, "
          f"mean degree {float((adj > 0).sum(1).mean()):.1f}")
    base = ExternalRunner(series, missing, adj, stride=a.stride)
    tr, va, te = base.sets()
    print(f"windows: {len(tr)} train, {len(va)} validation, {len(te)} test")
    iu = np.triu_indices(base.N, 1)
    c = np.corrcoef(np.nan_to_num(base.y, nan=np.nanmean(base.y)).T)
    print(f"mean pairwise node correlation {float(np.nanmean(c[iu])):.4f}")

    if a.probe:
        t0 = time.time()
        train_dyhgt(base, 0, epochs=1)
        print(f"one DyHGT epoch took {time.time() - t0:.0f}s")
        return

    path = TAB / "exp_i_external_metr_la.csv"
    rows, done = [], set()
    if path.exists():
        prev = pd.read_csv(path)
        rows = prev.to_dict("records")
        done = set(zip(prev.model, prev.seed, prev.missing_rate))

    rng_mask = np.random.default_rng(1234)
    runners = {}
    for rate in a.missing:
        extra = None
        if rate > 0:
            extra = rng_mask.random(series.shape) < rate
            extra[:base.va_end] = False      # evaluation time loss only
        runners[rate] = base if rate == 0 else ExternalRunner(
            series, missing, adj, extra_mask=extra, stride=a.stride)

    for seed in a.seeds:
        fitted = {}
        for kind in ("gru", "dcrnn", "stgcn", "transformer"):
            if all((kind, seed, r) in done for r in a.missing):
                continue
            t0 = time.time()
            fitted[kind] = train_point(base, kind, seed, a.epochs)
            print(f"  {kind} seed {seed} trained in {time.time() - t0:.0f}s", flush=True)
        if not all(("dyhgt", seed, r) in done for r in a.missing):
            t0 = time.time()
            fitted["dyhgt"] = train_dyhgt(base, seed, a.epochs)
            print(f"  dyhgt seed {seed} trained in {time.time() - t0:.0f}s", flush=True)

        for rate in a.missing:
            run = runners[rate]
            y_va, y_te = run.target(va), run.target(te)
            for kind, model in fitted.items():
                if (kind, seed, rate) in done:
                    continue
                if kind == "dyhgt":
                    pi, mu, sg = predict_dyhgt(run, model, te)
                    mean, _ = mdn_moments(torch.tensor(pi), torch.tensor(mu), torch.tensor(sg))
                    s = score(y_te, mean.numpy(), None, pi, mu, sg)
                else:
                    pv = predict_point(run, model, kind, va)
                    pt = predict_point(run, model, kind, te)
                    s = score(y_te, pt, gaussian_scale(y_va, pv))
                s.update(model=kind, seed=seed, missing_rate=rate)
                rows.append(s)
                pd.DataFrame(rows).to_csv(path, index=False)
                print(f"  {kind:11s} seed {seed} miss {rate:.1f} "
                      f"crps {s['crps']:.3f} mape {s['mape']:.2f} picp {s['picp95']:.3f}",
                      flush=True)

        for kind, fn in (("ha", historical_average), ("ridge", ridge_ar)):
            for rate in a.missing:
                if (kind, seed, rate) in done or seed != a.seeds[0]:
                    continue
                run = runners[rate]
                pv = fn(run, va)
                pt = fn(run, te)
                s = score(run.target(te), pt, gaussian_scale(run.target(va), pv))
                s.update(model=kind, seed=seed, missing_rate=rate)
                rows.append(s)
                pd.DataFrame(rows).to_csv(path, index=False)
                print(f"  {kind:11s} seed {seed} miss {rate:.1f} crps {s['crps']:.3f}",
                      flush=True)

    print("Experiment I complete")


if __name__ == "__main__":
    main()
