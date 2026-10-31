"""Experiment A: fault detection, root cause attribution, and reconstruction.

The DyHGT model here is trained with the full multi-task objective

    L = lf Lfault + lr Lroot + lm Lmask + ly Lforecast + lc Lcalibration + ls Lstability

so that the results reported for Experiment A and for the ablation study come
from one jointly trained system rather than from separate specialised models.
Loss weights are taken from the configuration file and were not tuned; that is
stated in the manuscript.
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
import torch.nn.functional as F

from dyhgt import metrics as M
from dyhgt.data import (N_MODEL, N_TRANSIT, adj_id_for_steps, build_features,
                        build_relation_adjacency, load_arrays, make_windows,
                        node_type_index, pe_matrix, split_bounds, to_model_nodes)
from dyhgt.graph import build_topology, eigen_centrality, normalized_adjacency
from dyhgt.models import DyHGT, brier, mdn_moments, mdn_nll
from dyhgt.utils import PROC, TAB, ensure_dirs, load_config, save_json, set_seed

ART_MASK = None
A_DONE = set()


# ---------------------------------------------------------------------------
class Bench:
    def __init__(self, cfg):
        self.cfg = cfg
        self.arr = load_arrays()
        self.tr_end, self.va_end, self.T = split_bounds(self.arr["splits"])
        self.x, self.rel, self.filled, self.nanmask = build_features(self.arr)
        self.mapping = self.arr["mapping"]
        self.xm = to_model_nodes(self.x, self.mapping)
        self.relm = to_model_nodes(self.rel[..., None], self.mapping)[..., 0]
        self.adj_stack, self.windows = build_relation_adjacency(
            self.arr["travel_observed"], self.tr_end, cfg)
        self.pe = pe_matrix(cfg)
        self.ntype = torch.tensor(node_type_index())
        self.W = cfg["fault_model"]["window"]
        self.H = cfg["forecast"]["horizon"]
        self.y_true = self.arr["travel_true"]
        blk = self.y_true[:self.tr_end, :N_TRANSIT]
        self.y_mu, self.y_sd = float(blk.mean()), float(blk.std() + 1e-6)
        self.fault = (self.arr["fault_true"] > 0).astype(np.float32)
        self.root = self.arr["root_id"]
        self.event = self.arr["event_id"]

    def sets(self):
        w, h = self.W, self.H
        return (make_windows(self.T, w, h, 0, self.tr_end),
                make_windows(self.T, w, h, self.tr_end, self.va_end),
                make_windows(self.T, w, h, self.va_end, self.T))

    def batch(self, starts, mask_frac=0.0, rng=None, fixed_mask=None):
        xb = np.stack([self.xm[s:s + self.W] for s in starts]).copy()
        qb = np.stack([self.relm[s:s + self.W] for s in starts])
        fb = np.stack([self.fault[s:s + self.W] for s in starts])
        yb = np.stack([self.y_true[s:s + self.W] for s in starts])
        recon_mask = np.zeros(fb.shape, dtype=bool)
        if fixed_mask is not None:
            recon_mask = np.stack([fixed_mask[s:s + self.W] for s in starts])
            xb[:, :, :N_TRANSIT, 0][recon_mask] = 0.0
            xb[:, :, :N_TRANSIT, 1][recon_mask] = 1.0
        elif mask_frac > 0 and rng is not None:
            observed = np.stack([~self.nanmask[s:s + self.W] for s in starts])
            recon_mask = observed & (rng.random(observed.shape) < mask_frac)
            xb[:, :, :N_TRANSIT, 0][recon_mask] = 0.0
            xb[:, :, :N_TRANSIT, 1][recon_mask] = 1.0
        t_last = starts + self.W - 1
        fc_t = starts + self.W + self.H - 1
        y_fc = self.y_true[fc_t][:, :N_TRANSIT]
        aid = adj_id_for_steps(t_last, self.windows)
        return (torch.tensor(xb), torch.tensor(qb), torch.tensor(fb),
                torch.tensor(((yb[:, :, :N_TRANSIT] - self.y_mu) / self.y_sd).astype(np.float32)),
                torch.tensor(recon_mask), torch.tensor(((y_fc - self.y_mu) / self.y_sd).astype(np.float32)),
                aid, t_last)

    def root_targets(self, t_last):
        """Root node index at the last step, or minus one when no event is active."""
        out = np.full(len(t_last), -1, dtype=np.int64)
        for i, t in enumerate(t_last):
            r = self.root[t]
            act = np.where(r >= 0)[0]
            if len(act):
                vals, counts = np.unique(r[act], return_counts=True)
                out[i] = int(vals[np.argmax(counts)])
        return out


def make_model(b: Bench, seed, use_graph=True, use_reliability=True, heterogeneous=True):
    set_seed(seed)
    fm = b.cfg["fault_model"]
    return DyHGT(n_feat=b.xm.shape[-1], n_nodes=N_MODEL, n_rel=b.adj_stack.shape[1],
                 pe=b.pe, d=fm["d_model"], heads=fm["n_heads"], layers=fm["n_layers"],
                 k_mix=b.cfg["forecast"]["mdn_components"], use_graph=use_graph,
                 use_reliability=use_reliability, heterogeneous=heterogeneous)


def adj_tensor(b: Bench, heterogeneous=True):
    adj = torch.tensor(b.adj_stack, dtype=torch.float32)
    if not heterogeneous:
        merged = np.clip(b.adj_stack.sum(axis=1, keepdims=True), 0, 1)
        adj = torch.tensor(merged, dtype=torch.float32)
    return adj


def train_multitask(b: Bench, seed, epochs=None, use_graph=True, use_reliability=True,
                    heterogeneous=True, use_imputation=True, use_calibration=True,
                    static_graph=False, multitask=True, patience_limit=3):
    fm = b.cfg["fault_model"]
    model = make_model(b, seed, use_graph, use_reliability, heterogeneous)
    adj = adj_tensor(b, heterogeneous)
    if static_graph:      # every step sees the base topology, closures ignored
        adj = adj[:1].repeat(adj.shape[0], 1, 1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=fm["lr"])
    tr, va, _ = b.sets()
    rng = np.random.default_rng(seed)
    epochs = epochs or fm["epochs"]
    best, best_state, patience = np.inf, None, 0
    train_multitask.last_epoch = 0
    train_multitask.last_val = float("nan")
    mask_frac = 0.2 if use_imputation else 0.0
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(tr))
        for i in range(0, len(tr), fm["batch_size"]):
            sel = tr[order[i:i + fm["batch_size"]]]
            xb, qb, fb, yb, rmask, yfc, aid, t_last = b.batch(sel, mask_frac, rng)
            rt = torch.tensor(b.root_targets(t_last))
            opt.zero_grad()
            total = 0.0
            for a in np.unique(aid):
                s = aid == a
                out = model(xb[s], adj[a], qb[s], b.ntype)
                fl = out["fault_logit"][..., :N_TRANSIT]
                lf = F.binary_cross_entropy_with_logits(fl, fb[s])
                loss = fm["lambda_fault"] * lf
                rs = out["root_score"][:, -1, :N_TRANSIT]
                tgt = rt[s]
                if (tgt >= 0).any():
                    loss = loss + fm["lambda_root"] * F.cross_entropy(rs[tgt >= 0], tgt[tgt >= 0])
                if multitask:
                    rm = rmask[s]
                    if rm.any():
                        loss = loss + fm["lambda_mask"] * F.smooth_l1_loss(
                            out["recon"][..., :N_TRANSIT][rm], yb[s][rm])
                    pi = out["pi"][:, :N_TRANSIT]; mu = out["mu"][:, :N_TRANSIT]
                    sg = out["sigma"][:, :N_TRANSIT]
                    loss = loss + fm["lambda_forecast"] * mdn_nll(pi, mu, sg, yfc[s])
                    if use_calibration:
                        loss = loss + fm["lambda_calibration"] * brier(fl, fb[s])
                    loss = loss + fm["lambda_stability"] * (fl[:, 1:] - fl[:, :-1]).pow(2).mean()
                total = total + loss * s.mean()
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        vl = validate(b, model, adj, va, mask_frac, rng)
        train_multitask.last_epoch = ep + 1
        train_multitask.last_val = float(vl)
        if vl < best - 1e-4:
            best, best_state, patience = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= patience_limit:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, adj


@torch.no_grad()
def validate(b, model, adj, idx, mask_frac, rng):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(idx), 128):
        sel = idx[i:i + 128]
        xb, qb, fb, yb, rmask, yfc, aid, _ = b.batch(sel, mask_frac, rng)
        for a in np.unique(aid):
            s = aid == a
            out = model(xb[s], adj[a], qb[s], b.ntype)
            fl = out["fault_logit"][..., :N_TRANSIT]
            tot += float(F.binary_cross_entropy_with_logits(fl, fb[s])) * int(s.sum())
            n += int(s.sum())
    return tot / max(n, 1)


@torch.no_grad()
def predict_multitask(b, model, adj, idx, mask_frac=0.0, seed=0, fixed_mask=None):
    model.eval()
    rng = np.random.default_rng(seed)
    P, Rs, Rec, Rm, Y, T = [], [], [], [], [], []
    for i in range(0, len(idx), 128):
        sel = idx[i:i + 128]
        xb, qb, fb, yb, rmask, yfc, aid, t_last = b.batch(sel, mask_frac, rng, fixed_mask)
        p = np.zeros((len(sel), b.W, N_TRANSIT), dtype=np.float32)
        rs = np.zeros((len(sel), N_TRANSIT), dtype=np.float32)
        rc = np.zeros((len(sel), b.W, N_TRANSIT), dtype=np.float32)
        for a in np.unique(aid):
            s = aid == a
            out = model(xb[s], adj[a], qb[s], b.ntype)
            p[s] = torch.sigmoid(out["fault_logit"][..., :N_TRANSIT] / out["temperature"]).numpy()
            rs[s] = torch.softmax(out["root_score"][:, -1, :N_TRANSIT], -1).numpy()
            rc[s] = out["recon"][..., :N_TRANSIT].numpy()
        P.append(p); Rs.append(rs); Rec.append(rc); Rm.append(rmask.numpy())
        Y.append(yb.numpy()); T.append(t_last)
    return (np.concatenate(P), np.concatenate(Rs), np.concatenate(Rec),
            np.concatenate(Rm), np.concatenate(Y), np.concatenate(T))


# ---------------------------------------------------------------------------
# Classical and graph feature baselines
# ---------------------------------------------------------------------------
def tabular_features(b: Bench, lags=(1, 2, 4)):
    base = np.stack([
        b.arr["signal"], np.log1p(b.arr["latency"]), b.arr["drift"],
        b.rel, np.nan_to_num(b.arr["travel_observed"], nan=b.y_mu) - b.y_mu,
        b.arr["missing_mask"].astype(float)], axis=-1)
    feats = [base]
    for L in lags:
        feats.append(np.concatenate([np.repeat(base[:1], L, axis=0), base[:-L]], axis=0))
    return np.concatenate(feats, axis=-1)


def graph_features(b: Bench, tab: np.ndarray):
    a = normalized_adjacency(build_topology(N_TRANSIT, b.cfg["graph"]["topology_seed"]))
    one = np.einsum("ij,tjf->tif", a, tab)
    two = np.einsum("ij,tjf->tif", a, one)
    return np.concatenate([tab, one, two], axis=-1)


TAB_STRIDE = 2   # every second step of the training block, for CPU budget


def run_tabular(b: Bench, seeds, rows=None):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tab = tabular_features(b)
    gtab = graph_features(b, tab)
    y = (b.arr["fault_true"] > 0).astype(int)
    tr = slice(0, b.tr_end); va = slice(b.tr_end, b.va_end); te = slice(b.va_end, b.T)

    def flat(arr, sl, stride=1):
        return arr[sl][::stride].reshape(-1, arr.shape[-1])

    specs = [
        ("logistic_regression", lambda s: make_pipeline(StandardScaler(),
            LogisticRegression(max_iter=400, class_weight="balanced", random_state=s)), tab),
        ("random_forest", lambda s: RandomForestClassifier(
            n_estimators=200, min_samples_leaf=4, class_weight="balanced",
            n_jobs=1, random_state=s), tab),
        ("hist_gradient_boosting", lambda s: HistGradientBoostingClassifier(
            max_iter=200, random_state=s), tab),
        ("mlp", lambda s: make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=60, random_state=s)), tab),
        ("random_forest_graph_features", lambda s: RandomForestClassifier(
            n_estimators=200, min_samples_leaf=4, class_weight="balanced",
            n_jobs=1, random_state=s), gtab),
    ]
    rows = rows if rows is not None else []
    prob_grids = {}
    for name, ctor, X in specs:
        for seed in seeds:
            if (name, seed) in A_DONE:
                continue
            t0 = time.time()
            clf = ctor(seed).fit(flat(X, tr, TAB_STRIDE), y[tr][::TAB_STRIDE].ravel())
            pv = clf.predict_proba(flat(X, va))[:, 1]
            pt = clf.predict_proba(flat(X, te))[:, 1]
            thr = best_threshold(y[va].ravel(), pv)
            s = M.classification_scores(y[te].ravel(), pt, thr)
            s.update(model=name, seed=seed, threshold=float(thr),
                     train_seconds=round(time.time() - t0, 2))
            rows.append(s)
            pd.DataFrame(rows).to_csv(TAB / "exp_a_fault_detection.csv", index=False)
            print(f"  {name} seed {seed} f1={s['f1']:.3f}", flush=True)
            if seed == seeds[0]:
                g = np.full((X.shape[0], N_TRANSIT), np.nan)
                g[b.va_end:b.T] = pt.reshape(-1, N_TRANSIT)
                prob_grids[name] = g
    return rows, prob_grids


def best_threshold(y, p):
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return float(thr[int(np.argmax(f1[:-1]))]) if len(thr) else 0.5


# ---------------------------------------------------------------------------
# Reconstruction baselines
# ---------------------------------------------------------------------------
def reconstruction_baselines(b: Bench, mask, y_true):
    a = normalized_adjacency(build_topology(N_TRANSIT, b.cfg["graph"]["topology_seed"]))
    obs = b.arr["travel_observed"][:, :N_TRANSIT].copy()
    obs[mask] = np.nan
    out = {}
    out["global_mean"] = np.full_like(obs, b.y_mu)
    period = 96
    clim = np.zeros_like(obs)
    for k in range(period):
        sel = np.arange(len(obs)) % period == k
        trsel = sel.copy(); trsel[b.tr_end:] = False
        clim[sel] = np.nanmean(np.where(np.isnan(obs[trsel]), np.nan, obs[trsel]), axis=0)
    out["hour_of_week_climatology"] = np.nan_to_num(clim, nan=b.y_mu)
    lin = pd.DataFrame(obs).interpolate(limit_direction="both").to_numpy()
    out["linear_interpolation"] = np.nan_to_num(lin, nan=b.y_mu)
    filled = np.where(np.isnan(obs), np.nan_to_num(clim, nan=b.y_mu), obs)
    diff = filled.copy()
    for _ in range(3):
        diff = np.where(np.isnan(obs), diff @ a.T / np.maximum(a.sum(1), 1e-9), obs)
        diff = np.nan_to_num(diff, nan=b.y_mu)
    out["graph_diffusion"] = diff
    lo = pd.DataFrame(obs).ffill().bfill().to_numpy()
    out["last_observation_carried_forward"] = np.nan_to_num(lo, nan=b.y_mu)
    from sklearn.impute import KNNImputer
    sub = obs[::2]
    knn = KNNImputer(n_neighbors=5).fit(sub)
    out["knn_imputer"] = knn.transform(obs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--skip-tabular", action="store_true")
    ap.add_argument("--patience", type=int, default=3,
                    help="early stopping patience, 0 disables it so the epoch budget binds")
    a = ap.parse_args()
    cfg = load_config(); ensure_dirs()
    seeds = a.seeds if a.seeds else cfg["model_seeds"]
    b = Bench(cfg)
    tr, va, te = b.sets()
    global ART_MASK
    rng0 = np.random.default_rng(0)
    obs_ok0 = ~np.isnan(b.arr["travel_observed"][:, :N_TRANSIT])
    ART_MASK = obs_ok0 & (rng0.random(obs_ok0.shape) < 0.2)
    ART_MASK[:b.va_end] = False
    rows, prob_grids = [], {}
    fd_path = TAB / "exp_a_fault_detection.csv"
    done = set()
    if fd_path.exists():
        prev = pd.read_csv(fd_path)
        rows = prev.to_dict("records"); done = set(zip(prev.model, prev.seed))
    A_DONE.update(done)

    if not a.skip_tabular:
        trows, pg = run_tabular(b, seeds, rows)
        rows = trows
        prob_grids.update(pg)

    root_rows, recon_rows = [], []
    for seed in seeds:
        if ("dyhgt_marl", seed) in A_DONE:
            continue
        t0 = time.time()
        model, adj = train_multitask(
            b, seed, epochs=a.epochs,
            patience_limit=(a.patience or (a.epochs or b.cfg["fault_model"]["epochs"]) + 1))
        torch.save(model.state_dict(), PROC / f"dyhgt_multitask_seed{seed}.pt")
        pv = predict_multitask(b, model, adj, va)
        pt = predict_multitask(b, model, adj, te)
        yv = b.fault[va[:, None] + np.arange(b.W)[None, :]][:, :, :N_TRANSIT]
        yt = b.fault[te[:, None] + np.arange(b.W)[None, :]][:, :, :N_TRANSIT]
        thr = best_threshold(yv.ravel(), pv[0].ravel())
        s = M.classification_scores(yt.ravel().astype(int), pt[0].ravel(), thr)
        s.update(model="dyhgt_marl", seed=seed, threshold=float(thr),
                 train_seconds=round(time.time() - t0, 2))
        rows.append(s)
        pd.DataFrame(rows).to_csv(TAB / "exp_a_fault_detection.csv", index=False)
        print(f"  dyhgt_marl seed {seed} f1={s['f1']:.3f} recall={s['recall']:.3f}", flush=True)
        if seed == seeds[0]:
            g = np.full((b.T, N_TRANSIT), np.nan)
            g[pt[5]] = pt[0][:, -1, :]
            prob_grids["dyhgt_marl"] = g

        # root cause attribution on test cascade events
        rt = b.root_targets(pt[5])
        sel = rt >= 0
        if sel.sum() > 5:
            root_rows.append({"model": "dyhgt_marl", "seed": seed,
                              "top1": M.topk_accuracy(pt[1][sel], rt[sel], 1),
                              "top3": M.topk_accuracy(pt[1][sel], rt[sel], 3),
                              "n_events": int(sel.sum())})
            cent = eigen_centrality(build_topology(N_TRANSIT, cfg["graph"]["topology_seed"]))
            faulty = b.fault[pt[5]][:, :N_TRANSIT][sel] > 0
            root_rows.append({"model": "highest_centrality_faulty_node", "seed": seed,
                              "top1": M.topk_accuracy(faulty * cent[None, :], rt[sel], 1),
                              "top3": M.topk_accuracy(faulty * cent[None, :], rt[sel], 3),
                              "n_events": int(sel.sum())})
            rng = np.random.default_rng(seed)
            root_rows.append({"model": "random_faulty_node", "seed": seed,
                              "top1": M.topk_accuracy(faulty * rng.random(faulty.shape), rt[sel], 1),
                              "top3": M.topk_accuracy(faulty * rng.random(faulty.shape), rt[sel], 3),
                              "n_events": int(sel.sum())})

        # masked reconstruction on the shared artificial mask, read out at the
        # last step of each window so every masked cell is scored exactly once
        pr = predict_multitask(b, model, adj, te, fixed_mask=ART_MASK)
        rm_last = pr[3][:, -1, :]
        pred_last = pr[2][:, -1, :] * b.y_sd + b.y_mu
        true_last = pr[4][:, -1, :] * b.y_sd + b.y_mu
        if rm_last.any():
            recon_rows.append({"model": "dyhgt_marl", "seed": seed,
                               "mae": M.mae(true_last[rm_last], pred_last[rm_last]),
                               "rmse": M.rmse(true_last[rm_last], pred_last[rm_last]),
                               "n_masked": int(rm_last.sum()), "causal": True})
        # per mechanism breakdown of the naturally missing cells
        mech = b.arr["missing_mechanism"][pr[5]][:, :N_TRANSIT]
        for code, label in [(2, "MNAR_signal"), (3, "gateway_outage"), (4, "topology_closure")]:
            sel_m = mech == code
            if sel_m.sum() > 20:
                recon_rows.append({"model": f"dyhgt_marl[{label}]", "seed": seed,
                                   "mae": M.mae(true_last[sel_m], pred_last[sel_m]),
                                   "rmse": M.rmse(true_last[sel_m], pred_last[sel_m]),
                                   "n_masked": int(sel_m.sum()), "causal": True})
    pd.DataFrame(rows).to_csv(TAB / "exp_a_fault_detection.csv", index=False)
    pd.DataFrame(root_rows).to_csv(TAB / "exp_a_root_cause.csv", index=False)

    # reconstruction baselines evaluated on the same artificial mask
    art = ART_MASK
    y_true = b.y_true[:, :N_TRANSIT]
    for name, pred in reconstruction_baselines(b, art, y_true).items():
        recon_rows.append({"model": name, "seed": 0,
                           "mae": M.mae(y_true[art], pred[art]),
                           "rmse": M.rmse(y_true[art], pred[art]),
                           "n_masked": int(art.sum()),
                           "causal": name in ("global_mean", "hour_of_week_climatology",
                                              "last_observation_carried_forward")})
    pd.DataFrame(recon_rows).to_csv(TAB / "exp_a_reconstruction.csv", index=False)

    # detection delay on test cascade events, first seed probability grids
    ev = b.arr["events"]
    ev_test = ev[(ev.StartStep >= b.va_end) & (ev.RootIndex >= 0)]
    starts = [(int(r.StartStep), int(r.RootIndex)) for r in ev_test.itertuples()]
    delay_rows = []
    for name, grid in prob_grids.items():
        g = np.nan_to_num(grid, nan=0.0)
        for thr in (0.5,):
            d = M.detection_delay(g, [(t, n) for t, n in starts if t + 12 < len(g)], thr)
            d.update(model=name, threshold=thr)
            delay_rows.append(d)
    pd.DataFrame(delay_rows).to_csv(TAB / "exp_a_detection_delay.csv", index=False)
    save_json({"n_test_events": int(len(ev_test))}, TAB / "exp_a_meta.json")
    print("Experiment A complete")


if __name__ == "__main__":
    main()
