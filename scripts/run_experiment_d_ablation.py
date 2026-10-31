"""Experiment D: end to end ablation.

Model side switches are retrained from scratch. Policy side switches are read
from the Experiment C result file so that the ablation table reports one
consistent set of numbers rather than a second, separately tuned policy run.
Results are appended incrementally so a interrupted run can resume.
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

from dyhgt import metrics as M
from dyhgt.data import N_TRANSIT
from dyhgt.utils import TAB, ensure_dirs, load_config

import run_experiment_a_faults as A

VARIANTS = [
    ("full_model", dict()),
    ("static_graph", dict(static_graph=True)),
    ("homogeneous_relations", dict(heterogeneous=False)),
    ("no_reliability_conditioning", dict(use_reliability=False)),
    ("no_imputation_head", dict(use_imputation=False)),
    ("no_calibration_term", dict(use_calibration=False)),
    ("single_task_fault_only", dict(multitask=False)),
    ("no_graph_temporal_only", dict(use_graph=False)),
]


def evaluate(b, model, adj, va, te, seed):
    pv = A.predict_multitask(b, model, adj, va)
    pt = A.predict_multitask(b, model, adj, te)
    yv = b.fault[va[:, None] + np.arange(b.W)[None, :]][:, :, :N_TRANSIT]
    yt = b.fault[te[:, None] + np.arange(b.W)[None, :]][:, :, :N_TRANSIT]
    thr = A.best_threshold(yv.ravel(), pv[0].ravel())
    out = M.classification_scores(yt.ravel().astype(int), pt[0].ravel(), thr)

    # forecast quality of the same trained instance
    fc = []
    with torch.no_grad():
        model.eval()
        for i in range(0, len(te), 128):
            sel = te[i:i + 128]
            xb, qb, fb, yb, rmask, yfc, aid, t_last = b.batch(sel)
            pi = np.zeros((len(sel), N_TRANSIT, model.k_mix), dtype=np.float32)
            mu = np.zeros_like(pi); sg = np.zeros_like(pi)
            for a_ in np.unique(aid):
                s = aid == a_
                o = model(xb[s], adj[a_], qb[s], b.ntype)
                pi[s] = o["pi"][:, :N_TRANSIT].numpy()
                mu[s] = o["mu"][:, :N_TRANSIT].numpy()
                sg[s] = o["sigma"][:, :N_TRANSIT].numpy()
            fc.append((pi, mu * b.y_sd + b.y_mu, sg * b.y_sd, yfc.numpy() * b.y_sd + b.y_mu))
    pi = np.concatenate([f[0] for f in fc]); mu = np.concatenate([f[1] for f in fc])
    sg = np.concatenate([f[2] for f in fc]); y = np.concatenate([f[3] for f in fc])
    mean = (pi * mu).sum(-1)
    out.update(forecast_mae=M.mae(y, mean), forecast_crps=M.crps_mixture(y, pi, mu, sg),
               forecast_picp95=M.picp(y, M.mixture_quantile(0.025, pi, mu, sg),
                                      M.mixture_quantile(0.975, pi, mu, sg)),
               forecast_ece=M.calibration_error(M.pit_mixture(y, pi, mu, sg).ravel()))

    # reconstruction on the shared artificial mask
    pr = A.predict_multitask(b, model, adj, te, fixed_mask=A.ART_MASK)
    rm = pr[3][:, -1, :]
    if rm.any():
        pred = pr[2][:, -1, :] * b.y_sd + b.y_mu
        truth = pr[4][:, -1, :] * b.y_sd + b.y_mu
        out["reconstruction_mae"] = M.mae(truth[rm], pred[rm])
    rt = b.root_targets(pt[5])
    sel = rt >= 0
    out["root_top1"] = M.topk_accuracy(pt[1][sel], rt[sel], 1) if sel.sum() > 5 else float("nan")
    out.update(threshold=float(thr), seed=seed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--variants", nargs="*", default=None)
    a = ap.parse_args()
    cfg = load_config(); ensure_dirs()
    b = A.Bench(cfg)
    tr, va, te = b.sets()
    rng0 = np.random.default_rng(0)
    ok = ~np.isnan(b.arr["travel_observed"][:, :N_TRANSIT])
    A.ART_MASK = ok & (rng0.random(ok.shape) < 0.2)
    A.ART_MASK[:b.va_end] = False

    path = TAB / "exp_d_ablation_model.csv"
    rows, done = [], set()
    if path.exists():
        prev = pd.read_csv(path)
        rows = prev.to_dict("records")
        done = set(zip(prev.variant, prev.seed))

    wanted = [v for v in VARIANTS if a.variants is None or v[0] in a.variants]
    for name, kw in wanted:
        for seed in a.seeds:
            if (name, seed) in done:
                continue
            t0 = time.time()
            model, adj = A.train_multitask(b, seed, epochs=a.epochs, **kw)
            r = evaluate(b, model, adj, va, te, seed)
            r.update(variant=name, train_seconds=round(time.time() - t0, 2))
            rows.append(r)
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"  {name} seed {seed} f1={r['f1']:.3f} crps={r['forecast_crps']:.3f} "
                  f"({r['train_seconds']:.0f}s)", flush=True)

    # policy side switches, taken from Experiment C
    pol = TAB / "exp_c_policy.csv"
    if pol.exists():
        dfc = pd.read_csv(pol)
        keep = {"decentralized_constrained": "full_policy",
                "centralized_constrained": "centralized_decision_policy",
                "decentralized_utility_max": "no_fairness_constraint",
                "decentralized_constrained_no_abstention": "no_abstention_or_escalation",
                "rule_based_legacy": "rule_based_legacy"}
        sub = dfc[dfc.policy.isin(keep)].copy()
        sub["variant"] = sub.policy.map(keep)
        sub.to_csv(TAB / "exp_d_ablation_policy.csv", index=False)
    print("Experiment D complete")


if __name__ == "__main__":
    main()
