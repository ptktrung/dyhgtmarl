"""Does the early stopping rule understate the encoder's fault detection score?

Experiment G exposed a problem with the stopping rule. On the revised corpus it
halted at the same epoch whatever budget was set, and when it was disabled the
same ten epoch budget produced a materially better model. The fault detection
comparison of Section VI-A was run with that rule in place, so the headline
negative result has to be checked against a run without it before it can be
attributed to the architecture.

This script retrains the multi-task fault head on whichever regime is active,
once with the rule as published and once with it disabled, on the same seed,
the same budget, the same features, the same split and the same threshold
selection procedure. Everything other than the stopping rule is held fixed.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from dyhgt import metrics as M
from dyhgt.data import N_TRANSIT
from dyhgt.utils import REGIME, TAB, ensure_dirs, load_config
import run_experiment_a_faults as A


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    a = ap.parse_args()

    cfg = load_config()
    ensure_dirs()
    b = A.Bench(cfg)
    rng0 = np.random.default_rng(0)
    ok = ~np.isnan(b.arr["travel_observed"][:, :N_TRANSIT])
    A.ART_MASK = ok & (rng0.random(ok.shape) < 0.2)
    A.ART_MASK[:b.va_end] = False
    _, va, te = b.sets()

    yv = b.fault[va[:, None] + np.arange(b.W)[None, :]][:, :, :N_TRANSIT]
    yt = b.fault[te[:, None] + np.arange(b.W)[None, :]][:, :, :N_TRANSIT]

    path = TAB / "exp_h_early_stopping.csv"
    rows, done = [], set()
    if path.exists():
        prev = pd.read_csv(path)
        rows = prev.to_dict("records")
        done = set(zip(prev.seed, prev.stopping))

    for seed in a.seeds:
        for label, limit in (("patience_3", 3), ("disabled", a.epochs + 1)):
            if (seed, label) in done:
                continue
            t0 = time.time()
            model, adj = A.train_multitask(b, seed, epochs=a.epochs, patience_limit=limit)
            pv = A.predict_multitask(b, model, adj, va)
            pt = A.predict_multitask(b, model, adj, te)
            thr = A.best_threshold(yv.ravel(), pv[0].ravel())
            s = M.classification_scores(yt.ravel().astype(int), pt[0].ravel(), thr)
            s.update(seed=seed, stopping=label, epochs_cap=a.epochs,
                     epochs_run=int(getattr(A.train_multitask, "last_epoch", a.epochs)),
                     regime=REGIME, train_seconds=round(time.time() - t0, 1),
                     threshold=float(thr))
            rows.append(s)
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"  {REGIME} seed={seed} {label}: ran {s['epochs_run']} epochs, "
                  f"f1={s['f1']:.3f} precision={s['precision']:.3f} "
                  f"recall={s['recall']:.3f} ({s['train_seconds']:.0f}s)", flush=True)

    d = pd.DataFrame(rows)
    g = d.groupby("stopping").f1.mean()
    if {"patience_3", "disabled"} <= set(g.index):
        print(f"mean F1 with the published rule {g['patience_3']:.3f}, "
              f"with it disabled {g['disabled']:.3f}")
    print("Early stopping check complete")


if __name__ == "__main__":
    main()
