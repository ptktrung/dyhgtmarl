"""Experiment G: separate training budget from architecture.

Both reviews observe that the neural models were trained for ten to fourteen
epochs while the tabular baselines were fitted to convergence, and argue that
the fault detection comparison may therefore measure under-training rather
than architecture. This script holds everything else fixed and varies only the
epoch budget, so the two explanations can be told apart from the data.

If F1 rises steeply with budget, the comparison was budget limited. If it
plateaus well below the baselines, the architecture is the limit on this
corpus and the conclusion stands.

Early stopping is disabled here. With the usual patience of three the loop
terminates at the same epoch whatever cap is set, so every setting returns the
identical model and the sweep measures nothing. The epoch actually reached and
the final validation loss are recorded alongside the scores, so a reader can
see that the budget bound rather than the stopping rule.
"""
import argparse, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from dyhgt import metrics as M
from dyhgt.data import N_TRANSIT
from dyhgt.utils import REGIME, TAB, ensure_dirs, load_config
import run_experiment_a_faults as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, nargs="*", default=[10, 25, 40])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=0,
                    help="early stopping patience, 0 disables it so the budget binds")
    a = ap.parse_args()
    cfg = load_config(); ensure_dirs()
    b = A.Bench(cfg)
    rng0 = np.random.default_rng(0)
    ok = ~np.isnan(b.arr["travel_observed"][:, :N_TRANSIT])
    A.ART_MASK = ok & (rng0.random(ok.shape) < 0.2)
    A.ART_MASK[:b.va_end] = False
    tr, va, te = b.sets()

    path = TAB / "exp_g_budget.csv"
    rows, done = [], set()
    if path.exists():
        prev = pd.read_csv(path)
        rows = prev.to_dict("records")
        done = set(zip(prev.epochs, prev.seed))

    for ep in a.epochs:
        if (ep, a.seed) in done:
            continue
        t0 = time.time()
        model, adj = A.train_multitask(b, a.seed, epochs=ep,
                                       patience_limit=(a.patience or ep + 1))
        pv = A.predict_multitask(b, model, adj, va)
        pt = A.predict_multitask(b, model, adj, te)
        yv = b.fault[va[:, None] + np.arange(b.W)[None, :]][:, :, :N_TRANSIT]
        yt = b.fault[te[:, None] + np.arange(b.W)[None, :]][:, :, :N_TRANSIT]
        thr = A.best_threshold(yv.ravel(), pv[0].ravel())
        s = M.classification_scores(yt.ravel().astype(int), pt[0].ravel(), thr)
        s.update(epochs=ep, seed=a.seed, regime=REGIME,
                 epochs_run=int(getattr(A.train_multitask, "last_epoch", ep)),
                 final_val_loss=round(float(getattr(A.train_multitask, "last_val", float("nan"))), 5),
                 patience=a.patience,
                 train_seconds=round(time.time() - t0, 1), threshold=float(thr))
        rows.append(s)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  {REGIME} epochs={ep} ran={s['epochs_run']} f1={s['f1']:.3f} "
              f"precision={s['precision']:.3f} recall={s['recall']:.3f} "
              f"val={s['final_val_loss']:.4f} ({s['train_seconds']:.0f}s)", flush=True)
    print("Experiment G complete")


if __name__ == "__main__":
    main()
