"""Detection delay on test cascade events, on an absolute step grid.

Written as a separate script so the delay can be recomputed without repeating
the full Experiment A sweep.
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd, torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dyhgt import metrics as M
from dyhgt.data import N_TRANSIT
from dyhgt.utils import PROC, TAB, load_config
import run_experiment_a_faults as A


def main():
    cfg = load_config()
    b = A.Bench(cfg)
    tab = A.tabular_features(b)
    gtab = A.graph_features(b, tab)
    y = (b.arr["fault_true"] > 0).astype(int)
    tr, va, te = slice(0, b.tr_end), slice(b.tr_end, b.va_end), slice(b.va_end, b.T)
    grids = {}
    for name, ctor, X in [
        ("logistic_regression", lambda: make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=400, class_weight="balanced", random_state=0)), tab),
        ("hist_gradient_boosting", lambda: HistGradientBoostingClassifier(
            max_iter=200, random_state=0), tab),
        ("random_forest_graph_features", lambda: RandomForestClassifier(
            n_estimators=120, min_samples_leaf=4, class_weight="balanced",
            n_jobs=1, random_state=0), gtab)]:
        clf = ctor().fit(X[tr][::2].reshape(-1, X.shape[-1]), y[tr][::2].ravel())
        g = np.zeros((b.T, N_TRANSIT))
        g[b.va_end:b.T] = clf.predict_proba(X[te].reshape(-1, X.shape[-1]))[:, 1].reshape(-1, N_TRANSIT)
        grids[name] = g
        print("fitted", name, flush=True)

    ck = sorted(PROC.glob("dyhgt_multitask_seed*.pt"))
    if ck:
        rng0 = np.random.default_rng(0)
        ok = ~np.isnan(b.arr["travel_observed"][:, :N_TRANSIT])
        A.ART_MASK = ok & (rng0.random(ok.shape) < 0.2); A.ART_MASK[:b.va_end] = False
        model, adj = A.train_multitask(b, 0, epochs=0)
        model.load_state_dict(torch.load(ck[0]))
        _, _, tew = b.sets()
        pt = A.predict_multitask(b, model, adj, tew)
        g = np.zeros((b.T, N_TRANSIT)); g[pt[5]] = pt[0][:, -1, :]
        grids["dyhgt_marl"] = g
        print("loaded dyhgt checkpoint", flush=True)

    ev = b.arr["events"]
    ev_t = ev[(ev.StartStep >= b.va_end) & (ev.RootIndex >= 0) & (ev.StartStep + 12 < b.T)]
    starts = [(int(r.StartStep), int(r.RootIndex)) for r in ev_t.itertuples()]
    rows = []
    for name, g in grids.items():
        for thr in (0.5,):
            d = M.detection_delay(g, starts, thr, horizon=12)
            d.update(model=name, threshold=thr, n_events=len(starts),
                     step_minutes=15)
            rows.append(d)
    pd.DataFrame(rows).to_csv(TAB / "exp_a_detection_delay.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
