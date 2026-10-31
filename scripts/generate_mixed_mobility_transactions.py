"""Graph-linked transaction stream with a known latent validity label."""
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from dyhgt.datagen import fit_transaction_marginals, generate_transactions
from dyhgt.graph import build_node_mapping
from dyhgt.utils import CITY, PROC, RAW, REGIME, ensure_dirs, load_config, raw_file, save_json, require

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(); ensure_dirs()
    n = a.n or cfg["transactions"]["n"]
    seed = a.seed if a.seed is not None else cfg["transactions"]["seed"]
    src = pd.read_csv(require(raw_file("transactions"),
                              "Place the cohort transaction CSV in data/raw."))
    marg = fit_transaction_marginals(src)
    save_json(marg, PROC / "transaction_marginals.json")
    fault = np.load(require(PROC / "fault_true.npy", "Run inject_faults_and_missingness.py first."))
    drift = np.load(PROC / "drift.npy")
    mapping = build_node_mapping()
    tx = generate_transactions(n, marg, mapping, fault, drift, seed=seed,
                               cfg=cfg, regime=REGIME)
    tx.to_csv(PROC / "transactions.csv", index=False)
    print("transactions", tx.shape, "flag rate", round(float(tx.IsError.mean()), 4),
          "true valid", round(float(tx.TrueValid.mean()), 4))

if __name__ == "__main__":
    main()
