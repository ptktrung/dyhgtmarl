"""Export the trained encoder's calibrated fault probability over all steps.

Section IV-F originally fed the decision layer a standalone logistic estimate,
which left the encoder and the decision layer uncoupled at inference. This
script writes the DyHGT fault head's own probability grid so the decision
layer can consume it instead, which is what the architecture claims.

The grid covers every step the sliding window can score. Steps before the
first full window fall back to the training block mean, and the file records
how many did.
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, torch
from dyhgt.data import N_TRANSIT
from dyhgt.utils import PROC, load_config, save_json
import run_experiment_a_faults as A


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true",
                    help="train a checkpoint if none is present")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3,
                    help="early stopping patience, 0 disables it so the epoch budget binds")
    args = ap.parse_args()
    cfg = load_config()
    b = A.Bench(cfg)
    ck = sorted(PROC.glob("dyhgt_multitask_seed*.pt"))
    if not ck and not args.retrain:
        sys.exit(f"no multi-task checkpoint in {PROC}. Run Experiment A, or pass --retrain.")
    rng0 = np.random.default_rng(0)
    ok = ~np.isnan(b.arr["travel_observed"][:, :N_TRANSIT])
    A.ART_MASK = ok & (rng0.random(ok.shape) < 0.2)
    A.ART_MASK[:b.va_end] = False
    if ck:
        model, adj = A.train_multitask(b, 0, epochs=0)
        model.load_state_dict(torch.load(ck[0]))
        src = ck[0].name
    else:
        print(f"training a checkpoint for {args.epochs} epochs")
        model, adj = A.train_multitask(b, 0, epochs=args.epochs,
                                       patience_limit=(args.patience or args.epochs + 1))
        torch.save(model.state_dict(), PROC / "dyhgt_multitask_seed0.pt")
        src = "dyhgt_multitask_seed0.pt (trained here)"

    grid = np.full((b.T, N_TRANSIT), np.nan)
    idx = np.arange(0, b.T - b.W - b.H + 1)
    pr = A.predict_multitask(b, model, adj, idx)
    grid[pr[5]] = pr[0][:, -1, :]
    filled = int(np.isnan(grid).sum())
    base = float(np.nanmean(grid[:b.tr_end]))
    grid = np.nan_to_num(grid, nan=base)
    np.save(PROC / "dyhgt_fault_probability.npy", grid)
    save_json({"checkpoint": src, "steps": int(b.T), "nodes": int(N_TRANSIT),
               "cells_backfilled_with_training_mean": filled,
               "mean_probability": round(float(grid.mean()), 5)},
              PROC / "dyhgt_fault_probability_meta.json")
    print(f"wrote dyhgt_fault_probability.npy {grid.shape}, "
          f"{filled} cells backfilled, mean {grid.mean():.4f}")


if __name__ == "__main__":
    main()
