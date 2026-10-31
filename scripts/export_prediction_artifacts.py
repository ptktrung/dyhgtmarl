"""Export prediction arrays used by the figure scripts.

Reloads the checkpoints written by Experiments A and B instead of retraining,
so the figures show exactly the models that produced the result tables.
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, torch
from dyhgt import metrics as M
from dyhgt.data import N_TRANSIT
from dyhgt.utils import PROC, RES, TAB, load_config

def main():
    cfg = load_config()
    import run_experiment_b_forecast as B
    import run_experiment_a_faults as A

    ckpt_b = RES / "dyhgt_forecast_seed0.pt"
    if ckpt_b.exists():
        run = B.Runner(cfg)
        model, adj = B.train_dyhgt(run, 0, epochs=0)
        model.load_state_dict(torch.load(ckpt_b))
        _, _, te = run.window_sets()
        pi, mu, sg = B.predict_dyhgt(run, model, adj, te)
        y, t = run.target(te)
        mean = (pi * mu).sum(-1)
        lo = M.mixture_quantile(0.025, pi, mu, sg)
        hi = M.mixture_quantile(0.975, pi, mu, sg)
        pit = M.pit_mixture(y, pi, mu, sg)
        np.savez_compressed(PROC / "forecast_example.npz", y=y, mean=mean, lo=lo, hi=hi,
                            pit=pit, steps=t, pi=pi, mu=mu, sigma=sg)
        print("wrote forecast_example.npz", y.shape)
    else:
        print("no forecast checkpoint, skipping forecast export")

    ckpt_a = sorted(PROC.glob("dyhgt_multitask_seed*.pt"))
    if ckpt_a:
        b = A.Bench(cfg)
        rng0 = np.random.default_rng(0)
        ok = ~np.isnan(b.arr["travel_observed"][:, :N_TRANSIT])
        A.ART_MASK = ok & (rng0.random(ok.shape) < 0.2)
        A.ART_MASK[:b.va_end] = False
        model, adj = A.train_multitask(b, 0, epochs=0)
        model.load_state_dict(torch.load(ckpt_a[0]))
        _, _, te = b.sets()
        pr = A.predict_multitask(b, model, adj, te, fixed_mask=A.ART_MASK)
        np.savez_compressed(PROC / "reconstruction_example.npz",
                            steps=pr[5],
                            recon=pr[2][:, -1, :] * b.y_sd + b.y_mu,
                            truth=pr[4][:, -1, :] * b.y_sd + b.y_mu,
                            mask=pr[3][:, -1, :], fault_prob=pr[0][:, -1, :])
        print("wrote reconstruction_example.npz")
    else:
        print("no multitask checkpoint, skipping reconstruction export")

if __name__ == "__main__":
    main()
