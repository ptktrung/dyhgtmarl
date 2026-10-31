"""Experiment E: function approximation check on the decision layer.

Section VI-D of the manuscript reports the decision layer as a set of tabular
action value functions, one per zone. A reviewer can reasonably ask whether the
conclusions of Experiment C follow from the policy or from the table. This
script answers that question by replacing the table with a linear softmax
policy trained by REINFORCE with a learned value baseline, holding everything
else fixed: the same constrained process, the same state information, the same
seven actions, the same financial reward, the same dual ascent on the fairness,
latency and risk budgets, the same chronological split and the same seeds.

The policy is a genuine function approximator. Parameters are shared across
states through a feature map that combines the six discretized state buckets
with four continuous quantities, so the policy generalizes between states that
the tabular variant treats as unrelated.
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
from dyhgt.utils import PROC, TAB, ensure_dirs, load_config, save_json, set_seed

import run_experiment_c_policy as C

ACTIONS = C.ACTIONS
N_ACT = len(ACTIONS)
I_CORR = ACTIONS.index("corroborate")
I_DEFER = ACTIONS.index("defer")

# bucket cardinalities, matching C.STATE_DIMS
DIMS = (2, 2, 4, 3, 3)          # zone, error flag, hardware, fault prob, reliability
N_CORR = 3                       # corroboration signal, 2 means not yet observed


def feature_matrix(feats: np.ndarray, cont: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Map state to a feature vector.

    One hot indicators for the six discrete components, four continuous
    quantities, and a bias term. The continuous terms are what make this a
    function approximator rather than a re-encoded lookup table: two states in
    different buckets still share gradient signal through them.
    """
    n = len(feats)
    blocks = []
    for k, d in enumerate(DIMS):
        oh = np.zeros((n, d), dtype=np.float64)
        oh[np.arange(n), np.clip(feats[:, k], 0, d - 1)] = 1.0
        blocks.append(oh)
    oh = np.zeros((n, N_CORR), dtype=np.float64)
    oh[np.arange(n), corr] = 1.0
    blocks.append(oh)
    blocks.append(cont)
    blocks.append(np.ones((n, 1)))
    return np.hstack(blocks)


def continuous_features(tx: pd.DataFrame, fault_prob_tx: np.ndarray,
                        rel_tx: np.ndarray) -> np.ndarray:
    hw = tx.HardwareScore.values.astype(float)
    fare = tx.Fare.values.astype(float)
    return np.stack([
        hw,
        fault_prob_tx,
        rel_tx,
        (np.log(fare) - np.log(7000.0)) / np.log(15000.0 / 7000.0),
    ], axis=1)


def softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)


class LinearActor:
    """Softmax policy and linear value baseline over the shared feature map."""

    def __init__(self, n_feat: int, n_groups: int, rng, lr_pi=0.05, lr_v=0.05):
        self.w = np.zeros((n_groups, N_ACT, n_feat))
        self.v = np.zeros((n_groups, n_feat))
        self.lr_pi, self.lr_v = lr_pi, lr_v
        self.rng = rng
        self.n_groups = n_groups

    def probs(self, phi, group, allowed_mask):
        z = np.einsum("bf,baf->ba", phi, self.w[group])
        z = np.where(allowed_mask, z, -1e9)
        return softmax_rows(z)

    def value(self, phi, group):
        return np.einsum("bf,bf->b", phi, self.v[group])

    def update(self, phi, group, actions, advantage, returns, probs):
        onehot = np.zeros_like(probs)
        onehot[np.arange(len(actions)), actions] = 1.0
        grad_logits = (onehot - probs) * advantage[:, None]
        for g in range(self.n_groups):
            sel = group == g
            if not sel.any():
                continue
            self.w[g] += self.lr_pi * np.einsum("ba,bf->af", grad_logits[sel], phi[sel]) / sel.sum()
            td = returns[sel] - self.value(phi[sel], group[sel])
            self.v[g] += self.lr_v * (td[:, None] * phi[sel]).mean(axis=0)


def episode_pass(actor, env, idx, feats, cont, group, allowed_mask, rng, greedy=False):
    """One vectorized sweep. Returns trajectory pieces and realized outcomes."""
    n = len(idx)
    corr0 = np.full(n, 2, dtype=int)
    phi0 = feature_matrix(feats, cont, corr0)
    p0 = actor.probs(phi0, group, allowed_mask)
    a0 = p0.argmax(axis=1) if greedy else np.array(
        [rng.choice(N_ACT, p=p0[i]) for i in range(n)])

    took_corr = a0 == I_CORR
    corr1 = corr0.copy()
    for i in np.where(took_corr)[0]:
        corr1[i] = env.corroboration(idx[i])

    final = a0.copy()
    phi1 = p1 = a1 = None
    if took_corr.any():
        mask2 = allowed_mask.copy()
        mask2[:, I_CORR] = False          # one corroboration per record
        phi1 = feature_matrix(feats[took_corr], cont[took_corr], corr1[took_corr])
        p1 = actor.probs(phi1, group[took_corr], mask2[took_corr])
        a1 = p1.argmax(axis=1) if greedy else np.array(
            [rng.choice(N_ACT, p=p1[j]) for j in range(p1.shape[0])])
        final[took_corr] = a1

    rewards = np.zeros(n)
    accepted = np.zeros(n, dtype=bool)
    risk = np.zeros(n)
    latency = np.zeros(n)
    reviewed = np.zeros(n)
    for j in range(n):
        name = ACTIONS[final[j]]
        if name == "corroborate":
            name = "defer"
        lat = C.ACTION_LATENCY_MS[ACTIONS[a0[j]]]
        if took_corr[j]:
            lat += C.ACTION_LATENCY_MS[name]
        r, e, rv, ac = env.outcome(idx[j], name)
        if took_corr[j]:
            r -= env.corr_cost
        rewards[j], accepted[j], latency[j], reviewed[j] = r, bool(ac), lat, rv
        risk[j] = 1.0 if (ac and not env.valid[idx[j]]) else 0.0
    return dict(phi0=phi0, p0=p0, a0=a0, took_corr=took_corr, phi1=phi1, p1=p1, a1=a1,
                rewards=rewards, accepted=accepted, risk=risk, latency=latency,
                reviewed=reviewed, final=final)


def train_actor(env, feats, cont, cfg, seed, decentralized=True, fairness_budget=0.05,
                risk_budget=0.02, latency_budget=None, epochs=40, batch=6000):
    pc = cfg["policy"]
    rng = np.random.default_rng(seed + 17)
    n = len(feats)
    n_feat = sum(DIMS) + N_CORR + cont.shape[1] + 1
    n_groups = 2 if decentralized else 1
    actor = LinearActor(n_feat, n_groups, rng)
    group_all = feats[:, 0].astype(int) if decentralized else np.zeros(n, dtype=int)
    allowed_all = np.ones((n, N_ACT), dtype=bool)
    scale = float(np.median(env.fare))
    gamma = pc["gamma"]
    lam_f = lam_r = lam_l = 0.0
    hist = []

    for ep in range(epochs):
        sel = rng.permutation(n)[:min(batch, n)]
        out = episode_pass(actor, env, sel, feats[sel], cont[sel], group_all[sel],
                           allowed_all[sel], rng)
        shaped = out["rewards"] / scale - lam_r * out["risk"] - lam_l * out["latency"] / 10.0
        if fairness_budget is not None:
            grp = feats[sel, 0].astype(int)
            shaped -= lam_f * 0.5 * ((~out["accepted"]) & (grp == 1))
            shaped -= lam_f * 0.05 * (out["accepted"] & (grp == 0))

        # terminal step update
        if out["took_corr"].any():
            phi1, p1, a1 = out["phi1"], out["p1"], out["a1"]
            g1 = group_all[sel][out["took_corr"]]
            ret1 = shaped[out["took_corr"]]
            adv1 = ret1 - actor.value(phi1, g1)
            actor.update(phi1, g1, a1, adv1, ret1, p1)

        ret0 = np.where(out["took_corr"], gamma * shaped, shaped)
        adv0 = ret0 - actor.value(out["phi0"], group_all[sel])
        actor.update(out["phi0"], group_all[sel], out["a0"], adv0, ret0, out["p0"])

        if fairness_budget is not None:
            gap = M.fairness_gaps(M.group_rates(
                out["accepted"].astype(float), env.zone[sel],
                env.valid[sel]))["equal_opportunity_gap"]
            lam_f = max(0.0, lam_f + pc["lagrange_lr"] * (gap - fairness_budget))
            lam_r = max(0.0, lam_r + pc["lagrange_lr"] * (out["risk"].mean() - risk_budget))
            if latency_budget:
                lam_l = max(0.0, lam_l + 0.1 * (out["latency"].mean() - latency_budget))
            hist.append({"epoch": ep, "lambda_fairness": lam_f, "lambda_risk": lam_r,
                         "lambda_latency": lam_l, "equal_opportunity_gap": gap,
                         "mean_reward_vnd": float(out["rewards"].mean())})
    return actor, group_all, pd.DataFrame(hist)


def greedy_fn(actor, feats, cont, group):
    """Wrap the learned policy for C.run_policy so scoring is byte identical."""
    cache = {}

    def fn(i, corr, allowed):
        key = (i, corr)
        if key not in cache:
            phi = feature_matrix(feats[i:i + 1], cont[i:i + 1], np.array([corr]))
            mask = np.ones((1, N_ACT), dtype=bool)
            if corr != 2:
                mask[0, I_CORR] = False
            p = actor.probs(phi, group[i:i + 1], mask)
            cache[key] = ACTIONS[int(p.argmax())]
        return cache[key]
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    cfg = load_config(); ensure_dirs()
    seeds = a.seeds if a.seeds else cfg["model_seeds"]

    tx = pd.read_csv(PROC / "transactions.csv")
    splits = pd.read_csv(PROC / "splits.csv")
    tr_end = int((splits.Split == "train").sum())
    va_end = tr_end + int((splits.Split == "val").sum())

    fault_prob = C.upstream_reliability_estimate(tr_end)
    bench = pd.read_csv(PROC / "integrated_benchmark.csv",
                        usecols=["Step", "UnifiedNodeID", "HardwareScore"])
    hw_node = bench.pivot(index="Step", columns="UnifiedNodeID", values="HardwareScore").to_numpy()
    feats_all, fp_all = C.discretize(tx, fault_prob, hw_node)
    rel_all = hw_node[tx.Step.values, tx.NodeIndex.values]
    cont_all = continuous_features(tx, fp_all, rel_all)

    tr_mask = tx.Step.values < tr_end
    te_mask = tx.Step.values >= va_end

    path = TAB / "exp_e_policy_gradient.csv"
    rows, done = [], set()
    if path.exists():
        prev = pd.read_csv(path)
        rows = prev.to_dict("records")
        done = set(zip(prev.policy, prev.seed))

    hists = []
    for seed in seeds:
        set_seed(seed)
        env_tr = C.Env(tx[tr_mask].reset_index(drop=True), feats_all[tr_mask], cfg,
                       np.random.default_rng(seed))
        env_te = C.Env(tx[te_mask].reset_index(drop=True), feats_all[te_mask], cfg,
                       np.random.default_rng(seed + 5000))
        f_tr, f_te = feats_all[tr_mask], feats_all[te_mask]
        c_tr, c_te = cont_all[tr_mask], cont_all[te_mask]

        for label, dec, budget in [
                ("pg_decentralized_constrained", True, 0.05),
                ("pg_decentralized_utility_max", True, None),
                ("pg_centralized_constrained", False, 0.05)]:
            if (label, seed) in done:
                continue
            t0 = time.time()
            actor, _, hist = train_actor(env_tr, f_tr, c_tr, cfg, seed,
                                         decentralized=dec, fairness_budget=budget,
                                         latency_budget=cfg["policy"]["latency_budget_ms"],
                                         epochs=a.epochs)
            g_te = f_te[:, 0].astype(int) if dec else np.zeros(len(f_te), dtype=int)
            res = C.run_policy(env_te, greedy_fn(actor, f_te, c_te, g_te), list(ACTIONS))
            row = C.summarize(env_te, res, label, seed,
                              extra={"train_seconds": round(time.time() - t0, 2),
                                     "representation": "linear_function_approximation",
                                     "n_parameters": int(actor.w.size + actor.v.size)})
            rows.append(row)
            pd.DataFrame(rows).to_csv(path, index=False)
            if not hist.empty:
                hist["policy"] = label; hist["seed"] = seed
                hists.append(hist)
            print(f"  {label} seed {seed} revenue={row['revenue_recovered_pct']:.2f} "
                  f"gap={row['equal_opportunity_gap']:.4f} ({row['train_seconds']:.0f}s)",
                  flush=True)

    if hists:
        pd.concat(hists).to_csv(TAB / "exp_e_dual_variables.csv", index=False)

    # Cost sensitivity. The headline run converges on blanket acceptance because
    # the latent validity rate is high and the penalty for a false accept is
    # only twice the fare. This sweep raises that penalty and records the point
    # at which selective actions start to pay for themselves.
    sweep_path = TAB / "exp_e_cost_sensitivity.csv"
    srows = []
    for mult in [2.0, 5.0, 10.0, 20.0, 40.0]:
        cfg_m = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
        cfg_m["policy"]["false_accept_multiplier"] = mult
        for seed in seeds[:3]:
            set_seed(seed)
            env_tr = C.Env(tx[tr_mask].reset_index(drop=True), feats_all[tr_mask], cfg_m,
                           np.random.default_rng(seed))
            env_te = C.Env(tx[te_mask].reset_index(drop=True), feats_all[te_mask], cfg_m,
                           np.random.default_rng(seed + 5000))
            actor, _, _ = train_actor(env_tr, feats_all[tr_mask], cont_all[tr_mask], cfg_m,
                                      seed, decentralized=True, fairness_budget=0.05,
                                      latency_budget=cfg_m["policy"]["latency_budget_ms"],
                                      epochs=a.epochs)
            g_te = feats_all[te_mask][:, 0].astype(int)
            res = C.run_policy(env_te, greedy_fn(actor, feats_all[te_mask],
                                                 cont_all[te_mask], g_te), list(ACTIONS))
            srows.append(C.summarize(env_te, res, "pg_decentralized_constrained", seed,
                                     extra={"false_accept_multiplier": mult}))
            pd.DataFrame(srows).to_csv(sweep_path, index=False)
        m = pd.DataFrame(srows)
        m = m[m.false_accept_multiplier == mult]
        print(f"  multiplier {mult:>5}: accept={m.frac_accept.mean():.3f} "
              f"abstain={m.abstention_rate.mean():.3f} "
              f"revenue={m.revenue_recovered_pct.mean():.2f}", flush=True)
    save_json({"note": "Linear softmax policy trained by REINFORCE with a learned value "
                       "baseline. Same constrained process, state information, actions, "
                       "reward, dual ascent, split and seeds as Experiment C.",
               "n_features": int(sum(DIMS) + N_CORR + cont_all.shape[1] + 1),
               "epochs": a.epochs, "seeds": list(seeds)},
              TAB / "exp_e_meta.json")
    print("Experiment E complete")


if __name__ == "__main__":
    main()
