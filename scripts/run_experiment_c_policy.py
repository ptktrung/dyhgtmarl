"""Experiment C: fairness aware decentralized decision policies.

The environment is the graph linked transaction stream. Every transaction has a
latent validity label that the policy cannot see, so accept, reject, impute,
corroborate, defer, isolate and escalate all carry real consequences that can
be scored. Constraints on fairness, decision latency and risk are enforced with
a dual ascent scheme rather than as post hoc filtering.

Per action latency values are modelling assumptions of the simulator and are
labelled as such. Measured wall clock decision time is reported separately and
is a genuine measurement of this implementation on the machine that ran it.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from dyhgt import metrics as M
from dyhgt.utils import PROC, TAB, ensure_dirs, load_config, save_json, set_seed

ACTIONS = ["accept", "reject", "impute", "corroborate", "defer", "isolate", "escalate"]
# nominal per decision latency in milliseconds, a simulator assumption
ACTION_LATENCY_MS = {"accept": 0.4, "reject": 0.4, "impute": 1.2, "corroborate": 2.0,
                     "defer": 0.6, "isolate": 0.5, "escalate": 0.8}
TERMINAL = {"accept", "reject", "impute", "defer", "isolate", "escalate"}


# ---------------------------------------------------------------------------
def upstream_reliability_estimate(train_end, source: str = "logistic"):
    """Node fault probability supplied to the decision layer.

    Two sources are available. The logistic surrogate is fitted on training
    telemetry alone and keeps the decision layer independent of the encoder.
    The dyhgt source reads the calibrated fault probability produced by the
    trained multi-task encoder, which couples the two halves of the system as
    the architecture claims. Neither sees the ground truth fault label.
    """
    if source == "dyhgt":
        grid = PROC / "dyhgt_fault_probability.npy"
        if grid.exists():
            return np.load(grid)
        raise FileNotFoundError(
            f"{grid} not found. Run scripts/export_fault_probability.py first.")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    sig = np.load(PROC / "signal.npy"); lat = np.load(PROC / "latency.npy")
    dri = np.load(PROC / "drift.npy"); flt = (np.load(PROC / "fault_true.npy") > 0).astype(int)
    X = np.stack([sig, np.log1p(lat), dri], axis=-1)
    Xtr = X[:train_end].reshape(-1, 3); ytr = flt[:train_end].ravel()
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300,
                                                             class_weight="balanced")).fit(Xtr, ytr)
    p = clf.predict_proba(X.reshape(-1, 3))[:, 1].reshape(flt.shape)
    return p


def discretize(tx: pd.DataFrame, fault_prob: np.ndarray, hw_node: np.ndarray):
    hw_b = np.digitize(tx.HardwareScore.values, [0.30, 0.60, 0.85])
    fp = fault_prob[tx.Step.values, tx.NodeIndex.values]
    fp_b = np.digitize(fp, [0.25, 0.60])
    rel = hw_node[tx.Step.values, tx.NodeIndex.values]
    rel_b = np.digitize(rel, [0.55, 0.75])
    zone_b = (tx.Zone.values == "Periphery").astype(int)
    err_b = tx.IsError.values.astype(int)
    return np.stack([zone_b, err_b, hw_b, fp_b, rel_b], axis=1).astype(int), fp


STATE_DIMS = (2, 2, 4, 3, 3, 3)   # last dim is the corroboration signal, 2 means unseen


def state_index(feat_row, corr=2):
    idx = 0
    for v, d in zip(list(feat_row) + [corr], STATE_DIMS):
        idx = idx * d + int(v)
    return idx


N_STATES = int(np.prod(STATE_DIMS))


# ---------------------------------------------------------------------------
class Env:
    def __init__(self, tx, feats, cfg, rng):
        self.tx = tx
        self.feats = feats
        self.cfg = cfg
        self.rng = rng
        self.fare = tx.Fare.values.astype(float)
        self.valid = tx.TrueValid.values.astype(bool)
        self.hw = tx.HardwareScore.values.astype(float)
        self.zone = tx.Zone.values
        pc = cfg["policy"]
        self.review = float(pc["review_cost_vnd"])
        self.corr_cost = float(pc["corroborate_cost_vnd"])
        # the stress regime sets a per record multiplier, so the cost of a
        # false accept differs by zone
        self.fa = float(pc["false_accept_multiplier"])
        if "FalseAcceptMultiplier" in tx.columns:
            base = tx["FalseAcceptMultiplier"].to_numpy(dtype=float)
            # the cost sensitivity sweep varies the configured multiplier, so the
            # per record vector is scaled by the same factor and keeps its
            # zone structure instead of being overridden by it
            self.fa_vec = base * (self.fa / float(base.mean()))
        else:
            self.fa_vec = np.full(len(tx), self.fa)

    def corroboration(self, i):
        """A second noisy observation of validity, accurate when hardware is good."""
        acc = 0.55 + 0.40 * self.hw[i]
        signal = self.valid[i] if self.rng.random() < acc else (not self.valid[i])
        return int(signal)

    def outcome(self, i, action):
        """Return revenue, integrity error flag, review flag, accepted flag."""
        fare, valid, hw = self.fare[i], self.valid[i], self.hw[i]
        fa = self.fa_vec[i]
        if action == "accept":
            return (fare if valid else -fa * fare), (0 if valid else 1), 0, 1
        if action == "reject":
            return 0.0, (1 if valid else 0), 0, 0
        if action == "impute":
            ok = self.rng.random() < min(0.35 + 0.65 * hw, 0.99)
            if not ok:
                return -0.25 * fare, 1, 0, 1
            return (fare if valid else -fa * fare), (0 if valid else 1), 0, 1
        if action == "defer":
            return (fare - self.review if valid else -self.review), 0, 1, int(valid)
        if action == "escalate":
            return (fare - 1.6 * self.review if valid else -1.6 * self.review), 0, 1, int(valid)
        if action == "isolate":
            return 0.0, (1 if valid else 0), 0, 0
        raise ValueError(action)


def run_policy(env: Env, policy_fn, allowed, track=True):
    n = len(env.tx)
    rev = np.zeros(n); err = np.zeros(n, dtype=int); rev_flag = np.zeros(n, dtype=int)
    acc = np.zeros(n, dtype=int); lat = np.zeros(n); acts = np.empty(n, dtype=object)
    t0 = time.perf_counter()
    for i in range(n):
        corr = 2
        total_lat = 0.0
        while True:
            a = policy_fn(i, corr, allowed)
            total_lat += ACTION_LATENCY_MS[a]
            if a == "corroborate" and corr == 2:
                corr = env.corroboration(i)
                continue
            if a == "corroborate":
                a = "defer"
                total_lat += ACTION_LATENCY_MS["defer"]
            r, e, rv, ac = env.outcome(i, a)
            if corr != 2:
                r -= env.corr_cost
            rev[i], err[i], rev_flag[i], acc[i], lat[i], acts[i] = r, e, rv, ac, total_lat, a
            break
    wall = (time.perf_counter() - t0) / n * 1000.0
    return {"revenue": rev, "integrity_error": err, "reviewed": rev_flag,
            "accepted": acc, "latency_ms": lat, "action": acts,
            "wall_ms_per_decision": wall}


def summarize(env: Env, res: dict, label: str, seed: int, extra=None):
    fare_total = env.fare.sum()
    accepted = res["accepted"].astype(bool)
    rates = M.group_rates(accepted.astype(float), env.zone, env.valid)
    gaps = M.fairness_gaps(rates)
    out = {
        "policy": label, "seed": seed,
        "revenue_recovered_pct": float(res["revenue"].sum() / fare_total * 100.0),
        "gross_revenue_pct": float(env.fare[accepted & env.valid].sum() / fare_total * 100.0),
        "decision_error_rate": float(res["integrity_error"].mean()),
        "false_accept_rate": float((accepted & ~env.valid).mean()),
        "valid_rejection_rate": float((~accepted & env.valid).mean()),
        "abstention_rate": float(np.isin(res["action"], ["defer", "escalate", "isolate"]).mean()),
        "review_workload_pct": float(res["reviewed"].mean() * 100.0),
        "nominal_latency_ms": float(res["latency_ms"].mean()),
        "measured_wall_ms_per_decision": float(res["wall_ms_per_decision"]),
    }
    out.update(gaps)
    for z, r in rates.items():
        out[f"accept_rate_{z}"] = r["acceptance_rate"]
        out[f"tpr_{z}"] = r["true_positive_rate"]
    for a in ACTIONS:
        out[f"frac_{a}"] = float((res["action"] == a).mean())
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------------------
def rule_based_factory(env: Env, feats):
    def fn(i, corr, allowed):
        if env.tx.IsError.values[i] and env.hw[i] < 0.9:
            return "reject"
        return "accept"
    return fn


class QLearner:
    def __init__(self, n_agents, allowed, cfg, seed):
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((n_agents, N_STATES, len(ACTIONS)))
        self.allowed = [ACTIONS.index(a) for a in allowed]
        pc = cfg["policy"]
        self.alpha, self.gamma, self.eps = pc["alpha"], pc["gamma"], pc["epsilon"]

    def act(self, agent, s, greedy=False):
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.choice(self.allowed))
        qa = self.q[agent, s]
        mask = np.full(len(ACTIONS), -np.inf)
        mask[self.allowed] = qa[self.allowed]
        return int(np.argmax(mask))

    def update(self, agent, s, a, r, s2, done):
        target = r if done else r + self.gamma * self.q[agent, s2, self.allowed].max()
        self.q[agent, s, a] += self.alpha * (target - self.q[agent, s, a])


def train_q(env: Env, feats, cfg, seed, allowed, decentralized=True,
            fairness_budget=None, risk_budget=0.02, latency_budget=None, episodes=None):
    pc = cfg["policy"]
    episodes = episodes or pc["episodes"]
    n_agents = 2 if decentralized else 1
    agent_of = feats[:, 0] if decentralized else np.zeros(len(feats), dtype=int)
    ql = QLearner(n_agents, allowed, cfg, seed)
    rng = np.random.default_rng(seed + 991)
    lam_f = lam_r = lam_l = 0.0
    scale = float(np.median(env.fare))
    hist = []
    n = len(env.tx)
    batch = min(n, 6000)
    for ep in range(episodes):
        order = rng.permutation(n)[:batch]
        acc = np.zeros(n, dtype=bool); used = np.zeros(n, dtype=bool)
        lat_sum, risk_sum = 0.0, 0.0
        for i in order:
            corr = 2
            s = state_index(feats[i], corr)
            steps = 0
            while True:
                a = ql.act(int(agent_of[i]), s)
                name = ACTIONS[a]
                lat_sum += ACTION_LATENCY_MS[name]
                if name == "corroborate" and corr == 2 and steps == 0:
                    corr = env.corroboration(i)
                    s2 = state_index(feats[i], corr)
                    ql.update(int(agent_of[i]), s, a, -env.corr_cost / scale, s2, False)
                    s, steps = s2, steps + 1
                    continue
                if name == "corroborate":
                    name = "defer"
                    a = ACTIONS.index("defer")
                r, e, rv, ac = env.outcome(i, name)
                acc[i], used[i] = bool(ac), True
                risk = 1.0 if (ac and not env.valid[i]) else 0.0
                risk_sum += risk
                shaped = r / scale - lam_r * risk - lam_l * ACTION_LATENCY_MS[name] / 10.0
                if fairness_budget is not None:
                    # per decision fairness surrogate: reward accepting valid
                    # records from the group currently below the other group
                    grp = int(feats[i, 0])
                    shaped -= lam_f * ((1 - ac) if grp == 1 else 0.0) * 0.5
                    shaped -= lam_f * (ac if grp == 0 else 0.0) * 0.05
                ql.update(int(agent_of[i]), s, a, shaped, 0, True)
                break
        if fairness_budget is not None:
            sel = used
            gap = M.fairness_gaps(M.group_rates(acc[sel].astype(float), env.zone[sel],
                                                env.valid[sel]))["equal_opportunity_gap"]
            lam_f = max(0.0, lam_f + pc["lagrange_lr"] * (gap - fairness_budget))
            lam_r = max(0.0, lam_r + pc["lagrange_lr"] * (risk_sum / batch - risk_budget))
            if latency_budget:
                lam_l = max(0.0, lam_l + 0.1 * (lat_sum / batch - latency_budget))
            hist.append({"episode": ep, "lambda_fairness": lam_f, "lambda_risk": lam_r,
                         "lambda_latency": lam_l, "equal_opportunity_gap": gap})
    return ql, agent_of, pd.DataFrame(hist)


def greedy_factory(ql, agent_of, feats):
    def fn(i, corr, allowed):
        s = state_index(feats[i], corr)
        return ACTIONS[ql.act(int(agent_of[i]), s, greedy=True)]
    return fn


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--fault-source", choices=["logistic", "dyhgt"], default="logistic",
                    dest="fault_source",
                    help="where the decision layer gets its upstream fault probability")
    a = ap.parse_args()
    cfg = load_config(); ensure_dirs()
    seeds = a.seeds if a.seeds else cfg["model_seeds"]
    tx = pd.read_csv(PROC / "transactions.csv")
    splits = pd.read_csv(PROC / "splits.csv")
    tr_end = int((splits.Split == "train").sum())
    va_end = tr_end + int((splits.Split == "val").sum())

    fault_prob = upstream_reliability_estimate(tr_end, a.fault_source)
    bench = pd.read_csv(PROC / "integrated_benchmark.csv",
                        usecols=["Step", "UnifiedNodeID", "HardwareScore"])
    hw_node = bench.pivot(index="Step", columns="UnifiedNodeID", values="HardwareScore").to_numpy()

    feats_all, fp_all = discretize(tx, fault_prob, hw_node)
    tr_mask = tx.Step.values < tr_end
    te_mask = tx.Step.values >= va_end

    # the coupled variant writes to its own file so the two fault probability
    # sources can be compared rather than overwriting one another
    sfx = "" if a.fault_source == "logistic" else f"_{a.fault_source}"
    out_policy = TAB / f"exp_c_policy{sfx}.csv"
    out_pareto = TAB / f"exp_c_pareto{sfx}.csv"
    rows, pareto_rows, dual_hist = [], [], []
    done_seeds = set()
    if out_policy.exists():
        prev = pd.read_csv(out_policy)
        rows = prev.to_dict("records"); done_seeds = set(prev.seed.unique().tolist())
    if out_pareto.exists():
        pareto_rows = pd.read_csv(out_pareto).to_dict("records")
    print("policy seeds already done:", sorted(done_seeds), flush=True)
    full = set(ACTIONS)
    no_abstain = [x for x in ACTIONS if x not in ("defer", "escalate", "isolate")]

    for seed in seeds:
        if seed in done_seeds:
            continue
        set_seed(seed)
        rng = np.random.default_rng(seed)
        env_tr = Env(tx[tr_mask].reset_index(drop=True), feats_all[tr_mask], cfg, rng)
        env_te = Env(tx[te_mask].reset_index(drop=True), feats_all[te_mask], cfg,
                     np.random.default_rng(seed + 5000))
        f_tr, f_te = feats_all[tr_mask], feats_all[te_mask]

        res = run_policy(env_te, rule_based_factory(env_te, f_te), list(full))
        rows.append(summarize(env_te, res, "rule_based_legacy", seed))

        configs = [
            ("centralized_utility_max", False, None, list(ACTIONS)),
            ("centralized_constrained", False, 0.05, list(ACTIONS)),
            ("decentralized_constrained", True, 0.05, list(ACTIONS)),
            ("decentralized_constrained_no_abstention", True, 0.05, no_abstain),
            ("decentralized_utility_max", True, None, list(ACTIONS)),
        ]
        for label, dec, budget, allowed in configs:
            ql, agent_of_tr, hist = train_q(env_tr, f_tr, cfg, seed, allowed,
                                            decentralized=dec, fairness_budget=budget,
                                            latency_budget=cfg["policy"]["latency_budget_ms"],
                                            episodes=a.episodes)
            agent_of_te = f_te[:, 0] if dec else np.zeros(len(f_te), dtype=int)
            res = run_policy(env_te, greedy_factory(ql, agent_of_te, f_te), allowed)
            rows.append(summarize(env_te, res, label, seed))
            if not hist.empty:
                hist["policy"] = label; hist["seed"] = seed
                dual_hist.append(hist)

        # fairness utility trade off curve
        for eps_f in cfg["policy"]["fairness_budgets"]:
            ql, _, _ = train_q(env_tr, f_tr, cfg, seed, list(ACTIONS), decentralized=True,
                               fairness_budget=eps_f, episodes=a.episodes)
            res = run_policy(env_te, greedy_factory(ql, f_te[:, 0], f_te), list(ACTIONS))
            s = summarize(env_te, res, "decentralized_constrained", seed,
                          extra={"fairness_budget": eps_f})
            pareto_rows.append(s)
        for r in rows[-7:]:
            r.setdefault("fault_source", a.fault_source)
        pd.DataFrame(rows).to_csv(out_policy, index=False)
        pd.DataFrame(pareto_rows).to_csv(out_pareto, index=False)
        print(f"  policy seed {seed} done", flush=True)

    pd.DataFrame(rows).to_csv(out_policy, index=False)
    pd.DataFrame(pareto_rows).to_csv(out_pareto, index=False)
    if dual_hist:
        pd.concat(dual_hist).to_csv(TAB / "exp_c_dual_variables.csv", index=False)
    save_json({"n_transactions_total": int(len(tx)),
               "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
               "true_valid_rate": float(tx.TrueValid.mean()),
               "flag_rate": float(tx.IsError.mean()),
               "fault_probability_source": a.fault_source,
               "note": "Per action latency values are simulator assumptions, not hardware measurements."},
              TAB / "exp_c_meta.json")
    print("Experiment C complete")


if __name__ == "__main__":
    main()
