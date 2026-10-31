"""Metrics used across the four experiments.

Probabilistic scores are computed in closed form for a Gaussian and for a
Gaussian mixture so that deterministic baselines and the mixture density head
are scored on the same footing. Deterministic baselines are given a Gaussian
predictive distribution whose variance is fitted on the validation residuals,
which is fairer than letting CRPS collapse onto the absolute error.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Point forecasting
# ---------------------------------------------------------------------------
def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mape(y, p, floor=1.0):
    d = np.maximum(np.abs(y), floor)
    return float(np.mean(np.abs((y - p) / d)) * 100.0)


# ---------------------------------------------------------------------------
# Probabilistic forecasting
# ---------------------------------------------------------------------------
def crps_gaussian(y, mu, sigma):
    sigma = np.maximum(sigma, 1e-6)
    z = (y - mu) / sigma
    return float(np.mean(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def _a_term(mu, sigma):
    sigma = np.maximum(sigma, 1e-6)
    z = mu / sigma
    return mu * (2 * norm.cdf(z) - 1) + 2 * sigma * norm.pdf(z)


def crps_mixture(y, pi, mu, sigma):
    """Closed form CRPS of a Gaussian mixture, Grimit et al. form."""
    y = y[..., None]
    first = np.sum(pi * _a_term(y - mu, sigma), axis=-1)
    m_i = mu[..., :, None]
    m_j = mu[..., None, :]
    s_ij = np.sqrt(sigma[..., :, None] ** 2 + sigma[..., None, :] ** 2)
    pair = np.sum(pi[..., :, None] * pi[..., None, :] * _a_term(m_i - m_j, s_ij), axis=(-2, -1))
    return float(np.mean(first - 0.5 * pair))


def mixture_cdf(y, pi, mu, sigma):
    return np.sum(pi * norm.cdf((y[..., None] - mu) / np.maximum(sigma, 1e-6)), axis=-1)


def mixture_quantile(qlevel, pi, mu, sigma, lo=-50.0, hi=250.0, iters=60):
    a = np.full(pi.shape[:-1], lo)
    b = np.full(pi.shape[:-1], hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        c = mixture_cdf(m, pi, mu, sigma)
        a = np.where(c < qlevel, m, a)
        b = np.where(c < qlevel, b, m)
    return 0.5 * (a + b)


def picp(y, lower, upper):
    return float(np.mean((y >= lower) & (y <= upper)))


def interval_width(lower, upper):
    return float(np.mean(upper - lower))


def calibration_error(pit_values, n_bins=20):
    """Expected calibration error for a probabilistic regression forecast.

    Uses the probability integral transform. A perfectly calibrated forecast
    has uniform PIT values, so the score is the mean absolute gap between
    nominal and empirical coverage across a grid of quantile levels.
    """
    levels = np.linspace(0.05, 0.95, n_bins)
    emp = np.array([np.mean(pit_values <= q) for q in levels])
    return float(np.mean(np.abs(emp - levels)))


def pit_gaussian(y, mu, sigma):
    return norm.cdf((y - mu) / np.maximum(sigma, 1e-6))


def pit_mixture(y, pi, mu, sigma):
    return mixture_cdf(y, pi, mu, sigma)


# ---------------------------------------------------------------------------
# Classification and detection
# ---------------------------------------------------------------------------
def classification_scores(y, prob, threshold=0.5):
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 precision_recall_fscore_support)
    pred = (prob >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    out = {"precision": float(p), "recall": float(r), "f1": float(f),
           "pr_auc": float(average_precision_score(y, prob)) if y.sum() else float("nan"),
           "roc_auc": float(roc_auc_score(y, prob)) if 0 < y.sum() < len(y) else float("nan")}
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    out["false_alarm_rate"] = float(fp / max(tn + fp, 1))
    out["n_false_alarms"] = fp
    out["n_missed"] = int(((pred == 0) & (y == 1)).sum())
    return out


def detection_delay(prob_grid, event_starts, threshold=0.5, horizon=12):
    """Steps between the true onset of an event and the first alarm.

    Events never raised within the horizon are counted as misses and reported
    separately rather than being silently dropped from the mean.
    """
    delays, missed = [], 0
    for (t0, node) in event_starts:
        window = prob_grid[t0:t0 + horizon, node]
        hit = np.where(window >= threshold)[0]
        if len(hit):
            delays.append(int(hit[0]))
        else:
            missed += 1
    return {"mean_delay_steps": float(np.mean(delays)) if delays else float("nan"),
            "median_delay_steps": float(np.median(delays)) if delays else float("nan"),
            "detected": len(delays), "undetected": missed}


def topk_accuracy(scores, true_idx, k=1):
    order = np.argsort(-scores, axis=1)[:, :k]
    return float(np.mean([t in order[i] for i, t in enumerate(true_idx)]))


# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------
def group_rates(decisions_accept, group, valid):
    out = {}
    for g in np.unique(group):
        sel = group == g
        out[g] = {
            "acceptance_rate": float(decisions_accept[sel].mean()),
            "true_positive_rate": float(decisions_accept[sel & valid].mean()) if (sel & valid).sum() else float("nan"),
            "false_positive_rate": float(decisions_accept[sel & ~valid].mean()) if (sel & ~valid).sum() else float("nan"),
            "n": int(sel.sum()),
        }
    return out


def fairness_gaps(rates):
    keys = list(rates)
    def gap(field):
        vals = [rates[k][field] for k in keys]
        return float(max(vals) - min(vals))
    return {"demographic_parity_gap": gap("acceptance_rate"),
            "equal_opportunity_gap": gap("true_positive_rate"),
            "false_positive_rate_gap": gap("false_positive_rate")}


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return (float("nan"), float("nan"))
    draws = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return (float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2)))
