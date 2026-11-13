"""Ground-truth evaluation of causal estimates (possible because the synthetic generator knows the truth)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..config import ARMS, K


def causal_metrics(P_hat: np.ndarray, P_true: np.ndarray) -> pd.DataFrame:
    """Per intervention: ATE error, PEHE, rank correlation and uplift-ranking quality."""
    tau_h, tau_t = P_hat - P_hat[:, [0]], P_true - P_true[:, [0]]
    rows = []
    for a in range(1, K):
        rho = spearmanr(tau_h[:, a], tau_t[:, a]).statistic if tau_h[:, a].std() > 1e-9 else 0.0
        rows.append({
            "arm": ARMS[a], "ate_true": tau_t[:, a].mean(), "ate_hat": tau_h[:, a].mean(),
            "ate_abs_error": abs(tau_h[:, a].mean() - tau_t[:, a].mean()),
            "pehe": float(np.sqrt(np.mean((tau_h[:, a] - tau_t[:, a]) ** 2))),
            "pehe_constant_ate_baseline": float(np.std(tau_t[:, a])),      # error if you predicted the true ATE for everyone
            "spearman_tau": float(rho), "uplift_ratio": uplift_ratio(tau_h[:, a], tau_t[:, a]),
            "p0_rmse": float(np.sqrt(np.mean((P_hat[:, 0] - P_true[:, 0]) ** 2))),
        })
    return pd.DataFrame(rows).set_index("arm")


def uplift_curve(score: np.ndarray, tau_true: np.ndarray, n_points: int = 50):
    """Cumulative TRUE gain when treating the top-q share of tenants ranked by `score`."""
    order = np.argsort(-score)
    cum = np.cumsum(tau_true[order])
    q = np.linspace(0, 1, n_points + 1)[1:]
    idx = np.maximum((q * len(score)).astype(int) - 1, 0)
    return np.r_[0, q], np.r_[0, cum[idx]]


def uplift_ratio(score, tau_true) -> float:
    """Area under the (true) uplift curve of `score` relative to the perfect ordering, random ordering = 0."""
    q, g = uplift_curve(score, tau_true)
    _, g_best = uplift_curve(tau_true, tau_true)
    g_rand = q * tau_true.sum()
    num, den = np.trapezoid(g - g_rand, q), np.trapezoid(g_best - g_rand, q)
    return float(num / den) if den > 1e-12 else float("nan")


def cate_calibration(tau_hat: np.ndarray, tau_true: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Sort tenants by predicted effect, cut into deciles, compare mean predicted vs mean TRUE effect per decile."""
    df = pd.DataFrame({"hat": tau_hat, "true": tau_true})
    df["bin"] = pd.qcut(df["hat"].rank(method="first"), bins, labels=False)
    return df.groupby("bin").agg(mean_pred=("hat", "mean"), mean_true=("true", "mean"), n=("hat", "size")).reset_index()


def decision_agreement(nv_hat: np.ndarray, nv_true: np.ndarray, feas: np.ndarray) -> dict:
    a_hat = np.where(feas, nv_hat, -np.inf).argmax(1)
    a_true = np.where(feas, nv_true, -np.inf).argmax(1)
    n = len(a_hat)
    loss = nv_true[np.arange(n), a_true] - nv_true[np.arange(n), a_hat]
    return {"top1_agreement": float((a_hat == a_true).mean()), "mean_value_loss_per_tenant": float(loss.mean())}
