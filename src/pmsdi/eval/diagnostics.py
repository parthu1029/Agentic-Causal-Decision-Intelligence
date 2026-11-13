"""Checks you can run on REAL data too (no ground truth needed): overlap, balance, refutations, E-value."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from scipy import stats

from ..config import ARMS, K
from ..models.causal import _clf, _lr_prop


def propensity_oof(X: pd.DataFrame, A: np.ndarray, seed=0) -> np.ndarray:
    return cross_val_predict(_lr_prop(), np.asarray(X, float), A, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                             method="predict_proba")


def overlap_table(e: np.ndarray, A: np.ndarray) -> pd.DataFrame:
    rows = []
    for a in range(K):
        w = 1.0 / np.clip(e[A == a, a], 1e-3, 1)
        rows.append({"arm": ARMS[a], "n": int((A == a).sum()), "e_min": e[:, a].min(), "e_p1": np.quantile(e[:, a], 0.01),
                     "e_median": np.median(e[:, a]), "e_max": e[:, a].max(), "share_e_below_0.05": float((e[:, a] < 0.05).mean()),
                     "ess_ratio": float(w.sum() ** 2 / (w**2).sum() / len(w))})
    return pd.DataFrame(rows).set_index("arm")


def covariate_balance(X: pd.DataFrame, A: np.ndarray, e: np.ndarray, arm: int) -> pd.DataFrame:
    """Standardised mean difference of every covariate, treated(arm) vs control, raw and IPW-weighted."""
    m = (A == arm) | (A == 0)
    Z = X[m]
    t = (A[m] == arm)
    w = np.where(t, 1 / np.clip(e[m, arm], 0.02, 1), 1 / np.clip(e[m, 0], 0.02, 1))
    rows = []
    for c in X.columns:
        z = Z[c].to_numpy(float)
        def wm(x, ww): return np.average(x, weights=ww)
        def wv(x, ww): return np.average((x - wm(x, ww)) ** 2, weights=ww)
        sd = np.sqrt((z[t].var() + z[~t].var()) / 2) + 1e-12
        sdw = np.sqrt((wv(z[t], w[t]) + wv(z[~t], w[~t])) / 2) + 1e-12
        rows.append({"feature": c, "smd_raw": (z[t].mean() - z[~t].mean()) / sd,
                     "smd_ipw": (wm(z[t], w[t]) - wm(z[~t], w[~t])) / sdw})
    return pd.DataFrame(rows).set_index("feature")


def aipw_all_arms(X, A, Y, e=None, seed=0, n_splits=5, fast=True):
    """Cross-fitted AIPW (doubly-robust) estimate of ATE(a vs no_action) for every arm in one pass."""
    Z = np.asarray(X, float)
    n = len(Z)
    mu = np.zeros((n, K))
    ee = np.zeros((n, K)) if e is None else np.asarray(e)
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(Z, A * 2 + Y):
        for a in range(K):
            idx = tr[A[tr] == a]
            mu[te, a] = _clf(seed).fit(Z[idx], Y[idx]).predict_proba(Z[te])[:, 1]
        if e is None:
            ee[te] = _lr_prop().fit(Z[tr], A[tr]).predict_proba(Z[te])
    ee = np.clip(ee, 0.03, 1)
    ind = np.eye(K)[A]
    phi = mu + ind * (Y[:, None] - mu) / ee
    out = []
    for a in range(1, K):
        psi = phi[:, a] - phi[:, 0]
        se = psi.std(ddof=1) / np.sqrt(n)
        out.append({"arm": ARMS[a], "ate": psi.mean(), "se": se, "ci_lo": psi.mean() - 1.96 * se, "ci_hi": psi.mean() + 1.96 * se,
                    "p_arm": phi[:, a].mean(), "p_control": phi[:, 0].mean()})
    return pd.DataFrame(out).set_index("arm")


def naive_ate(A, Y) -> pd.Series:
    m0 = Y[A == 0].mean()
    return pd.Series({ARMS[a]: Y[A == a].mean() - m0 for a in range(1, K)})


def rct_benchmark(A, Y, is_rct: np.ndarray) -> pd.DataFrame:
    """Difference-in-means inside the randomised pilot: an assumption-free reference for the observational estimate."""
    a, y = A[is_rct], Y[is_rct]
    rows = []
    for k in range(1, K):
        y1, y0 = y[a == k], y[a == 0]
        d = y1.mean() - y0.mean()
        se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
        rows.append({"arm": ARMS[k], "ate_rct": d, "se_rct": se, "ci_lo_rct": d - 1.96 * se, "ci_hi_rct": d + 1.96 * se})
    return pd.DataFrame(rows).set_index("arm")


def refutation_suite(X, A, Y, n_placebo=8, n_subset=8, seed=0) -> pd.DataFrame:
    """Three DoWhy-style refutation tests implemented with our AIPW estimator.
       placebo  : shuffle the action labels -> estimated effect must collapse to ~0
       random   : add a pure-noise covariate -> estimate must not move
       subset   : re-estimate on random 80 % subsamples -> estimate must be stable"""
    rng = np.random.default_rng(seed)
    base = aipw_all_arms(X, A, Y, seed=seed)
    rows = []
    placebo = np.array([aipw_all_arms(X, rng.permutation(A), Y, seed=seed + i + 1)["ate"].to_numpy() for i in range(n_placebo)])
    Xn = X.copy()
    Xn["__noise__"] = rng.normal(size=len(X))
    rnd = aipw_all_arms(Xn, A, Y, seed=seed)
    sub = []
    for i in range(n_subset):
        idx = rng.choice(len(X), int(0.8 * len(X)), replace=False)
        sub.append(aipw_all_arms(X.iloc[idx], A[idx], Y[idx], seed=seed + i)["ate"].to_numpy())
    sub = np.array(sub)
    for j, arm in enumerate(ARMS[1:]):
        t_stat = placebo[:, j].mean() / (placebo[:, j].std(ddof=1) / np.sqrt(len(placebo)) + 1e-12)
        rows.append({"arm": arm, "ate_original": base.loc[arm, "ate"],
                     "placebo_mean": placebo[:, j].mean(), "placebo_pass(|mean|<2*se_orig)": abs(placebo[:, j].mean()) < 2 * base.loc[arm, "se"],
                     "random_cause_ate": rnd.loc[arm, "ate"], "random_cause_pass(|diff|<se)": abs(rnd.loc[arm, "ate"] - base.loc[arm, "ate"]) < base.loc[arm, "se"],
                     "subset_mean": sub[:, j].mean(), "subset_sd": sub[:, j].std(ddof=1),
                     "subset_pass(orig within 2sd)": abs(sub[:, j].mean() - base.loc[arm, "ate"]) < 2 * sub[:, j].std(ddof=1) + 1e-9})
    return pd.DataFrame(rows).set_index("arm")


def e_value(p_arm: float, p_control: float, ci_lo: float | None = None) -> dict:
    """VanderWeele-Ding E-value on the risk-ratio scale: how strong would an UNMEASURED confounder have to be
    (association with both action and outcome) to fully explain away the estimated effect?"""
    def ev(rr):
        rr = max(rr, 1 / rr) if rr > 0 else 1.0
        return rr + np.sqrt(rr * (rr - 1)) if rr > 1 else 1.0
    rr = p_arm / p_control
    out = {"risk_ratio": rr, "e_value_point": ev(rr)}
    if ci_lo is not None:
        rr_lo = (p_control + ci_lo) / p_control if ci_lo > 0 else 1.0
        out["e_value_ci_bound"] = ev(rr_lo) if ci_lo > 0 else 1.0
    return out
