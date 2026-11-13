"""Multi-arm heterogeneous-treatment-effect estimators.

All estimators share one API:
    est.fit(X, A, Y, propensity=None)       X DataFrame, A int arm index (0 = no_action), Y renewed (0/1)
    est.predict_p(X) -> (n, K)              estimated P(renew | x, do(arm)) for every arm
    tau_hat = est.predict_p(X) - est.predict_p(X)[:, [0]]

Estimators
    NaiveMeans   - per-arm mean of Y (what you get if you ignore confounding)           [BIASED on logged data]
    SLearner     - one outcome model with the arm as a feature
    TLearner     - one outcome model per arm
    DRLearner    - doubly-robust pseudo-outcomes (cross-fitted) + final regression      [own implementation]
    EconMLDR     - same idea via Microsoft EconML (optional dependency)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import K


def _clf(seed=0):
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=150, min_samples_leaf=30,
                                          l2_regularization=1.0, random_state=seed)


def _reg(seed=0):
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=60,
                                         l2_regularization=2.0, random_state=seed)


def _final_reg(seed=0):
    """Final stage smooths very noisy pseudo-outcomes -> shallow, heavily regularised trees."""
    return HistGradientBoostingRegressor(max_depth=2, learning_rate=0.03, max_iter=120, min_samples_leaf=200,
                                         l2_regularization=5.0, random_state=seed)


def _lr_prop():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))


class NaiveMeans:
    name = "Naive (diff. in means)"

    def fit(self, X, A, Y, propensity=None):
        self.m_ = np.array([Y[A == a].mean() if (A == a).any() else np.nan for a in range(K)])
        return self

    def predict_p(self, X):
        return np.tile(self.m_, (len(X), 1))


class SLearner:
    name = "S-learner"

    def __init__(self, seed=0):
        self.seed = seed

    def _aug(self, X, a):
        Z = np.asarray(X, float)
        oh = np.zeros((len(Z), K))
        oh[:, a] = 1
        return np.hstack([Z, oh])

    def fit(self, X, A, Y, propensity=None):
        Z = np.asarray(X, float)
        oh = np.eye(K)[A]
        self.m_ = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250, min_samples_leaf=30,
                                                 l2_regularization=1.0, random_state=self.seed).fit(np.hstack([Z, oh]), Y)
        return self

    def predict_p(self, X):
        return np.column_stack([self.m_.predict_proba(self._aug(X, a))[:, 1] for a in range(K)])


class TLearner:
    name = "T-learner"

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, A, Y, propensity=None):
        Z = np.asarray(X, float)
        self.m_ = [_clf(self.seed).fit(Z[A == a], Y[A == a]) for a in range(K)]
        return self

    def predict_p(self, X):
        Z = np.asarray(X, float)
        return np.column_stack([m.predict_proba(Z)[:, 1] for m in self.m_])


class DRLearner:
    """Cross-fitted doubly-robust learner.
    1. out-of-fold outcome models mu_a(x) = P(Y=1 | x, A=a)  and propensity e_a(x) = P(A=a | x)
    2. pseudo-outcome  phi_a = mu_a(x) + 1[A=a] (Y - mu_a(x)) / e_a(x)       (E[phi_a | x] = P(renew | x, do(a)))
    3. regress phi_a on x  ->  p_hat_a(x).    Consistent if EITHER the outcome OR the propensity model is right."""
    name = "DR-learner (ours)"

    def __init__(self, n_splits=5, clip=0.05, seed=0):
        self.n_splits, self.clip, self.seed = n_splits, clip, seed

    def fit(self, X, A, Y, propensity=None):
        Z = np.asarray(X, float)
        n = len(Z)
        mu = np.zeros((n, K))
        e = np.zeros((n, K))
        skf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed)
        strat = A * 2 + Y
        for tr, te in skf.split(Z, strat):
            for a in range(K):
                idx = tr[A[tr] == a]
                mu[te, a] = _clf(self.seed).fit(Z[idx], Y[idx]).predict_proba(Z[te])[:, 1]
            if propensity is None:
                pm = _lr_prop().fit(Z[tr], A[tr])
                e[te] = pm.predict_proba(Z[te])
        if propensity is not None:
            e = np.asarray(propensity, float)
        e = np.clip(e, self.clip, 1.0)
        ind = np.eye(K)[A]
        phi = mu + ind * (Y[:, None] - mu) / e
        self.phi_, self.e_, self.mu_ = phi, e, mu
        # final stage: smooth the noisy pseudo-outcomes into p_hat_a(x); effects are modelled as difference to arm 0
        self.base_ = _final_reg(self.seed).fit(Z, phi[:, 0])
        self.tau_ = [None] + [_final_reg(self.seed + a).fit(Z, phi[:, a] - phi[:, 0]) for a in range(1, K)]
        return self

    def predict_p(self, X):
        Z = np.asarray(X, float)
        p0 = np.clip(self.base_.predict(Z), 0.02, 0.98)
        out = [p0] + [np.clip(p0 + self.tau_[a].predict(Z), 0.01, 0.99) for a in range(1, K)]
        return np.column_stack(out)


class EconMLDR:
    """Thin wrapper around econml.dr.DRLearner (multiple discrete treatments)."""
    name = "EconML DRLearner"

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, A, Y, propensity=None):
        from econml.dr import DRLearner as _DR
        Z = np.asarray(X, float)
        self.base_ = _clf(self.seed).fit(Z[A == 0], Y[A == 0])
        self.m_ = _DR(model_propensity=_lr_prop(), model_regression=_reg(self.seed),
                      model_final=_final_reg(self.seed), cv=5, random_state=self.seed)
        self.m_.fit(Y, A, X=Z)
        return self

    def predict_p(self, X):
        Z = np.asarray(X, float)
        p0 = np.clip(self.base_.predict_proba(Z)[:, 1], 0.02, 0.98)
        cols = [p0]
        for a in range(1, K):
            cols.append(np.clip(p0 + self.m_.effect(Z, T0=0, T1=a), 0.01, 0.99))
        return np.column_stack(cols)


def available_estimators(seed=0, include_econml=True) -> dict:
    est = {"naive": NaiveMeans(), "s_learner": SLearner(seed), "t_learner": TLearner(seed), "dr_learner": DRLearner(seed=seed)}
    if include_econml:
        try:
            import econml  # noqa: F401
            est["econml_dr"] = EconMLDR(seed)
        except Exception:
            pass
    return est


def crossfit_predict_p(factory, X: pd.DataFrame, A, Y, n_splits=4, seed=0):
    """Out-of-fold p_hat for every logged row - needed for off-policy evaluation without in-sample optimism."""
    Z = X
    out = np.zeros((len(Z), K))
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(Z, A * 2 + Y):
        m = factory().fit(Z.iloc[tr], A[tr], Y[tr])
        out[te] = m.predict_p(Z.iloc[te])
    return out
