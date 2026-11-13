"""One reusable experiment: generate -> features -> fit estimators -> decide -> score against ground truth."""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import ARMS, K, Config
from .data.features import build_features
from .data.synthetic_pms import generate
from .decision import interventions as iv
from .decision import optimizer as opt
from .decision import policies as pol
from .eval import causal_eval, policy_eval
from .models import causal

warnings.filterwarnings("ignore")

POLICY_ORDER = ["No action", "Random feasible", "B1 Rule-based", "B2 Risk-only alert", "B3 Risk + rules",
                "A Causal (no constraints)", "B Causal + constraints", "Oracle (true effects)"]
CONSTRAINT_AWARE = {"B Causal + constraints", "Oracle (true effects)"}


def split_xy(tables, oracle, F):
    hist = (oracle["cohort"] == "history").to_numpy()
    Xh, Xc = F[hist], F[~hist]
    A = np.array([ARMS.index(a) for a in oracle.loc[hist, "logged_action"]])
    Y = tables["renewal_outcomes"].set_index("tenant_id").loc[Xh.index, "renewed"].to_numpy().astype(int)
    return Xh, Xc, A, Y, hist


def fit_risk_model(Xh, Y, seed=0):
    """Renewal-risk baseline: P(non-renewal | x) learned from history."""
    Xh = Xh.copy()
    Xh.columns = [str(col) for col in Xh.columns]

    return HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.06,
        max_iter=200,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=seed
    ).fit(Xh, 1 - Y)


def build_proposals(estimator_name, P_hat, risk, Xc, cfg, P_true, rng):
    rent = Xc["rent"].to_numpy()
    V, cost, feas, budget = iv.retained_value(rent, cfg), iv.costs(rent, cfg), iv.feasibility(Xc, cfg), iv.budget_for(rent, cfg)
    nv_hat, nv_true = opt.net_value(P_hat, V, cost), opt.net_value(P_true, V, cost)
    props = {
        "No action": pol.policy_no_action(len(Xc)),
        "Random feasible": pol.policy_random(len(Xc), feas, rng),
        "B1 Rule-based": pol.policy_rule_based(Xc),
        "B2 Risk-only alert": pol.policy_risk_only(risk, cfg),
        "B3 Risk + rules": pol.policy_risk_plus_rules(Xc, risk, cfg),
        "A Causal (no constraints)": pol.policy_causal(nv_hat),
        "B Causal + constraints": pol.policy_causal_constrained(nv_hat, feas, cost, budget),
        "Oracle (true effects)": pol.policy_causal_constrained(nv_true, feas, cost, budget),
    }
    return props, dict(V=V, cost=cost, feas=feas, budget=budget, nv_hat=nv_hat, nv_true=nv_true)


def run_experiment(cfg: Config, calib: dict | None = None, estimators=("naive", "s_learner", "t_learner", "dr_learner"),
                   headline="dr_learner", with_econml=False, verbose=False, data=None):
    """data=(tables, oracle, world) re-uses an already generated / database-loaded world."""
    t0 = time.time()
    tables, oracle, world = data if data is not None else generate(cfg, calib)
    F = build_features(tables)
    Xh, Xc, A, Y, hist = split_xy(tables, oracle, F)
    P_true = oracle.loc[~hist, [f"p_{a}" for a in ARMS]].to_numpy()
    u = oracle.loc[~hist, "u"].to_numpy()
    all_est = causal.available_estimators(cfg.seed, include_econml=with_econml)
    risk_model = fit_risk_model(Xh, Y, cfg.seed)
    risk = risk_model.predict_proba(Xc)[:, 1]
    rng = np.random.default_rng(cfg.seed + 1)
    results = {"causal": {}, "P_hat": {}, "policy": {}, "per_tenant": {}}
    names = list(estimators) + (["econml_dr"] if with_econml and "econml_dr" in all_est else [])
    for name in names:
        est = all_est[name].fit(Xh, A, Y)
        P_hat = est.predict_p(Xc)
        results["P_hat"][name] = P_hat
        results["causal"][name] = causal_eval.causal_metrics(P_hat, P_true)
        props, ctx = build_proposals(name, P_hat, risk, Xc, cfg, P_true, np.random.default_rng(cfg.seed + 1))
        pdf, per = policy_eval.evaluate_policies(Xc, P_true, u, props, cfg, CONSTRAINT_AWARE, seed=cfg.seed)
        results["policy"][name], results["per_tenant"][name] = pdf, per
        results.setdefault("ctx", ctx)
        results.setdefault("decision_quality", {})[name] = causal_eval.decision_agreement(ctx["nv_hat"], ctx["nv_true"], ctx["feas"])
        if verbose:
            print(f"  [{name}] done  ({time.time() - t0:.0f}s)")
    results.update(dict(tables=tables, oracle=oracle, world=world, F=F, Xh=Xh, Xc=Xc, A=A, Y=Y, hist=hist,
                        P_true=P_true, u=u, risk=risk, risk_model=risk_model, estimators=all_est, cfg=cfg))
    results["headline"] = headline
    return results
