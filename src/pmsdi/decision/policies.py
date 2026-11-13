"""The five decision policies from slide 21 (+ reference policies).  Each policy maps tenant context -> action index."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ARMS, Config
from . import optimizer as opt

IDX = {a: i for i, a in enumerate(ARMS)}


def policy_no_action(n: int) -> np.ndarray:
    return np.zeros(n, dtype=int)


def policy_random(n: int, feas: np.ndarray, rng) -> np.ndarray:
    out = np.zeros(n, dtype=int)
    for i in range(n):
        out[i] = rng.choice(np.nonzero(feas[i])[0])
    return out


def policy_rule_based(F: pd.DataFrame) -> np.ndarray:
    """Baseline 1 - hand-written playbook, no model.  Feasibility is applied later by the execution layer."""
    out = np.zeros(len(F), dtype=int)
    open_r = F["open_requests"].to_numpy() > 0
    pricey = F["rent_vs_market"].to_numpy() > 0.05
    unhappy = (F["satisfaction_score"].to_numpy() <= 3) | (F["avg_sentiment_90d"].to_numpy() < 0)
    out[unhappy] = IDX["proactive_outreach"]
    out[pricey] = IDX["rent_concession"]
    out[open_r] = IDX["service_priority"]
    return out


def flag_high_risk(risk: np.ndarray, cfg: Config) -> np.ndarray:
    thr = np.quantile(risk, 1 - cfg.risk_flag_top_share)
    return risk >= thr


def policy_risk_only(risk: np.ndarray, cfg: Config) -> np.ndarray:
    """Baseline 2 - an alert system: every flagged tenant gets the manager's default reaction (a call)."""
    return np.where(flag_high_risk(risk, cfg), IDX["proactive_outreach"], 0)


def policy_risk_plus_rules(F: pd.DataFrame, risk: np.ndarray, cfg: Config) -> np.ndarray:
    """Baseline 3 - flag with the churn model, then pick the action with the hand-written playbook."""
    return np.where(flag_high_risk(risk, cfg), policy_rule_based(F), 0)


def policy_causal(nv_hat: np.ndarray) -> np.ndarray:
    """Proposed A - argmax of estimated net value; ignores business rules and the budget."""
    return opt.select_unconstrained(nv_hat)


def policy_causal_constrained(nv_hat, feas, cost, budget) -> np.ndarray:
    """Proposed B - estimated net value + feasibility rules + portfolio budget (MILP)."""
    return opt.select_budgeted(nv_hat, feas, cost, budget)
