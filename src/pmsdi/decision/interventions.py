"""Intervention library, business rules (feasibility) and economics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ARMS, K, Config

DESCRIPTIONS = {
    "no_action": "Do nothing; let the normal renewal process run.",
    "service_priority": "Escalate and fast-track the tenant's open maintenance requests.",
    "rent_concession": "Offer a one-off renewal credit (cost = fraction of one month's rent).",
    "proactive_outreach": "Manager calls / meets the tenant before the renewal offer to address concerns.",
}


def costs(rent: np.ndarray, cfg: Config) -> np.ndarray:
    """(n, K) intervention cost, relative to monthly rent so it is currency independent."""
    rel = np.array([cfg.cost_rel[a] for a in ARMS])
    return np.asarray(rent, float)[:, None] * rel[None, :]


def retained_value(rent: np.ndarray, cfg: Config) -> np.ndarray:
    """Retained-rent proxy: revenue of the next lease term + turnover cost avoided if the tenant renews."""
    return np.asarray(rent, float) * (cfg.lease_months + cfg.turnover_months)


def feasibility(F: pd.DataFrame, cfg: Config) -> np.ndarray:
    """(n, K) boolean mask of actions the *policy* allows right now.  no_action is always allowed."""
    feas = np.ones((len(F), K), dtype=bool)
    if cfg.service_requires_open_request:
        feas[:, 1] = F["open_requests"].to_numpy() > 0
    ok = F["late_payments_12m"].to_numpy() <= cfg.concession_max_late_payments
    if cfg.concession_requires_zero_arrears:
        ok &= F["has_arrears"].to_numpy() == 0
    feas[:, 2] = ok
    return feas


def infeasible_reason(row: pd.Series, action: str, cfg: Config) -> str:
    if action == "service_priority" and cfg.service_requires_open_request and row["open_requests"] <= 0:
        return "no open maintenance request to prioritise"
    if action == "rent_concession":
        if cfg.concession_requires_zero_arrears and row["has_arrears"] > 0:
            return "tenant has outstanding rent arrears"
        if row["late_payments_12m"] > cfg.concession_max_late_payments:
            return f"more than {cfg.concession_max_late_payments} late payments in 12 months"
    return ""


def budget_for(rent: np.ndarray, cfg: Config) -> float:
    return float(cfg.budget_pct_rent_roll * np.sum(rent))
