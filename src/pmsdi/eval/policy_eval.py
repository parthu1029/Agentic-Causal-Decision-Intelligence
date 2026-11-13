"""Policy-level evaluation: policy value, regret, uplift, business value, compliance."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ARMS, K, Config
from ..decision import interventions as iv
from ..decision import optimizer as opt


def evaluate_policies(F: pd.DataFrame, P_true: np.ndarray, u: np.ndarray, proposals: dict, cfg: Config,
                      constraint_aware: set, seed: int = 0, n_boot: int = 300) -> pd.DataFrame:
    """proposals: name -> action index array (what the policy WANTS to do).
    Policies in `constraint_aware` are trusted to respect rules/budget; for all others the execution layer
    blocks infeasible proposals and anything beyond the budget (random order), exactly as a real PMS would."""
    rng = np.random.default_rng(seed)
    n = len(F)
    rent = F["rent"].to_numpy()
    V, cost, feas = iv.retained_value(rent, cfg), iv.costs(rent, cfg), iv.feasibility(F, cfg)
    budget = iv.budget_for(rent, cfg)
    nv_true = opt.net_value(P_true, V, cost)
    ar = np.arange(n)
    Y_pot = (u[:, None] < P_true).astype(int)
    oracle_act = opt.select_budgeted(nv_true, feas, cost, budget)
    base_nv = nv_true[:, 0]
    rows, per_tenant = [], {}
    for name, act in proposals.items():
        act = np.asarray(act)
        infeas = ~feas[ar, act]
        if name in constraint_aware:
            ex, b_i, b_b = act, np.zeros(n, bool), np.zeros(n, bool)
        else:
            ex, b_i, b_b = opt.enforce(act, feas, cost, budget, rng)
        v = nv_true[ar, ex]
        per_tenant[name] = v
        spend = cost[ar, ex].sum()
        realized = Y_pot[ar, ex] * V - cost[ar, ex]
        lo, hi = _boot_mean(v, rng, n_boot)
        rows.append({
            "policy": name, "net_value_per_tenant": v.mean(), "ci_lo": lo, "ci_hi": hi,
            "uplift_vs_no_action_per_tenant": (v - base_nv).mean(),
            "regret_vs_oracle_per_tenant": nv_true[ar, oracle_act].mean() - v.mean(),
            "extra_renewals_expected": float((P_true[ar, ex] - P_true[:, 0]).sum()),
            "retained_rent_proxy_per_tenant": float((P_true[ar, ex] * V).mean()),
            "spend_total": spend, "spend_pct_budget": 100 * spend / budget, "share_treated": float((ex != 0).mean()),
            "proposals_violating_rules_pct": 100 * float(infeas.mean()), "blocked_by_budget_pct": 100 * float(b_b.mean()),
            "realized_net_value_per_tenant": float(realized.mean()),
        })
    df = pd.DataFrame(rows).set_index("policy")
    top = nv_true[ar, oracle_act].mean() - base_nv.mean()
    df["pct_of_oracle_gain"] = 100 * df["uplift_vs_no_action_per_tenant"] / top if top > 0 else np.nan
    return df, per_tenant


def _boot_mean(x, rng, n_boot):
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    m = x[idx].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def paired_diff_ci(a: np.ndarray, b: np.ndarray, seed: int = 0, n_boot: int = 1000):
    """Bootstrap CI for mean(a-b) - use to claim 'policy A beats baseline B' with uncertainty."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))
