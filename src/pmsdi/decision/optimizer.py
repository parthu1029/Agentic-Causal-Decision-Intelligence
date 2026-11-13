"""Constraint-aware optimisation.

Per tenant i choose one action a from the FEASIBLE set to maximise expected net value
        NV[i,a] = p_hat[i,a] * V[i] - cost[i,a]
and, at portfolio level, keep  sum_i cost[i, a_i] <= budget.
That is a multiple-choice knapsack; we solve it exactly as a MILP with HiGHS (scipy.optimize.milp)."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp


def net_value(p: np.ndarray, V: np.ndarray, cost: np.ndarray) -> np.ndarray:
    return p * V[:, None] - cost


def select_unconstrained(nv: np.ndarray) -> np.ndarray:
    return nv.argmax(axis=1)


def select_feasible(nv: np.ndarray, feas: np.ndarray) -> np.ndarray:
    return np.where(feas, nv, -np.inf).argmax(axis=1)


def select_budgeted(nv: np.ndarray, feas: np.ndarray, cost: np.ndarray, budget: float, time_limit: float = 60.0) -> np.ndarray:
    """Exact multiple-choice knapsack.  Falls back to a Lagrangian greedy if the solver fails."""
    n, K = nv.shape
    free = select_feasible(nv, feas)
    if cost[np.arange(n), free].sum() <= budget + 1e-9:
        return free                                    # budget not binding -> per-tenant optimum is optimal
    ii, aa = np.nonzero(feas)
    nvar = len(ii)
    c = -nv[ii, aa]
    rows = sparse.csr_matrix((np.ones(nvar), (ii, np.arange(nvar))), shape=(n, nvar))
    budget_row = sparse.csr_matrix(cost[ii, aa][None, :])
    cons = [LinearConstraint(rows, 1, 1), LinearConstraint(budget_row, 0, budget)]
    res = milp(c, constraints=cons, integrality=np.ones(nvar), bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "mip_rel_gap": 1e-5})
    if res.x is None:
        return _greedy(nv, feas, cost, budget)
    x = np.round(res.x).astype(int)
    out = np.zeros(n, dtype=int)                       # default no_action
    chosen = x == 1
    out[ii[chosen]] = aa[chosen]
    return out


def _greedy(nv, feas, cost, budget):
    """Fallback: start from no_action, add upgrades by best marginal value per unit cost."""
    n, K = nv.shape
    out = np.zeros(n, dtype=int)
    gain = nv - nv[:, [0]]
    gain = np.where(feas, gain, -np.inf)
    gain[:, 0] = -np.inf
    best = gain.argmax(1)
    g = gain[np.arange(n), best]
    c = cost[np.arange(n), best]
    order = np.argsort(-(g / np.maximum(c, 1e-9)))
    spent = 0.0
    for i in order:
        if g[i] <= 0:
            break
        if spent + c[i] <= budget:
            out[i] = best[i]
            spent += c[i]
    return out


def enforce(actions: np.ndarray, feas: np.ndarray, cost: np.ndarray, budget: float, rng: np.random.Generator):
    """The execution layer: what really happens when a policy proposes `actions`.
    Infeasible proposals are blocked; tenants are processed in random order and anything that would exceed
    the remaining budget is blocked too.  Blocked => no_action."""
    n = len(actions)
    executed = actions.copy()
    blocked_infeasible = ~feas[np.arange(n), actions]
    executed[blocked_infeasible] = 0
    spent, blocked_budget = 0.0, np.zeros(n, dtype=bool)
    for i in rng.permutation(n):
        a = executed[i]
        if a == 0:
            continue
        c = cost[i, a]
        if spent + c > budget + 1e-9:
            executed[i] = 0
            blocked_budget[i] = True
        else:
            spent += c
    return executed, blocked_infeasible, blocked_budget
