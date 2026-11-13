import itertools

import numpy as np

from pmsdi.config import ARMS
from pmsdi.decision import interventions as iv
from pmsdi.decision import optimizer as opt


def test_milp_matches_brute_force_and_respects_budget():
    rng = np.random.default_rng(1)
    for _ in range(5):
        n, K = 6, 4
        nv = rng.normal(5, 10, (n, K)); nv[:, 0] = 0
        cost = rng.uniform(1, 10, (n, K)); cost[:, 0] = 0
        feas = rng.uniform(size=(n, K)) < 0.8; feas[:, 0] = True
        budget = 10.0
        best = max(sum(nv[i, a] for i, a in enumerate(c)) for c in itertools.product(range(K), repeat=n)
                   if all(feas[i, a] for i, a in enumerate(c)) and sum(cost[i, a] for i, a in enumerate(c)) <= budget)
        sel = opt.select_budgeted(nv, feas, cost, budget)
        assert abs(sum(nv[i, a] for i, a in enumerate(sel)) - best) < 1e-6
        assert sum(cost[i, a] for i, a in enumerate(sel)) <= budget + 1e-9 and feas[np.arange(n), sel].all()


def test_enforce_blocks_infeasible_and_over_budget():
    rng = np.random.default_rng(0)
    n = 200
    cost = np.column_stack([np.zeros(n), np.full(n, 10.0), np.full(n, 20.0), np.full(n, 5.0)])
    feas = rng.uniform(size=(n, 4)) < 0.5; feas[:, 0] = True
    act = rng.integers(0, 4, n)
    ex, bi, bb = opt.enforce(act, feas, cost, 300.0, rng)
    assert cost[np.arange(n), ex].sum() <= 300.0 and feas[np.arange(n), ex].all()
    assert (bi == ~feas[np.arange(n), act]).all()


def test_feasibility_rules(world, cfg):
    _, _, _, F = world
    feas = iv.feasibility(F, cfg)
    assert feas[:, 0].all()
    assert ((F["open_requests"].to_numpy() > 0) == feas[:, 1]).all()
    ok = (F["late_payments_12m"] <= cfg.concession_max_late_payments) & (F["has_arrears"] == 0)
    assert (ok.to_numpy() == feas[:, 2]).all() and feas[:, 3].all()
