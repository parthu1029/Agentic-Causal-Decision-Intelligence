"""Off-policy evaluation (OPE): estimate how a NEW policy would have performed using only logged data.
This is how you evaluate a decision policy on real PMS history where counterfactual outcomes are never observed.
On the synthetic data we can then compare the OPE estimate against the true (oracle) policy value."""
from __future__ import annotations

import numpy as np


def ope_value(actions, A, Y, e_logged, p_hat, V, cost) -> dict:
    """Value = E[ V * 1{renewed} - cost(action) ] per tenant, for deterministic target `actions`.
    Reward uses the known cost of the executed action, so only the renewal indicator needs correcting."""
    n = len(A)
    ar = np.arange(n)
    match = (A == actions).astype(float)
    w = match / np.clip(e_logged[ar, A], 0.03, 1)
    reward = Y * V - cost[ar, A]
    ips = float(np.mean(w * reward))
    snips = float(np.sum(w * reward) / np.sum(w)) if w.sum() > 0 else float("nan")
    direct = float(np.mean(p_hat[ar, actions] * V - cost[ar, actions]))
    dr = float(np.mean(V * (p_hat[ar, actions] + w * (Y - p_hat[ar, A])) - cost[ar, actions]))
    return {"ips": ips, "snips": snips, "direct": direct, "dr": dr}
