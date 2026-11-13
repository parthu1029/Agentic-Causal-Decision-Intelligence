import numpy as np

from pmsdi.pipeline import POLICY_ORDER, run_experiment
import pytest


@pytest.fixture(scope="module")
def R(cfg, world):
    tables, oracle, w, _ = world
    return run_experiment(cfg, data=(tables, oracle, w), estimators=("naive", "s_learner", "dr_learner"))


def test_adjusted_estimators_beat_naive_on_confounded_logs(R):
    naive = R["causal"]["naive"]["ate_abs_error"].mean()
    for est in ("s_learner", "dr_learner"):
        assert R["causal"][est]["ate_abs_error"].mean() < naive / 3


def test_dr_recovers_average_effects(R):
    c = R["causal"]["dr_learner"]
    assert (c["ate_abs_error"] < 0.03).all() and (np.sign(c["ate_hat"]) == np.sign(c["ate_true"])).all()


def test_policy_ordering(R):
    p = R["policy"]["dr_learner"].loc[POLICY_ORDER]
    nv = p["net_value_per_tenant"]
    assert nv["Oracle (true effects)"] >= nv["B Causal + constraints"] > nv["No action"]
    assert nv["B Causal + constraints"] > nv["B3 Risk + rules"]
    assert p.loc["B Causal + constraints", "proposals_violating_rules_pct"] == 0
    assert p.loc["A Causal (no constraints)", "proposals_violating_rules_pct"] > 0       # unconstrained policy asks for forbidden actions
    assert (p["spend_pct_budget"] <= 100.0001).all()
    assert p.loc["Oracle (true effects)", "regret_vs_oracle_per_tenant"] == 0


def test_off_policy_estimate_close_to_truth(world, cfg):
    """With (near-)random logging, plain IPS must be nearly unbiased for any policy."""
    from pmsdi.config import Config
    from pmsdi.data.features import build_features
    from pmsdi.data.synthetic_pms import generate
    from pmsdi.decision import interventions as iv
    from pmsdi.eval.ope import ope_value
    c2 = Config.from_dict({**cfg.to_dict(), "logging_strength": 0.0, "n_history": 8000, "n_current": 200})
    t, o, _ = generate(c2)
    F = build_features(t)
    h = (o["cohort"] == "history").to_numpy()
    Fh = F[h]
    from pmsdi.config import ARMS
    A = np.array([ARMS.index(a) for a in o.loc[h, "logged_action"]])
    Y = t["renewal_outcomes"].set_index("tenant_id").loc[Fh.index, "renewed"].to_numpy()
    e = o.loc[h, [f"e_{a}" for a in ARMS]].to_numpy()
    P = o.loc[h, [f"p_{a}" for a in ARMS]].to_numpy()
    V, cost = iv.retained_value(Fh["rent"].to_numpy(), c2), iv.costs(Fh["rent"].to_numpy(), c2)
    act = np.full(len(Fh), 2)
    truth = (P[:, 2] * V - cost[:, 2]).mean()
    est = ope_value(act, A, Y, e, P, V, cost)
    assert abs(est["ips"] - truth) < 0.05 * abs(truth) and abs(est["dr"] - truth) < 0.02 * abs(truth)
