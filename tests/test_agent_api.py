import numpy as np
import pandas as pd

from pmsdi.agent.graph import Agent
from pmsdi.data import db


def _tenant_with(service, need_blocked=False):
    for tid in service.tenants().index[:400]:
        est = service.estimate(int(tid))
        if need_blocked == bool((~est["feasible"]).any()):
            return int(tid)
    raise AssertionError("no suitable tenant")


def test_recommendation_is_feasible_and_best(service):
    tid = _tenant_with(service, need_blocked=True)
    r = service.recommend(tid)
    t = r["table"]
    assert t.set_index("action").loc[r["recommended"], "feasible"]
    assert r["recommended"] == t[t["feasible"]].sort_values("net_value").iloc[-1]["action"]


def test_agent_full_flow_and_audit(service):
    ag = Agent(service)
    assert ag.engine_name in ("langgraph", "sequential-fallback")
    tid = _tenant_with(service, need_blocked=False)
    st = ag.recommend(tid)
    assert {"trigger", "summary", "explanation", "recommended"} <= set(st) and len(st["trace"]) == 7
    out = ag.decide(st, True, "tester", st["recommended"], "unit test")
    assert out["execution"]["approved_action"] == st["recommended"] and out["monitor"]["status"] == "awaiting_outcome"
    res = service.simulate_outcome(tid)
    assert res["renewed"] in (0, 1)
    events = set(db.read_audit(service.engine, tid)["event"])
    assert {"recommendation", "approved_and_executed", "monitoring_started", "outcome_recorded"} <= events
    assert len(service.monitor()) >= 1


def test_guardrail_blocks_forbidden_action_even_if_human_approves(service):
    ag = Agent(service)
    tid = _tenant_with(service, need_blocked=True)
    st = ag.recommend(tid)
    bad = st["estimates"][~st["estimates"]["feasible"]].iloc[0]["action"]
    out = ag.decide(st, True, "tester", bad)
    assert "violates policy" in out["execution"]["error"]
    n = pd.read_sql_query("SELECT COUNT(*) c FROM interventions WHERE tenant_id=%d AND assigned_by='agent_human_approved'" % tid, service.engine)["c"][0]
    assert n == 0


def test_rejection_executes_nothing(service):
    ag = Agent(service)
    tid = int(service.tenants().index[5])
    out = ag.decide(ag.recommend(tid), False, "tester", note="no")
    assert "execution" not in out or not out.get("execution")
    assert "rejected" in set(db.read_audit(service.engine, tid)["event"])


def test_portfolio_plan_respects_budget(service):
    plan, s = service.portfolio_plan(0.03)
    assert s["spend"] <= s["budget"] + 1e-6 and s["expected_net_gain"] > 0
    F = service.F.loc[plan.index]
    feas = __import__("pmsdi.decision.interventions", fromlist=["x"]).feasibility(F, service.cfg)
    idx = plan["action"].map({a: i for i, a in enumerate(__import__("pmsdi.config", fromlist=["x"]).ARMS)}).to_numpy()
    assert feas[np.arange(len(plan)), idx].all()


def test_rest_api(service):
    from fastapi.testclient import TestClient
    from pmsdi import api
    api._state["svc"], api._state["agent"] = service, Agent(service)
    c = TestClient(api.app)
    assert c.get("/health").json()["status"] == "ok"
    tid = int(c.get("/tenants?limit=3").json()[2]["tenant_id"])
    r = c.get(f"/tenants/{tid}/recommendation").json()
    assert r["recommended_action"] in {e["action"] for e in r["estimates"] if e["feasible"]}
    d = c.post(f"/tenants/{tid}/decision", json={"approved": True, "approver": "api-test", "action": r["recommended_action"]}).json()
    assert d["execution"]["approved_action"] == r["recommended_action"]
    assert c.post("/tenants/%d/decision" % tid, json={"approved": True}).status_code == 409
    assert c.get("/audit", params={"tenant_id": tid}).json()
