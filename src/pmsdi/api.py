"""REST API (FastAPI).  Run:  uvicorn pmsdi.api:app --reload   ->  http://localhost:8000/docs"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agent.graph import Agent
from .service import DecisionService

app = FastAPI(title="Tenant-retention decision intelligence", version="1.0")
_state: dict = {}
_pending: dict[int, dict] = {}


def svc() -> DecisionService:
    if "svc" not in _state:
        _state["svc"] = DecisionService()
        _state["agent"] = Agent(_state["svc"])
    return _state["svc"]


class Decision(BaseModel):
    approved: bool
    approver: str = "manager"
    action: str | None = None
    note: str = ""


@app.get("/health")
def health():
    s = svc()
    return {"status": "ok", "tenants_loaded": int(len(s.F)), "agent_engine": _state["agent"].engine_name}


@app.get("/tenants")
def tenants(limit: int = 20):
    return svc().tenants(limit=limit).reset_index().to_dict("records")


@app.get("/tenants/{tenant_id}/context")
def context(tenant_id: int):
    s = svc()
    if tenant_id not in s.F.index:
        raise HTTPException(404, "unknown tenant")
    ev = s.evidence(tenant_id)
    return {"features": ev["features"].to_dict(), "lease": {k: str(v) for k, v in ev["lease"].items()},
            "open_or_recent_requests": ev["requests"].astype(str).to_dict("records"),
            "recent_messages": ev["messages"].astype(str).to_dict("records")}


@app.get("/tenants/{tenant_id}/recommendation")
def recommendation(tenant_id: int):
    s = svc()
    if tenant_id not in s.F.index:
        raise HTTPException(404, "unknown tenant")
    st = _state["agent"].recommend(tenant_id)
    _pending[tenant_id] = st
    return {"tenant_id": tenant_id, "trigger": st["trigger"], "summary": st["summary"], "risk_non_renewal": st["risk"],
            "recommended_action": st["recommended"], "explanation": st["explanation"],
            "estimates": st["estimates"].round(4).to_dict("records"), "trace": st["trace"]}


@app.post("/tenants/{tenant_id}/decision")
def decision(tenant_id: int, d: Decision):
    svc()
    if tenant_id not in _pending:
        raise HTTPException(409, "request a recommendation first")
    out = _state["agent"].decide(_pending.pop(tenant_id), d.approved, d.approver, d.action, d.note)
    return {"trace": out["trace"], "execution": out.get("execution"), "monitor": out.get("monitor")}


@app.get("/audit")
def audit(tenant_id: int | None = None):
    from .data import db
    return db.read_audit(svc().engine, tenant_id).astype(str).to_dict("records")
