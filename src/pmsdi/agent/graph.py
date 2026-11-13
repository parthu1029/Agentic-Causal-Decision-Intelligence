"""LangGraph workflow  (slide 12):  Trigger -> Investigate -> Summarize -> Generate -> Estimate -> Optimize -> Explain
 ... [human approval] ...  -> Execute -> Monitor & Learn.

Two compiled graphs share one state schema:
  recommend graph : everything up to the explanation, then STOP (the human decides)
  execute graph   : approval gate -> execute (mock PMS tool) -> monitor
Splitting at the approval gate makes human-in-the-loop explicit and trivially testable, and it works inside
Streamlit/FastAPI where the 'human' answers in a later HTTP request.  If langgraph is not installed the same node
functions run sequentially (identical behaviour)."""
from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd

from ..config import ARM_LABELS, ARMS
from ..service import DecisionService, explain
from . import llm


class AgentState(TypedDict, total=False):
    tenant_id: int
    overrides: dict
    trigger: str
    evidence: dict
    summary: str
    candidates: list
    blocked: list
    estimates: pd.DataFrame
    recommended: str
    risk: float
    explanation: str
    decision: dict          # {"approved": bool, "action": str, "approver": str, "note": str}
    execution: dict
    monitor: dict
    trace: list


def _t(state, msg):
    return list(state.get("trace", [])) + [msg]


def build_nodes(svc: DecisionService) -> dict:
    cfg = svc.cfg

    def trigger(s: AgentState):
        f = svc.F.loc[s["tenant_id"]]
        why = []
        if f["days_to_lease_end"] <= 90:
            why.append(f"renewal window open ({int(f['days_to_lease_end'])} days to lease end)")
        if f["open_requests"] > 0 and f["max_open_age_days"] > 30:
            why.append("maintenance request open for more than 30 days")
        if f["avg_sentiment_90d"] < -0.3:
            why.append("negative tenant sentiment in the last 90 days")
        reason = "; ".join(why) or "manual analysis requested"
        return {"trigger": reason, "trace": _t(s, f"TRIGGER: {reason}")}

    def investigate(s: AgentState):
        ev = svc.evidence(s["tenant_id"])
        return {"evidence": ev, "trace": _t(s, "INVESTIGATE: pulled lease, payments, service requests, messages, survey via PMS tools")}

    def summarize(s: AgentState):
        f, cur = s["evidence"]["features"], cfg.currency
        txt = (f"Tenant {s['tenant_id']}: rent {cur}{f['rent']:,.0f} ({f['rent_vs_market']:+.1%} vs market), "
               f"{int(f['late_payments_12m'])} late payments in 12 months"
               f"{', with outstanding arrears' if f['has_arrears'] else ''}. "
               f"{int(f['open_requests'])} open maintenance request(s) (oldest {int(f['max_open_age_days'])} days), "
               f"average resolution {f['avg_resolution_days']:.0f} days. "
               f"Average message sentiment {f['avg_sentiment_90d']:+.2f} over {int(f['n_comms_90d'])} message(s) in 90 days; "
               f"satisfaction score {f['satisfaction_score']:.1f}/5. {int(f['days_to_lease_end'])} days to lease end.")
        return {"summary": llm.polish(txt), "trace": _t(s, "SUMMARIZE: built evidence-based tenant summary")}

    def generate(s: AgentState):
        est = svc.estimate(s["tenant_id"], s.get("overrides"))
        cand = est.loc[est["feasible"], "action"].tolist()
        blocked = [(r["action"], r["reason_blocked"]) for _, r in est[~est["feasible"]].iterrows()]
        return {"estimates": est, "candidates": cand, "blocked": blocked,
                "trace": _t(s, f"GENERATE: feasible actions {cand}; blocked {[b[0] for b in blocked]}")}

    def estimate(s: AgentState):
        return {"risk": svc.risk(s["tenant_id"], s.get("overrides")),
                "trace": _t(s, "ESTIMATE: renewal risk + counterfactual renewal probability under every action (DR-learner)")}

    def optimize(s: AgentState):
        est = s["estimates"]
        ok = est[est["feasible"]]
        best = ok.loc[ok["net_value"].idxmax(), "action"]
        return {"recommended": best, "trace": _t(s, f"OPTIMIZE: argmax net value over feasible set -> {best}")}

    def explain_node(s: AgentState):
        txt = explain(s["estimates"], s["recommended"], cfg.currency)
        svc.audit(s["tenant_id"], "recommendation", {"recommended": s["recommended"], "risk": s["risk"],
                                                     "trigger": s["trigger"],
                                                     "p_renew": s["estimates"].set_index("action")["p_renew"].round(4).to_dict()})
        return {"explanation": txt, "trace": _t(s, "EXPLAIN: rationale generated, recommendation logged; awaiting human approval")}

    def approval_gate(s: AgentState):
        d = s.get("decision") or {}
        return {"trace": _t(s, f"APPROVAL: {'approved' if d.get('approved') else 'rejected / not provided'} by {d.get('approver', 'n/a')}")}

    def route_after_gate(s: AgentState):
        return "execute" if (s.get("decision") or {}).get("approved") else "end"

    def execute(s: AgentState):
        d = s["decision"]
        chosen = d.get("action") or s["recommended"]
        feas = s["estimates"].set_index("action")["feasible"]
        if not bool(feas[chosen]):                                    # guardrail: even humans cannot bypass policy rules here
            rec = {"error": f"{chosen} violates policy: {s['estimates'].set_index('action').loc[chosen, 'reason_blocked']}"}
            svc.audit(s["tenant_id"], "execution_blocked", rec)
            return {"execution": rec, "trace": _t(s, "EXECUTE: BLOCKED by policy guardrail")}
        rec = svc.execute(s["tenant_id"], s["recommended"], chosen, d.get("approver", "unknown"), d.get("note", ""))
        return {"execution": rec, "trace": _t(s, f"EXECUTE: mock PMS tool applied '{chosen}'")}

    def monitor(s: AgentState):
        info = {"status": "awaiting_outcome", "tracked": s["execution"].get("approved_action")}
        svc.audit(s["tenant_id"], "monitoring_started", info)
        return {"monitor": info, "trace": _t(s, "MONITOR: outcome tracking started (compare predicted vs realised on renewal)")}

    return dict(trigger=trigger, investigate=investigate, summarize=summarize, generate=generate, estimate=estimate,
                optimize=optimize, explain=explain_node, approval_gate=approval_gate, route=route_after_gate,
                execute=execute, monitor=monitor)


RECOMMEND_ORDER = ["trigger", "investigate", "summarize", "generate", "estimate", "optimize", "explain"]
EXECUTE_ORDER = ["approval_gate", "execute", "monitor"]


class Agent:
    def __init__(self, svc: DecisionService, use_langgraph: bool = True):
        self.svc = svc
        self.nodes = build_nodes(svc)
        self.rec_graph = self.exec_graph = None
        if use_langgraph:
            try:
                from langgraph.graph import END, START, StateGraph
                g = StateGraph(AgentState)
                for n in RECOMMEND_ORDER:
                    g.add_node(n, self.nodes[n])
                g.add_edge(START, RECOMMEND_ORDER[0])
                for a, b in zip(RECOMMEND_ORDER, RECOMMEND_ORDER[1:]):
                    g.add_edge(a, b)
                g.add_edge(RECOMMEND_ORDER[-1], END)
                self.rec_graph = g.compile()
                h = StateGraph(AgentState)
                for n in EXECUTE_ORDER:
                    h.add_node(n, self.nodes[n])
                h.add_edge(START, "approval_gate")
                h.add_conditional_edges("approval_gate", self.nodes["route"], {"execute": "execute", "end": END})
                h.add_edge("execute", "monitor")
                h.add_edge("monitor", END)
                self.exec_graph = h.compile()
            except Exception:
                self.rec_graph = self.exec_graph = None

    @property
    def engine_name(self):
        return "langgraph" if self.rec_graph is not None else "sequential-fallback"

    def recommend(self, tenant_id: int, overrides: dict | None = None) -> AgentState:
        state: AgentState = {"tenant_id": int(tenant_id), "overrides": overrides or {}, "trace": []}
        if self.rec_graph is not None:
            return self.rec_graph.invoke(state)
        for n in RECOMMEND_ORDER:
            state.update(self.nodes[n](state))
        return state

    def decide(self, state: AgentState, approved: bool, approver: str, action: str | None = None, note: str = "") -> AgentState:
        state = dict(state)
        state["decision"] = {"approved": approved, "approver": approver, "action": action, "note": note}
        if not approved:
            self.svc.reject(state["tenant_id"], state["recommended"], approver, note)
        if self.exec_graph is not None:
            return self.exec_graph.invoke(state)
        state.update(self.nodes["approval_gate"](state))
        if approved:
            state.update(self.nodes["execute"](state))
            state.update(self.nodes["monitor"](state))
        return state
