"""DecisionService - the application layer used by the agent, the REST API and the dashboard.
Wraps: database (PMS tools), trained models, optimizer, audit trail."""
from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import insert, text, update

from .config import ARM_LABELS, ARMS, ARTIFACT_DIR, K, SYNTH_DIR, Config
from .data import db
from .data.features import FEATURES, parse_dates
from .decision import interventions as iv
from .decision import optimizer as opt

IDX = {a: i for i, a in enumerate(ARMS)}


class DecisionService:
    def __init__(self, engine=None, artifacts_path=None, load_oracle=True, oracle=None):
        self.engine = engine or db.get_engine()
        art = joblib.load(artifacts_path or ARTIFACT_DIR / "models.joblib")
        self.cfg = Config.from_dict(art["cfg"])
        self.est, self.est_alt, self.risk_model = art["dr"], art.get("s_learner"), art["risk_model"]
        self.F = db.read_features(self.engine)[FEATURES]
        self.current_ids = db.read_table(self.engine, "tenants").query("cohort == 'current'")["tenant_id"].to_numpy()
        self._P = None
        oracle_path = SYNTH_DIR / "oracle.csv.gz"
        if oracle is not None:
            self.oracle = oracle
        else:
            self.oracle = pd.read_csv(oracle_path, index_col=0) if (load_oracle and oracle_path.exists()) else None

    # ------------------------------------------------------------------ batch predictions (cached)
    def _batch(self):
        if self._P is None:
            Fc = self.F.loc[self.current_ids]
            self._P = {"ids": self.current_ids, "P": self.est.predict_p(Fc),
                       "risk": self.risk_model.predict_proba(Fc)[:, 1]}
        return self._P

    def tenants(self, cohort="current", limit=None) -> pd.DataFrame:
        b = self._batch()
        df = self.F.loc[b["ids"], ["rent", "days_to_lease_end", "open_requests", "late_payments_12m", "satisfaction_score"]].copy()
        df["risk_non_renewal"] = b["risk"]
        st = pd.read_sql_query(text("SELECT tenant_id, MAX(status) AS action_status FROM interventions GROUP BY tenant_id"), self.engine)
        df = df.join(st.set_index("tenant_id")["action_status"]).fillna({"action_status": "pending"})
        df = df.sort_values("risk_non_renewal", ascending=False)
        return df.head(limit) if limit else df

    # ------------------------------------------------------------------ PMS tools (read side)
    def evidence(self, tenant_id: int) -> dict:
        def q(sql, **p):
            return pd.read_sql_query(text(sql), self.engine, params=p)
        tid = int(tenant_id)
        lease = q("SELECT * FROM leases WHERE tenant_id=:t", t=tid)
        unit = q("SELECT u.bedrooms,u.area_sqft,u.market_rent,p.city,p.age_years,p.violations_per_100_units_12m "
                 "FROM tenants t JOIN units u ON t.unit_id=u.unit_id JOIN properties p ON u.property_id=p.property_id WHERE t.tenant_id=:t", t=tid)
        asof = pd.to_datetime(lease["decision_date"].iloc[0])
        pay = q("SELECT due_date,amount_due,amount_paid,days_late FROM payments WHERE tenant_id=:t ORDER BY due_date DESC LIMIT 12", t=tid)
        req = q("SELECT category,priority,created_date,resolved_date FROM service_requests WHERE tenant_id=:t ORDER BY created_date DESC LIMIT 10", t=tid)
        com = q("SELECT sent_at,channel,sentiment,response_hours,message_text FROM communications WHERE tenant_id=:t ORDER BY sent_at DESC LIMIT 8", t=tid)
        sv = q("SELECT survey_date,score FROM satisfaction_surveys WHERE tenant_id=:t ORDER BY survey_date DESC LIMIT 1", t=tid)
        past = q("SELECT action,status,assigned_by,cost,executed_at FROM interventions WHERE tenant_id=:t", t=tid)
        for df_, cols in ((pay, ["due_date"]), (req, ["created_date", "resolved_date"]), (com, ["sent_at"]),
                          (sv, ["survey_date"]), (past, ["executed_at"]), (lease, ["start_date", "end_date", "decision_date"])):
            for c in cols:
                df_[c] = pd.to_datetime(df_[c])
        return {"tenant_id": tid, "lease": lease.iloc[0].to_dict(), "unit": unit.iloc[0].to_dict(), "decision_date": asof,
                "payments": pay, "requests": req, "messages": com, "survey": sv, "actions": past,
                "features": self.F.loc[tid]}

    # ------------------------------------------------------------------ analytics
    def estimate(self, tenant_id: int, overrides: dict | None = None) -> pd.DataFrame:
        row = self.F.loc[[int(tenant_id)]].copy()
        for k, v in (overrides or {}).items():
            row[k] = float(v)
        row["has_arrears"] = (row["arrears_ratio"] > 0.02).astype(float)
        P = self.est.predict_p(row)[0]
        P_alt = self.est_alt.predict_p(row)[0] if self.est_alt is not None else P
        rent = row["rent"].to_numpy()
        V, cost, feas = iv.retained_value(rent, self.cfg)[0], iv.costs(rent, self.cfg)[0], iv.feasibility(row, self.cfg)[0]
        nv = P * V - cost
        out = pd.DataFrame({
            "action": ARMS, "label": [ARM_LABELS[a] for a in ARMS], "p_renew": P, "uplift": P - P[0], "cost": cost,
            "retained_value": V, "net_value": nv, "net_value_vs_no_action": nv - nv[0], "feasible": feas,
            "p_renew_alt_estimator": P_alt, "estimator_spread": np.abs(P - P_alt),
            "reason_blocked": [iv.infeasible_reason(row.iloc[0], a, self.cfg) if not feas[i] else "" for i, a in enumerate(ARMS)],
        })
        return out

    def risk(self, tenant_id: int, overrides: dict | None = None) -> float:
        row = self.F.loc[[int(tenant_id)]].copy()
        for k, v in (overrides or {}).items():
            row[k] = float(v)
        return float(self.risk_model.predict_proba(row)[0, 1])

    def recommend(self, tenant_id: int, overrides: dict | None = None) -> dict:
        est = self.estimate(tenant_id, overrides)
        ok = est[est["feasible"]]
        best = ok.loc[ok["net_value"].idxmax()]
        return {"tenant_id": int(tenant_id), "recommended": best["action"], "table": est, "risk": self.risk(tenant_id, overrides),
                "explanation": explain(est, best["action"], self.cfg.currency)}

    def portfolio_plan(self, budget_pct: float | None = None, ids=None) -> tuple[pd.DataFrame, dict]:
        b = self._batch()
        ids = b["ids"] if ids is None else ids
        pos = pd.Index(b["ids"]).get_indexer(ids)
        P = b["P"][pos]
        F = self.F.loc[ids]
        rent = F["rent"].to_numpy()
        V, cost, feas = iv.retained_value(rent, self.cfg), iv.costs(rent, self.cfg), iv.feasibility(F, self.cfg)
        pct = self.cfg.budget_pct_rent_roll if budget_pct is None else budget_pct
        budget = pct * rent.sum()
        nv = opt.net_value(P, V, cost)
        act = opt.select_budgeted(nv, feas, cost, budget)
        ar = np.arange(len(ids))
        df = pd.DataFrame({"tenant_id": ids, "action": np.array(ARMS)[act], "p_renew_if_action": P[ar, act], "uplift": P[ar, act] - P[:, 0],
                           "cost": cost[ar, act], "net_value_gain": nv[ar, act] - nv[:, 0], "risk": b["risk"][pos]}).set_index("tenant_id")
        summ = {"budget": budget, "spend": float(cost[ar, act].sum()), "tenants": len(ids), "treated": int((act != 0).sum()),
                "expected_extra_renewals": float((P[ar, act] - P[:, 0]).sum()), "expected_net_gain": float((nv[ar, act] - nv[:, 0]).sum())}
        return df, summ

    # ------------------------------------------------------------------ PMS tools (write side) + audit
    def audit(self, tenant_id: int, event: str, payload: dict):
        db.log_audit(self.engine, tenant_id, event, payload)

    def execute(self, tenant_id: int, recommended: str, chosen: str, approver: str, note: str = "") -> dict:
        """Mock PMS execution. The executed action is whatever the human approved - which may differ from the recommendation."""
        tid = int(tenant_id)
        rent = float(self.F.loc[tid, "rent"])
        cost = float(iv.costs(np.array([rent]), self.cfg)[0, IDX[chosen]])
        now = datetime.now()
        with self.engine.begin() as con:
            con.execute(insert(db.md.tables["interventions"]).values(
                tenant_id=tid, action=chosen, assigned_by="agent_human_approved", logged_propensity=None, cost=cost,
                status="executed", executed_at=now))
        rec = {"tenant_id": tid, "recommended": recommended, "approved_action": chosen, "overridden": recommended != chosen,
               "approver": approver, "cost": cost, "note": note}
        self.audit(tid, "approved_and_executed", rec)
        return rec

    def reject(self, tenant_id: int, recommended: str, approver: str, note: str = "") -> dict:
        rec = {"tenant_id": int(tenant_id), "recommended": recommended, "approver": approver, "note": note}
        self.audit(tenant_id, "rejected", rec)
        return rec

    def simulate_outcome(self, tenant_id: int) -> dict:
        """DEMO ONLY: draw the renewal outcome from the hidden ground truth so 'monitor & learn' can be shown.
        In a real PMS this row would arrive when the tenant actually renews or leaves."""
        if self.oracle is None:
            raise RuntimeError("oracle.csv.gz not found - outcome simulation is only available with the synthetic world")
        tid = int(tenant_id)
        act = pd.read_sql_query(text("SELECT action FROM interventions WHERE tenant_id=:t ORDER BY intervention_id DESC LIMIT 1"),
                                self.engine, params={"t": tid})
        action = act["action"].iloc[0] if len(act) else "no_action"
        o = self.oracle.loc[tid]
        renewed = int(o["u"] < o[f"p_{action}"])
        with self.engine.begin() as con:
            con.execute(insert(db.md.tables["renewal_outcomes"]).values(
                tenant_id=tid, lease_id=tid, renewed=renewed, source="simulated_demo", decided_at=datetime.now()))
            con.execute(update(db.md.tables["leases"]).where(db.md.tables["leases"].c.tenant_id == tid)
                        .values(status="renewed" if renewed else "not_renewed"))
        rec = {"tenant_id": tid, "action": action, "renewed": renewed, "source": "simulated_demo"}
        self.audit(tid, "outcome_recorded", rec)
        return rec

    def monitor(self) -> pd.DataFrame:
        """Predicted vs realised outcomes for decisions executed through the agent (needs outcomes)."""
        q = """SELECT i.tenant_id, i.action, o.renewed FROM interventions i JOIN renewal_outcomes o ON o.tenant_id=i.tenant_id
               WHERE i.assigned_by='agent_human_approved' AND o.source='simulated_demo'"""
        d = pd.read_sql_query(text(q), self.engine)
        if d.empty:
            return d
        P = self.est.predict_p(self.F.loc[d["tenant_id"]])
        d["p_pred"] = P[np.arange(len(d)), d["action"].map(IDX).to_numpy()]
        d["p_pred_no_action"] = P[:, 0]
        if self.oracle is not None:
            d["p_true"] = [self.oracle.loc[t, f"p_{a}"] for t, a in zip(d["tenant_id"], d["action"])]
        return d


def explain(est: pd.DataFrame, best: str, cur: str = "$") -> str:
    """Plain-language rationale built only from computed numbers (no free-text generation)."""
    nv0 = est.loc[est["action"] == "no_action", "net_value"].iloc[0]
    b = est[est["action"] == best].iloc[0]
    lines = [f"Recommended: **{ARM_LABELS[best]}** - estimated renewal {b['p_renew']:.1%} "
             f"({b['uplift']:+.1%} vs no action), cost {cur}{b['cost']:,.0f}, "
             f"net value {cur}{b['net_value_vs_no_action']:+,.0f} vs doing nothing."]
    for _, r in est[est["action"] != best].iterrows():
        if r["action"] == "no_action":
            continue
        if not r["feasible"]:
            lines.append(f"- {ARM_LABELS[r['action']]}: not allowed - {r['reason_blocked']}.")
        else:
            why = "lower expected net value" if r["net_value"] < b["net_value"] else "equal value"
            lines.append(f"- {ARM_LABELS[r['action']]}: {r['uplift']:+.1%} renewal for {cur}{r['cost']:,.0f}; "
                         f"net {cur}{r['net_value_vs_no_action']:+,.0f} -> {why}.")
    if best == "no_action":
        lines.append("No feasible intervention is expected to pay for itself for this tenant.")
    return "\n".join(lines)


def reset_demo(svc: "DecisionService"):
    """Remove everything the live demo wrote (agent actions, simulated outcomes, audit log) so the demo can be repeated."""
    with svc.engine.begin() as con:
        con.execute(text("UPDATE leases SET status='pending' WHERE tenant_id IN "
                         "(SELECT tenant_id FROM renewal_outcomes WHERE source='simulated_demo')"))
        con.execute(text("DELETE FROM renewal_outcomes WHERE source='simulated_demo'"))
        con.execute(text("DELETE FROM interventions WHERE assigned_by='agent_human_approved'"))
        con.execute(text("DELETE FROM audit_log"))
