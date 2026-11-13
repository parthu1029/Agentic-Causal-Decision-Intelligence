"""Feature store: turns the raw PMS tables into one model-ready row per tenant, *as of the decision date*.

This is the "Feature Store" box of the architecture (slide 10).  It only uses information that
existed on or before each lease's decision_date, so there is no leakage from the outcome."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "rent", "area_sqft", "bedrooms", "property_age", "violations_per_100_units", "rent_vs_market",
    "tenure_months", "days_to_lease_end",
    "late_payments_12m", "avg_days_late", "arrears_ratio", "has_arrears",
    "requests_12m", "open_requests", "avg_resolution_days", "max_open_age_days",
    "n_comms_90d", "avg_sentiment_90d", "min_sentiment_90d", "complaint_msgs_90d", "avg_response_hours",
    "satisfaction_score", "survey_missing",
]

DATE_COLS = {
    "tenants": ["move_in_date"], "leases": ["start_date", "end_date", "decision_date"],
    "payments": ["due_date", "paid_date"], "service_requests": ["created_date", "resolved_date"],
    "communications": ["sent_at"], "satisfaction_surveys": ["survey_date"],
    "interventions": ["executed_at"], "renewal_outcomes": ["decided_at"],
}


def parse_dates(tables: dict) -> dict:
    for name, cols in DATE_COLS.items():
        if name in tables:
            for c in cols:
                if c in tables[name]:
                    tables[name][c] = pd.to_datetime(tables[name][c])
    return tables


def build_features(t: dict) -> pd.DataFrame:
    L = t["leases"][["lease_id", "tenant_id", "unit_id", "monthly_rent", "end_date", "decision_date"]]
    U = t["units"][["unit_id", "property_id", "bedrooms", "area_sqft", "market_rent"]]
    P = t["properties"][["property_id", "age_years", "violations_per_100_units_12m"]]
    T = t["tenants"][["tenant_id", "move_in_date"]]
    base = L.merge(U, on="unit_id").merge(P, on="property_id").merge(T, on="tenant_id")
    base["rent"] = base["monthly_rent"]
    base["rent_vs_market"] = base["rent"] / base["market_rent"] - 1.0
    base["tenure_months"] = (base["decision_date"] - base["move_in_date"]).dt.days / 30.4
    base["days_to_lease_end"] = (base["end_date"] - base["decision_date"]).dt.days
    base = base.rename(columns={"age_years": "property_age", "violations_per_100_units_12m": "violations_per_100_units"})
    base = base.set_index("tenant_id")
    asof = base["decision_date"]
    out = base[["rent", "area_sqft", "bedrooms", "property_age", "violations_per_100_units", "rent_vs_market",
                "tenure_months", "days_to_lease_end"]].copy()
    day = pd.Timedelta(days=1)

    # ---- payments (last 12 months)
    pay = t["payments"].merge(asof.rename("asof"), left_on="tenant_id", right_index=True)
    pay = pay[(pay["due_date"] <= pay["asof"]) & (pay["due_date"] > pay["asof"] - 365 * day)].copy()
    pay["late"] = (pay["days_late"] > 5).astype(int)
    pay["short"] = (pay["amount_due"] - pay["amount_paid"]).clip(lower=0)
    g = pay.groupby("tenant_id")
    out["late_payments_12m"] = g["late"].sum()
    out["avg_days_late"] = g["days_late"].mean()
    out["arrears_ratio"] = g["short"].sum() / out["rent"]

    # ---- service requests
    rq = t["service_requests"].merge(asof.rename("asof"), left_on="tenant_id", right_index=True)
    rq = rq[(rq["created_date"] <= rq["asof"]) & (rq["created_date"] > rq["asof"] - 365 * day)].copy()
    rq["is_open"] = rq["resolved_date"].isna() | (rq["resolved_date"] > rq["asof"])
    rq["res_days"] = np.where(~rq["is_open"], (rq["resolved_date"] - rq["created_date"]).dt.days, np.nan)
    rq["open_age"] = np.where(rq["is_open"], (rq["asof"] - rq["created_date"]).dt.days, np.nan)
    g = rq.groupby("tenant_id")
    out["requests_12m"] = g.size()
    out["open_requests"] = g["is_open"].sum()
    out["avg_resolution_days"] = g["res_days"].mean()
    out["max_open_age_days"] = g["open_age"].max()

    # ---- communications (last 90 days)
    cm = t["communications"].merge(asof.rename("asof"), left_on="tenant_id", right_index=True)
    cm = cm[(cm["sent_at"] <= cm["asof"]) & (cm["sent_at"] > cm["asof"] - 90 * day)].copy()
    cm["complaint"] = (cm["sentiment"] < -0.3).astype(int)
    g = cm.groupby("tenant_id")
    out["n_comms_90d"] = g.size()
    out["avg_sentiment_90d"] = g["sentiment"].mean()
    out["min_sentiment_90d"] = g["sentiment"].min()
    out["complaint_msgs_90d"] = g["complaint"].sum()
    out["avg_response_hours"] = g["response_hours"].mean()

    # ---- satisfaction survey (latest before the decision date)
    sv = t["satisfaction_surveys"].merge(asof.rename("asof"), left_on="tenant_id", right_index=True)
    sv = sv[sv["survey_date"] <= sv["asof"]].sort_values("survey_date")
    out["satisfaction_score"] = sv.groupby("tenant_id")["score"].last()

    # ---- fill gaps with "no event" semantics (documented), never with outcome information
    for c in ["late_payments_12m", "arrears_ratio", "requests_12m", "open_requests", "n_comms_90d", "complaint_msgs_90d"]:
        out[c] = out[c].fillna(0.0)
    out["avg_days_late"] = out["avg_days_late"].fillna(0.0)
    out["max_open_age_days"] = out["max_open_age_days"].fillna(0.0)
    out["avg_sentiment_90d"] = out["avg_sentiment_90d"].fillna(0.0)
    out["min_sentiment_90d"] = out["min_sentiment_90d"].fillna(0.0)
    out["avg_resolution_days"] = out["avg_resolution_days"].fillna(out["avg_resolution_days"].median())
    out["avg_response_hours"] = out["avg_response_hours"].fillna(out["avg_response_hours"].median())
    out["survey_missing"] = out["satisfaction_score"].isna().astype(int)
    out["satisfaction_score"] = out["satisfaction_score"].fillna(out["satisfaction_score"].median())
    out["has_arrears"] = (out["arrears_ratio"] > 0.02).astype(int)
    return out[FEATURES].astype(float)
