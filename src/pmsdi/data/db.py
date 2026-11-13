"""Relational PMS schema (SQLite by default, PostgreSQL via DATABASE_URL).

Entity-relationship summary
    properties 1--* units 1--1 tenants 1--* leases
    tenants 1--* payments | service_requests | communications | satisfaction_surveys
    tenants 1--* interventions (actions) 1--* renewal_outcomes
    audit_log records every proposed / approved / executed action.
"""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
from sqlalchemy import (Column, DateTime, Float, ForeignKey, Index, Integer, MetaData, String, Table, Text,
                        create_engine, delete, insert, select, text)

from ..config import database_url
from .features import parse_dates

md = MetaData()
Table("properties", md, Column("property_id", Integer, primary_key=True), Column("city", String(60)),
      Column("n_units", Integer), Column("age_years", Float), Column("violations_per_100_units_12m", Float),
      Column("opex_ratio", Float), Column("market_index", Float))
Table("units", md, Column("unit_id", Integer, primary_key=True), Column("property_id", ForeignKey("properties.property_id")),
      Column("bedrooms", Integer), Column("area_sqft", Float), Column("market_rent", Float))
Table("tenants", md, Column("tenant_id", Integer, primary_key=True), Column("unit_id", ForeignKey("units.unit_id")),
      Column("move_in_date", DateTime), Column("cohort", String(12)))
Table("leases", md, Column("lease_id", Integer, primary_key=True), Column("tenant_id", ForeignKey("tenants.tenant_id")),
      Column("unit_id", ForeignKey("units.unit_id")), Column("start_date", DateTime), Column("end_date", DateTime),
      Column("decision_date", DateTime), Column("monthly_rent", Float), Column("status", String(20)),
      Column("decided_at", DateTime))
Table("payments", md, Column("payment_id", Integer, primary_key=True), Column("tenant_id", ForeignKey("tenants.tenant_id")),
      Column("lease_id", ForeignKey("leases.lease_id")), Column("due_date", DateTime), Column("amount_due", Float),
      Column("amount_paid", Float), Column("paid_date", DateTime), Column("days_late", Integer))
Table("service_requests", md, Column("request_id", Integer, primary_key=True),
      Column("tenant_id", ForeignKey("tenants.tenant_id")), Column("unit_id", ForeignKey("units.unit_id")),
      Column("category", String(30)), Column("created_date", DateTime), Column("resolved_date", DateTime),
      Column("priority", String(10)))
Table("communications", md, Column("comm_id", Integer, primary_key=True), Column("tenant_id", ForeignKey("tenants.tenant_id")),
      Column("sent_at", DateTime), Column("channel", String(20)), Column("sentiment", Float),
      Column("response_hours", Float), Column("message_text", Text))
Table("satisfaction_surveys", md, Column("survey_id", Integer, primary_key=True),
      Column("tenant_id", ForeignKey("tenants.tenant_id")), Column("survey_date", DateTime), Column("score", Integer))
Table("interventions", md, Column("intervention_id", Integer, primary_key=True),
      Column("tenant_id", ForeignKey("tenants.tenant_id")), Column("action", String(30)), Column("assigned_by", String(30)),
      Column("logged_propensity", Float), Column("cost", Float), Column("status", String(20)),
      Column("executed_at", DateTime))
Table("renewal_outcomes", md, Column("outcome_id", Integer, primary_key=True, autoincrement=True),
      Column("tenant_id", ForeignKey("tenants.tenant_id")), Column("lease_id", Integer), Column("renewed", Integer),
      Column("source", String(30)), Column("decided_at", DateTime))
Table("audit_log", md, Column("audit_id", Integer, primary_key=True, autoincrement=True), Column("ts", DateTime),
      Column("tenant_id", Integer), Column("event", String(40)), Column("payload", Text))
Table("feature_store", md, Column("tenant_id", Integer, primary_key=True))   # columns added dynamically (see write_features)
Index("ix_pay_tenant", md.tables["payments"].c.tenant_id)
Index("ix_req_tenant", md.tables["service_requests"].c.tenant_id)
Index("ix_com_tenant", md.tables["communications"].c.tenant_id)

LOAD_ORDER = ["properties", "units", "tenants", "leases", "payments", "service_requests", "communications",
              "satisfaction_surveys", "interventions", "renewal_outcomes", "audit_log"]


def get_engine(url: str | None = None):
    return create_engine(url or database_url(), future=True)


def reset_schema(engine):
    md.drop_all(engine)
    md.create_all(engine)


def write_tables(engine, tables: dict, chunksize: int = 20000):
    reset_schema(engine)
    for name in LOAD_ORDER:
        df = tables.get(name)
        if df is None or df.empty:
            continue
        cols = [c.name for c in md.tables[name].columns if c.name in df.columns]
        df[cols].to_sql(name, engine, if_exists="append", index=False, chunksize=chunksize)


def write_features(engine, F: pd.DataFrame):
    F.reset_index().to_sql("feature_store", engine, if_exists="replace", index=False, chunksize=20000)


def read_table(engine, name: str) -> pd.DataFrame:
    df = pd.read_sql_table(name, engine)
    return parse_dates({name: df})[name]


def load_tables(engine) -> dict:
    return {n: read_table(engine, n) for n in LOAD_ORDER}


def read_features(engine) -> pd.DataFrame:
    return pd.read_sql_table("feature_store", engine).set_index("tenant_id")


def log_audit(engine, tenant_id: int, event: str, payload: dict):
    with engine.begin() as con:
        con.execute(insert(md.tables["audit_log"]).values(
            ts=datetime.now(), tenant_id=int(tenant_id), event=event, payload=json.dumps(payload, default=str)))


def read_audit(engine, tenant_id: int | None = None) -> pd.DataFrame:
    q = "SELECT * FROM audit_log" + (f" WHERE tenant_id = {int(tenant_id)}" if tenant_id is not None else "")
    df = pd.read_sql_query(text(q + " ORDER BY audit_id DESC"), engine)
    return df
