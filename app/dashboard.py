"""Tenant Retention & Risk Dashboard  -  streamlit run app/dashboard.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from pmsdi.config import ARM_LABELS, ARMS, FIG_DIR, OUT_DIR, PROCESSED_DIR, ARTIFACT_DIR
from pmsdi.data import db

st.set_page_config(page_title="Tenant Retention Decision Intelligence", page_icon=None, layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1250px;}
.rec-card {border-left: 5px solid #0e7c66; background: #e4f0ec; padding: 1rem 1.2rem; border-radius: 4px;}
.warn-card {border-left: 5px solid #c27c0e; background: #f6ecd9; padding: .8rem 1.1rem; border-radius: 4px; font-size: .92rem;}
.small {font-size: .85rem; color: #4b5563;}
</style>""", unsafe_allow_html=True)


def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def show_img(path, caption=None):
    try:
        st.image(str(path), caption=caption, width="stretch")
    except TypeError:
        st.image(str(path), caption=caption, use_container_width=True)


@st.cache_resource(show_spinner="Loading models and database ...")
def get_service():
    from pmsdi.agent.graph import Agent
    from pmsdi.service import DecisionService
    svc = DecisionService()
    return svc, Agent(svc)


if not (ARTIFACT_DIR / "models.joblib").exists() or not (ROOT / "data" / "pms.db").exists():
    st.error("No database / trained models found. Run `python scripts/run_all.py` (or steps 02 and 03) first.")
    st.stop()

svc, agent = get_service()
cfg, cur = svc.cfg, svc.cfg.currency
demo = json.loads((PROCESSED_DIR / "demo_tenants.json").read_text()) if (PROCESSED_DIR / "demo_tenants.json").exists() else {}

# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("### Decision Intelligence")
st.sidebar.caption("Cross-module PMS reasoning and counterfactual intervention.")
page = st.sidebar.radio("Navigate", ["Tenant decision", "Portfolio plan", "What-if explorer", "Evidence and evaluation", "Audit and monitoring"])
evaluator = False
if svc.oracle is not None:
    evaluator = st.sidebar.toggle("Evaluator mode: reveal hidden ground truth", value=False,
                                  help="Only possible because the PMS is synthetic. Shows the TRUE renewal probabilities and hidden concern.")
st.sidebar.markdown("---")
st.sidebar.caption(f"Agent engine: {agent.engine_name}")
if st.sidebar.button("Reset demo (clear actions, outcomes, audit)"):
    from pmsdi.service import reset_demo
    reset_demo(svc)
    st.session_state.clear()
    st.sidebar.success("Demo state cleared.")


def pick_tenant(key):
    opts = {"Enter a tenant id": None} | {f"{k}  (#{v['tenant_id']})": v["tenant_id"] for k, v in demo.items()}
    choice = st.selectbox("Demo tenant", list(opts), key=f"{key}_demo")
    top = svc.tenants(limit=1).index[0]
    default = opts[choice] if opts[choice] else int(top)
    return int(st.number_input("Tenant id", min_value=int(svc.current_ids.min()), max_value=int(svc.current_ids.max()),
                               value=int(default), key=f"{key}_tid_{default}"))


def true_row(tid):
    return svc.oracle.loc[tid] if (evaluator and svc.oracle is not None) else None


# ================================================================== PAGE 1
if page == "Tenant decision":
    st.title("Tenant Retention & Risk Dashboard")
    st.caption("Identify tenants at risk and evaluate counterfactual interventions - then approve, edit or reject the recommendation.")
    c1, c2 = st.columns([3, 1])
    with c1:
        tid = pick_tenant("p1")
    with c2:
        st.write("")
        st.write("")
        run = st.button("Run agentic analysis", type="primary")
    if run:
        with st.status("Agent running ...", expanded=True) as status:
            state = agent.recommend(tid)
            for line in state["trace"]:
                st.write(line)
            status.update(label="Analysis complete", state="complete", expanded=False)
        st.session_state["state"], st.session_state["done"] = state, None
    state = st.session_state.get("state")
    if not state or state["tenant_id"] != tid:
        st.info("Select a tenant and press **Run agentic analysis**.")
        st.stop()

    est, best = state["estimates"], state["recommended"]
    st.markdown(f"**Trigger:** {state['trigger']}  |  **Predicted non-renewal risk:** {state['risk']:.0%}")
    st.write(state["summary"])

    st.subheader("Causal estimations")
    cols = st.columns(len(ARMS))
    base = est.iloc[0]["p_renew"]
    for col, (_, r) in zip(cols, est.iterrows()):
        label = "Base renewal probability" if r["action"] == "no_action" else f"If {ARM_LABELS[r['action']].lower()}"
        col.metric(label, f"{r['p_renew']:.1%}", None if r["action"] == "no_action" else f"{r['uplift'] * 100:+.1f} pp",
                   delta_color="normal" if r["feasible"] else "off")
        if not r["feasible"]:
            col.caption(f"blocked: {r['reason_blocked']}")
    t = true_row(tid)
    if t is not None:
        st.markdown(f"<div class='warn-card'><b>Ground truth (evaluator mode)</b> - hidden concern: <b>{t['concern']}</b>. "
                    + " | ".join(f"{ARM_LABELS[a]}: true P(renew) {t['p_' + a]:.1%}" for a in ARMS) + "</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small'>Individual estimates are noisy: the DR-learner and the S-learner disagree by "
                f"{est['estimator_spread'].max():.1%} at most for this tenant. Treat them as ranked evidence, not exact probabilities.</div>",
                unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Financial trade-offs")
        tbl = est[["label", "p_renew", "uplift", "cost", "net_value_vs_no_action", "feasible", "estimator_spread"]].rename(columns={
            "label": "Action", "p_renew": "P(renew)", "uplift": "Uplift", "cost": f"Cost ({cur})",
            "net_value_vs_no_action": f"Net value vs no action ({cur})", "feasible": "Allowed", "estimator_spread": "Estimator spread"})
        show_df(tbl.style.format({"P(renew)": "{:.1%}", "Uplift": "{:+.1%}", f"Cost ({cur})": "{:,.0f}",
                                  f"Net value vs no action ({cur})": "{:+,.0f}", "Estimator spread": "{:.1%}"}), hide_index=True)
        st.bar_chart(est.set_index("label")["net_value_vs_no_action"], height=220)
    with right:
        st.subheader("Agent recommendation")
        st.markdown(f"<div class='rec-card'>{state['explanation'].replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
        st.caption("Net value = P(renew) x retained rent (12-month lease + one month of avoided turnover) - intervention cost.")

    ev = state["evidence"]
    tabs = st.tabs(["Payments", "Service requests", "Messages", "Past actions"])
    with tabs[0]:
        show_df(ev["payments"], hide_index=True)
    with tabs[1]:
        show_df(ev["requests"], hide_index=True)
    with tabs[2]:
        show_df(ev["messages"], hide_index=True)
    with tabs[3]:
        show_df(ev["actions"], hide_index=True) if len(ev["actions"]) else st.caption("No earlier actions for this tenant.")

    st.subheader("Human-in-the-loop approval")
    done = st.session_state.get("done")
    if done is None:
        labels = {a: ARM_LABELS[a] + ("" if est.set_index("action").loc[a, "feasible"] else "  (blocked by policy)") for a in ARMS}
        a1, a2, a3 = st.columns([2, 1, 2])
        choice = a1.selectbox("Finalised action to execute in PMS", ARMS, index=ARMS.index(best), format_func=lambda a: labels[a])
        who = a2.text_input("Approver", value="property.manager")
        note = a3.text_input("Note (optional)")
        b1, b2, _ = st.columns([1, 1, 3])
        if b1.button("Approve and execute", type="primary"):
            out = agent.decide(state, True, who, choice, note)
            st.session_state["done"] = out
            st.rerun()
        if b2.button("Reject"):
            st.session_state["done"] = agent.decide(state, False, who, None, note)
            st.rerun()
    else:
        if done.get("execution", {}).get("error"):
            st.error(done["execution"]["error"])
        elif done.get("execution"):
            x = done["execution"]
            st.success(f"Executed **{ARM_LABELS[x['approved_action']]}** for tenant {x['tenant_id']} (cost {cur}{x['cost']:,.0f}); "
                       f"{'manager OVERRODE the recommendation' if x['overridden'] else 'matches the recommendation'}.")
            if st.button("Simulate renewal outcome (demo only)"):
                r = svc.simulate_outcome(tid)
                st.write(f"Outcome recorded: tenant {'**renewed**' if r['renewed'] else '**did not renew**'} (drawn from the hidden ground truth).")
        else:
            st.warning("Recommendation rejected - logged in the audit trail, nothing executed.")
        with st.expander("Execution trace"):
            st.write("\n\n".join(done["trace"]))
        if st.button("Analyse another action for this tenant"):
            st.session_state["done"] = None
            st.rerun()

# ================================================================== PAGE 2
elif page == "Portfolio plan":
    st.title("Portfolio plan under a budget")
    st.write("The optimiser picks one allowed action per tenant to maximise expected net value without exceeding the budget "
             "(an exact multiple-choice knapsack, solved as a MILP).")
    pct = st.slider("Budget as % of one month of total rent roll", 0.0, 15.0, float(cfg.budget_pct_rent_roll * 100), 0.5) / 100
    plan, s = svc.portfolio_plan(pct)
    m = st.columns(5)
    m[0].metric("Tenants in renewal window", f"{s['tenants']:,}")
    m[1].metric("Tenants treated", f"{s['treated']:,}")
    m[2].metric("Spend / budget", f"{cur}{s['spend']:,.0f}", f"{s['spend'] / max(s['budget'], 1):.0%} of budget", delta_color="off")
    m[3].metric("Expected extra renewals", f"{s['expected_extra_renewals']:.0f}")
    m[4].metric("Expected net gain", f"{cur}{s['expected_net_gain']:,.0f}")
    st.bar_chart(plan["action"].map(ARM_LABELS).value_counts(), height=240)
    if evaluator:
        ids = plan.index
        o = svc.oracle.loc[ids]
        ar = np.arange(len(ids))
        act = plan["action"].map({a: i for i, a in enumerate(ARMS)}).to_numpy()
        Pt = o[[f"p_{a}" for a in ARMS]].to_numpy()
        V = svc.F.loc[ids, "rent"].to_numpy() * (cfg.lease_months + cfg.turnover_months)
        true_gain = ((Pt[ar, act] - Pt[:, 0]) * V).sum() - plan["cost"].sum()
        st.markdown(f"<div class='warn-card'><b>Ground truth:</b> true extra renewals {(Pt[ar, act] - Pt[:, 0]).sum():.0f}; "
                    f"true net gain {cur}{true_gain:,.0f} (model expected {cur}{s['expected_net_gain']:,.0f}).</div>", unsafe_allow_html=True)
    st.subheader("Highest-value actions")
    show_df(plan[plan["action"] != "no_action"].sort_values("net_value_gain", ascending=False).head(50)
            .style.format({"p_renew_if_action": "{:.1%}", "uplift": "{:+.1%}", "cost": "{:,.0f}", "net_value_gain": "{:+,.0f}", "risk": "{:.0%}"}))
    st.divider()
    st.subheader("Tenant non-renewal risk ranking")

    risk_df = svc.tenants()

    min_risk = st.slider(
        "Minimum predicted non-renewal risk",
        0.0, 1.0, 0.0, 0.05
    )

    risk_df = risk_df[risk_df["risk_non_renewal"] >= min_risk]

    st.caption(f"Showing {len(risk_df):,} current tenants matching the filter.")

    show_df(
        risk_df.head(100).reset_index().rename(
            columns={"index": "tenant_id"}
        ).style.format({
            "rent": "{:,.0f}",
            "risk_non_renewal": "{:.1%}",
            "satisfaction_score": "{:.1f}"
        }),
        hide_index=True
    )

# ================================================================== PAGE 3
elif page == "What-if explorer":
    st.title("What-if explorer")
    st.markdown("<div class='warn-card'>This edits the tenant's <b>observed features</b> (e.g. 'what if there were no open requests?') and re-runs the model. "
                "It is a scenario tool, not a causal claim about changing that feature; the causal claims are the per-action estimates.</div>", unsafe_allow_html=True)
    tid = pick_tenant("p3")
    f = svc.F.loc[tid]
    c = st.columns(3)
    ov = {
        "open_requests": c[0].slider("Open maintenance requests", 0, 6, int(f["open_requests"])),
        "avg_resolution_days": c[0].slider("Average resolution time (days)", 1, 60, int(min(f["avg_resolution_days"], 60))),
        "late_payments_12m": c[1].slider("Late payments in 12 months", 0, 12, int(min(f["late_payments_12m"], 12))),
        "arrears_ratio": c[1].slider("Arrears (share of one month's rent)", 0.0, 1.0, float(min(f["arrears_ratio"], 1.0)), 0.05),
        "rent_vs_market": c[2].slider("Rent vs market", -0.25, 0.30, float(np.clip(f["rent_vs_market"], -0.25, 0.30)), 0.01),
        "avg_sentiment_90d": c[2].slider("Average message sentiment", -1.0, 1.0, float(f["avg_sentiment_90d"]), 0.05),
        "satisfaction_score": c[0].slider("Satisfaction (1-5)", 1.0, 5.0, float(f["satisfaction_score"]), 0.5),
    }
    r0, r1 = svc.recommend(tid), svc.recommend(tid, ov)
    a, b = st.columns(2)
    for col, title, r in ((a, "Current situation", r0), (b, "Scenario", r1)):
        with col:
            st.subheader(title)
            st.metric("Predicted non-renewal risk", f"{r['risk']:.0%}")
            st.markdown(f"<div class='rec-card'>{r['explanation'].replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
            show_df(r["table"][["label", "p_renew", "uplift", "net_value_vs_no_action", "feasible"]].style.format(
                {"p_renew": "{:.1%}", "uplift": "{:+.1%}", "net_value_vs_no_action": "{:+,.0f}"}), hide_index=True)

# ================================================================== PAGE 4
elif page == "Evidence and evaluation":
    st.title("Evidence and evaluation")
    if not (OUT_DIR / "EVALUATION_REPORT.md").exists():
        st.warning("Run `python scripts/04_evaluate.py` to generate the evaluation outputs.")
        st.stop()
    t = st.tabs(["Risk models", "Causal vs truth", "Decision policies", "Logged-data checks", "Off-policy evaluation", "Sensitivity", "Full report"])

    def csv(name, **kw):
        p = OUT_DIR / name
        return pd.read_csv(p, **kw) if p.exists() else None
    with t[0]:
        st.write("Renewal-risk baselines. AUROC = ranking quality; AUPRC = precision/recall trade-off (chance level = prevalence).")
        d = csv("risk_baselines.csv", index_col=0)
        show_df(d[["accuracy", "precision", "recall", "f1", "auroc", "auprc", "brier", "ece"]].style.format("{:.3f}")) if d is not None else None
        c = csv("prediction_vs_ceiling.csv", index_col=0)
        show_df(c.style.format("{:.3f}")) if c is not None else None
        for f in ("01_roc_pr.png", "03_confusion_matrix.png"):
            if (FIG_DIR / f).exists():
                show_img(FIG_DIR / f)
    with t[1]:
        d = csv("causal_summary.csv", index_col=0)
        show_df(d.style.format("{:.3f}")) if d is not None else None
        for f in ("08_cate_calibration.png", "09_uplift_service_priority.png", "09_uplift_rent_concession.png"):
            if (FIG_DIR / f).exists():
                show_img(FIG_DIR / f)
    with t[2]:
        d = csv("policy_results.csv", index_col=0)
        show_df(d[["net_value_per_tenant", "uplift_vs_no_action_per_tenant", "regret_vs_oracle_per_tenant", "pct_of_oracle_gain", "spend_pct_budget",
                   "proposals_violating_rules_pct", "blocked_by_budget_pct"]].style.format("{:,.1f}")) if d is not None else None
        show_img(FIG_DIR / "10_policy_value.png") if (FIG_DIR / "10_policy_value.png").exists() else None
        d = csv("policy_paired_comparisons.csv", index_col=0)
        show_df(d.style.format({"mean_gain_per_tenant": "{:,.1f}", "ci_lo": "{:,.1f}", "ci_hi": "{:,.1f}"})) if d is not None else None
    with t[3]:
        st.write("Checks that need no ground truth, so they can be repeated on real PMS history.")
        for f in ("07_association_vs_causation.png", "05_overlap.png", "06_covariate_balance.png"):
            if (FIG_DIR / f).exists():
                show_img(FIG_DIR / f)
        for n in ("refutations.csv", "e_values.csv"):
            d = csv(n)
            show_df(d) if d is not None else None
    with t[4]:
        show_img(FIG_DIR / "11_off_policy_evaluation.png") if (FIG_DIR / "11_off_policy_evaluation.png").exists() else None
        d = csv("ope.csv")
        show_df(d.style.format({c: "{:,.0f}" for c in d.columns if c != "policy"})) if d is not None else None
    with t[5]:
        d = csv("sensitivity_results.csv")
        if d is None:
            st.info("Run `python scripts/05_sensitivity.py` for this tab.")
        else:
            show_df(d.round(3))
            for f in ("12_sens_hidden_confounding.png", "13_sens_policy_advantage.png"):
                if (FIG_DIR / f).exists():
                    show_img(FIG_DIR / f)
    with t[6]:
        st.markdown((OUT_DIR / "EVALUATION_REPORT.md").read_text())

# ================================================================== PAGE 5
else:
    st.title("Audit and monitoring")
    a = db.read_audit(svc.engine)
    st.subheader("Audit trail")
    st.caption("Every recommendation, approval/rejection, execution and outcome. The approved action can differ from the recommended one.")
    show_df(a, hide_index=True) if len(a) else st.info("Nothing logged yet - run an analysis on the Tenant decision page.")
    st.subheader("Monitor and learn: predicted vs realised")
    m = svc.monitor()
    if m.empty:
        st.info("No executed decisions with recorded outcomes yet. Execute an action and press 'Simulate renewal outcome'.")
    else:
        k = st.columns(3)
        k[0].metric("Decisions tracked", len(m))
        k[1].metric("Mean predicted P(renew)", f"{m['p_pred'].mean():.1%}")
        k[2].metric("Realised renewal rate", f"{m['renewed'].mean():.1%}")
        if "p_true" in m and evaluator:
            st.caption(f"Hidden truth: mean true P(renew) under the chosen actions = {m['p_true'].mean():.1%}")
        show_df(m, hide_index=True)
        st.caption("A handful of decisions says little: compare predicted and realised rates over hundreds of decisions "
                   "(and by action) before concluding the policy over- or under-estimates.")
