# Agentic Property-Management Decision Intelligence for Tenant Retention

Working implementation of the Review-II slides: a synthetic-but-calibrated PMS, renewal-risk baselines, multi-arm causal
(counterfactual) estimators, a constraint-aware optimiser, a LangGraph agent with human approval + audit trail, a REST API,
a Streamlit dashboard, and an evaluation suite that scores everything against known ground truth.

> **Honest status.** All outcomes in this repository come from a *synthetic* world whose treatment effects are *assumptions*
> (see `docs/ASSUMPTIONS.md`). The code is real and tested (`pytest`: 18 tests); the numbers show that the method works *when those
> assumptions hold* and how it degrades when they do not. They are not evidence about real tenants.

## Quick start (about 3 minutes)

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-optional.txt                 # optional: EconML, DoWhy
python scripts/run_all.py                                # calibrate -> generate DB -> train -> demo tenants -> evaluate (~2 min)
streamlit run app/dashboard.py                           # the demo UI
```
Other entry points: `python scripts/run_all.py --quick` (small smoke test), `--with-sensitivity` (+3 min), `--with-dowhy`,
`uvicorn pmsdi.api:app --app-dir src` (REST, docs at `/docs`), `python -m pytest -q`.
Tested on Python 3.12 (versions in `requirements-lock.txt`). Default database is SQLite (no setup); PostgreSQL via `DATABASE_URL` (see `docker-compose.yml` - **untested**).

## Slide -> code -> output map

| Slide | What it promises | Code | Output / where to see it |
|---|---|---|---|
| 14-15 data strategy / pipeline | public data -> clean -> map -> synthetic PMS | `scripts/00-02`, `data/calibrate.py`, `data/synthetic_pms.py`, `data/db.py` | `data/pms.db`, `data/processed/calibration.json`, `docs/DATA_SOURCES.md` |
| 10 layer 2 | entity resolution, unified graph, **feature store** | `data/db.py` (relational schema), `data/features.py` | table `feature_store` (23 features, as-of decision date) |
| 10 layer 3 / 16 | risk models, AUROC / AUPRC | `models/churn_baseline.py`, `scripts/03`, `scripts/06` (your Kaggle churn CSV) | `outputs/risk_baselines.csv`, figures 01-04 |
| 10 layer 4 / 9 / 13 | intervention library, **causal engine**, counterfactual simulator, optimiser | `decision/interventions.py`, `models/causal.py` (S/T/DR + EconML), `decision/optimizer.py` (exact MILP) | `outputs/causal_summary.csv`, figures 07-09 |
| 21 | 5 policies + metrics (value, regret, uplift, business value, compliance) | `decision/policies.py`, `eval/policy_eval.py` | `outputs/policy_results.csv`, figure 10 |
| 12 | agentic workflow Trigger ... Monitor | `agent/graph.py` (LangGraph), `service.py` (PMS tools) | trace shown in the dashboard; `audit_log` table |
| 17-20 | dashboard, human-in-the-loop, audit | `app/dashboard.py`, `api.py` | `streamlit run app/dashboard.py` |
| 13 | EconML / DoWhy | `models/causal.py::EconMLDR`, `scripts/08_dowhy_check.py` | `outputs/causal_summary.csv`, `outputs/dowhy_refutations_*.csv` |
| (extra) | is the causal estimate trustworthy on *logged* data? | `eval/diagnostics.py`, `eval/ope.py` | `outputs/EVALUATION_REPORT.md` sections 4-5, `scripts/05_sensitivity.py` |

**Not implemented (be upfront in the review):** anomaly detection, vendor/market data, a real NLP model (message sentiment is a synthetic field),
financial-impact models beyond the net-value formula, a React front end (the dashboard in your screenshots is Streamlit, so I built Streamlit),
the HUD operating-cost calibration (needs a file with those columns), the "landlord disputes" data, PostgreSQL testing, real downloads of the public datasets
(the loaders are tested on fabricated files with the same column names; run them on the real files and check `sources_used`).

## How the pieces fit

```
public CSVs --calibrate--> calibration.json --> synthetic_pms.generate() --> tables (SQLite/Postgres) --> feature_store
                                                      |                                                       |
                                                 oracle.csv.gz  (truth, hidden)                  history rows: X, logged action A, outcome Y
                                                      |                                                       |
                                                      |        risk model P(non-renewal|X)       causal model p_hat_a(X) for EVERY action a
                                                      |                                                       |
                                                      |                 net value NV_a = p_hat_a * V - cost_a
                                                      |        optimise: argmax over ALLOWED actions, total spend <= budget  (MILP)
                                                      |                                                       |
                                          scoring against truth <--------  agent: investigate -> recommend -> HUMAN APPROVES -> mock PMS action -> audit -> outcome
```

* **Decision rule** `a* = argmax_{a in allowed} p_hat_a(x) * V - cost_a`, with `V = rent x (12 + 1)` (next lease + avoided turnover).
* **Causal estimator (production):** cross-fitted *doubly-robust* learner - outcome models + propensity model -> pseudo-outcomes -> smoothed regression.
  Compared against naive means, S-learner, T-learner and EconML's DRLearner.
* **Counterfactual** = the model's `p_hat_a(x)` for actions that were *not* taken for that tenant.

## Reference run (seed 42, default config)

Risk prediction on the decision cohort: AUROC 0.736 / AUPRC 0.665 (chance 0.50 / 0.43; *ceiling* using the true probabilities 0.803 / 0.740). Do **not** expect the 0.91 of your Kaggle benchmark: a real renewal is still a coin flip even if you know the probability.

Effect of each action on P(renew) in the logged history:

| action | truth | naive (ignores confounding) | doubly-robust | randomised pilot only |
|---|---|---|---|---|
| service_priority | +0.041 | **-0.114** | +0.037 | +0.047 (CI -0.02..+0.11) |
| rent_concession | +0.056 | **-0.114** | +0.036 | +0.053 (CI -0.01..+0.12) |
| proactive_outreach | +0.030 | **-0.102** | +0.019 | +0.088 (CI +0.03..+0.15) |

Naive says every action *hurts* (managers treated the riskiest tenants); adjustment recovers the sign and rough size.
Average-effect error: naive 0.128, S-learner 0.029, **DR 0.012**. Per-tenant effect error (PEHE) is only about as good as predicting the average
(0.068 vs 0.072): individual effects are weakly identifiable - claim policy value, not per-tenant precision.

Policy value on 3,000 tenants (net value per tenant; regret is vs the best feasible within-budget plan):

| policy | net value | gain vs no action | % of oracle gain | rule violations |
|---|---|---|---|---|
| No action | 10,822 | 0 | 0 % | - |
| B1 Rule-based | 11,127 | +305 | 28 % | 6.8 % |
| B2 Risk-only alert | 11,051 | +229 | 21 % | 0 % |
| B3 Risk + rules | 11,146 | +324 | 30 % | 6.3 % |
| A Causal (no constraints) | 11,404 | +582 | 54 % | 19.8 % |
| **B Causal + constraints** | **11,494** | **+672** | **62 %** | **0 %** |
| Oracle (true effects) | 11,896 | +1,074 | 100 % | 0 % |

Paired bootstrap, B - B3 = +348 per tenant (95 % CI 290-404). Sensitivity (`outputs/sensitivity_results.csv`): B stays ahead of B3 for effect sizes 0.5x-1.5x and
up to strong hidden confounding, but advantage shrinks (e.g. effect 0.5x: ~43-55 % of oracle gain vs ~30 % for B3) and ATE error grows with hidden confounding.
Worth saying aloud: the S-learner is about as good as the DR-learner here; DR's advantage is its robustness guarantee, not a large empirical margin.

## Folder structure

```
app/dashboard.py            Streamlit UI (5 pages)             scripts/00..08, run_all.py   pipeline steps
src/pmsdi/config.py         all assumptions in one place        tests/                       18 pytest tests
src/pmsdi/data/             calibrate, synthetic_pms, db, features
src/pmsdi/models/           churn_baseline, causal              docs/                        data, assumptions, demo, Q&A
src/pmsdi/decision/         interventions, optimizer, policies  outputs/                     reference run (figures, CSVs, report)
src/pmsdi/eval/             causal_eval, policy_eval, diagnostics, ope, plots
src/pmsdi/agent/            graph (LangGraph), llm (optional wording)       src/pmsdi/service.py, api.py
```

## Fix these on your slides before the next review
1. **Slide 18 figures are labelled "ILLUSTRATIVE DUMMY RESULTS" / "SYNTHETIC DUMMY PREDICTIONS"** in the PDF. A reviewer who reads the footer will notice. Regenerate from real predictions
   (`scripts/06_kaggle_churn_benchmark.py --csv <your churn csv> --target <col> --positive <value>` writes the table, curves and a *consistent* confusion matrix from one split).
2. The slide-18 confusion matrix (130/20/9/21) implies precision 51 % / accuracy 83.9 %, not the 60 % / 87.2 % in the slide-17 table - they come from different runs.
3. The dashboard screenshot shows flat 50 % -> 58 % uplifts and "no_action" recommended: placeholder logic. The dashboard here uses real model output.
4. Slides say React; the screenshots are Streamlit. Say Streamlit.
5. The churn table (900 rows, 16.7 % positives) cannot be reproduced without the original CSV - keep it labelled "generic benchmark, not tenant retention".

## Suggested split among the three of you (edit to match who really did what)
* **Data / PMS backend:** `data/*`, `scripts/00-02`, `docs/DATA_SOURCES.md`, DB + feature store.
* **ML / causal / evaluation:** `models/*`, `decision/optimizer.py`, `eval/*`, `scripts/03-06`, `docs/ASSUMPTIONS.md`.
* **Agent / UI / integration:** `agent/*`, `service.py`, `api.py`, `app/dashboard.py`, `tests/test_agent_api.py`, `docs/DEMO_SCRIPT.md`.
Each person should be able to run and explain their own modules; everyone should be able to explain `docs/ASSUMPTIONS.md`.
