"""Step 4 - the full evaluation: prediction, causal estimation (vs ground truth), policy value / regret,
identification diagnostics, refutations, E-values and off-policy evaluation.  Writes outputs/*.csv, figures, report."""
import _bootstrap  # noqa: F401
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from pmsdi.config import ARMS, K, OUT_DIR, SYNTH_DIR, Config
from pmsdi.data import db
from pmsdi.data.features import FEATURES
from pmsdi.decision import interventions as iv
from pmsdi.decision import optimizer as opt
from pmsdi.eval import causal_eval, diagnostics as dg, ope as opem, plots, policy_eval
from pmsdi.models import causal
from pmsdi.pipeline import POLICY_ORDER, fit_risk_model, run_experiment, split_xy
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")
t0 = time.time()
cfg = Config.load()
engine = db.get_engine()
tables = db.load_tables(engine)
oracle = pd.read_csv(SYNTH_DIR / "oracle.csv.gz", index_col=0)
print("running experiment on database contents ...")
R = run_experiment(cfg, data=(tables, oracle, {}), with_econml=True, verbose=True)
H = R["headline"]
Xh, Xc, A, Y, hist = R["Xh"], R["Xc"], R["A"], R["Y"], R["hist"]
P_true, u = R["P_true"], R["u"]
md = []
def md_table(df, fmt="{:.3f}"):
    d = df.rename_axis(df.index.name or "item").reset_index()
    out = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    for _, r in d.iterrows():
        out += "| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |\n"
    return out

# ------------------------------------------------------------------ 1. prediction
y_true_noact = 1 - (u < P_true[:, 0]).astype(int)                 # actual outcome had nobody been treated
auc_model = roc_auc_score(y_true_noact, R["risk"]); ap_model = average_precision_score(y_true_noact, R["risk"])
auc_ceiling = roc_auc_score(y_true_noact, 1 - P_true[:, 0]); ap_ceiling = average_precision_score(y_true_noact, 1 - P_true[:, 0])
pred = pd.DataFrame({"AUROC": [auc_model, auc_ceiling], "AUPRC": [ap_model, ap_ceiling]}, index=["risk model (HGB)", "ceiling: true probabilities"])
pred.loc["chance"] = [0.5, y_true_noact.mean()]
pred.to_csv(OUT_DIR / "prediction_vs_ceiling.csv")
md += ["## 1. Renewal-risk prediction\n", "Evaluated on the decision cohort against the outcome that would occur with NO action.\n",
       md_table(pred), "\nThe ceiling row scores the *true* renewal probabilities: no model can beat it, because each renewal is still a coin flip. "
       "AUROC near the ceiling means the risk model has extracted what the features contain.\n"]

# ------------------------------------------------------------------ 2. causal estimation vs truth
rows, summ = [], []
for name, df in R["causal"].items():
    d = df.copy(); d.insert(0, "estimator", name); rows.append(d.reset_index())
    summ.append({"estimator": name, "mean_ate_abs_error": df["ate_abs_error"].mean(), "mean_pehe": df["pehe"].mean(),
                 "pehe_if_constant_ate": df["pehe_constant_ate_baseline"].mean(), "mean_spearman": df["spearman_tau"].mean(),
                 "mean_uplift_ratio": df["uplift_ratio"].mean(), **R["decision_quality"][name]})
cm_all = pd.concat(rows); cm_all.to_csv(OUT_DIR / "causal_metrics_by_arm.csv", index=False)
cs = pd.DataFrame(summ).set_index("estimator"); cs.to_csv(OUT_DIR / "causal_summary.csv")
md += ["## 2. Causal / counterfactual estimates vs ground truth\n",
       "ATE error = |estimated - true average effect|. PEHE = RMS error of the per-tenant effect (lower is better; compare with "
       "`pehe_if_constant_ate`, the error you would make by giving everyone the true average effect). Spearman / uplift ratio measure whether the *ranking* of tenants by effect is right "
       "(0 = random, 1 = perfect).\n", md_table(cs)]
cal = {arm: causal_eval.cate_calibration((R["P_hat"][H] - R["P_hat"][H][:, [0]])[:, a], (P_true - P_true[:, [0]])[:, a])
       for a, arm in enumerate(ARMS) if a > 0}
for arm, d in cal.items():
    d.to_csv(OUT_DIR / f"cate_calibration_{arm}.csv", index=False)
plots.cate_calibration(cal, H)
for a, arm in enumerate(ARMS):
    if a == 0: continue
    curves = {n: causal_eval.uplift_curve((P - P[:, [0]])[:, a], (P_true - P_true[:, [0]])[:, a]) for n, P in R["P_hat"].items() if n != "naive"}
    curves["perfect (oracle ranking)"] = causal_eval.uplift_curve((P_true - P_true[:, [0]])[:, a], (P_true - P_true[:, [0]])[:, a])
    plots.uplift_curves(curves, arm, f"09_uplift_{arm}.png")

# ------------------------------------------------------------------ 3. decision policies
pol_tbl = R["policy"][H].loc[POLICY_ORDER]
pol_tbl.to_csv(OUT_DIR / "policy_results.csv")
per = R["per_tenant"][H]
cmp_rows = []
for a_, b_ in [("B Causal + constraints", "B3 Risk + rules"), ("B Causal + constraints", "A Causal (no constraints)"),
               ("B Causal + constraints", "B1 Rule-based"), ("A Causal (no constraints)", "B3 Risk + rules")]:
    d, lo, hi = policy_eval.paired_diff_ci(per[a_], per[b_])
    cmp_rows.append({"comparison": f"{a_}  minus  {b_}", "mean_gain_per_tenant": d, "ci_lo": lo, "ci_hi": hi, "significant": lo > 0 or hi < 0})
pc = pd.DataFrame(cmp_rows).set_index("comparison"); pc.to_csv(OUT_DIR / "policy_paired_comparisons.csv")
by_est = pd.DataFrame({e: {"pct_of_oracle_gain (policy B)": R["policy"][e].loc["B Causal + constraints", "pct_of_oracle_gain"],
                           "regret_per_tenant": R["policy"][e].loc["B Causal + constraints", "regret_vs_oracle_per_tenant"]} for e in R["policy"]}).T
by_est.to_csv(OUT_DIR / "policy_by_estimator.csv")
plots.policy_bars(pol_tbl)
md += [f"## 3. Decision policies (estimator = {H})\n",
       "Net value = P(renew) x retained rent - intervention cost, evaluated with the TRUE probabilities. Non-constraint-aware policies go "
       "through the execution layer, which blocks rule violations and anything over budget. 'Oracle' = best possible feasible, within-budget plan.\n",
       md_table(pol_tbl[["net_value_per_tenant", "ci_lo", "ci_hi", "uplift_vs_no_action_per_tenant", "regret_vs_oracle_per_tenant", "pct_of_oracle_gain",
                          "spend_pct_budget", "proposals_violating_rules_pct", "blocked_by_budget_pct"]], "{:.1f}"),
       "\nPaired bootstrap comparisons:\n", md_table(pc, "{:.1f}"), "\nPolicy B by estimator:\n", md_table(by_est, "{:.2f}")]

# ------------------------------------------------------------------ 4. identification diagnostics on the LOGGED history (no oracle needed)
inter = tables["interventions"].set_index("tenant_id").loc[Xh.index]
is_rct = (inter["assigned_by"] == "randomized_pilot").to_numpy()
obs = ~is_rct
Xo, Ao, Yo = Xh[obs], A[obs], Y[obs]
e = dg.propensity_oof(Xo, Ao)
ov = dg.overlap_table(e, Ao); ov.to_csv(OUT_DIR / "overlap.csv"); plots.overlap(e, Ao)
bal = {ARMS[a]: dg.covariate_balance(Xo, Ao, e, a) for a in range(1, K)}
for arm, d in bal.items(): d.to_csv(OUT_DIR / f"balance_{arm}.csv")
plots.love(bal)
aipw = dg.aipw_all_arms(Xo, Ao, Yo, e=None)
naive = dg.naive_ate(Ao, Yo)
rct = dg.rct_benchmark(A, Y, is_rct)
Ph = oracle.loc[Xh.index, [f"p_{a}" for a in ARMS]].to_numpy()
truth_h = pd.Series({ARMS[a]: (Ph[:, a] - Ph[:, 0]).mean() for a in range(1, K)})
eff = pd.DataFrame({"truth": truth_h, "naive": naive, "aipw": aipw["ate"], "aipw_lo": aipw["ci_lo"], "aipw_hi": aipw["ci_hi"],
                    "rct": rct["ate_rct"], "rct_lo": rct["ci_lo_rct"], "rct_hi": rct["ci_hi_rct"]})
eff.to_csv(OUT_DIR / "identification_effects.csv"); plots.effect_comparison(eff)
ev = pd.DataFrame({arm: dg.e_value(aipw.loc[arm, "p_arm"], aipw.loc[arm, "p_control"], aipw.loc[arm, "ci_lo"]) for arm in aipw.index}).T
ev.to_csv(OUT_DIR / "e_values.csv")
print("running refutation suite ...")
ref = dg.refutation_suite(Xo, Ao, Yo, n_placebo=6, n_subset=6); ref.to_csv(OUT_DIR / "refutations.csv")
md += ["## 4. Can we trust the causal estimates on logged data? (checks that work on REAL data)\n",
       "Effect of each action on P(renew), estimated three ways, next to the hidden truth:\n", md_table(eff, "{:+.3f}"),
       "\n**Overlap** (every action must have a realistic chance of being chosen for every kind of tenant):\n", md_table(ov),
       "\n**Refutations** - placebo (shuffled actions -> effect ~0), random extra covariate (estimate unchanged), 80% subsets (stable):\n", md_table(ref),
       "\n**E-values** - strength of an unmeasured confounder needed to explain away the effect:\n", md_table(ev)]

# ------------------------------------------------------------------ 5. off-policy evaluation (counterfactual evaluation from logs)
print("off-policy evaluation ...")
rent_h = Xh["rent"].to_numpy(); Vh, ch, fh = iv.retained_value(rent_h, cfg), iv.costs(rent_h, cfg), iv.feasibility(Xh, cfg)
e_all = dg.propensity_oof(Xh, A)
P_cf = causal.crossfit_predict_p(lambda: causal.DRLearner(seed=cfg.seed), Xh, A, Y, n_splits=4)
risk_oof = cross_val_predict(HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.06, random_state=0),
                             Xh, 1 - Y, cv=StratifiedKFold(4, shuffle=True, random_state=0), method="predict_proba")[:, 1]
from pmsdi.decision import policies as pol
def mask(act): return np.where(fh[np.arange(len(act)), act], act, 0)
targets = {"No action": np.zeros(len(Xh), int), "B1 Rule-based": mask(pol.policy_rule_based(Xh)),
           "B3 Risk + rules": mask(pol.policy_risk_plus_rules(Xh, risk_oof, cfg)),
           "Causal, feasible (cross-fit)": opt.select_feasible(opt.net_value(P_cf, Vh, ch), fh),
           "Oracle, feasible": opt.select_feasible(opt.net_value(Ph, Vh, ch), fh)}
rows = []
for nm, act in targets.items():
    est_ = opem.ope_value(act, A, Y, e_all, P_cf, Vh, ch)
    truth = float((Ph[np.arange(len(act)), act] * Vh - ch[np.arange(len(act)), act]).mean())
    rows.append({"policy": nm, "truth": truth, **est_})
ope_df = pd.DataFrame(rows); ope_df.to_csv(OUT_DIR / "ope.csv", index=False); plots.ope(ope_df)
md += ["## 5. Off-policy evaluation (estimating a policy's value from logs only)\n",
       "TRUE = oracle value; IPS / SNIPS / DR / direct use only the logged actions and outcomes. Use this on real PMS history, where no oracle exists.\n",
       md_table(ope_df.set_index("policy"), "{:.0f}")]

(OUT_DIR / "EVALUATION_REPORT.md").write_text("# Evaluation report (auto-generated)\n\n" + "\n".join(md))
json.dump({"policy_B_pct_of_oracle_gain": float(pol_tbl.loc["B Causal + constraints", "pct_of_oracle_gain"]),
           "policy_B3_pct_of_oracle_gain": float(pol_tbl.loc["B3 Risk + rules", "pct_of_oracle_gain"])}, open(OUT_DIR / "summary.json", "w"))
print(f"\nreport -> {OUT_DIR / 'EVALUATION_REPORT.md'}   ({time.time() - t0:.0f}s)")
