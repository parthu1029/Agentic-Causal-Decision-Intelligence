"""Optional - the same identification + refutation workflow with Microsoft DoWhy for ONE action vs no_action.
DoWhy writes the causal assumptions down explicitly (graph -> estimand -> estimate -> refute), which is what a
reviewer expects to see in a causal-inference project.   usage: python scripts/08_dowhy_check.py [--arm rent_concession]"""
import _bootstrap  # noqa: F401
import argparse
import warnings

import numpy as np
import pandas as pd

from pmsdi.config import ARMS, OUT_DIR, SYNTH_DIR, Config
from pmsdi.data import db
from pmsdi.data.features import FEATURES
from pmsdi.pipeline import split_xy

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="rent_concession", choices=ARMS[1:])
ap.add_argument("--n", type=int, default=5000)
ap.add_argument("--sims", type=int, default=20)
a = ap.parse_args()
from dowhy import CausalModel

engine = db.get_engine()
tables = db.load_tables(engine)
F = db.read_features(engine)[FEATURES]
oracle = pd.read_csv(SYNTH_DIR / "oracle.csv.gz", index_col=0)
Xh, _, A, Y, _ = split_xy(tables, oracle, F)
inter = tables["interventions"].set_index("tenant_id").loc[Xh.index]
keep = ((A == 0) | (A == ARMS.index(a.arm))) & (inter["assigned_by"] == "manager_log").to_numpy()   # observational rows only
df = Xh[keep].copy()
df["treated"] = (A[keep] != 0).astype(int)
df["renewed"] = Y[keep]
df = df.sample(min(a.n, len(df)), random_state=0).reset_index(drop=True)
confounders = [c for c in FEATURES if df[c].std() > 0]
print(f"{a.arm} vs no_action: n={len(df)}, treated share={df['treated'].mean():.2f}, adjusting for {len(confounders)} observed covariates")

model = CausalModel(data=df, treatment="treated", outcome="renewed", common_causes=confounders)
estimand = model.identify_effect(proceed_when_unidentifiable=True)
print("\nIDENTIFIED ESTIMAND (backdoor adjustment - assumes NO unmeasured confounding):")
try:
    print("  backdoor adjustment set:", sorted(estimand.get_backdoor_variables()))
except Exception:
    print("  (estimand identified; print `estimand` for the full expression)")
est = model.estimate_effect(estimand, method_name="backdoor.linear_regression", test_significance=True)
print(f"\nDoWhy ATE (linear-probability backdoor regression) = {est.value:+.4f}")
naive = df.groupby("treated")["renewed"].mean()
print(f"naive difference in means = {naive[1] - naive[0]:+.4f}")
truth = (oracle.loc[Xh.index[keep], f"p_{a.arm}"] - oracle.loc[Xh.index[keep], "p_no_action"]).mean()
print(f"TRUE average effect on these tenants (oracle) = {truth:+.4f}")
rows = []
for name, kw in [("random_common_cause", {}), ("placebo_treatment_refuter", {"placebo_type": "permute"}), ("data_subset_refuter", {"subset_fraction": 0.8})]:
    r = model.refute_estimate(estimand, est, method_name=name, num_simulations=a.sims, **kw)
    print(f"\n{name}: original {est.value:+.4f} -> new {r.new_effect:+.4f}  (p-value {getattr(r, 'refutation_result', {}).get('p_value') if isinstance(getattr(r, 'refutation_result', None), dict) else 'n/a'})")
    rows.append({"refuter": name, "original_effect": est.value, "new_effect": r.new_effect, "p_value": getattr(r, "refutation_result", {}).get("p_value") if isinstance(getattr(r, "refutation_result", None), dict) else None})
pd.DataFrame(rows).to_csv(OUT_DIR / f"dowhy_refutations_{a.arm}.csv", index=False)
