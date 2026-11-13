"""Step 5 - how fragile are the conclusions?  Re-generate the world under different ASSUMPTIONS and re-run the whole pipeline.
  effect_scale        0.5x / 1x / 1.5x  true treatment effects
  hidden_confounding  past managers also reacted to something we cannot see (0 = no, 2 = strongly)
  logging_strength    how strongly past actions depended on observed risk (0 = effectively randomised)
Honest finding to look for: adjustment fixes OBSERVED confounding but not HIDDEN confounding."""
import _bootstrap  # noqa: F401
import argparse
import time

import pandas as pd

from pmsdi.config import FIG_DIR, OUT_DIR, Config
from pmsdi.data import calibrate
from pmsdi.eval import plots
from pmsdi.pipeline import run_experiment

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="smaller grid (about 1 minute)")
args = ap.parse_args()
base = Config.load()
base.n_history, base.n_current = 8000, 2000
calib = calibrate.load()
grid = [(es, hc, 1.0) for es in ([1.0] if args.quick else [0.5, 1.0, 1.5]) for hc in ([0.0, 2.0] if args.quick else [0.0, 1.0, 2.0])]
grid += [(1.0, 0.0, ls) for ls in ([0.0, 2.0] if args.quick else [0.0, 2.0, 3.0])]
rows = []
t0 = time.time()
for es, hc, ls in grid:
    cfg = Config.from_dict(base.to_dict())
    cfg.effect_scale, cfg.hidden_confounding, cfg.logging_strength = es, hc, ls
    R = run_experiment(cfg, calib, estimators=("naive", "s_learner", "dr_learner"))
    for est in R["causal"]:
        c, p = R["causal"][est], R["policy"][est]
        rows.append({"effect_scale": es, "hidden_confounding": hc, "logging_strength": ls, "estimator": est,
                     "mean_ate_abs_error": c["ate_abs_error"].mean(), "mean_pehe": c["pehe"].mean(),
                     "B_pct_of_oracle_gain": p.loc["B Causal + constraints", "pct_of_oracle_gain"],
                     "B3_pct_of_oracle_gain": p.loc["B3 Risk + rules", "pct_of_oracle_gain"],
                     "B_minus_B3_per_tenant": p.loc["B Causal + constraints", "net_value_per_tenant"] - p.loc["B3 Risk + rules", "net_value_per_tenant"]})
    print(f"effect_scale={es} hidden_confounding={hc} logging_strength={ls}  ({time.time() - t0:.0f}s)")
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "sensitivity_results.csv", index=False)
pd.set_option("display.width", 220)
print(df.round(3).to_string(index=False))
sub = df[(df.logging_strength == 1.0) & (df.estimator != "naive")]
plots.sensitivity(sub[sub.effect_scale == 1.0], "hidden_confounding", "mean_ate_abs_error", "estimator",
                  "ATE error as unmeasured confounding grows", "12_sens_hidden_confounding.png")
plots.sensitivity(sub[sub.estimator == "dr_learner"], "hidden_confounding", "B_minus_B3_per_tenant", "effect_scale",
                  "Policy B advantage over B3 (DR-learner)", "13_sens_policy_advantage.png")
