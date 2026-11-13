"""Step 3 - (a) renewal-risk benchmark with AUROC/AUPRC,  (b) causal estimators,  (c) save serving artifacts."""
import _bootstrap  # noqa: F401
import joblib
import numpy as np
import pandas as pd

from pmsdi.config import ARTIFACT_DIR, OUT_DIR, SYNTH_DIR, Config
from pmsdi.data import db
from pmsdi.data.features import FEATURES
from pmsdi.eval import plots
from pmsdi.models import causal
from pmsdi.models.churn_baseline import benchmark
from pmsdi.pipeline import fit_risk_model, split_xy

cfg = Config.load()
engine = db.get_engine()
tables = db.load_tables(engine)
F = db.read_features(engine)[FEATURES]
oracle = pd.read_csv(SYNTH_DIR / "oracle.csv.gz", index_col=0)
Xh, Xc, A, Y, hist = split_xy(tables, oracle, F)

print("== (a) renewal-risk baselines (positive class = NON-renewal) ==")
metrics, preds, curves, meta = benchmark(Xh, 1 - Y, seed=cfg.seed)
pd.set_option("display.width", 200)
print(metrics.drop(columns=["tn", "fp", "fn", "tp"]).round(4))
print(f"prevalence of non-renewal in test split = {meta['prevalence']:.3f}  (AUPRC chance level)")
metrics.to_csv(OUT_DIR / "risk_baselines.csv")
plots.roc_pr(curves, meta["prevalence"], metrics)
plots.model_bars(metrics)
best = metrics["auroc"].idxmax()
plots.confusion(metrics.loc[best], f"{best} (same split as table)")
plots.reliability(meta["y_test"], preds[best], best)

print("\n== (b) causal estimators on logged history ==")
dr = causal.DRLearner(seed=cfg.seed).fit(Xh, A, Y)
sl = causal.SLearner(cfg.seed).fit(Xh, A, Y)
risk = fit_risk_model(Xh, Y, cfg.seed)
joblib.dump({"dr": dr, "s_learner": sl, "risk_model": risk, "features": FEATURES, "cfg": cfg.to_dict()}, ARTIFACT_DIR / "models.joblib")
print("saved ->", ARTIFACT_DIR / "models.joblib")
