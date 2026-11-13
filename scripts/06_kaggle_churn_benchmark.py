"""Re-create slide 16/17/18 properly from YOUR Kaggle churn CSV: one split, one set of predictions, all metrics
(incl. AUROC and AUPRC), ROC/PR curves and a confusion matrix that is consistent with the table.
usage: python scripts/06_kaggle_churn_benchmark.py --csv data/raw/churn.csv --target Churn --positive Yes"""
import _bootstrap  # noqa: F401
import argparse

import numpy as np
import pandas as pd

from pmsdi.config import OUT_DIR
from pmsdi.eval import plots
from pmsdi.models.churn_baseline import benchmark

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--target", required=True, help="name of the churn column")
ap.add_argument("--positive", default="1", help="value of the target that means 'churned' (e.g. Yes / 1 / True)")
ap.add_argument("--drop", nargs="*", default=[], help="id-like columns to drop")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
df = pd.read_csv(a.csv).drop(columns=a.drop, errors="ignore")
y = (df[a.target].astype(str).str.lower() == a.positive.lower()).astype(int).to_numpy()
X = pd.get_dummies(df.drop(columns=[a.target]), drop_first=True).astype(float)
X = X.fillna(X.median())
metrics, preds, curves, meta = benchmark(X, y, seed=a.seed)
pd.set_option("display.width", 200)
print(f"rows={len(df)}  positive share={y.mean():.2%}  (AUPRC chance level = {meta['prevalence']:.3f})")
print(metrics.drop(columns=["tn", "fp", "fn", "tp"]).round(4))
metrics.to_csv(OUT_DIR / "kaggle_churn_benchmark.csv")
plots.roc_pr(curves, meta["prevalence"], metrics)
plots.model_bars(metrics)
best = metrics["auroc"].idxmax()
plots.confusion(metrics.loc[best], f"{best}")
print("figures -> outputs/figures/01_roc_pr.png, 02_model_comparison.png, 03_confusion_matrix.png")
