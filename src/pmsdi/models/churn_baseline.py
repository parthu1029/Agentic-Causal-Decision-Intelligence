"""Renewal-risk baselines (slide 16): six classifiers, evaluated with ALL metrics from the same predictions.
Positive class = NON-renewal (churn = 1).  AUPRC's chance level equals the positive-class prevalence."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def make_classifiers(seed: int = 42) -> dict:
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=30, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=10, n_jobs=-1, random_state=seed),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=10, n_jobs=-1, random_state=seed),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                                        subsample=0.8, random_state=seed),
    }


def expected_calibration_error(y, p, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def metrics_at(y, p, thr: float = 0.5) -> dict:
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat), "f1": f1_score(y, yhat), "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p), "brier": brier_score_loss(y, p), "ece": expected_calibration_error(y, p),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def benchmark(X: pd.DataFrame, y: np.ndarray, seed: int = 42, test_size: float = 0.2):
    """Returns (metrics DataFrame, predictions dict, curves dict, split indices).  One split -> one set of predictions
    -> every number, plot and confusion matrix in the report comes from the SAME predictions."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    rows, preds, curves, fitted = [], {}, {}, {}
    for name, clf in make_classifiers(seed).items():
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        rows.append({"model": name, **metrics_at(yte, p)})
        preds[name] = p
        fpr, tpr, _ = roc_curve(yte, p)
        pr, rc, _ = precision_recall_curve(yte, p)
        curves[name] = {"fpr": fpr, "tpr": tpr, "precision": pr, "recall": rc}
        fitted[name] = clf
    df = pd.DataFrame(rows).set_index("model")
    meta = {"y_test": yte, "test_index": Xte.index, "prevalence": float(np.mean(yte)), "fitted": fitted}
    return df, preds, curves, meta
