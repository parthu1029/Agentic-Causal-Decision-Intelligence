"""All figures used in the report / slides.  Every function saves a PNG and returns its path."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import ARMS, FIG_DIR

C = {"ink": "#1f2a37", "teal": "#0e7c66", "amber": "#c27c0e", "red": "#b4372d", "grey": "#8a94a0", "blue": "#2b5c8a"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "font.size": 10, "axes.titleweight": "bold", "figure.dpi": 130})


def _save(fig, name):
    p = Path(FIG_DIR) / name
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


def roc_pr(curves: dict, prevalence: float, metrics: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, c in curves.items():
        ax[0].plot(c["fpr"], c["tpr"], label=f"{name} ({metrics.loc[name, 'auroc']:.3f})")
        ax[1].plot(c["recall"], c["precision"], label=f"{name} ({metrics.loc[name, 'auprc']:.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color=C["grey"])
    ax[0].set(title="ROC curve (AUROC)", xlabel="False-positive rate", ylabel="True-positive rate")
    ax[1].axhline(prevalence, ls="--", color=C["grey"], label=f"chance = prevalence {prevalence:.2f}")
    ax[1].set(title="Precision-recall curve (AUPRC)", xlabel="Recall", ylabel="Precision")
    for a in ax:
        a.legend(fontsize=7)
    return _save(fig, "01_roc_pr.png")


def model_bars(metrics: pd.DataFrame):
    cols = ["accuracy", "f1", "auroc", "auprc"]
    fig, ax = plt.subplots(figsize=(9, 4))
    metrics[cols].plot.bar(ax=ax, color=[C["grey"], C["amber"], C["teal"], C["blue"]], width=0.8)
    ax.set(title="Renewal-risk baselines: same test split, same predictions", ylabel="score", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=25)
    return _save(fig, "02_model_comparison.png")


def confusion(m: pd.Series, title: str):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center", color="white" if v > cm.max() / 2 else C["ink"])
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Renews", "Leaves"], yticklabels=["Renews", "Leaves"],
           xlabel="Predicted", ylabel="Actual", title=title)
    ax.grid(False)
    return _save(fig, "03_confusion_matrix.png")


def reliability(y, p, name):
    fig, ax = plt.subplots(figsize=(4.2, 4))
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    xs = [p[idx == b].mean() for b in range(10) if (idx == b).sum() > 10]
    ys = [y[idx == b].mean() for b in range(10) if (idx == b).sum() > 10]
    ax.plot([0, 1], [0, 1], "--", color=C["grey"])
    ax.plot(xs, ys, "o-", color=C["teal"])
    ax.set(title=f"Calibration: {name}", xlabel="Predicted P(non-renewal)", ylabel="Observed rate")
    return _save(fig, "04_risk_calibration.png")


def overlap(e, A):
    fig, axs = plt.subplots(1, len(ARMS), figsize=(12, 2.8), sharey=True)
    for a, ax in enumerate(axs):
        ax.hist(e[A == a, a], bins=30, alpha=0.7, color=C["teal"], label="treated with this arm")
        ax.hist(e[A != a, a], bins=30, alpha=0.5, color=C["grey"], label="others")
        ax.set_title(ARMS[a], fontsize=9)
        ax.set_xlim(0, 1)
    axs[0].legend(fontsize=7)
    fig.suptitle("Overlap: estimated P(arm | covariates). Mass near 0 = the data cannot tell us the effect there", fontsize=9)
    return _save(fig, "05_overlap.png")


def love(bal: dict):
    fig, axs = plt.subplots(1, len(bal), figsize=(12, 4.4), sharey=True)
    for ax, (arm, df) in zip(np.atleast_1d(axs), bal.items()):
        y = np.arange(len(df))
        ax.scatter(df["smd_raw"].abs(), y, color=C["red"], label="before weighting")
        ax.scatter(df["smd_ipw"].abs(), y, color=C["teal"], label="after IPW")
        ax.axvline(0.1, ls="--", color=C["grey"])
        ax.set_yticks(y)
        ax.set_yticklabels(df.index, fontsize=6)
        ax.set_title(f"{arm} vs no_action", fontsize=9)
        ax.set_xlabel("|standardised mean difference|")
    np.atleast_1d(axs)[0].legend(fontsize=7)
    return _save(fig, "06_covariate_balance.png")


def effect_comparison(tbl: pd.DataFrame):
    """tbl index = arm; columns: truth, naive, aipw, aipw_lo, aipw_hi, rct, rct_lo, rct_hi."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(tbl))
    ax.bar(x - 0.3, tbl["truth"], 0.2, label="TRUTH (oracle)", color=C["ink"])
    ax.bar(x - 0.1, tbl["naive"], 0.2, label="naive difference in means", color=C["red"])
    ax.errorbar(x + 0.1, tbl["aipw"], yerr=[tbl["aipw"] - tbl["aipw_lo"], tbl["aipw_hi"] - tbl["aipw"]], fmt="s", color=C["teal"],
                label="doubly-robust (adjusted)", capsize=3)
    ax.errorbar(x + 0.3, tbl["rct"], yerr=[tbl["rct"] - tbl["rct_lo"], tbl["rct_hi"] - tbl["rct"]], fmt="D", color=C["amber"],
                label="randomised pilot only", capsize=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(tbl.index)
    ax.set(ylabel="change in P(renew)  (0.05 = 5 percentage points)", title="Association vs causation: effect of each action on renewal")
    ax.legend(fontsize=7)
    return _save(fig, "07_association_vs_causation.png")


def cate_calibration(cal: dict, name: str):
    fig, axs = plt.subplots(1, len(cal), figsize=(11, 3.6))
    for ax, (arm, df) in zip(np.atleast_1d(axs), cal.items()):
        lim = [min(df.mean_pred.min(), df.mean_true.min()) - 0.01, max(df.mean_pred.max(), df.mean_true.max()) + 0.01]
        ax.plot(lim, lim, "--", color=C["grey"])
        ax.plot(df["mean_pred"], df["mean_true"], "o-", color=C["teal"])
        ax.set(title=arm, xlabel="predicted effect (decile mean)", ylabel="TRUE effect (decile mean)")
    fig.suptitle(f"CATE calibration - {name}", fontsize=10)
    return _save(fig, "08_cate_calibration.png")


def uplift_curves(curves: dict, arm: str, fname: str = "09_uplift_curve.png"):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, (q, g) in curves.items():
        ax.plot(q, g, label=name, lw=2 if name.startswith("perfect") else 1.3)
    ax.set(title=f"Uplift curve - {arm}", xlabel="share of tenants treated (ranked by predicted effect)", ylabel="cumulative TRUE gain in renewals")
    ax.legend(fontsize=7)
    return _save(fig, fname)


def policy_bars(df: pd.DataFrame):
    d = df.copy()
    fig, ax = plt.subplots(figsize=(9, 4.4))
    colors = [C["ink"] if "Oracle" in i else C["teal"] if i.startswith(("A", "B Causal")) else C["grey"] for i in d.index]
    y = d["uplift_vs_no_action_per_tenant"]
    ax.barh(d.index, y, color=colors, xerr=[y - (d["ci_lo"] - d.loc["No action", "net_value_per_tenant"]),
                                                 (d["ci_hi"] - d.loc["No action", "net_value_per_tenant"]) - y], ecolor=C["ink"], capsize=2)
    ax.invert_yaxis()
    ax.set(title="Policy value: net value gained per tenant vs doing nothing (95% bootstrap CI)", xlabel="net value per tenant")
    return _save(fig, "10_policy_value.png")


def ope(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    pols = df["policy"].unique()
    x = np.arange(len(pols))
    for k, (est, col) in enumerate([("truth", C["ink"]), ("ips", C["red"]), ("snips", C["amber"]), ("dr", C["teal"])]):
        ax.bar(x + (k - 1.5) * 0.2, [df[df.policy == p][est].iloc[0] for p in pols], 0.2, label=est.upper() if est != "truth" else "TRUE value", color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(pols, rotation=15, fontsize=8)
    lo = min(df[["truth", "ips", "snips", "dr"]].min()) * 0.98
    ax.set(ylim=(lo, None), title="Off-policy evaluation from logged data vs the true value", ylabel="net value per tenant")
    ax.legend(fontsize=7)
    return _save(fig, "11_off_policy_evaluation.png")


def sensitivity(df: pd.DataFrame, x: str, y: str, hue: str, title: str, name: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    for h, g in df.groupby(hue):
        ax.plot(g[x], g[y], "o-", label=f"{hue}={h}")
    ax.set(title=title, xlabel=x, ylabel=y)
    ax.legend(fontsize=7)
    return _save(fig, name)
