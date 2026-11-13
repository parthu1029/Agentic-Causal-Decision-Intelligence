"""Step 7 - choose the tenants to show in the live demo: one clear success for each kind of tenant + one honest miss.
Uses the hidden ground truth ONLY to pick illustrative cases; the dashboard still shows model estimates."""
import _bootstrap  # noqa: F401
import json

import numpy as np
import pandas as pd

from pmsdi.config import ARMS, PROCESSED_DIR
from pmsdi.decision import interventions as iv
from pmsdi.service import DecisionService

svc = DecisionService()
cfg, ids = svc.cfg, svc.current_ids
F = svc.F.loc[ids]
P_hat = svc._batch()["P"]
o = svc.oracle.loc[ids]
P_true = o[[f"p_{a}" for a in ARMS]].to_numpy()
rent = F["rent"].to_numpy()
V, cost, feas = iv.retained_value(rent, cfg), iv.costs(rent, cfg), iv.feasibility(F, cfg)
nv_true, nv_hat = P_true * V[:, None] - cost, P_hat * V[:, None] - cost
best_true = np.where(feas, nv_true, -np.inf).argmax(1)
best_hat = np.where(feas, nv_hat, -np.inf).argmax(1)
gain_true = nv_true[np.arange(len(ids)), best_true] - nv_true[:, 0]
risk = svc._batch()["risk"]
picks = {}

def pick(label, mask, agree=True):
    m = mask & ((best_hat == best_true) if agree else (best_hat != best_true)) & (risk > 0.45)
    if m.sum() == 0:
        return
    i = np.where(m)[0][np.argmax(np.where(m, gain_true, -1)[m])]
    picks[label] = {"tenant_id": int(ids[i]), "model_recommends": ARMS[best_hat[i]], "truth_best": ARMS[best_true[i]],
                    "hidden_concern": o.iloc[i]["concern"], "true_gain": float(gain_true[i]), "risk": float(risk[i])}

c = o["concern"].to_numpy()
pick("Maintenance-driven tenant (service priority wins)", (c == "maintenance") & (best_true == 1))
pick("Price-sensitive tenant (concession wins)", (c == "price") & (best_true == 2))
pick("Communication-driven tenant (outreach wins)", (c == "communication") & (best_true == 3))
pick("Relocating tenant (nothing helps - do nothing)", (c == "relocation") & (best_true == 0), agree=True)
pick("Concession blocked by policy (arrears / late payments)", (~feas[:, 2]) & (c == "price"))
pick("Honest miss: model and truth disagree", (best_hat != best_true) & (gain_true > 0), agree=False)
(PROCESSED_DIR / "demo_tenants.json").write_text(json.dumps(picks, indent=2))
for k, v in picks.items():
    print(f"{k:60s} tenant {v['tenant_id']}  model={v['model_recommends']:<19s} truth={v['truth_best']:<19s} concern={v['hidden_concern']}")
