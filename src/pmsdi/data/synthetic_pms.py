"""Calibrated synthetic longitudinal PMS with KNOWN potential outcomes.

Why synthetic?  In real data you only ever see the outcome of the action that was taken.  Here the
generator knows, for every tenant, the renewal probability under *every* intervention
(`oracle` table), so causal estimators and decision policies can be scored against the truth.

Real public data enters through `calibration` (rent levels, unit sizes, violation rates, complaint mix ...)
produced by `pmsdi.data.calibrate`.  Everything else is a documented ASSUMPTION (see docs/ASSUMPTIONS.md).

Generation order (so that the tables really "explain" the features):
   latent traits -> hidden tenant concern -> raw event tables (payments, requests, messages, surveys)
   -> features (feature store) -> true renewal probabilities p_a(x, concern) -> logged action + outcome
"""
from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from ..config import ARMS, K, Config, SYNTH_DIR
from .features import build_features

CONCERNS = ["maintenance", "price", "communication", "relocation", "none"]

DEFAULT_CALIB = {
    "bhk_probs": {"1": 0.15, "2": 0.42, "3": 0.33, "4": 0.10},
    "rent_median_by_bhk": {"1": 900.0, "2": 1250.0, "3": 1650.0, "4": 2100.0},
    "sqft_median_by_bhk": {"1": 550.0, "2": 800.0, "3": 1100.0, "4": 1450.0},
    "rent_sigma": 0.25,
    "cities": ["Chennai", "Bengaluru", "Hyderabad", "Pune"],
    "violations_per_100_units_12m": {"mean": 6.0, "dispersion": 1.5},
    "request_resolution_median_days": 6.0,
    "opex_ratio": 0.38,
    "concern_mix": {"maintenance": 0.26, "price": 0.20, "communication": 0.14, "relocation": 0.12, "none": 0.28},
    "sources_used": [],
}

# Effect of each intervention (rows) on the LOGIT of renewal for each hidden concern (columns).
# This is the assumed causal structure: an intervention helps mainly when it addresses the tenant's real concern.
TAU = np.array([
    #  maint  price  comm   reloc  none
    [0.00, 0.00, 0.00, 0.00, 0.00],     # no_action
    [1.00, 0.10, 0.10, 0.00, 0.05],     # service_priority
    [0.15, 1.10, 0.10, 0.05, 0.10],     # rent_concession
    [0.10, 0.10, 0.90, 0.02, -0.10],    # proactive_outreach (slightly annoying to already-happy tenants)
])

TEXTS = {
    "neg": ["The repair request is still not fixed, this is the third time I am writing.",
            "No response from the office for days. Please resolve the issue.",
            "Rent is higher than similar flats nearby, I am considering other options."],
    "neu": ["Please confirm the date of the maintenance visit.",
            "Could you share the renewal terms for next year?"],
    "pos": ["Thanks for the quick fix last week.", "Happy with the service, please send the renewal paperwork."],
}


def merge_calibration(calib: dict | None) -> dict:
    cal = deepcopy(DEFAULT_CALIB)
    for k, v in (calib or {}).items():
        cal[k] = v
    return cal


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax_sample(rng, logits):
    g = -np.log(-np.log(rng.uniform(size=logits.shape)))
    return np.argmax(logits + g, axis=1)


def make_properties(rng, cfg: Config, cal: dict):
    n = cfg.n_properties
    q = rng.normal(size=n)                      # hidden maintenance quality (higher = better run building)
    age = np.clip(rng.gamma(4, 8, n), 2, 120)
    vm, disp = cal["violations_per_100_units_12m"]["mean"], cal["violations_per_100_units_12m"]["dispersion"]
    mu = vm * np.exp(-0.5 * q + 0.01 * (age - 30))
    viol = rng.gamma(shape=disp, scale=mu / disp)
    props = pd.DataFrame({
        "property_id": np.arange(1, n + 1),
        "city": rng.choice(cal["cities"], n),
        "n_units": rng.integers(20, 250, n),
        "age_years": age.round(0),
        "violations_per_100_units_12m": viol.round(2),
        "opex_ratio": np.clip(rng.normal(cal["opex_ratio"], 0.05, n), 0.2, 0.6).round(3),
        "market_index": np.exp(rng.normal(0, 0.18, n)).round(3),
    })
    return props, q


def _gen_tables(rng, cfg, cal, props, q_prop, n, id0, cohort, as_of):
    """Raw event tables + hidden traits for one cohort of n tenants (1 unit, 1 lease each)."""
    day = np.timedelta64(1, "D")
    tid = np.arange(id0, id0 + n)
    pidx = rng.integers(0, len(props), n)
    bhk = rng.choice([1, 2, 3, 4], n, p=[cal["bhk_probs"][str(b)] for b in (1, 2, 3, 4)])
    sq_med = np.array([cal["sqft_median_by_bhk"][str(b)] for b in bhk])
    rent_med = np.array([cal["rent_median_by_bhk"][str(b)] for b in bhk])
    area = sq_med * np.exp(rng.normal(0, 0.15, n))
    market = rent_med * props["market_index"].to_numpy()[pidx] * (area / sq_med) ** 0.5 * np.exp(rng.normal(0, 0.08, n))
    gap = rng.normal(0.02, 0.08, n)                       # rent relative to market
    rent = np.round(market * (1 + gap), 0)
    tenure = np.clip(6 + rng.gamma(2.0, 14.0, n), 6, 120)
    stress = rng.normal(size=n)                           # hidden financial stress
    q = q_prop[pidx]

    # ---- hidden concern (what is really driving the renewal decision)
    mix = cal["concern_mix"]
    lg = np.zeros((n, 5))
    lg[:, 0] = np.log(mix["maintenance"]) - 0.9 * q
    lg[:, 1] = np.log(mix["price"]) + 8.0 * (gap - 0.02) + 0.25 * stress
    lg[:, 2] = np.log(mix["communication"])
    lg[:, 3] = np.log(mix["relocation"])
    lg[:, 4] = np.log(mix["none"])
    concern = _softmax_sample(rng, lg)
    m, p_, c_, r_, n_ = [(concern == i).astype(float) for i in range(5)]
    sat_latent = np.clip(3.8 - 0.9 * m - 0.6 * c_ - 0.5 * p_ - 0.25 * r_ + 0.3 * n_ + 0.3 * q + rng.normal(0, 0.55, n), 1, 5)

    if isinstance(as_of, str):
        as_of_arr = np.full(n, np.datetime64(as_of), dtype="datetime64[D]")
    else:
        as_of_arr = as_of
    dte = rng.integers(cfg.decision_window_days[0], cfg.decision_window_days[1] + 1, n)
    end = as_of_arr + dte.astype("timedelta64[D]")
    start = end - np.timedelta64(365, "D")
    move_in = as_of_arr - (tenure * 30.4).astype(int).astype("timedelta64[D]")

    unit_ids = tid.copy()
    units = pd.DataFrame({"unit_id": unit_ids, "property_id": props["property_id"].to_numpy()[pidx], "bedrooms": bhk,
                          "area_sqft": area.round(0), "market_rent": market.round(0)})
    tenants = pd.DataFrame({"tenant_id": tid, "unit_id": unit_ids, "move_in_date": move_in, "cohort": cohort})
    leases = pd.DataFrame({"lease_id": tid, "tenant_id": tid, "unit_id": unit_ids, "start_date": start, "end_date": end,
                           "decision_date": as_of_arr, "monthly_rent": rent,
                           "status": "pending", "decided_at": pd.NaT})

    # ---- payments: 12 monthly rows per tenant
    months = np.arange(1, 13)
    base_month = as_of_arr.astype("datetime64[M]")
    due = (base_month[:, None] - months[None, :].astype("timedelta64[M]")).astype("datetime64[D]")  # (n,12)
    late_p = _sigmoid(-1.5 + 0.9 * stress + 0.3 * p_)
    late = rng.uniform(size=(n, 12)) < late_p[:, None]
    dl = np.where(late, 1 + rng.geometric(0.18, size=(n, 12)), 0)
    due_amt = np.repeat(rent[:, None], 12, axis=1)
    paid_amt = due_amt.copy()
    arrears = rng.uniform(size=n) < _sigmoid(-3.0 + 1.0 * stress)
    frac = rng.uniform(0.3, 1.0, n)
    paid_amt[:, 0] = np.where(arrears, np.round(rent * (1 - frac)), rent)
    paid_date = due + dl.astype("timedelta64[D]")
    unpaid_first = arrears & (paid_amt[:, 0] == 0)
    dl[:, 0] = np.where(arrears, (as_of_arr - due[:, 0]).astype(int), dl[:, 0])
    payments = pd.DataFrame({
        "tenant_id": np.repeat(tid, 12), "lease_id": np.repeat(tid, 12),
        "due_date": due.ravel(), "amount_due": due_amt.ravel(), "amount_paid": paid_amt.ravel(),
        "paid_date": paid_date.ravel(), "days_late": dl.ravel(),
    })
    payments.loc[(payments["amount_paid"] == 0).to_numpy(), "paid_date"] = pd.NaT
    payments.insert(0, "payment_id", np.arange(1, len(payments) + 1))

    # ---- service requests
    lam = 1.5 * np.exp(-0.35 * q + 0.7 * m) * (0.8 + 0.2 * bhk)
    nreq = rng.poisson(lam)
    owner = np.repeat(np.arange(n), nreq)
    created = as_of_arr[owner] - rng.integers(1, 366, len(owner)).astype("timedelta64[D]")
    stuck = rng.uniform(size=len(owner)) < (0.08 + 0.40 * m[owner])
    med = cal["request_resolution_median_days"] * (1 + 0.5 * m[owner]) * np.exp(-0.2 * q[owner])
    dur = np.where(stuck, rng.exponential(70, len(owner)) + 20, np.exp(rng.normal(np.log(med), 0.6)))
    resolved = created + np.ceil(dur).astype(int).astype("timedelta64[D]")
    is_open = resolved > as_of_arr[owner]
    cats = np.array(["plumbing", "electrical", "appliance", "pest", "structural", "noise"])
    reqs = pd.DataFrame({
        "request_id": np.arange(1, len(owner) + 1), "tenant_id": tid[owner], "unit_id": unit_ids[owner],
        "category": rng.choice(cats, len(owner)), "created_date": created,
        "resolved_date": np.where(is_open, np.datetime64("NaT"), resolved),
        "priority": rng.choice(["low", "normal", "high"], len(owner), p=[0.3, 0.55, 0.15]),
    })

    # ---- communications (last 180 days; features use the last 90)
    nmsg = rng.poisson(2 * (1.6 + 1.8 * c_ + 0.6 * m))
    own = np.repeat(np.arange(n), nmsg)
    mu_s = 0.25 - 0.65 * c_ - 0.35 * m - 0.15 * p_ + 0.20 * n_
    sent = np.clip(rng.normal(mu_s[own], 0.35), -1, 1)
    resp_med = 14.0 * np.exp(-0.3 * q[own]) * (1 + 0.8 * c_[own])
    bucket = np.where(sent < -0.3, "neg", np.where(sent > 0.3, "pos", "neu"))
    texts = [TEXTS[b][i % len(TEXTS[b])] for i, b in enumerate(bucket)]
    comms = pd.DataFrame({
        "comm_id": np.arange(1, len(own) + 1), "tenant_id": tid[own],
        "sent_at": as_of_arr[own] - rng.integers(0, 180, len(own)).astype("timedelta64[D]"),
        "channel": rng.choice(["email", "phone", "portal", "whatsapp"], len(own)),
        "sentiment": sent.round(3), "response_hours": np.exp(rng.normal(np.log(resp_med), 0.6)).round(1),
        "message_text": texts,
    })

    # ---- satisfaction survey (not everybody answers)
    has = rng.uniform(size=n) < np.where(r_ == 1, 0.5, 0.7)
    surveys = pd.DataFrame({
        "survey_id": np.arange(1, has.sum() + 1), "tenant_id": tid[has],
        "survey_date": as_of_arr[has] - rng.integers(10, 150, has.sum()).astype("timedelta64[D]"),
        "score": np.clip(np.round(sat_latent[has] + rng.normal(0, 0.4, has.sum())), 1, 5).astype(int),
    })
    hidden = pd.DataFrame({"tenant_id": tid, "concern": np.array(CONCERNS)[concern], "sat_latent": sat_latent,
                           "property_quality": q, "financial_stress": stress})
    return dict(units=units, tenants=tenants, leases=leases, payments=payments, service_requests=reqs,
                communications=comms, satisfaction_surveys=surveys), hidden


# ---------------------------------------------------------------- outcome model (ground truth)
# fixed standardisation constants (population values of the generator) used by the outcome + logging models
ZC = {"sat": (3.25, 0.8), "late": (1.1, 1.4), "gap": (0.02, 0.08), "tenure": (35.0, 20.0), "open": (0.25, 0.6)}


def _z(x, mu, sd):
    return (np.asarray(x, float) - mu) / sd


def base_logit(F: pd.DataFrame, hidden: pd.DataFrame, rng, noise_sd=0.4):
    """Renewal logit WITHOUT intercept under 'no action'. Mix of observed features and hidden traits."""
    concern = hidden["concern"].to_numpy()
    shift = pd.Series(concern).map({"maintenance": -0.6, "price": -0.7, "communication": -0.5,
                                    "relocation": -1.6, "none": 0.5}).to_numpy()
    l0 = (0.55 * _z(hidden["sat_latent"], *ZC["sat"]) - 0.45 * _z(F["late_payments_12m"], *ZC["late"])
          - 0.50 * _z(F["rent_vs_market"], *ZC["gap"]) + 0.30 * _z(F["tenure_months"], *ZC["tenure"])
          - 0.30 * _z(F["open_requests"], *ZC["open"]) - 0.25 * F["has_arrears"].to_numpy() + shift
          + rng.normal(0, noise_sd, len(F)))
    return np.asarray(l0)


def tau_logit(F: pd.DataFrame, hidden: pd.DataFrame, effect_scale: float):
    ci = pd.Series(hidden["concern"].to_numpy()).map({c: i for i, c in enumerate(CONCERNS)}).to_numpy()
    out = np.zeros((len(F), K))
    mod = [np.ones(len(F)),
           np.where(F["open_requests"].to_numpy() > 0, 1.0, 0.35),
           np.clip(0.7 + 4.0 * F["rent_vs_market"].to_numpy(), 0.3, 1.5),
           np.ones(len(F))]
    for a in range(K):
        out[:, a] = effect_scale * TAU[a, ci] * mod[a]
    return out


def logging_policy_probs(F: pd.DataFrame, hidden: pd.DataFrame, cfg: Config):
    """How past managers chose actions: they react to *observed* risk signals (=> confounding),
    optionally also to the hidden concern (=> unmeasured confounding)."""
    ls, hc, eps = cfg.logging_strength, cfg.hidden_confounding, cfg.logging_epsilon
    z_late = _z(F["late_payments_12m"], *ZC["late"])
    z_sat = _z(F["satisfaction_score"], *ZC["sat"])
    z_gap = _z(F["rent_vs_market"], *ZC["gap"])
    opn = np.minimum(F["open_requests"].to_numpy(), 2)
    sent = F["avg_sentiment_90d"].to_numpy()
    r_obs = 0.35 * z_late - 0.5 * z_sat + 0.45 * opn - 0.6 * sent + 0.4 * z_gap + 0.3 * F["has_arrears"].to_numpy()
    c = hidden["concern"].to_numpy()
    S = np.zeros((len(F), K))
    S[:, 0] = 0.8 - 1.1 * ls * r_obs
    S[:, 1] = -0.8 + ls * (1.1 * opn + 0.5 * r_obs) + 1.5 * hc * (c == "maintenance")
    S[:, 2] = -1.2 + ls * (0.9 * z_gap + 0.6 * r_obs) + 1.5 * hc * (c == "price")
    S[:, 3] = -0.6 + ls * (-0.9 * sent + 0.4 * r_obs) + 1.5 * hc * (c == "communication")
    S -= S.max(axis=1, keepdims=True)
    P = np.exp(S)
    P /= P.sum(axis=1, keepdims=True)
    return (1 - eps) * P + eps / K


def generate(cfg: Config | None = None, calib: dict | None = None, verbose: bool = False):
    """Returns (tables, oracle, world).  `oracle` holds ground truth and must never be fed to models."""
    cfg = cfg or Config()
    cal = merge_calibration(calib)
    rng = np.random.default_rng(cfg.seed)
    props, q_prop = make_properties(rng, cfg, cal)

    # decision dates: history spread over ~2 years, current cohort = "today"
    n_h, n_c = cfg.n_history, cfg.n_current
    hist_asof = np.datetime64("2024-01-01") + rng.integers(0, 850, n_h).astype("timedelta64[D]")
    t_h, hid_h = _gen_tables(rng, cfg, cal, props, q_prop, n_h, 1, "history", hist_asof)
    t_c, hid_c = _gen_tables(rng, cfg, cal, props, q_prop, n_c, n_h + 1, "current", "2026-10-18")

    tables = {"properties": props}
    for k in t_h:
        tables[k] = pd.concat([t_h[k], t_c[k]], ignore_index=True)
        # keep surrogate ids unique across cohorts
    for k, idc in [("payments", "payment_id"), ("service_requests", "request_id"),
                   ("communications", "comm_id"), ("satisfaction_surveys", "survey_id")]:
        tables[k][idc] = np.arange(1, len(tables[k]) + 1)
    hidden = pd.concat([hid_h, hid_c], ignore_index=True).set_index("tenant_id")

    F = build_features(tables)
    hidden = hidden.loc[F.index]
    is_h = F.index.to_numpy() <= n_h

    # ---- ground-truth renewal probabilities under each action
    l0 = base_logit(F, hidden, rng)
    b0 = brentq(lambda b: _sigmoid(l0[is_h] + b).mean() - cfg.target_renewal_rate, -8, 8)
    tl = tau_logit(F, hidden, cfg.effect_scale)
    Pm = _sigmoid(l0[:, None] + b0 + tl)                          # (n, K) true P(renew | x, hidden, a)
    u = rng.uniform(size=len(F))
    Y_pot = (u[:, None] < Pm).astype(int)                          # coupled potential outcomes Y_a

    # ---- logged actions + observed outcomes for the history cohort
    probs = logging_policy_probs(F, hidden, cfg)
    A = np.array([rng.choice(K, p=p) for p in probs])
    rand = rng.uniform(size=len(F)) < cfg.rct_fraction
    A = np.where(rand, rng.integers(0, K, len(F)), A)
    e_true = np.where(rand[:, None], 1.0 / K, probs)               # true logging propensities (oracle only)
    assigned_by = np.where(rand, "randomized_pilot", "manager_log")

    oracle = pd.DataFrame(Pm, index=F.index, columns=[f"p_{a}" for a in ARMS])
    for a in range(K):
        oracle[f"y_{ARMS[a]}"] = Y_pot[:, a]
        oracle[f"e_{ARMS[a]}"] = e_true[:, a]
    oracle["concern"] = hidden["concern"]
    oracle["u"] = u
    oracle["logged_action"] = np.where(is_h, np.array(ARMS)[A], "")
    oracle["cohort"] = np.where(is_h, "history", "current")

    hist_idx = F.index[is_h]
    A_h = A[is_h]
    rent_h = F.loc[hist_idx, "rent"].to_numpy()
    cost_h = np.array([cfg.cost_rel[ARMS[a]] for a in A_h]) * rent_h
    tables["interventions"] = pd.DataFrame({
        "intervention_id": np.arange(1, len(hist_idx) + 1), "tenant_id": hist_idx, "action": np.array(ARMS)[A_h],
        "assigned_by": assigned_by[is_h],
        "logged_propensity": np.where(rand[is_h], 1.0 / K, np.nan),   # only known for the randomized pilot
        "cost": cost_h.round(2), "status": "executed",
        "executed_at": tables["leases"].set_index("tenant_id").loc[hist_idx, "decision_date"].to_numpy(),
    })
    y_obs = Y_pot[is_h, :][np.arange(is_h.sum()), A_h]
    tables["renewal_outcomes"] = pd.DataFrame({
        "tenant_id": hist_idx, "lease_id": hist_idx, "renewed": y_obs, "source": "historical",
        "decided_at": tables["leases"].set_index("tenant_id").loc[hist_idx, "end_date"].to_numpy(),
    })
    lease = tables["leases"].set_index("tenant_id")
    lease.loc[hist_idx, "status"] = np.where(y_obs == 1, "renewed", "not_renewed")
    lease.loc[hist_idx, "decided_at"] = lease.loc[hist_idx, "end_date"]
    tables["leases"] = lease.reset_index()[["lease_id", "tenant_id", "unit_id", "start_date", "end_date", "decision_date",
                                            "monthly_rent", "status", "decided_at"]]
    tables["audit_log"] = pd.DataFrame(columns=["audit_id", "ts", "tenant_id", "event", "payload"])
    world = {"b0": float(b0), "config": cfg.to_dict(), "calibration_sources": cal.get("sources_used", [])}
    if verbose:
        print(f"history n={n_h}  current n={n_c}  mean P(renew|no action)={Pm[is_h, 0].mean():.3f}")
    return tables, oracle, world


def save_world(world: dict, path=None):
    (path or SYNTH_DIR / "world.json").write_text(json.dumps(world, indent=2, default=str))
