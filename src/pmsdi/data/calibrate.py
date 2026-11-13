"""Turn public datasets into `calibration.json`, the *only* channel through which real data touches the
synthetic generator.  Every loader is optional: if a file is missing we keep the documented default and
record that in `sources_used`, so you always know what was calibrated from real data.

Expected files (see docs/DATA_SOURCES.md for where to download them) in data/raw/:
  rent_listings*.csv / House_Rent_Dataset.csv   Kaggle rental listings        -> rent, sqft, BHK mix
  nyc_hpd_violations*.csv                       NYC HPD violations            -> violation rate, closing time
  nyc_hpd_complaints*.csv                       NYC HPD complaints & problems -> complaint category mix
  hud_multifamily*.csv                          HUD multifamily properties    -> property size / opex ratio (if present)
  ahs_reasons_for_moving*.csv  (optional)       survey table you tabulated    -> concern mix
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROCESSED_DIR, RAW_DIR
from .synthetic_pms import DEFAULT_CALIB


def _find(df: pd.DataFrame, *patterns) -> str | None:
    for p in patterns:
        for c in df.columns:
            if re.search(p, str(c), flags=re.I):
                return c
    return None


def _first_file(patterns) -> Path | None:
    for p in patterns:
        hits = sorted(RAW_DIR.glob(p))
        if hits:
            return hits[0]
    return None


def calibrate_rent(path: Path) -> dict:
    df = pd.read_csv(path)
    c_rent = _find(df, r"^rent$", r"monthly.?rent", r"price", r"rent")
    c_area = _find(df, r"^size$", r"sq.?f", r"area_?sqft", r"square", r"^area$")
    c_bhk = _find(df, r"^bhk$", r"bedroom", r"^beds?$", r"rooms")
    c_city = _find(df, r"^city$")
    df = df[pd.to_numeric(df[c_rent], errors="coerce").notna()].copy()
    df[c_rent] = pd.to_numeric(df[c_rent])
    df = df[df[c_rent] > 0]
    lo, hi = df[c_rent].quantile([0.01, 0.99])
    df = df[(df[c_rent] >= lo) & (df[c_rent] <= hi)]            # trim extreme listings
    out = {"sources_used": [f"rent:{path.name} (n={len(df)})"]}
    if c_bhk:
        b = pd.to_numeric(df[c_bhk], errors="coerce").clip(1, 4).round().dropna().astype(int)
        df = df.loc[b.index].assign(_b=b)
        probs = df["_b"].value_counts(normalize=True).reindex([1, 2, 3, 4]).fillna(0.01)
        out["bhk_probs"] = {str(k): float(v / probs.sum()) for k, v in probs.items()}
        med = df.groupby("_b")[c_rent].median().reindex([1, 2, 3, 4]).interpolate(limit_direction="both")
        out["rent_median_by_bhk"] = {str(k): float(v) for k, v in med.items()}
        res = np.log(df[c_rent]) - np.log(df["_b"].map(med))
        out["rent_sigma"] = float(res.std())
        if c_area:
            a = pd.to_numeric(df[c_area], errors="coerce")
            sq = df.assign(_a=a).groupby("_b")["_a"].median().reindex([1, 2, 3, 4]).interpolate(limit_direction="both")
            out["sqft_median_by_bhk"] = {str(k): float(v) for k, v in sq.items()}
    if c_city:
        out["cities"] = [str(x) for x in df[c_city].value_counts().head(6).index]
    return out


def calibrate_violations(path: Path) -> dict:
    """NYC HPD violations export.  Uses issue/close dates to calibrate how long repairs stay open and
    how many violations a building gets per year (per 100 units if a unit count column exists)."""
    df = pd.read_csv(path, low_memory=False)
    c_b = _find(df, r"^buildingid$", r"building", r"bbl")
    c_in = _find(df, r"inspectiondate", r"novissued", r"approveddate", r"issue")
    c_cl = _find(df, r"currentstatusdate", r"closed", r"certified", r"^status.?date")
    out = {"sources_used": [f"violations:{path.name} (rows={len(df)})"]}
    if c_b and c_in:
        d = pd.to_datetime(df[c_in], errors="coerce")
        yrs = max((d.max() - d.min()).days / 365.0, 1.0)
        per_b = df.groupby(c_b).size() / yrs
        mean, var = float(per_b.mean()), float(per_b.var())
        disp = float(np.clip(mean**2 / max(var - mean, 1e-6), 0.3, 5.0))
        # NYC buildings average ~50 units: report per-100-units rate
        out["violations_per_100_units_12m"] = {"mean": float(np.clip(mean * 2.0, 1.0, 40.0)), "dispersion": disp}
    if c_in and c_cl:
        dur = (pd.to_datetime(df[c_cl], errors="coerce") - pd.to_datetime(df[c_in], errors="coerce")).dt.days
        dur = dur[(dur > 0) & (dur < 720)]
        if len(dur) > 100:
            out["request_resolution_median_days"] = float(np.clip(dur.median(), 2, 60))
    return out


def calibrate_complaints(path: Path) -> dict:
    df = pd.read_csv(path, low_memory=False)
    c_t = _find(df, r"majorcategory", r"minorcategory", r"complainttype", r"type", r"category")
    out = {"sources_used": [f"complaints:{path.name} (rows={len(df)})"]}
    if c_t:
        vc = df[c_t].astype(str).str.lower().value_counts(normalize=True)
        out["complaint_category_mix"] = {k: float(v) for k, v in vc.head(10).items()}
    return out


def calibrate_hud(path: Path) -> dict:
    df = pd.read_csv(path, low_memory=False)
    out = {"sources_used": [f"hud:{path.name} (rows={len(df)})"]}
    c_u = _find(df, r"total.?unit", r"units", r"unit_count")
    c_opex = _find(df, r"operating.?(cost|expense)", r"opex")
    c_rev = _find(df, r"(gross|total).?(rent|income|revenue)")
    if c_opex and c_rev:
        r = (pd.to_numeric(df[c_opex], errors="coerce") / pd.to_numeric(df[c_rev], errors="coerce")).dropna()
        r = r[(r > 0.1) & (r < 1.0)]
        if len(r) > 50:
            out["opex_ratio"] = float(r.median())
    if c_u:
        u = pd.to_numeric(df[c_u], errors="coerce").dropna()
        out["unit_count_quantiles"] = {str(q): float(u.quantile(q)) for q in (0.1, 0.5, 0.9)}
    return out


def calibrate_concern_mix(path: Path) -> dict:
    """Optional: a small CSV (reason,share) tabulated from AHS 'reasons for moving'.
    reason must be one of maintenance / price / communication / relocation / none."""
    df = pd.read_csv(path)
    mix = dict(zip(df.iloc[:, 0].str.lower(), df.iloc[:, 1].astype(float)))
    keys = DEFAULT_CALIB["concern_mix"].keys()
    if set(keys) <= set(mix):
        s = sum(mix[k] for k in keys)
        return {"concern_mix": {k: mix[k] / s for k in keys}, "sources_used": [f"concern_mix:{path.name}"]}
    return {}


def run(save: bool = True) -> dict:
    cal = {k: v for k, v in DEFAULT_CALIB.items()}
    cal["sources_used"] = []
    jobs = [
        (calibrate_rent, ["rent_listings*.csv", "House_Rent_Dataset.csv", "*house*rent*.csv"]),
        (calibrate_violations, ["nyc_hpd_violations*.csv", "Housing_Maintenance_Code_Violations*.csv"]),
        (calibrate_complaints, ["nyc_hpd_complaints*.csv", "Housing_Maintenance_Code_Complaints*.csv"]),
        (calibrate_hud, ["hud_multifamily*.csv", "Multifamily_Properties*.csv"]),
        (calibrate_concern_mix, ["ahs_reasons_for_moving*.csv"]),
    ]
    for fn, pats in jobs:
        f = _first_file(pats)
        if f is None:
            cal["sources_used"].append(f"{fn.__name__}: NOT FOUND -> defaults kept")
            continue
        try:
            res = fn(f)
            cal["sources_used"] += res.pop("sources_used", [])
            cal.update(res)
        except Exception as exc:   # a malformed file must never silently corrupt the calibration
            cal["sources_used"].append(f"{fn.__name__}: FAILED on {f.name} ({exc!r}) -> defaults kept")
    if save:
        (PROCESSED_DIR / "calibration.json").write_text(json.dumps(cal, indent=2))
    return cal


def load() -> dict | None:
    p = PROCESSED_DIR / "calibration.json"
    return json.loads(p.read_text()) if p.exists() else None


if __name__ == "__main__":
    c = run()
    print(json.dumps(c["sources_used"], indent=2))
