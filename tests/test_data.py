import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from pmsdi.config import ARMS
from pmsdi.data import calibrate, db
from pmsdi.data.features import FEATURES, build_features


def test_foreign_keys_and_sizes(world, cfg):
    t, oracle, _, F = world
    ids = set(t["tenants"]["tenant_id"])
    for name in ["leases", "payments", "service_requests", "communications", "satisfaction_surveys", "interventions", "renewal_outcomes"]:
        assert set(t[name]["tenant_id"]) <= ids, name
    assert set(t["units"]["property_id"]) <= set(t["properties"]["property_id"])
    assert len(F) == cfg.n_history + cfg.n_current == len(oracle)
    assert F.isna().sum().sum() == 0 and list(F.columns) == FEATURES


def test_truth_is_consistent(world, cfg):
    t, oracle, _, F = world
    P = oracle[[f"p_{a}" for a in ARMS]].to_numpy()
    assert ((P > 0) & (P < 1)).all()
    h = oracle["cohort"] == "history"
    assert abs(P[h.to_numpy(), 0].mean() - cfg.target_renewal_rate) < 0.01       # intercept was calibrated to the target
    # observed outcome == potential outcome of the logged action
    logged = oracle.loc[h, "logged_action"]
    y_pot = np.array([oracle.loc[i, f"y_{a}"] for i, a in logged.items()])
    y_obs = t["renewal_outcomes"].set_index("tenant_id").loc[logged.index, "renewed"].to_numpy()
    assert (y_pot == y_obs).all()
    # coupled potential outcomes: higher p => never a lower outcome (shared noise u)
    y = oracle[[f"y_{a}" for a in ARMS]].to_numpy()
    assert ((P[:, 1] >= P[:, 0]) <= (y[:, 1] >= y[:, 0])).all()


def test_logged_data_is_confounded(world):
    """Past managers treated riskier tenants => naive comparison says actions HURT although the truth is positive."""
    t, oracle, _, F = world
    h = (oracle["cohort"] == "history").to_numpy()
    A = np.array([ARMS.index(a) for a in oracle.loc[h, "logged_action"]])
    Y = t["renewal_outcomes"].set_index("tenant_id").loc[F.index[h], "renewed"].to_numpy()
    P = oracle.loc[h, [f"p_{a}" for a in ARMS]].to_numpy()
    for a in range(1, 4):
        assert Y[A == a].mean() - Y[A == 0].mean() < 0 < (P[:, a] - P[:, 0]).mean()


def test_db_roundtrip_gives_identical_features(world):
    t, _, _, F = world
    eng = create_engine("sqlite://")
    db.write_tables(eng, t)
    F2 = build_features(db.load_tables(eng))
    assert np.allclose(F.to_numpy(), F2.to_numpy())


def test_calibration_loaders_parse_kaggle_like_files(tmp_path):
    rng = np.random.default_rng(0)
    n = 600
    bhk = rng.choice([1, 2, 3, 4], n, p=[.2, .4, .3, .1])
    rent = (8000 * bhk * np.exp(rng.normal(0, .3, n))).round()
    pd.DataFrame({"Posted On": "2022-05-01", "BHK": bhk, "Rent": rent, "Size": 400 * bhk, "City": rng.choice(["Chennai", "Mumbai"], n)}) \
        .to_csv(tmp_path / "House_Rent_Dataset.csv", index=False)
    cal = calibrate.calibrate_rent(tmp_path / "House_Rent_Dataset.csv")
    assert 7000 < cal["rent_median_by_bhk"]["1"] < 10000 and abs(sum(cal["bhk_probs"].values()) - 1) < 1e-6
    assert set(cal["cities"]) == {"Chennai", "Mumbai"}
    d = pd.date_range("2024-01-01", periods=2000, freq="D")
    pd.DataFrame({"BuildingID": rng.integers(1, 200, 2000), "InspectionDate": d,
                  "CurrentStatusDate": d + pd.to_timedelta(rng.integers(3, 40, 2000), unit="D")}).to_csv(tmp_path / "v.csv", index=False)
    v = calibrate.calibrate_violations(tmp_path / "v.csv")
    assert 2 <= v["request_resolution_median_days"] <= 60 and v["violations_per_100_units_12m"]["mean"] > 0
