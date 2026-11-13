import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from pmsdi.config import ARMS, Config            # noqa: E402
from pmsdi.data import db                         # noqa: E402
from pmsdi.data.features import build_features    # noqa: E402
from pmsdi.data.synthetic_pms import generate     # noqa: E402
from pmsdi.models import causal                   # noqa: E402
from pmsdi.pipeline import fit_risk_model, split_xy  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return Config(n_history=5000, n_current=1500, n_properties=60, seed=7)


@pytest.fixture(scope="session")
def world(cfg):
    tables, oracle, w = generate(cfg)
    return tables, oracle, w, build_features(tables)


@pytest.fixture(scope="session")
def service(cfg, world, tmp_path_factory):
    """A full DecisionService on an in-memory SQLite database with freshly trained models."""
    from pmsdi.service import DecisionService
    tables, oracle, _, F = world
    Xh, Xc, A, Y, _ = split_xy(tables, oracle, F)
    art = tmp_path_factory.mktemp("art") / "models.joblib"
    joblib.dump({"dr": causal.DRLearner(seed=7).fit(Xh, A, Y), "s_learner": causal.SLearner(7).fit(Xh, A, Y),
                 "risk_model": fit_risk_model(Xh, Y, 7), "features": list(F.columns), "cfg": cfg.to_dict()}, art)
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=__import__("sqlalchemy").pool.StaticPool)
    db.write_tables(eng, tables)
    db.write_features(eng, F)
    return DecisionService(engine=eng, artifacts_path=art, oracle=oracle)
