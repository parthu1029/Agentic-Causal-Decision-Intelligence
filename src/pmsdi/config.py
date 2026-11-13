"""Central configuration.  Every number that is an *assumption* lives here so it can be
documented, changed, and swept in the sensitivity analysis."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTH_DIR = DATA_DIR / "synthetic"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
ARTIFACT_DIR = OUT_DIR / "artifacts"
for _d in (RAW_DIR, PROCESSED_DIR, SYNTH_DIR, OUT_DIR, FIG_DIR, ARTIFACT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# The intervention library.  Index 0 is always the control ("do nothing").
ARMS = ["no_action", "service_priority", "rent_concession", "proactive_outreach"]
ARM_LABELS = {
    "no_action": "No action",
    "service_priority": "Prioritise maintenance",
    "rent_concession": "Rent concession",
    "proactive_outreach": "Proactive outreach",
}
K = len(ARMS)


@dataclass
class Config:
    # ---- reproducibility / size
    seed: int = 42
    n_properties: int = 120
    n_history: int = 12000       # past tenants with logged actions + outcomes (training data)
    n_current: int = 3000        # tenants in the renewal window now (decision + evaluation cohort)
    # ---- world assumptions (documented in docs/ASSUMPTIONS.md)
    target_renewal_rate: float = 0.58     # mean P(renew | no action); replace by an industry benchmark
    effect_scale: float = 1.0             # multiplies every true treatment effect (logit scale)
    rct_fraction: float = 0.15            # share of history assigned at random (pilot)
    logging_strength: float = 1.0         # how strongly past managers' choices depended on risk (confounding)
    hidden_confounding: float = 0.0       # extra dependence of past choices on the HIDDEN tenant concern
    logging_epsilon: float = 0.15         # exploration mass in the logging policy (guarantees overlap)
    # ---- economics (all relative to the tenant's monthly rent => currency independent)
    lease_months: int = 12
    turnover_months: float = 1.0          # vacancy + make-ready cost avoided when a tenant renews
    cost_rel: dict = field(default_factory=lambda: {
        "no_action": 0.0, "service_priority": 0.25, "rent_concession": 0.50, "proactive_outreach": 0.05})
    budget_pct_rent_roll: float = 0.06    # portfolio budget = pct * total monthly rent roll of the cohort
    currency: str = "$"
    # ---- policy / business rules used at decision time
    concession_max_late_payments: int = 2
    concession_requires_zero_arrears: bool = True
    service_requires_open_request: bool = True
    decision_window_days: tuple = (30, 90)
    risk_flag_top_share: float = 0.40     # baselines 2/3 flag the riskiest 40 %

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        cfg = cls()
        path = Path(path) if path else ROOT / "config.yaml"
        if path.exists():
            raw = yaml.safe_load(path.read_text()) or {}
            names = {f.name for f in fields(cls)}
            for k, v in raw.items():
                if k in names:
                    setattr(cfg, k, tuple(v) if k == "decision_window_days" else v)
        if os.environ.get("PMSDI_QUICK"):                 # smoke-test sizes (used by `run_all.py --quick` and CI)
            cfg.n_history, cfg.n_current, cfg.n_properties = 5000, 1500, 60
        return cfg

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        names = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in d.items() if k in names})
        cfg.decision_window_days = tuple(cfg.decision_window_days)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)


def database_url() -> str:
    """SQLite by default (zero setup).  Set DATABASE_URL=postgresql+psycopg2://user:pw@host/db for PostgreSQL."""
    return os.environ.get("DATABASE_URL", f"sqlite:///{(DATA_DIR / 'pms.db').as_posix()}")
