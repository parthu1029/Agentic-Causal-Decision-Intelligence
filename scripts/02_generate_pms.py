"""Step 2 - build the synthetic longitudinal PMS, write it to the database, materialise the feature store,
and save the hidden ground truth (oracle) separately.   Re-run with a different seed/config to regenerate."""
import _bootstrap  # noqa: F401
import time

from pmsdi.config import SYNTH_DIR, Config
from pmsdi.data import calibrate, db
from pmsdi.data.features import build_features
from pmsdi.data.synthetic_pms import generate, save_world

cfg = Config.load()
calib = calibrate.load()
print("calibration:", "loaded from data/processed/calibration.json" if calib else "none (documented defaults)")
t0 = time.time()
tables, oracle, world = generate(cfg, calib, verbose=True)
engine = db.get_engine()
db.write_tables(engine, tables)
F = build_features(tables)
db.write_features(engine, F)
oracle.to_csv(SYNTH_DIR / "oracle.csv.gz")
save_world(world)
print(f"\nDatabase: {engine.url}")
for k, v in tables.items():
    print(f"  {k:22s} {len(v):>9,d} rows")
print(f"  {'feature_store':22s} {len(F):>9,d} rows x {F.shape[1]} features")
print(f"Ground truth (NEVER shown to models) -> {SYNTH_DIR / 'oracle.csv.gz'}")
print(f"done in {time.time() - t0:.0f}s")
