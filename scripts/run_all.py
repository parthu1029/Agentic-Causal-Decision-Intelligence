"""Run the whole pipeline in order.   python scripts/run_all.py [--quick] [--with-sensitivity] [--with-dowhy]
--quick  : smaller world (5k history / 1.5k current) for a ~2 minute smoke test."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--with-sensitivity", action="store_true")
ap.add_argument("--with-dowhy", action="store_true")
a = ap.parse_args()
here = Path(__file__).parent
env = dict(os.environ, **({"PMSDI_QUICK": "1"} if a.quick else {}))
steps = ["01_calibrate.py", "02_generate_pms.py", "03_train_models.py", "07_pick_demo_tenants.py", "04_evaluate.py"]
if a.with_sensitivity:
    steps.append("05_sensitivity.py")
if a.with_dowhy:
    steps.append("08_dowhy_check.py")
t0 = time.time()
for s in steps:
    print(f"\n{'=' * 78}\n>>> {s}\n{'=' * 78}", flush=True)
    r = subprocess.run([sys.executable, str(here / s)] + (["--quick"] if (a.quick and s == "05_sensitivity.py") else []), env=env)
    if r.returncode:
        sys.exit(f"step {s} failed")
print(f"\nAll done in {time.time() - t0:.0f}s.  Start the dashboard:  streamlit run app/dashboard.py")
