"""Step 0 (optional) - fetch the public datasets that can be downloaded without a login.
NYC Open Data (Socrata) needs no key for moderate volumes.  Kaggle needs your API token (see docs/DATA_SOURCES.md);
HUD files are downloaded manually from the links in docs/DATA_SOURCES.md.   Files land in data/raw/.
(This script could not be executed in the build sandbox - outbound network was restricted - so check the first run.)"""
import _bootstrap  # noqa: F401
import argparse
import subprocess
import sys

import pandas as pd

from pmsdi.config import RAW_DIR

SOCRATA = {
    # HPD Housing Maintenance Code Violations (dataset id wvxf-dwi5) and Complaints and Problems (ygpa-z7cr)
    "nyc_hpd_violations.csv": "https://data.cityofnewyork.us/resource/wvxf-dwi5.csv",
    "nyc_hpd_complaints.csv": "https://data.cityofnewyork.us/resource/ygpa-z7cr.csv",
}

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=200000, help="rows to pull per NYC dataset (Socrata $limit)")
ap.add_argument("--kaggle", action="store_true", help="also try the Kaggle CLI for the rental-listings dataset")
args = ap.parse_args()

for fname, url in SOCRATA.items():
    out = RAW_DIR / fname
    if out.exists():
        print("exists, skipping:", out.name)
        continue
    print("downloading", url, f"(limit {args.rows}) ...")
    try:
        df = pd.read_csv(f"{url}?$limit={args.rows}", low_memory=False)
        df.to_csv(out, index=False)
        print("  saved", out, df.shape)
    except Exception as exc:
        print("  FAILED:", exc, "\n  -> download manually from the NYC Open Data page and save as", out)

if args.kaggle:
    cmd = ["kaggle", "datasets", "download", "-d", "iamsouravbanerjee/house-rent-prediction-dataset", "-p", str(RAW_DIR), "--unzip"]
    print("running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        print("Kaggle CLI failed:", exc, "\n  -> download House_Rent_Dataset.csv manually into data/raw/")
print("\nNow run: python scripts/01_calibrate.py")
