"""Step 1 - read whatever public datasets are in data/raw/ and write data/processed/calibration.json."""
import _bootstrap  # noqa: F401
import json

from pmsdi.data import calibrate

cal = calibrate.run()
print("Calibration sources:")
for s in cal["sources_used"]:
    print("  -", s)
print("\nrent medians by BHK:", cal["rent_median_by_bhk"])
print("saved -> data/processed/calibration.json")
