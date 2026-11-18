#!/usr/bin/env python3
"""Attack-free residual levels of the detector (threshold selection, Section 3.1).

Runs the proposed controller in the attack-free scenario and records the link-consistency
residuals r_ij^c and the actuation residual ||a_hat_i^u|| after a 2 s transient.
Writes results/residual_levels.json.  Usage: python scripts/residual_levels.py [--runs 50]
"""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "src"))
import run_extended as R          # noqa: E402
from ruav import detector as D    # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--runs", type=int, default=50); a = ap.parse_args()
rc, au = [], []
_orig = D.Detector.update
def _rec(self, *args, **kw):
    out = _orig(self, *args, **kw)
    if self.t > 2.0:
        rc.extend(out.residuals["link"].values()); au.append(out.residuals["actuation"])
    return out
D.Detector.update = _rec
for k in range(a.runs):
    R.sim.run(R.base_cfg(), "proposed", "clean", R.SEED0 + k)
rc, au = np.asarray(rc), np.asarray(au)
res = {"runs": a.runs,
       "link": {"max": float(rc.max()), "q9999": float(np.quantile(rc, 0.9999)), "q99": float(np.quantile(rc, 0.99))},
       "actuation": {"max": float(au.max()), "q9999": float(np.quantile(au, 0.9999)), "q99": float(np.quantile(au, 0.99))}}
json.dump(res, open(os.path.join(ROOT, "results", "residual_levels.json"), "w"), indent=1)
print(json.dumps(res))
