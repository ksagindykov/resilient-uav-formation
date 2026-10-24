#!/usr/bin/env python3
"""Effect of violating the sensing-radius design rule (Section 3.2): compares the campaign
with R_s = 8 m (results/mc_results.jsonl) and R_s = 6 m (results/mc_results_Rs6.jsonl,
produced by run_monte_carlo.py --sensing-radius 6 --out results/mc_results_Rs6.jsonl).
Writes results/design_rule.json."""
import json, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = {}
for tag, fn in (("Rs8", "mc_results.jsonl"), ("Rs6", "mc_results_Rs6.jsonl")):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "results", fn))]
    pr = [r for r in rows if r["method"] == "proposed"]
    ev = [e for r in pr for e in r["events"] if e[0] == "comm_fdi"]
    comm = [r["peak_Emax"] for r in pr if r["scenario"] == "comm"]
    out[tag] = {"fdi_events": len(ev), "fdi_detected": int(sum(e[1] for e in ev)),
                "fdi_detection_rate": float(np.mean([e[1] for e in ev])),
                "comm_peak_Emax_mean": float(np.mean(comm))}
json.dump(out, open(os.path.join(ROOT, "results", "design_rule.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
