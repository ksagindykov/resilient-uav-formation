#!/usr/bin/env python3
"""Monte Carlo campaign of Section 6: every method x scenario x seed (paired seeds:
all methods see identical attacks, noise seeds and initial conditions).

Usage: python scripts/run_monte_carlo.py [--runs 50] [--workers 2] [--out results/mc_results.jsonl]
"""
import argparse, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from multiprocessing import Pool
import numpy as np, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ruav import sim  # noqa: E402

SCENARIOS = ["clean", "sensor", "comm", "actuator", "hybrid"]
SEED0 = 20261024

_cfg = None


def _init(sensing_radius=None):
    global _cfg
    cert = json.load(open(os.path.join(ROOT, "certificates", "benchmark.json")))
    cy = yaml.safe_load(open(os.path.join(ROOT, "configs", "hybrid_attack.yaml")))
    _cfg = sim.default_config(cy, cert["cbf"]["mu_pair"], cert["cbf"]["mu_leader"])
    if sensing_radius is not None:      # design-rule study (Section 6): R_s below the neighbour distance
        _cfg.cbf.sensing_radius = sensing_radius


def _job(args):
    method, scenario, seed = args
    r = sim.run(_cfg, method, scenario, seed)
    r["events"] = [(c, bool(d), None if (dl is None or np.isnan(dl)) else float(dl)) for c, d, dl in r["events"]]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "mc_results.jsonl"))
    ap.add_argument("--sensing-radius", type=float, default=None,
                    help="override R_s (used for results/mc_results_Rs6.jsonl with 6.0)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line); done.add((r["method"], r["scenario"], r["seed"]))
    jobs = [(m, s, SEED0 + k) for s in SCENARIOS for k in range(a.runs) for m in sim.METHODS
            if (m, s, SEED0 + k) not in done]
    print(f"{len(jobs)} runs to do ({len(done)} already in {a.out})", flush=True)
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.sensing_radius,)) as pool, open(a.out, "a") as fh:
        for n, r in enumerate(pool.imap_unordered(_job, jobs), 1):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if n % 50 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"{n}/{len(jobs)}  {el/60:.1f} min elapsed, ~{el/n*(len(jobs)-n)/60:.1f} min left", flush=True)


if __name__ == "__main__":
    main()
