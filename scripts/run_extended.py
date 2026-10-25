#!/usr/bin/env python3
"""Extended campaigns of Section 6 (beyond the main Monte Carlo study):

  quad      nonlinear quadrotor plant (attitude loop, thrust/tilt limits, motor lag)
  stealth   slowly growing spoofing and sub-threshold false data + actuation
  scale     formations of 10 and 20 followers (time-staggered attacks)
  sens      detector thresholds theta_c x theta_u, and attack amplitude scale

Every record is appended to results/ext_<study>.jsonl and carries its configuration.
Usage: python scripts/run_extended.py --study quad [--runs 30] [--workers 2]
"""
import argparse, copy, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from multiprocessing import Pool
import numpy as np, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ruav import sim  # noqa: E402

SEED0 = 20261024
SCALE_RS = {10: 9.0, 20: 14.0}      # sensing radius per formation size (design rule, Section 3.2)


QUAD_GAINS = (1.0, 2.0)               # outer-loop gains matched to the attitude-loop bandwidth
QUAD_TAU_INNER, ARM_TIME = 0.15, 5.0  # inner-loop model in the actuation residual; detector arming


def base_cfg(n=None, rs=None, plant="di"):
    cy = yaml.safe_load(open(os.path.join(ROOT, "configs", "hybrid_attack.yaml")))
    name = "benchmark.json" if n is None else f"benchmark_N{n}.json"
    if plant == "quad":
        name = "benchmark_quad.json"
    cert = json.load(open(os.path.join(ROOT, "certificates", name)))
    cfg = sim.default_config(cy, cert["cbf"]["mu_pair"], cert["cbf"]["mu_leader"])
    if n is not None:
        cfg.offsets, cfg.A, cfg.g = sim.make_benchmark(n)
        cfg.cbf.sensing_radius = rs
    if plant == "quad":
        cfg.plant = "quad"
        cfg.ctl = copy.replace(cfg.ctl, k_p=QUAD_GAINS[0], k_v=QUAD_GAINS[1])
        cfg.det = copy.replace(cfg.det, tau_inner=QUAD_TAU_INNER, arm_time=ARM_TIME)
    return cfg


def jobs_for(study, runs):
    seeds = [SEED0 + k for k in range(runs)]
    J = []
    if study == "quad":
        for s in ("clean", "sensor", "hybrid"):
            for m in ("proposed", "abl_no_detect", "B1_consensus", "B2_wmsr", "B3_consensus_cbf"):
                J += [dict(study=study, method=m, scenario=s, seed=k, plant="quad") for k in seeds]
    elif study == "stealth":
        for s in ("stealth_ramp", "stealth_sub"):
            for m in ("proposed", "abl_no_detect", "B1_consensus", "B2_wmsr", "B3_consensus_cbf"):
                J += [dict(study=study, method=m, scenario=s, seed=k) for k in seeds]
    elif study == "scale":
        for n in (10, 20):
            for m in ("proposed", "B1_consensus", "B2_wmsr", "B3_consensus_cbf"):
                J += [dict(study=study, method=m, scenario="hybrid", seed=k, n=n) for k in seeds]
    elif study == "sens":
        for tc in (0.3, 0.5, 0.8):
            for tu in (0.3, 0.4, 0.6):
                for s in ("clean", "hybrid"):
                    J += [dict(study=study, method="proposed", scenario=s, seed=k, theta_c=tc, theta_u=tu) for k in seeds]
        for sc in (0.25, 0.5, 1.0, 1.5):
            for m in ("proposed", "B1_consensus"):
                J += [dict(study=study, method=m, scenario="hybrid", seed=k, attack_scale=sc) for k in seeds]
    return J


def _job(j):
    n = j.get("n")
    cfg = base_cfg(n, SCALE_RS.get(n), j.get("plant", "di"))
    cfg.attack_scale = j.get("attack_scale", 1.0)
    if "theta_c" in j:
        cfg.det = copy.replace(cfg.det, theta_c=j["theta_c"], theta_u=j["theta_u"])
    r = sim.run(cfg, j["method"], j["scenario"], j["seed"])
    r["events"] = [(c, bool(d), None if (dl is None or np.isnan(dl)) else float(dl)) for c, d, dl in r["events"]]
    r.update({k: v for k, v in j.items() if k not in r})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, choices=["quad", "stealth", "scale", "sens"])
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    out = os.path.join(ROOT, "results", f"ext_{a.study}.jsonl")
    key = lambda d: json.dumps({k: d[k] for k in sorted(d) if k in ("method", "scenario", "seed", "plant", "n", "theta_c", "theta_u", "attack_scale")}, sort_keys=True)
    done = set()
    if os.path.exists(out):
        done = {key(json.loads(l)) for l in open(out)}
    jobs = [j for j in jobs_for(a.study, a.runs) if key(j) not in done]
    print(f"{a.study}: {len(jobs)} runs to do", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool, open(out, "a") as fh:
        for k, r in enumerate(pool.imap_unordered(_job, jobs), 1):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if k % 25 == 0 or k == len(jobs):
                el = time.time() - t0
                print(f"{k}/{len(jobs)} {el/60:.1f} min, ~{el/k*(len(jobs)-k)/60:.1f} min left", flush=True)


if __name__ == "__main__":
    main()
