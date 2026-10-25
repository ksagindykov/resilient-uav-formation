# resilient-uav-formation

Code accompanying the manuscript

> **Resilient Distributed Formation Control of Multi-UAV Systems under Sensor,
> Communication and Actuator Cyberattacks with Collision Avoidance Guarantees**
> K. Sagindykov, N. Mukusheva — submitted to *Drones* (MDPI), 2027.

The repository reproduces every number, table and figure of the paper.

## Status
| Module | File | Status |
|---|---|---|
| UAV double-integrator model | `src/ruav/dynamics.py` | done |
| Interaction graph, robustness check (Assumption 1(iv)) | `src/ruav/graph.py` | done |
| Time-staggered attack schedules | `src/ruav/attacks.py` | done |
| Residual detector and channel isolation | `src/ruav/detector.py` | done |
| Attack-aware estimator | `src/ruav/estimator.py` | done |
| Trust-weighted formation controller | `src/ruav/controller.py` | done |
| Robust CBF safety filter | `src/ruav/cbf.py` | done |
| Analysis constants, Lyapunov certificate (Section 4) | `src/ruav/analysis.py`, `scripts/compute_certificates.py` | done |
| Simulation, baselines B1–B3 and ablations | `src/ruav/sim.py` | done |
| Monte Carlo campaign (1750 runs, raw results in `results/`) | `scripts/run_monte_carlo.py` | done |
| Figures, tables, headline numbers | `scripts/make_figures.py`, `scripts/design_rule_check.py` | done |
| Nonlinear quadrotor plant (attitude on SO(3), thrust/tilt limits, motor lag) | `src/ruav/quad.py` | done |
| Extended studies: quadrotor, stealthy attacks, 10/20 followers, sensitivity (1770 runs) | `scripts/run_extended.py`, `scripts/make_extended.py` | done |
| PX4 SITL / flight tests | — | future work |

## Installation
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Reproducing the paper
```bash
python scripts/compute_certificates.py                               # Table of analysis constants (Section 4)
python scripts/run_monte_carlo.py --runs 50 --workers 4              # full campaign, ~1 h on 2 cores
python scripts/run_monte_carlo.py --runs 50 --sensing-radius 6 --out results/mc_results_Rs6.jsonl
python scripts/make_figures.py                                       # tables, figures, headline.json
python scripts/design_rule_check.py                                  # sensing-radius design rule
python scripts/compute_certificates.py --n 10 --rs 9  --tag N10       # certificates for the scalability study
python scripts/compute_certificates.py --n 20 --rs 14 --tag N20
python scripts/compute_certificates.py --kp 1 --kv 2 --tag quad      # certificate for the quadrotor gains
for s in quad stealth scale sens; do python scripts/run_extended.py --study $s --runs 30 --workers 4; done   # ~2 h on 2 cores
python scripts/make_extended.py                                      # extended tables, results/extended.json
```
All random numbers come from fixed seeds (`SEED0` in `scripts/run_monte_carlo.py`); every method sees the same attacks, noise and initial conditions for a given seed.
Raw results are written to `results/` and archived with each release on Zenodo.

## Citation
See `CITATION.cff`. After the Zenodo release, cite the DOI shown in the badge.

## License
MIT, see `LICENSE`.
