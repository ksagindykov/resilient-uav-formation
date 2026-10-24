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
| PX4 SITL + Gazebo | `docs/SITL.md` | step 5 |

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
```
All random numbers come from seeds listed in `configs/monte_carlo.yaml`.
Raw results are written to `results/` and archived with each release on Zenodo.

## Citation
See `CITATION.cff`. After the Zenodo release, cite the DOI shown in the badge.

## License
MIT, see `LICENSE`.
