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
| Baselines A–C | `src/ruav/baselines.py` | step 4 |
| Monte Carlo campaign | `scripts/run_monte_carlo.py` | step 4 |
| Figures and tables | `scripts/make_figures.py` | step 4 |
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
python scripts/run_scenario.py --config configs/hybrid_attack.yaml   # one run (step 4)
python scripts/run_monte_carlo.py --config configs/monte_carlo.yaml  # full campaign (step 4)
python scripts/make_figures.py                                       # tables and figures (step 4)
```
All random numbers come from seeds listed in `configs/monte_carlo.yaml`.
Raw results are written to `results/` and archived with each release on Zenodo.

## Citation
See `CITATION.cff`. After the Zenodo release, cite the DOI shown in the badge.

## License
MIT, see `LICENSE`.
