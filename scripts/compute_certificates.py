#!/usr/bin/env python3
"""Compute the analysis constants of Section 4 for the benchmark and store them in
certificates/benchmark.json (used in the paper, Table "analysis constants")."""
import json, os, sys
import numpy as np, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ruav import analysis, graph  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "hybrid_attack.yaml")))
A = np.array(cfg["graph"]["adjacency"]); g = np.array(cfg["graph"]["pinning"]); f = cfg["graph"]["f"]
kp, kv, M = 1.0, 1.8, 3
out = {"graph": {"leader_edge_connectivity": graph.leader_edge_connectivity(A, g), "f": f},
       "gains": {"k_p": kp, "k_v": kv, "m_levels": M},
       "mode_dependent_adt": analysis.adt_constants(A, g, f, M, kp, kv),
       "common_lyapunov": analysis.common_lyapunov_search(A, g, f, M, kp, kv)}
cbf = {"k0": 4.0, "k1": 4.0, "lambda1": 2.0, "lambda2": 2.0, "d_safe": 1.0,
       "a_u_bar": 1.0, "d_bar": 0.15, "a0_bar": 0.13, "sigma_bar": 0.05, "theta_bar": 0.05,
       "W_bar": 4.0, "R_s": 6.0, "u_max": float(4.0 * np.sqrt(3))}
for share, key, extra in ((0.5, "mu_pair", 0.0), (1.0, "mu_leader", cbf["a0_bar"])):
    cbf[key] = analysis.cbf_margin(cbf["sigma_bar"], cbf["theta_bar"], cbf["W_bar"], cbf["R_s"],
                                   cbf["u_max"], cbf["a_u_bar"] + cbf["d_bar"] + extra,
                                   cbf["k0"], cbf["k1"], share)
cbf["entry_condition_ok"] = analysis.sensing_radius_ok(cbf["R_s"], cbf["d_safe"], cbf["W_bar"], cbf["lambda1"])
cbf["activation_distance_pair"] = analysis.cbf_activation_distance(cbf["k0"], cbf["k1"],
                                    cbf["a_u_bar"] + cbf["d_bar"], cbf["mu_pair"])
out["cbf"] = cbf
os.makedirs(os.path.join(ROOT, "certificates"), exist_ok=True)
with open(os.path.join(ROOT, "certificates", "benchmark.json"), "w") as fh:
    json.dump(out, fh, indent=1)
summary = {k: v for k, v in out["common_lyapunov"].items() if k != "P"}
print(json.dumps({"adt": out["mode_dependent_adt"], "cqlf": summary, "cbf": cbf,
                  "graph": out["graph"]}, indent=1, default=float))
