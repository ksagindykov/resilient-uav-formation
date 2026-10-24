"""Robust exponential control barrier function (ECBF) safety filter, Section 3.5.

For a pair (i, j) with q = p_i - p_j, w = v_i - v_j and h = ||q||^2 - d_s^2,
    h_dot  = 2 q.w,
    h_ddot = 2 ||w||^2 + 2 q.(acc_i - acc_j).
The ECBF condition h_ddot + k1 h_dot + k0 h >= 0 keeps h >= 0 (Nguyen & Sreenath, 2016).
It is split between the two followers (reciprocal responsibility, Wang et al., 2017);
each side adds a robust margin for its own bounded actuation attack and disturbance.
Against the leader, which does not run the filter, the follower takes full responsibility
and also absorbs the bounded leader acceleration.

Only onboard relative measurements (rho, varrho) are used -- never the communication link.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class CBFParams:
    d_safe: float = 1.0       # d_s, m
    k0: float = 4.0           # ECBF gains: s^2 + k1 s + k0 must be Hurwitz (double root at -2)
    k1: float = 4.0
    a_u_bar: float = 1.0      # bound on actuation attack ||a_i^u||, m/s^2
    d_bar: float = 0.15       # bound on disturbance ||d_i||, m/s^2
    a0_bar: float = 0.2       # bound on leader acceleration, m/s^2
    meas_margin: float = 0.0  # mu_ij, margin for relative-sensing errors (derived in Section 4)
    u_box: float = 4.0        # per-axis bound u_max/sqrt(3) used in the QP of Section 3.5, m/s^2
    sensing_radius: float = 6.0


def pair_constraint(rho: np.ndarray, varrho: np.ndarray, prm: CBFParams, with_leader: bool):
    """Return (g, b) such that the constraint reads g . u_i <= b.

    rho = p_j - p_i, varrho = v_j - v_i (onboard relative measurements of neighbour j).
    """
    q = -np.asarray(rho, float)          # p_i - p_j
    w = -np.asarray(varrho, float)       # v_i - v_j
    nq = float(np.linalg.norm(q))
    h = nq**2 - prm.d_safe**2
    hd = 2.0 * q @ w
    if with_leader:
        share = 1.0
        disturb = prm.a_u_bar + prm.d_bar + prm.a0_bar
    else:
        share = 0.5
        disturb = prm.a_u_bar + prm.d_bar
    # 2 q.u_i >= -share*(2||w||^2 + k1 hd + k0 h) + 2||q|| disturb + mu
    rhs = -share * (2.0 * w @ w + prm.k1 * hd + prm.k0 * h) + 2.0 * nq * disturb + prm.meas_margin
    return -2.0 * q, -rhs                # -2q.u <= -rhs


def solve_qp(u_nom: np.ndarray, G: np.ndarray, b: np.ndarray, iters: int = 500, tol: float = 1e-9):
    """min ||u - u_nom||^2  s.t.  G u <= b   (Hildreth dual coordinate ascent).

    Returns (u, max_violation). max_violation > 0 means the constraint set was infeasible
    (or not converged); the caller decides on the fallback.
    """
    u_nom = np.asarray(u_nom, float)
    if G.size == 0:
        return u_nom.copy(), 0.0
    lam = np.zeros(G.shape[0])
    gg = np.einsum("ij,ij->i", G, G)
    u = u_nom.copy()
    for _ in range(iters):
        delta = 0.0
        for m in range(G.shape[0]):
            if gg[m] == 0.0:
                continue
            viol = G[m] @ u - b[m]
            new = max(0.0, lam[m] + viol / gg[m])
            step = new - lam[m]
            if step != 0.0:
                u -= step * G[m]
                lam[m] = new
                delta = max(delta, abs(step))
        if delta < tol:
            break
    return u, float(max(0.0, np.max(G @ u - b)))


def safety_filter(u_nom: np.ndarray, rel: dict, prm: CBFParams, leader_key=-1):
    """Minimally modify u_nom so that every pair constraint and the input box hold.

    rel: {j: (rho_ij, varrho_ij)} for all vehicles within the sensing radius;
         key `leader_key` denotes the leader.
    Returns (u_safe, info) with info = {'active': n_active, 'violation': v}.
    """
    rows, rhs = [], []
    for j, (rho, varrho) in rel.items():
        if np.linalg.norm(rho) > prm.sensing_radius:
            continue
        g, bb = pair_constraint(rho, varrho, prm, with_leader=(j == leader_key))
        rows.append(g)
        rhs.append(bb)
    for ax in range(3):                  # box |u_k| <= u_box
        e = np.zeros(3); e[ax] = 1.0
        rows += [e, -e]
        rhs += [prm.u_box, prm.u_box]
    G = np.array(rows); b = np.array(rhs)
    u, viol = solve_qp(u_nom, G, b)
    n_active = int(np.sum(np.abs(G @ u - b) < 1e-6))
    return u, {"active": n_active, "violation": viol}
