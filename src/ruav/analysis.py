"""Constants of the analysis in Section 4, computed for a given graph and gains.

Modes: the closed-loop error matrix of mode sigma is
    M_sigma = [[0, I], [-k_p Hbar_sigma, -k_v Hbar_sigma]]   (scalar per axis),
where Hbar_sigma = Lbar + Gbar has at most f attacked links with trust weight in
{0, 1/M, ..., 1}.  For each mode P_sigma solves M^T P + P M = -I, so that
V_sigma = xi^T P_sigma xi decays at rate c_sigma = 1 / (2 lambda_max(P_sigma))  (in V),
and the jump factor between modes is mu = max_{a,b} lambda_max(P_a) / lambda_min(P_b).
Theorem 2 requires the average dwell time tau_D > ln(mu) / c with c = min_sigma c_sigma.
"""
from __future__ import annotations
from itertools import combinations, product
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from .graph import in_links


def mode_matrices(A, g, f: int, m_levels: int, k_p: float, k_v: float):
    A = np.asarray(A, float); g = np.asarray(g, float)
    n = len(g)
    links = [(j, i) for i in range(n) for j in in_links(A, g, i)]
    levels = [k / m_levels for k in range(m_levels)]          # attacked/ramping link levels < 1
    modes = {(): None}
    for r in range(1, f + 1):
        for subset in combinations(links, r):
            for lv in product(levels, repeat=r):
                modes[tuple(zip(subset, lv))] = None
    out = {}
    for key in modes:
        A2, g2 = A.copy(), g.copy()
        for (j, i), lv in key:
            if j == -1:
                g2[i] *= lv
            else:
                A2[i, j] *= lv
        H = np.diag(A2.sum(1)) - A2 + np.diag(g2)
        M = np.block([[np.zeros((n, n)), np.eye(n)], [-k_p * H, -k_v * H]])
        out[key] = M
    return out


def adt_constants(A, g, f=1, m_levels=3, k_p=1.0, k_v=1.8):
    mats = mode_matrices(A, g, f, m_levels, k_p, k_v)
    P, c, hurwitz = {}, {}, True
    for key, M in mats.items():
        if np.max(np.linalg.eigvals(M).real) >= 0:
            hurwitz = False
            continue
        Pk = solve_continuous_lyapunov(M.T, -np.eye(M.shape[0]))
        Pk = 0.5 * (Pk + Pk.T)
        P[key] = Pk
        c[key] = 1.0 / (2.0 * np.linalg.eigvalsh(Pk).max())
    lmax = {k: np.linalg.eigvalsh(v).max() for k, v in P.items()}
    lmin = {k: np.linalg.eigvalsh(v).min() for k, v in P.items()}
    mu = max(lmax[a] / lmin[b] for a in P for b in P)
    cmin = min(c.values())
    return {"n_modes": len(mats), "all_hurwitz": hurwitz, "c": cmin, "mu": mu,
            "tau_D_min": float(np.log(mu) / cmin),
            "kappa": float(np.sqrt(max(lmax.values()) / min(lmin.values())))}


def cbf_activation_distance(k0, k1, delta_bar, mu_margin, share=0.5, d_safe=1.0, u_norm=0.0):
    """Smallest distance beyond which the pair constraint is inactive for vehicles at rest
    commanding ||u|| <= u_norm (Proposition 1)."""
    r = np.linspace(d_safe, 20.0, 20001)
    ok = share * k0 * (r**2 - d_safe**2) >= 2 * r * (u_norm + delta_bar) + mu_margin
    return float(r[np.argmax(ok)]) if ok.any() else np.inf


def cbf_margin(sig, vth, W, Rs, u_max, delta_bar, k0, k1, share):
    """mu_ij of Theorem 1 (relative-sensing errors sig, vth; relative speed bound W;
    sensing radius Rs; command bound u_max; delta_bar = a_u + d (+ a_0 for the leader))."""
    bracket = 2 * vth * (2 * W + vth) + 2 * k1 * (sig * (W + vth) + Rs * vth) + k0 * sig * (2 * Rs + sig)
    return 2 * sig * (u_max + delta_bar) + share * bracket


def sensing_radius_ok(Rs, d_safe, W, lam1):
    """Entry condition of Theorem 1: lam1 (Rs^2 - d_s^2) >= 2 Rs W."""
    return lam1 * (Rs**2 - d_safe**2) >= 2 * Rs * W


def common_lyapunov_search(A, g, f=1, m_levels=3, k_p=1.0, k_v=1.8, iters=4000, seed_eps=1e-3):
    """Projected-subgradient search for a common quadratic Lyapunov matrix P (trace 1) with
    M_sigma^T P + P M_sigma < 0 for every mode.  Returns a certificate dict; the certificate
    is valid iff 'max_eig' < 0 (re-checked from the returned P, not from the iterates)."""
    mats = list(mode_matrices(A, g, f, m_levels, k_p, k_v).values())
    n = mats[0].shape[0]
    P = solve_continuous_lyapunov(mats[0].T, -np.eye(n)); P = 0.5 * (P + P.T); P /= np.trace(P)
    best_lam, best_P = np.inf, P.copy()
    for k in range(iters):
        lam, v, Mw = -np.inf, None, None
        for M in mats:
            w, V = np.linalg.eigh(M.T @ P + P @ M)
            if w[-1] > lam:
                lam, v, Mw = w[-1], V[:, -1], M
        if lam < best_lam:
            best_lam, best_P = lam, P.copy()
        G = Mw @ np.outer(v, v) + np.outer(v, v) @ Mw.T
        P = P - (0.05 / np.sqrt(k + 1)) * 0.5 * (G + G.T)
        w, V = np.linalg.eigh(0.5 * (P + P.T))
        P = (V * np.maximum(w, seed_eps / n)) @ V.T
        P /= np.trace(P)
    P = best_P
    max_eig = max(np.linalg.eigvalsh(M.T @ P + P @ M).max() for M in mats)
    lP = np.linalg.eigvalsh(P)
    c = min(-np.linalg.eigvalsh(M.T @ P + P @ M).max() for M in mats) / lP.max()
    return {"valid": bool(max_eig < 0), "max_eig": float(max_eig), "n_modes": len(mats),
            "c": float(c), "kappa": float(np.sqrt(lP.max() / lP.min())),
            "lambda_min_P": float(lP.min()), "lambda_max_P": float(lP.max()), "P": P.tolist()}
