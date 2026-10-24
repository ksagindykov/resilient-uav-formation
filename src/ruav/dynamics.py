"""Outer-loop double-integrator UAV model, Eq. (1) of the paper.

    p_dot = v,   v_dot = u + a_u + d,   ||u|| <= u_max
"""
from __future__ import annotations
import numpy as np


def saturate(u: np.ndarray, u_max: float) -> np.ndarray:
    """Scale each 3-D command so that its Euclidean norm does not exceed u_max."""
    u = np.atleast_2d(u)
    n = np.linalg.norm(u, axis=1, keepdims=True)
    scale = np.minimum(1.0, u_max / np.maximum(n, 1e-12))
    return u * scale


def step(p: np.ndarray, v: np.ndarray, u: np.ndarray, a_u: np.ndarray,
         d: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretisation of the double integrator.

    p, v, u, a_u, d: arrays of shape (N, 3). Returns (p_next, v_next).
    """
    acc = u + a_u + d
    p_next = p + dt * v + 0.5 * dt**2 * acc
    v_next = v + dt * acc
    return p_next, v_next


def leader_circle(t: float, radius: float = 20.0, omega: float = 0.08,
                  z0: float = 10.0, z_amp: float = 2.0, z_omega: float = 0.04):
    """Leader reference used in the benchmark: horizontal circle with slow altitude oscillation.

    Returns (p0, v0, a0), each of shape (3,).
    """
    p0 = np.array([radius * np.cos(omega * t), radius * np.sin(omega * t),
                   z0 + z_amp * np.sin(z_omega * t)])
    v0 = np.array([-radius * omega * np.sin(omega * t), radius * omega * np.cos(omega * t),
                   z_amp * z_omega * np.cos(z_omega * t)])
    a0 = np.array([-radius * omega**2 * np.cos(omega * t), -radius * omega**2 * np.sin(omega * t),
                   -z_amp * z_omega**2 * np.sin(z_omega * t)])
    return p0, v0, a0
