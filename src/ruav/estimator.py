"""Attack-aware state estimation, Section 3.3.

Nominal mode: the navigation-filter output y_i is used directly (C = I_6).
Sensor alarm: the own state is reconstructed from trusted neighbours as the
coordinate-wise median of  z_ij - [rho_ij; varrho_ij]; with no trusted neighbour the
estimate is propagated by the double-integrator model (dead reckoning).
"""
from __future__ import annotations
import numpy as np


def estimate(y, sensor_alarm, msgs, rel, trusted, x_prev, u_prev, a_u_hat, dt):
    if not sensor_alarm:
        return np.asarray(y, float).copy()
    cands = [msgs[j] - np.concatenate(rel[j]) for j in trusted
             if msgs.get(j) is not None and j in rel]
    if cands:
        return np.median(np.array(cands), axis=0)
    p, v = x_prev[:3], x_prev[3:]
    acc = u_prev + a_u_hat
    return np.concatenate([p + dt * v + 0.5 * dt**2 * acc, v + dt * acc])
