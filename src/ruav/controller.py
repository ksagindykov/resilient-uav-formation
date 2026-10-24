"""Trust-weighted distributed formation controller, Sections 3.4 and 3.6.

u_i^nom = - k_p [ sum_j abar_ij((p_i - r_i) - (p_j - r_j)) + gbar_i (p_i - p_0 - r_i) ]
          - k_v [ sum_j abar_ij(v_i - v_j)               + gbar_i (v_i - v_0) ]
          - chi_i^u * a_u_hat_i
with abar_ij = a_ij * alpha_ij and gbar_i = g_i * alpha_i0.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ControlParams:
    k_p: float = 1.0
    k_v: float = 1.8
    tau_up: float = 3.0     # trust ramp-up time after a link alarm clears, s
    m_levels: int = 3       # trust takes values in {0, 1/M, ..., 1}  (finite mode set, Section 4)


class TrustWeights:
    """alpha_ij in {0, 1/M, ..., 1}: set to 0 on alarm; after the alarm clears it is raised
    by 1/M every tau_up/M seconds. The quantisation keeps the set of closed-loop modes finite,
    which is what the average dwell-time analysis of Section 4 requires."""
    def __init__(self, links: list, tau_up: float, m_levels: int = 3):
        self.alpha = {j: 1.0 for j in links}
        self.timer = {j: 0.0 for j in links}
        self.tau_up, self.m = tau_up, m_levels

    def update(self, link_alarm: dict, dt: float) -> dict:
        hold = self.tau_up / self.m
        for j, alarm in link_alarm.items():
            if alarm:
                self.alpha[j], self.timer[j] = 0.0, 0.0
            elif self.alpha[j] < 1.0:
                self.timer[j] += dt
                if self.timer[j] >= hold - 1e-12:
                    self.alpha[j] = min(1.0, round(self.alpha[j] * self.m + 1) / self.m)
                    self.timer[j] = 0.0
        return dict(self.alpha)


def formation_control(i, x_hat, offsets, msgs, alpha, leader_state, pinned, a_u_hat,
                      actuator_alarm, prm: ControlParams, leader_key=-1):
    p, v = x_hat[:3], x_hat[3:]
    ep, ev = np.zeros(3), np.zeros(3)
    for j, z in msgs.items():
        if j == leader_key or z is None:
            continue
        a = alpha.get(j, 0.0)
        ep += a * ((p - offsets[i]) - (z[:3] - offsets[j]))
        ev += a * (v - z[3:])
    if pinned and leader_state is not None:
        g = alpha.get(leader_key, 1.0)
        ep += g * (p - leader_state[:3] - offsets[i])
        ev += g * (v - leader_state[3:])
    u = -prm.k_p * ep - prm.k_v * ev
    if actuator_alarm:
        u = u - a_u_hat
    return u
