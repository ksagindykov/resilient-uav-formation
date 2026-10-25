"""Nonlinear quadrotor model with an onboard attitude loop (higher-fidelity plant, Section 6).

Each vehicle has position p, velocity v, attitude R in SO(3) and body rate w:
    p_dot = v,   v_dot = (T/m) R e3 - g e3 + d,   R_dot = R [w]x,   J w_dot = M - w x J w,
with collective thrust T following its set-point through a first-order motor lag, thrust
limited to [0, T_max] and the commanded tilt limited to tilt_max.  The formation controller
of the paper only issues an acceleration set-point a_cmd; the "autopilot" converts it into
thrust and attitude set-points and tracks the attitude with the geometric controller of
Lee, Leok and McClamroch (2010).  None of these inner-loop effects is modelled in the
analysis of Section 4: they act as unmodelled dynamics.
All vehicles are integrated together (vectorised) with a fixed inner step.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

E3 = np.array([0.0, 0.0, 1.0])


@dataclass
class QuadParams:
    mass: float = 1.5                 # kg
    J: tuple = (0.02, 0.02, 0.04)     # kg m^2
    g: float = 9.81
    thrust_max_ratio: float = 2.0     # T_max / (m g)
    tilt_max_deg: float = 35.0
    tau_motor: float = 0.05           # s, thrust lag
    k_R: float = 120.0                # attitude gains (per unit inertia)
    k_w: float = 20.0
    substeps: int = 5                 # inner steps per outer sample (4 ms inner step)


def _hat(w):
    """Batch skew-symmetric matrices, w: (n,3) -> (n,3,3)."""
    z = np.zeros(len(w))
    return np.stack([np.stack([z, -w[:, 2], w[:, 1]], -1),
                     np.stack([w[:, 2], z, -w[:, 0]], -1),
                     np.stack([-w[:, 1], w[:, 0], z], -1)], 1)


def _vee(S):
    return np.stack([S[:, 2, 1], S[:, 0, 2], S[:, 1, 0]], -1)


def _expm_so3(w, dt):
    """Rodrigues formula for exp([w dt]x), batch."""
    th = np.linalg.norm(w, axis=1) * dt
    K = _hat(w * dt)
    I = np.eye(3)[None]
    s = np.where(th > 1e-9, np.sin(th) / np.maximum(th, 1e-12), 1.0)[:, None, None]
    c = np.where(th > 1e-9, (1 - np.cos(th)) / np.maximum(th, 1e-12) ** 2, 0.5)[:, None, None]
    return I + s * K + c * (K @ K)


class QuadFleet:
    def __init__(self, p0: np.ndarray, v0: np.ndarray, prm: QuadParams = QuadParams()):
        n = len(p0)
        self.prm = prm
        self.p, self.v = p0.astype(float).copy(), v0.astype(float).copy()
        self.R = np.tile(np.eye(3), (n, 1, 1))
        self.w = np.zeros((n, 3))
        self.T = np.full(n, prm.mass * prm.g)
        self.J = np.array(prm.J)
        self.tilt_cos = np.cos(np.deg2rad(prm.tilt_max_deg))

    def _setpoints(self, a_cmd):
        prm = self.prm
        F = prm.mass * (a_cmd + prm.g * E3)                 # desired force
        F[:, 2] = np.maximum(F[:, 2], 0.1 * prm.mass * prm.g)
        # tilt limit: keep the vertical component, shrink the horizontal one
        nF = np.linalg.norm(F, axis=1)
        cos_t = F[:, 2] / nF
        bad = cos_t < self.tilt_cos
        if bad.any():
            h = F[bad, :2]
            hmax = F[bad, 2] * np.tan(np.deg2rad(prm.tilt_max_deg))
            F[bad, :2] = h * (hmax / np.linalg.norm(h, axis=1))[:, None]
        b3 = F / np.linalg.norm(F, axis=1, keepdims=True)
        b1c = np.tile(np.array([1.0, 0.0, 0.0]), (len(F), 1))
        b2 = np.cross(b3, b1c); b2 /= np.linalg.norm(b2, axis=1, keepdims=True)
        b1 = np.cross(b2, b3)
        Rd = np.stack([b1, b2, b3], axis=2)
        T_d = np.clip(np.einsum("ij,ij->i", F, self.R[:, :, 2]), 0.0, prm.thrust_max_ratio * prm.mass * prm.g)
        return Rd, T_d

    def step(self, a_cmd: np.ndarray, d: np.ndarray, dt: float):
        """Advance all vehicles by one outer sample dt with acceleration set-points a_cmd (n,3)
        and external acceleration disturbance d (n,3). Returns (p, v)."""
        prm = self.prm
        h = dt / prm.substeps
        for _ in range(prm.substeps):
            Rd, T_d = self._setpoints(a_cmd)
            eR = 0.5 * _vee(np.transpose(Rd, (0, 2, 1)) @ self.R - np.transpose(self.R, (0, 2, 1)) @ Rd)
            Jw = self.w * self.J
            M = self.J * (-prm.k_R * eR - prm.k_w * self.w) + np.cross(self.w, Jw)
            # dynamics (semi-implicit Euler)
            acc = (self.T / prm.mass)[:, None] * self.R[:, :, 2] - prm.g * E3 + d
            self.p = self.p + h * self.v + 0.5 * h * h * acc
            self.v = self.v + h * acc
            wdot = (M - np.cross(self.w, Jw)) / self.J
            self.w = self.w + h * wdot
            self.R = self.R @ _expm_so3(self.w, h)
            self.T = self.T + (h / prm.tau_motor) * (T_d - self.T)
        # re-orthonormalise
        U, _, Vt = np.linalg.svd(self.R)
        self.R = U @ Vt
        return self.p.copy(), self.v.copy()

    def tilt_deg(self):
        return np.rad2deg(np.arccos(np.clip(self.R[:, 2, 2], -1, 1)))
