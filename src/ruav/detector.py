"""Residual-based attack detection and channel isolation, Section 3.2.

All tests are local to follower i and use only y_i, received messages z_ij and onboard
relative measurements rho_ij.  Each raw test passes through a persistence filter: an alarm
is raised after `n_on` consecutive exceedances and cleared after `n_off` consecutive
non-exceedances, which bounds the switching frequency of the controller (Section 4).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DetectorParams:
    theta_c: float = 0.5      # link-consistency threshold, m
    theta_u: float = 0.4      # actuation-residual threshold, m/s^2
    tau_f: float = 0.5        # low-pass time constant for the actuation residual, s
    n_on: int = 5             # samples to raise an alarm
    n_off: int = 50           # samples to clear an alarm
    dos_samples: int = 5      # missing samples before a link is declared unavailable


class Persistence:
    """Counter-based hysteresis turning a raw boolean test into a stable alarm."""
    def __init__(self, n_on: int, n_off: int):
        self.n_on, self.n_off = n_on, n_off
        self.state, self.cnt = False, 0

    def update(self, raw: bool) -> bool:
        if raw != self.state:
            self.cnt += 1
            if self.cnt >= (self.n_on if raw else self.n_off):
                self.state, self.cnt = raw, 0
        else:
            self.cnt = 0
        return self.state


@dataclass
class DetectorOutput:
    sensor_alarm: bool
    actuator_alarm: bool
    link_alarm: dict            # j -> bool (FDI or unavailability)
    a_u_hat: np.ndarray         # estimated actuation attack (3,)
    residuals: dict = field(default_factory=dict)


class Detector:
    def __init__(self, prm: DetectorParams, neighbours: list, dt: float):
        self.prm, self.dt = prm, dt
        self.sensor = Persistence(prm.n_on, prm.n_off)
        self.actuator = Persistence(prm.n_on, prm.n_off)
        self.links = {j: Persistence(prm.n_on, prm.n_off) for j in neighbours}
        self.missing = {j: 0 for j in neighbours}
        self.a_u_hat = np.zeros(3)
        self.v_prev = None

    def update(self, y: np.ndarray, u_prev: np.ndarray, msgs: dict, rel: dict) -> DetectorOutput:
        """y = [p; v] own measurement (6,); msgs = {j: z_ij (6,) or None}; rel = {j: (rho, varrho)}."""
        prm = self.prm
        # (a) link consistency: received position vs own position + relative measurement
        r_c, received = {}, []
        for j, z in msgs.items():
            if z is None:
                self.missing[j] += 1
                continue
            self.missing[j] = 0
            if j in rel:
                r_c[j] = float(np.linalg.norm(z[:3] - y[:3] - rel[j][0]))
                received.append(j)
        bad = [j for j in received if r_c[j] > prm.theta_c]
        # (b) own-sensor test by majority: if most received links disagree, suspect y_i itself
        own_raw = len(received) >= 2 and len(bad) > len(received) / 2
        sensor_alarm = self.sensor.update(own_raw)
        link_alarm = {}
        for j, pers in self.links.items():
            fdi_raw = (j in bad) and not own_raw
            dos_raw = self.missing[j] >= prm.dos_samples
            link_alarm[j] = pers.update(fdi_raw or dos_raw)
        # (c) actuation residual: filtered unknown-input estimate a = dv/dt - u
        v = y[3:]
        if self.v_prev is not None and not sensor_alarm:
            raw_a = (v - self.v_prev) / self.dt - u_prev
            beta = self.dt / (prm.tau_f + self.dt)
            self.a_u_hat = (1 - beta) * self.a_u_hat + beta * raw_a
        self.v_prev = v.copy()
        act_raw = (not sensor_alarm) and float(np.linalg.norm(self.a_u_hat)) > prm.theta_u
        actuator_alarm = self.actuator.update(act_raw)
        return DetectorOutput(sensor_alarm, actuator_alarm, link_alarm, self.a_u_hat.copy(),
                              {"link": r_c, "actuation": float(np.linalg.norm(self.a_u_hat))})
