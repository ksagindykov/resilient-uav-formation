"""Time-staggered multi-channel attack schedules (Section 2.4, Definition 1, Assumption 1)."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import yaml

CHANNEL_TYPES = ("sensor", "comm_fdi", "comm_dos", "actuator")


@dataclass
class AttackEvent:
    """One activation interval [start, end) of one channel.

    channel: 'sensor' (target = follower i), 'comm_fdi' / 'comm_dos' (target = [i, j], link j -> i),
             'actuator' (target = follower i).
    bias:    constant 3-D (or 6-D for comm_fdi) offset; amp/freq add a sinusoidal part.
    """
    channel: str
    target: int | list[int]
    start: float
    end: float
    bias: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    amp: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    freq: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def __post_init__(self):
        if self.channel not in CHANNEL_TYPES:
            raise ValueError(f"unknown channel {self.channel}")
        if not self.end > self.start:
            raise ValueError("attack interval must have end > start")

    def active(self, t: float) -> bool:
        return self.start <= t < self.end

    def signal(self, t: float) -> np.ndarray:
        """Attack signal a_c(t) = sigma_c(t) * a_bar_c(t), Eq. (9)."""
        if not self.active(t):
            return np.zeros(len(self.bias))
        b, a, w = map(np.asarray, (self.bias, self.amp, self.freq))
        return b + a * np.sin(w * t)


@dataclass
class AttackSchedule:
    events: list[AttackEvent]

    @classmethod
    def from_yaml(cls, path: str) -> "AttackSchedule":
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return cls([AttackEvent(**e) for e in cfg.get("attacks", [])])

    def active_events(self, t: float) -> list[AttackEvent]:
        return [e for e in self.events if e.active(t)]

    def channel_types(self) -> set[str]:
        return {"comm" if e.channel.startswith("comm") else e.channel for e in self.events}

    def is_time_staggered_multichannel(self) -> bool:
        """Definition 1: at least two channel TYPES are attacked."""
        return len(self.channel_types()) >= 2

    def n_switches(self, t0: float, t: float) -> int:
        """n_a(t0, t): number of activations and deactivations on [t0, t) (Assumption 1(ii))."""
        times = [e.start for e in self.events] + [e.end for e in self.events]
        return int(sum(t0 <= s < t for s in times))

    def attacked_time(self, t0: float, t: float, dt: float = 1e-3) -> float:
        """|T_a ∩ [t0, t)|: total time with at least one active channel (Assumption 1(iii))."""
        grid = np.arange(t0, t, dt)
        return float(dt * sum(bool(self.active_events(s)) for s in grid))
