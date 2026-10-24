"""Closed-loop simulation of the leader--follower formation under time-staggered attacks.

One call of `run()` simulates one method in one scenario with one random seed and returns
the metrics used in the paper.  All randomness comes from `numpy.random.default_rng(seed)`.

Methods (Section 5 of the paper):
  proposed            detector + attack-aware estimator + trust weights + compensation + robust CBF
  abl_no_cbf          proposed without the safety filter
  abl_no_detect       robust CBF only (no detector, estimator, trust weights or compensation)
  abl_nonrobust_cbf   proposed with a non-robust CBF (no attack/disturbance/sensing margins)
  B1_consensus        leader--follower consensus formation control [Ren & Beard 2008]
  B2_wmsr             W-MSR-type trimmed formation control [LeBlanc et al. 2013; Dibaji & Ishii 2017]
  B3_consensus_cbf    consensus + reciprocal multi-robot CBF safety filter [Wang et al. 2017]
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from . import dynamics, cbf as cbfm, detector as detm, estimator as estm, controller as ctlm
from .attacks import AttackEvent

LEADER = -1

METHODS = {
    "proposed":          dict(detect=True,  trust=True,  comp=True,  cbf="robust",  wmsr=False),
    "abl_no_cbf":        dict(detect=True,  trust=True,  comp=True,  cbf=None,      wmsr=False),
    "abl_no_detect":     dict(detect=False, trust=False, comp=False, cbf="robust",  wmsr=False),
    "abl_nonrobust_cbf": dict(detect=True,  trust=True,  comp=True,  cbf="nominal", wmsr=False),
    "B1_consensus":      dict(detect=False, trust=False, comp=False, cbf=None,      wmsr=False),
    "B2_wmsr":           dict(detect=False, trust=False, comp=False, cbf=None,      wmsr=True),
    "B3_consensus_cbf":  dict(detect=False, trust=False, comp=False, cbf="nominal", wmsr=False),
}


@dataclass
class SimConfig:
    dt: float = 0.02
    horizon: float = 100.0
    offsets: np.ndarray = None
    A: np.ndarray = None
    g: np.ndarray = None
    d_safe: float = 1.0
    # noise models
    sigma_p: float = 0.03       # navigation-filter position noise (Gauss-Markov), m
    sigma_v: float = 0.02       # velocity noise, m/s
    tau_noise: float = 1.0      # correlation time of navigation noise, s
    sigma_rel: float = 0.02     # relative sensing noise std, clipped at rel_clip
    rel_clip: float = 0.05
    d_bar: float = 0.15         # disturbance bound, m/s^2
    p_loss: float = 0.01        # benign packet-loss probability
    init_spread: float = 0.5    # initial position spread around the formation, m
    # controller and protection
    ctl: ctlm.ControlParams = field(default_factory=ctlm.ControlParams)
    det: detm.DetectorParams = field(default_factory=detm.DetectorParams)
    cbf: cbfm.CBFParams = field(default_factory=cbfm.CBFParams)
    # metrics
    eps_rec: float = 0.3        # formation error regarded as recovered, m
    hold_rec: float = 5.0       # time max_i ||e_i|| must stay below eps_rec, s


def default_config(cfg_yaml: dict, mu_pair: float, mu_leader: float) -> SimConfig:
    c = SimConfig()
    c.offsets = np.array(cfg_yaml["formation"]["offsets"], float)
    c.A = np.array(cfg_yaml["graph"]["adjacency"], int)
    c.g = np.array(cfg_yaml["graph"]["pinning"], int)
    c.d_safe = float(cfg_yaml["formation"]["d_safe"])
    c.cbf = cbfm.CBFParams(d_safe=c.d_safe, a_u_bar=1.0, d_bar=c.d_bar, a0_bar=0.07,
                           meas_margin=mu_pair, meas_margin_leader=mu_leader, u_box=4.0,
                           sensing_radius=8.0)
    return c


def leader_state(t):
    p0, v0, a0 = dynamics.leader_circle(t)
    return np.concatenate([p0, v0]), a0


# ----------------------------------------------------------------------------- attacks
def random_attacks(scenario: str, rng, cfg: SimConfig) -> list[AttackEvent]:
    """Random attack schedule within the budget of Assumption 1.

    The adversary knows the formation geometry and points every attack so that the target is
    pushed towards one of its nearest neighbours (worst case for collision avoidance):
      sensor   -- spoofed own position b = -A d  (the controller then moves the vehicle by +A d)
      comm FDI -- falsified neighbour position s = +A d on one incoming link
      actuator -- additive acceleration +A d,  A <= a_u_bar
    where d is the unit vector from the target to a nearest neighbour in the formation.
    """
    n = len(cfg.g)
    R = cfg.offsets

    def towards_neighbour(i):
        dists = [(np.linalg.norm(R[k] - R[i]), k) for k in range(n) if k != i]
        dmin = min(dd for dd, _ in dists)
        ks = [k for dd, k in dists if dd <= dmin + 1e-9]
        k = ks[int(rng.integers(len(ks)))]
        d = R[k] - R[i]
        return d / np.linalg.norm(d)

    def sensor(t0):
        i = int(rng.integers(n))
        return AttackEvent("sensor", i, t0, t0 + rng.uniform(10, 20),
                           bias=list(-rng.uniform(2.0, 3.5) * towards_neighbour(i)))

    def comm(t0):
        i = int(rng.integers(n))
        srcs = list(np.flatnonzero(cfg.A[i])) + ([LEADER] if cfg.g[i] else [])
        j = int(srcs[int(rng.integers(len(srcs)))])
        if rng.random() < 0.7:
            b = rng.uniform(2.0, 3.5) * towards_neighbour(i)
            return AttackEvent("comm_fdi", [i, j], t0, t0 + rng.uniform(10, 20), bias=list(b) + [0, 0, 0])
        return AttackEvent("comm_dos", [i, j], t0, t0 + rng.uniform(5, 15))

    def actuator(t0):
        i = int(rng.integers(n))
        return AttackEvent("actuator", i, t0, t0 + rng.uniform(8, 15),
                           bias=list(rng.uniform(0.6, 0.95) * towards_neighbour(i)))

    if scenario == "clean":
        return []
    if scenario == "sensor":
        return [sensor(rng.uniform(15, 30))]
    if scenario == "comm":
        return [comm(rng.uniform(15, 30))]
    if scenario == "actuator":
        return [actuator(rng.uniform(15, 30))]
    if scenario == "hybrid":                      # time-staggered: random order and offsets
        t = rng.uniform(15, 20)
        evs = []
        for make in rng.permutation([sensor, comm, actuator]):
            e = make(t); evs.append(e)
            t = e.start + rng.uniform(4, 10)      # next channel starts while the previous may be active
        return evs
    raise ValueError(scenario)


# ----------------------------------------------------------------------------- W-MSR term
def wmsr_terms(own_c, own_v, vals_c, vals_v, leader=None, F=1):
    """W-MSR-type trimming per axis over the FOLLOWER neighbours: drop up to F largest values
    above the own value and up to F smallest below it; the leader term (if received) is kept,
    i.e. the leader is treated as trusted, as in leader-follower resilient consensus."""
    ep, ev = np.zeros(3), np.zeros(3)
    if vals_c:
        C = np.array(vals_c); V = np.array(vals_v)
        for ax in range(3):
            keep = np.ones(len(C), bool)
            above = np.where(C[:, ax] > own_c[ax])[0]
            below = np.where(C[:, ax] < own_c[ax])[0]
            if len(above):
                keep[above[np.argsort(-C[above, ax])[:F]]] = False
            if len(below):
                keep[below[np.argsort(C[below, ax])[:F]]] = False
            ep[ax] = np.sum(own_c[ax] - C[keep, ax])
            ev[ax] = np.sum(own_v[ax] - V[keep, ax])
    if leader is not None:
        ep += own_c - leader[0]; ev += own_v - leader[1]
    return ep, ev


# ----------------------------------------------------------------------------- simulation
def run(cfg: SimConfig, method: str, scenario: str, seed: int, record: bool = False,
        attacks: list | None = None) -> dict:
    flags = METHODS[method]
    rng = np.random.default_rng(seed)
    atk_rng = np.random.default_rng(seed + 10_000_019)          # same attacks for every method
    events = attacks if attacks is not None else random_attacks(scenario, atk_rng, cfg)
    n, dt = len(cfg.g), cfg.dt
    steps = int(round(cfg.horizon / dt))
    R = cfg.offsets
    in_nb = [list(np.flatnonzero(cfg.A[i])) for i in range(n)]

    x0, _ = leader_state(0.0)
    p = x0[:3] + R + rng.uniform(-cfg.init_spread, cfg.init_spread, (n, 3)) * np.array([1, 1, 0.3])
    v = np.tile(x0[3:], (n, 1)) + rng.normal(0, 0.1, (n, 3))
    eta_p = rng.normal(0, cfg.sigma_p, (n, 3)); eta_v = rng.normal(0, cfg.sigma_v, (n, 3))
    dist = np.zeros((n, 3))
    phi = np.exp(-dt / cfg.tau_noise)

    cbf_prm = cfg.cbf
    if flags["cbf"] == "nominal":
        cbf_prm = cbfm.CBFParams(d_safe=cfg.d_safe, k0=cfg.cbf.k0, k1=cfg.cbf.k1, a_u_bar=0.0,
                                 d_bar=0.0, a0_bar=0.0, u_box=cfg.cbf.u_box,
                                 sensing_radius=cfg.cbf.sensing_radius)
    nbrs_with_leader = [in_nb[i] + ([LEADER] if cfg.g[i] else []) for i in range(n)]
    dets = [detm.Detector(cfg.det, nbrs_with_leader[i], dt) for i in range(n)]
    trust = [ctlm.TrustWeights(nbrs_with_leader[i], cfg.ctl.tau_up, cfg.ctl.m_levels) for i in range(n)]
    x_hat = np.hstack([p, v])
    x_bcast = x_hat.copy()
    u_prev = np.zeros((n, 3))

    E_f = np.zeros(steps); E_max = np.zeros(steps); dmin_t = np.full(steps, np.inf)
    alarm_log = []               # (k, i, kind, j)
    prev_alarm = [dict(sensor=False, actuator=False, links={j: False for j in nbrs_with_leader[i]}) for i in range(n)]
    n_infeasible, n_active = 0, 0
    w_max = 0.0                  # largest relative speed within R_s (check of Assumption 2)
    trace = {"t": [], "E_f": [], "dmin": [], "alarms": []} if record else None

    for k in range(steps):
        t = k * dt
        xl, a0 = leader_state(t)
        active = [e for e in events if e.active(t)]
        a_y = np.zeros((n, 3)); a_u = np.zeros((n, 3)); fdi = {}; dos = set()
        for e in active:
            s = e.signal(t)
            if e.channel == "sensor":
                a_y[e.target] += s[:3]
            elif e.channel == "actuator":
                a_u[e.target] += s[:3]
            elif e.channel == "comm_fdi":
                fdi[tuple(e.target)] = s
            elif e.channel == "comm_dos":
                dos.add(tuple(e.target))
        # measurements
        eta_p = phi * eta_p + np.sqrt(1 - phi**2) * rng.normal(0, cfg.sigma_p, (n, 3))
        eta_v = phi * eta_v + np.sqrt(1 - phi**2) * rng.normal(0, cfg.sigma_v, (n, 3))
        y = np.hstack([p + eta_p + a_y, v + eta_v])
        all_p = np.vstack([p, xl[:3]]); all_v = np.vstack([v, xl[3:]])   # index n = leader

        u = np.zeros((n, 3))
        # relative measurements (vectorised): D[i, jj] = x_jj - x_i, jj = n is the leader
        Dp = all_p[None, :, :] - p[:, None, :]
        Dv = all_v[None, :, :] - v[:, None, :]
        dist_ij = np.sqrt(np.einsum("ijk,ijk->ij", Dp, Dp))
        inr = dist_ij <= cfg.cbf.sensing_radius
        if inr.any():
            w_max = max(w_max, float(np.sqrt(np.einsum("ijk,ijk->ij", Dv, Dv))[inr].max()))
        nz = np.clip(rng.normal(0, cfg.sigma_rel, (n, n + 1, 6)), -cfg.rel_clip, cfg.rel_clip)
        for i in range(n):
            rel = {}
            for jj in range(n + 1):
                if jj == i or dist_ij[i, jj] > cfg.cbf.sensing_radius:
                    continue
                rel[LEADER if jj == n else jj] = (Dp[i, jj] + nz[i, jj, :3], Dv[i, jj] + nz[i, jj, 3:])
            # received messages (one-step-delayed broadcasts)
            msgs = {}
            for j in nbrs_with_leader[i]:
                if (i, j) in dos or rng.random() < cfg.p_loss:
                    msgs[j] = None
                    continue
                if j == LEADER:
                    z = xl.copy()
                else:   # time-stamped message, propagated over the one-sample latency at the receiver
                    z = np.concatenate([x_bcast[j][:3] + dt * x_bcast[j][3:], x_bcast[j][3:]])
                if (i, j) in fdi:
                    z = z + fdi[(i, j)]
                msgs[j] = z
            # protection modules
            if flags["detect"]:
                out = dets[i].update(y[i], u_prev[i], msgs, rel)
                alpha = trust[i].update(out.link_alarm, dt) if flags["trust"] else {j: 1.0 for j in msgs}
                trusted = [j for j in msgs if alpha.get(j, 0) > 0 and not out.link_alarm.get(j, False)]
                x_hat[i] = estm.estimate(y[i], out.sensor_alarm, msgs, rel, trusted, x_hat[i],
                                         u_prev[i], out.a_u_hat if out.actuator_alarm else np.zeros(3), dt)
                act_alarm, a_hat = out.actuator_alarm and flags["comp"], out.a_u_hat
                cur = dict(sensor=out.sensor_alarm, actuator=out.actuator_alarm, links=out.link_alarm)
                for kind in ("sensor", "actuator"):
                    if cur[kind] and not prev_alarm[i][kind]:
                        alarm_log.append((k, i, kind, None))
                for j, a in cur["links"].items():
                    if a and not prev_alarm[i]["links"][j]:
                        alarm_log.append((k, i, "link", j))
                prev_alarm[i] = {"sensor": cur["sensor"], "actuator": cur["actuator"], "links": dict(cur["links"])}
            else:
                alpha = {j: 1.0 for j in msgs}
                x_hat[i] = y[i]
                act_alarm, a_hat = False, np.zeros(3)
            # nominal law
            if flags["wmsr"]:
                own_c = x_hat[i][:3] - R[i]
                vc, vv, lead = [], [], None
                for j, z in msgs.items():
                    if z is None:
                        continue
                    if j == LEADER:
                        lead = (z[:3], z[3:])
                    else:
                        vc.append(z[:3] - R[j]); vv.append(z[3:])
                ep, ev = wmsr_terms(own_c, x_hat[i][3:], vc, vv, lead)
                u_nom = -cfg.ctl.k_p * ep - cfg.ctl.k_v * ev
            else:
                u_nom = _formation(i, x_hat[i], R, msgs, alpha, bool(cfg.g[i]), a_hat, act_alarm, cfg.ctl)
            u_nom = np.clip(u_nom, -cbf_prm.u_box, cbf_prm.u_box)
            if flags["cbf"] is not None:
                u_i, info = cbfm.safety_filter(u_nom, rel, cbf_prm, leader_key=LEADER)
                n_infeasible += info["violation"] > 1e-6
                n_active += info["active"] > 0
            else:
                u_i = u_nom
            u[i] = u_i
        # disturbance: bounded Gauss-Markov
        dist = 0.99 * dist + rng.normal(0, 0.01, (n, 3))
        nd = np.linalg.norm(dist, axis=1, keepdims=True)
        dist = dist * np.minimum(1.0, cfg.d_bar / np.maximum(nd, 1e-12))
        p, v = dynamics.step(p, v, u, a_u, dist, dt)
        x_bcast = x_hat.copy()
        u_prev = u.copy()
        # metrics
        xl_next, _ = leader_state(t + dt)
        en = np.linalg.norm(p - xl_next[:3] - R, axis=1)
        E_f[k] = en.mean(); E_max[k] = en.max()
        allp = np.vstack([p, xl_next[:3]])
        diff = allp[:, None, :] - allp[None, :, :]
        dd = np.linalg.norm(diff, axis=2) + np.eye(n + 1) * 1e9
        dmin_t[k] = dd.min()
        if record:
            trace["t"].append(t + dt); trace["E_f"].append(E_f[k]); trace.setdefault("E_max", []).append(E_max[k]); trace["dmin"].append(dmin_t[k])

    res = _metrics(cfg, events, E_f, E_max, dmin_t, alarm_log, n_infeasible, n_active, steps, n, method, scenario, seed, trace, in_nb)
    res["w_max"] = w_max
    return res


def _formation(i, x_hat, R, msgs, alpha, pinned, a_hat, act_alarm, prm):
    p, v = x_hat[:3], x_hat[3:]
    ep, ev = np.zeros(3), np.zeros(3)
    for j, z in msgs.items():
        if z is None:
            continue
        a = alpha.get(j, 0.0)
        if a == 0.0:
            continue
        if j == LEADER:
            ep += a * (p - z[:3] - R[i]); ev += a * (v - z[3:])
        else:
            ep += a * ((p - R[i]) - (z[:3] - R[j])); ev += a * (v - z[3:])
    u = -prm.k_p * ep - prm.k_v * ev
    if act_alarm:
        u = u - a_hat
    return u


def _metrics(cfg, events, E_f, E_max, dmin_t, alarm_log, n_inf, n_act, steps, n, method, scenario, seed, trace, in_nb):
    dt = cfg.dt
    t = (np.arange(steps) + 1) * dt
    res = {"method": method, "scenario": scenario, "seed": seed,
           "d_min": float(dmin_t.min()), "collision": bool(dmin_t.min() < cfg.d_safe),
           "qp_infeasible_steps": int(n_inf), "cbf_active_frac": n_act / (steps * n)}
    if events:
        t_on = min(e.start for e in events); t_off = max(e.end for e in events)
        win = (t >= t_on) & (t <= t_off + 20.0)
        res["peak_Ef"] = float(E_f[win].max())
        res["peak_Emax"] = float(E_max[win].max())
        att = np.zeros(steps, bool)
        for e in events:
            att |= (t >= e.start) & (t < e.end)
        res["rms_Ef_attack"] = float(np.sqrt(np.mean(E_f[att] ** 2)))
        # recovery: first time after t_off from which E_f <= eps for hold_rec seconds
        hold = int(round(cfg.hold_rec / dt))
        idx = np.flatnonzero(t >= t_off)
        ok = E_max[idx] <= cfg.eps_rec
        t_rec = np.nan
        run_len = 0
        for m, flag in enumerate(ok):
            run_len = run_len + 1 if flag else 0
            if run_len >= hold:
                t_rec = t[idx[m - hold + 1]] - t_off
                break
        res["recovered"] = bool(not np.isnan(t_rec))
        res["T_rec"] = float(t_rec) if not np.isnan(t_rec) else float(cfg.horizon - t_off)
        # detection of each event
        dets = []
        for e in events:
            if e.channel == "sensor":
                hit = [k for (k, i, kind, j) in alarm_log if kind == "sensor" and i == e.target]
            elif e.channel == "actuator":
                hit = [k for (k, i, kind, j) in alarm_log if kind == "actuator" and i == e.target]
            else:
                ii, jj = e.target
                hit = [k for (k, i, kind, j) in alarm_log if kind == "link" and i == ii and j == jj]
            hit = [k * dt for k in hit if e.start <= k * dt <= e.end + 1.0]
            dets.append((e.channel, bool(hit), (min(hit) - e.start) if hit else np.nan))
        res["events"] = dets
    else:
        win = t >= 10.0
        res["peak_Ef"] = float(E_f[win].max())
        res["peak_Emax"] = float(E_max[win].max())
        res["rms_Ef_attack"] = float(np.sqrt(np.mean(E_f[win] ** 2)))
        res["events"] = []
    # unexplained alarms (false alarms)
    fa = 0
    for (k, i, kind, j) in alarm_log:
        tk = k * dt
        explained = False
        for e in events:
            if not (e.start - 0.1 <= tk <= e.end + 2.0):
                continue
            if e.channel in ("sensor", "actuator") and e.target == i:
                explained = True
            if e.channel == "sensor" and kind == "link" and j == e.target:
                explained = True
            if e.channel.startswith("comm") and kind == "link" and list(e.target) == [i, j]:
                explained = True
        fa += not explained
    res["false_alarms"] = int(fa)
    res["n_alarms"] = len(alarm_log)
    if trace is not None:
        res["trace"] = {kk: np.array(vv) for kk, vv in trace.items() if kk != "alarms"}
        res["trace"]["alarm_log"] = alarm_log
        res["trace"]["events"] = [(e.channel, e.target, e.start, e.end) for e in events]
    return res
