#!/usr/bin/env python3
"""Tables, figures and headline numbers of Section 6 from results/mc_results.jsonl.

Outputs (results/):
  summary.csv               mean [95% bootstrap CI] of every metric, per method and scenario
  headline.json             numbers quoted in the abstract and conclusions
  tex/*_en.tex, tex/*_ru.tex LaTeX tables included by the manuscript
  fig_timeseries.(pdf|png)  representative hybrid run
  fig_dmin.(pdf|png)        distribution of the minimum inter-vehicle distance
"""
import json, os, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ruav import sim  # noqa: E402

RES = os.path.join(ROOT, "results")
TEX = os.path.join(RES, "tex")
METHODS = ["proposed", "abl_no_cbf", "abl_no_detect", "abl_nonrobust_cbf",
           "B1_consensus", "B2_wmsr", "B3_consensus_cbf"]
NAME_EN = {"proposed": "Proposed", "abl_no_cbf": "Proposed w/o CBF",
           "abl_no_detect": "CBF only (no detection)", "abl_nonrobust_cbf": "Proposed, non-robust CBF",
           "B1_consensus": r"Consensus~\cite{ren2008distributed}", "B2_wmsr": r"W-MSR~\cite{leblanc2013resilient,dibaji2017resilient}",
           "B3_consensus_cbf": r"Consensus + CBF~\cite{wang2017safety}"}
NAME_RU = {"proposed": "Предложенный", "abl_no_cbf": "Предложенный без CBF",
           "abl_no_detect": "Только CBF (без обнаружения)", "abl_nonrobust_cbf": "Предложенный, нероб. CBF",
           "B1_consensus": NAME_EN["B1_consensus"].replace("Consensus", "Консенсус"),
           "B2_wmsr": NAME_EN["B2_wmsr"], "B3_consensus_cbf": NAME_EN["B3_consensus_cbf"].replace("Consensus", "Консенсус")}
PLAIN = {"proposed": "Proposed", "abl_no_detect": "CBF only (no detection)",
         "B1_consensus": "Consensus", "B2_wmsr": "W-MSR", "B3_consensus_cbf": "Consensus + CBF",
         "abl_no_cbf": "Proposed w/o CBF", "abl_nonrobust_cbf": "Non-robust CBF"}
SCEN = ["clean", "sensor", "comm", "actuator", "hybrid"]
SCEN_EN = {"clean": "No attack", "sensor": "Sensor", "comm": "Communication", "actuator": "Actuation", "hybrid": "Time-staggered"}
SCEN_RU = {"clean": "Без атаки", "sensor": "Сенсорная", "comm": "Коммуникац.", "actuator": "Исполнит.", "hybrid": "Разнесённая"}
INK, MUTED, GRID = "#2b2b29", "#6b6a63", "#e4e2dc"
SERIES = {"proposed": "#2a78d6", "B1_consensus": "#eb6834", "B3_consensus_cbf": "#1baf7a", "abl_no_detect": "#eda100"}


def load():
    rows = [json.loads(l) for l in open(os.path.join(RES, "mc_results.jsonl"))]
    by = defaultdict(list)
    for r in rows:
        by[(r["method"], r["scenario"])].append(r)
    for k in by:
        by[k].sort(key=lambda r: r["seed"])
    return by


def boot_ci(x, B=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (B, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for r, idx in enumerate(order):
        run = max(run, (m - r) * pvals[idx]); adj[idx] = min(1.0, run)
    return adj


def fmt_ci(t, nd=2):
    return f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"


def main():
    by = load()
    os.makedirs(TEX, exist_ok=True)
    summary, head = [], {}
    n_runs = len(by[("proposed", "hybrid")])
    head["n_runs_per_cell"] = n_runs
    head["n_total_runs"] = int(sum(len(v) for v in by.values()))
    for s in SCEN:
        for m in METHODS:
            rs = by[(m, s)]
            pe = boot_ci([r["peak_Emax"] for r in rs]); rm = boot_ci([r["rms_Ef_attack"] for r in rs])
            dm = [r["d_min"] for r in rs]
            row = dict(method=m, scenario=s, n=len(rs), peak_Emax=pe, rms_Ef=rm, d_min_mean=float(np.mean(dm)),
                       d_min_worst=float(np.min(dm)), collisions=int(sum(r["collision"] for r in rs)),
                       false_alarms=int(sum(r["false_alarms"] for r in rs)),
                       qp_infeasible=int(sum(r["qp_infeasible_steps"] > 0 for r in rs)),
                       cbf_active=float(np.mean([r["cbf_active_frac"] for r in rs])))
            if s != "clean":
                tr = [r["T_rec"] for r in rs]
                row.update(T_rec_median=float(np.median(tr)), T_rec=boot_ci(tr),
                           recovered=int(sum(r["recovered"] for r in rs)))
            summary.append(row)
    with open(os.path.join(RES, "summary.csv"), "w") as fh:
        keys = ["method", "scenario", "n", "peak_Emax", "rms_Ef", "d_min_mean", "d_min_worst", "collisions",
                "T_rec_median", "recovered", "false_alarms", "qp_infeasible", "cbf_active"]
        fh.write(",".join(keys) + "\n")
        for r in summary:
            fh.write(",".join(str(r.get(k, "")) .replace(",", ";") for k in keys) + "\n")
    S = {(r["method"], r["scenario"]): r for r in summary}

    # ---- paired tests vs proposed (peak worst-follower error), Holm over the 6 comparisons per scenario
    tests = {}
    for s in SCEN[1:]:
        ps, comp = [], [m for m in METHODS if m != "proposed"]
        for m in comp:
            a = [r["peak_Emax"] for r in by[("proposed", s)]]; b = [r["peak_Emax"] for r in by[(m, s)]]
            try:
                ps.append(wilcoxon(a, b).pvalue)
            except ValueError:
                ps.append(1.0)
        for m, p in zip(comp, holm(np.array(ps))):
            tests[(m, s)] = float(p)

    # ---- headline numbers (hybrid scenario, strongest baseline among B1-B3 by mean peak error)
    base = min(["B1_consensus", "B2_wmsr", "B3_consensus_cbf"], key=lambda m: S[(m, "hybrid")]["peak_Emax"][0])
    P, Bs = S[("proposed", "hybrid")], S[(base, "hybrid")]
    head.update(strongest_baseline=base,
                peak_reduction_pct=100 * (1 - P["peak_Emax"][0] / Bs["peak_Emax"][0]),
                trec_reduction_pct=100 * (1 - P["T_rec"][0] / Bs["T_rec"][0]) if Bs["T_rec"][0] > 0 else None,
                proposed_peak=P["peak_Emax"][0], baseline_peak=Bs["peak_Emax"][0],
                proposed_trec=P["T_rec"][0], baseline_trec=Bs["T_rec"][0],
                proposed_collisions_all=int(sum(S[("proposed", s)]["collisions"] for s in SCEN)),
                proposed_dmin_worst_all=float(min(S[("proposed", s)]["d_min_worst"] for s in SCEN)),
                proposed_qp_infeasible_runs=int(sum(S[("proposed", s)]["qp_infeasible"] for s in SCEN)),
                false_alarms_clean_proposed=S[("proposed", "clean")]["false_alarms"],
                d_safe=1.5)
    # detection statistics (proposed; detector is identical in all methods that use it)
    det = defaultdict(lambda: [0, 0, []])
    for s in SCEN[1:]:
        for r in by[("proposed", s)]:
            for ch, hit, dl in r["events"]:
                det[ch][0] += 1; det[ch][1] += int(hit)
                if hit and dl is not None:
                    det[ch][2].append(dl)
    head["detection"] = {ch: dict(n=v[0], rate=v[1] / v[0], median_delay=float(np.median(v[2])) if v[2] else None,
                                  p95_delay=float(np.percentile(v[2], 95)) if v[2] else None) for ch, v in det.items()}
    T_clean = len(by[("proposed", "clean")]) * 100.0 / 3600.0 * 6           # agent-hours
    head["false_alarm_rate_per_agent_hour"] = S[("proposed", "clean")]["false_alarms"] / T_clean
    with open(os.path.join(RES, "headline.json"), "w") as fh:
        json.dump(head, fh, indent=1)

    # ---- table A: hybrid scenario, all methods
    def table_hybrid(lang):
        nm = NAME_EN if lang == "en" else NAME_RU
        cap = (f"Time-staggered scenario: mean [95\\% bootstrap CI] over {n_runs} paired runs. $E_{{\\max}}$: peak worst-follower formation error; "
               f"$d_{{\\min}}$: minimum inter-vehicle distance (mean / worst run), $d_s={head['d_safe']}$~m; $T_{{\\mathrm{{rec}}}}$: recovery time; "
               "$p$: Holm-corrected Wilcoxon test against the proposed method on $E_{\\max}$.") if lang == "en" else (
               f"Сценарий с разнесёнными атаками: среднее [95\\% бутстреп-ДИ] по {n_runs} парным прогонам. $E_{{\\max}}$ --- пиковая ошибка худшего ведомого; "
               f"$d_{{\\min}}$ --- минимальное межаппаратное расстояние (среднее / худший прогон), $d_s={head['d_safe']}$~м; $T_{{\\mathrm{{rec}}}}$ --- время восстановления; "
               "$p$ --- критерий Уилкоксона с поправкой Холма относительно предложенного метода по $E_{\\max}$.")
        col = ("Method", "$E_{\\max}$, m", "$d_{\\min}$, m", "Viol.", "$T_{\\mathrm{rec}}$, s", "$p$") if lang == "en" else \
              ("Метод", "$E_{\\max}$, м", "$d_{\\min}$, м", "Наруш.", "$T_{\\mathrm{rec}}$, с", "$p$")
        L = [r"\begin{table}[t]", r"\centering", r"\caption{" + cap + r"}\label{tab:hybrid}", r"\footnotesize",
             r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%", r"\begin{tabular}{lccccc}", r"\toprule", " & ".join(col) + r" \\", r"\midrule"]
        for m in METHODS:
            r = S[(m, "hybrid")]
            p = "---" if m == "proposed" else (f"{tests[(m, 'hybrid')]:.1e}" if tests[(m, 'hybrid')] < 1e-3 else f"{tests[(m, 'hybrid')]:.3f}")
            L.append(f"{nm[m]} & {fmt_ci(r['peak_Emax'])} & {r['d_min_mean']:.2f} / {r['d_min_worst']:.2f} & "
                     f"{r['collisions']}/{r['n']} & {fmt_ci(r['T_rec'], 1)} & {p}" + r" \\")
            if m == "abl_nonrobust_cbf":
                L.append(r"\midrule")
        L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
        return "\n".join(L) + "\n"

    # ---- table B: peak error per scenario
    def table_scen(lang):
        nm = NAME_EN if lang == "en" else NAME_RU; sc = SCEN_EN if lang == "en" else SCEN_RU
        cap = ("Peak worst-follower formation error $E_{\\max}$ (m), mean [95\\% CI], and number of runs with $d_{\\min}<d_s$ (in parentheses) per scenario."
               if lang == "en" else "Пиковая ошибка худшего ведомого $E_{\\max}$ (м), среднее [95\\% ДИ], и число прогонов с $d_{\\min}<d_s$ (в скобках) по сценариям.")
        L = [r"\begin{table}[t]", r"\centering", r"\caption{" + cap + r"}\label{tab:scenarios}", r"\scriptsize",
             r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%", r"\begin{tabular}{l" + "c" * len(SCEN) + "}", r"\toprule",
             ("Method" if lang == "en" else "Метод") + " & " + " & ".join(sc[s] for s in SCEN) + r" \\", r"\midrule"]
        for m in METHODS:
            cells = [f"{S[(m, s)]['peak_Emax'][0]:.2f} [{S[(m, s)]['peak_Emax'][1]:.2f}, {S[(m, s)]['peak_Emax'][2]:.2f}] ({S[(m, s)]['collisions']})" for s in SCEN]
            L.append(nm[m] + " & " + " & ".join(cells) + r" \\")
            if m == "abl_nonrobust_cbf":
                L.append(r"\midrule")
        L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
        return "\n".join(L) + "\n"

    # ---- table C: detection
    def table_det(lang):
        chn = {"sensor": ("Own sensor", "Собственный датчик"), "comm_fdi": ("Link FDI", "Ложные данные в линии"),
               "comm_dos": ("Link DoS", "DoS линии"), "actuator": ("Actuation", "Исполнительный канал")}
        k = 0 if lang == "en" else 1
        cap = (f"Detection performance of the proposed detector over all attacked runs; false-alarm rate measured in the attack-free runs: "
               f"{head['false_alarm_rate_per_agent_hour']:.2f} per agent-hour." if lang == "en" else
               f"Качество обнаружения предложенного детектора по всем прогонам с атаками; частота ложных тревог по прогонам без атак: "
               f"{head['false_alarm_rate_per_agent_hour']:.2f} на аппарат-час.")
        col = ("Channel", "Events", "Detected, \\%", "Median delay, s", "95th perc. delay, s") if lang == "en" else \
              ("Канал", "Событий", "Обнаружено, \\%", "Медиана задержки, с", "95-й процентиль, с")
        L = [r"\begin{table}[t]", r"\centering", r"\caption{" + cap + r"}\label{tab:detection}", r"\footnotesize",
             r"\begin{tabular}{lcccc}", r"\toprule", " & ".join(col) + r" \\", r"\midrule"]
        for ch in ("sensor", "comm_fdi", "comm_dos", "actuator"):
            d = head["detection"].get(ch)
            if not d:
                continue
            md = "---" if d["median_delay"] is None else f"{d['median_delay']:.2f}"
            p95 = "---" if d["p95_delay"] is None else f"{d['p95_delay']:.2f}"
            L.append(f"{chn[ch][k]} & {d['n']} & {100*d['rate']:.1f} & {md} & {p95}" + r" \\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(L) + "\n"

    for lang in ("en", "ru"):
        for name, fn in (("table_hybrid", table_hybrid), ("table_scenarios", table_scen), ("table_detection", table_det)):
            open(os.path.join(TEX, f"{name}_{lang}.tex"), "w").write("% AUTO-GENERATED by scripts/make_figures.py\n" + fn(lang))

    # ---- figure: d_min distribution (strip plot), attacked scenarios pooled
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    rng = np.random.default_rng(0)
    for k, m in enumerate(METHODS):
        vals = [r["d_min"] for s in SCEN[1:] for r in by[(m, s)]]
        ax.scatter(k + rng.uniform(-0.18, 0.18, len(vals)), vals, s=9, color="#2a78d6" if m.startswith(("proposed", "abl")) else "#eb6834",
                   alpha=0.55, linewidths=0)
        ax.plot([k - 0.28, k + 0.28], [np.median(vals)] * 2, color=INK, lw=1.5)
    ax.axhline(1.5, color="#c62828", lw=1, ls="--"); ax.text(len(METHODS) - 0.5, 1.53, "$d_s$", color="#c62828", ha="right", va="bottom")
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels([PLAIN[m] for m in METHODS], rotation=20, ha="right")
    ax.set_ylabel("min. inter-vehicle distance, m"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig_dmin.{ext}"), dpi=200)
    plt.close(fig)

    # ---- figure: representative hybrid run (median-peak seed for the strongest baseline)
    cert = json.load(open(os.path.join(ROOT, "certificates", "benchmark.json")))
    cy = yaml.safe_load(open(os.path.join(ROOT, "configs", "hybrid_attack.yaml")))
    cfg = sim.default_config(cy, cert["cbf"]["mu_pair"], cert["cbf"]["mu_leader"])
    rs = by[(base, "hybrid")]
    seed = sorted(rs, key=lambda r: r["peak_Emax"])[len(rs) // 2]["seed"]
    head["timeseries_seed"] = seed
    show = ["proposed", "abl_no_detect", "B1_consensus", "B3_consensus_cbf"]
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 3.8), sharex=True)
    ev = None
    for m in show:
        r = sim.run(cfg, m, "hybrid", seed, record=True)
        tr = r["trace"]; ev = tr["events"]
        axs[0].plot(tr["t"], tr["E_max"], lw=1.4, color=SERIES[m], label=PLAIN[m])
        axs[1].plot(tr["t"], tr["dmin"], lw=1.4, color=SERIES[m], label=PLAIN[m])
    lab = {"sensor": "sensor", "comm_fdi": "link FDI", "comm_dos": "link DoS", "actuator": "actuation"}
    for k, (ch, tg, t0, t1) in enumerate(ev):
        for a in axs:
            a.axvspan(t0, t1, color="#9a9a94", alpha=0.10 + 0.05 * k, lw=0)
        axs[0].text(t0 + 0.3, axs[0].get_ylim()[1] * (0.95 - 0.12 * k), lab[ch], fontsize=7, color=MUTED, va="top")
    axs[1].axhline(1.5, color="#c62828", lw=1, ls="--")
    axs[0].set_ylabel("$E_{\\max}$, m"); axs[1].set_ylabel("$d_{\\min}$, m"); axs[1].set_xlabel("time, s")
    for a in axs:
        a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
    axs[0].legend(ncol=2, frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig_timeseries.{ext}"), dpi=200)
    plt.close(fig)
    with open(os.path.join(RES, "headline.json"), "w") as fh:
        json.dump(head, fh, indent=1)
    print(json.dumps(head, indent=1))


if __name__ == "__main__":
    main()
