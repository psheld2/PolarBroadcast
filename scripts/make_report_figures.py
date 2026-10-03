"""Figures for reports/superposition_alphabets.tex.

1. figures/qam_realized_n128.pdf: polar SUP and TDM rate pairs realized at n = 128 on the 4-point AWGN
   BC at 7.33 / 2.15 dB (qam_realized_7.33dB_2.15dB_n128.txt), with normal-approximation curves.
2. figures/crossover_binary_vs_quaternary.pdf: signed SUP - TDM gap vs blocklength in QPSK symbols,
   for the binary model (hard-decision Gray QPSK = two BSBC(0.01, 0.10) uses per symbol) and the
   quaternary symmetric model, from interior_gaps.csv (interior_gaps.py).

3. figures/gain_8ary_vs_n.pdf and figures/rate_control_8ary.pdf: 8PSK / 8QAM results, from hard8pc_*.txt
   and ratectl_*_13_8.txt (plotting only).
4. figures/polar_8ary_n128.pdf: polar simulations (polar_8ary_results.jsonl) against na_curves_8ary.csv.

Run from scripts/:  python3 make_report_figures.py [qam_realized crossover gain_8ary rate_control polar_8ary]
"""
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qam_bc import qpsk_moments, sup_region_R1, tdm_region_R1

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "reports", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9})
db = lambda x: 10 ** (x / 10)


def fig_qam_realized():
    pts = {"tdm": [], "sup": {}}
    for line in open(os.path.join(HERE, "qam_realized_7.33dB_2.15dB_n128.txt")):
        if m := re.match(r"TDM\s+P1=(\S+)\s+R2=(\S+)\s+R1=(\S+)", line):
            pts["tdm"].append((m[1], float(m[2]), float(m[3])))
        elif m := re.match(r"SUP\s+theta=\s*(\d+) beta=(\S+)\s+R2=(\S+)\s+R1=(\S+)", line):
            pts["sup"].setdefault(int(m[1]), []).append((float(m[2]), float(m[3]), float(m[4])))
    S1, S2, n, eps = db(7.33), db(2.15), 128, 1e-2
    R2 = np.linspace(0.0, qpsk_moments(S2)[0], 150)
    betas = np.concatenate([np.linspace(0.01, 0.3, 30), np.linspace(0.32, 0.98, 34)])
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.plot(R2, tdm_region_R1(n, eps, eps, S1, S2, R2), "k-", lw=1, label="TDM, equal power (NA)")
    ax.plot(R2, tdm_region_R1(n, eps, eps, S1, S2, R2, power_control=True), "k--", lw=1,
            label="TDM, power control (NA)")
    for th, ls in ((0, ":"), (90, "-.")):
        r1, _ = sup_region_R1(n, eps, eps, S1, S2, R2, betas, [math.radians(th)])
        ax.plot(R2, np.where(np.isfinite(r1), r1, np.nan), ls, color="C0" if th == 0 else "C3", lw=1,
                label=rf"SUP, $\theta={th}^\circ$ (NA)")
    for (th, mk), col in zip(((0, "v"), (30, "<"), (60, ">"), (90, "o")), ("C0", "C1", "C2", "C3")):
        p = sorted(pts["sup"][th], key=lambda t: t[1])
        ax.plot([q[1] for q in p], [q[2] for q in p], mk, color=col, ms=4,
                label=rf"polar SUP, $\theta={th}^\circ$")
    t = pts["tdm"]
    eq = [q for q in t if q[0] in ("None", "1.0")]
    pc = [q for q in t if q[0] not in ("None", "1.0")]
    ax.plot([q[1] for q in eq], [q[2] for q in eq], "ks", ms=4, label="polar TDM, equal power")
    ax.plot([q[1] for q in pc], [q[2] for q in pc], "kD", ms=3.5, mfc="none", label="polar TDM, power control")
    ax.set_xlabel(r"$R_2$ (bits/symbol)"); ax.set_ylabel(r"$R_1$ (bits/symbol)")
    ax.set_xlim(0, 1.35); ax.set_ylim(0, 1.95)
    ax.grid(True, lw=0.3, alpha=0.6)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right", ncol=1)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "qam_realized_n128.pdf"))


def fig_crossover():
    rows = [l.strip().split(",") for l in open(os.path.join(HERE, "interior_gaps.csv"))][1:]
    data = {}
    for model, m, g in rows:
        data.setdefault(model, []).append((int(m), float(g)))
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    b = [(m - 1, 2 * g) for m, g in data["BSBC(0.01;0.10)"] if np.isfinite(g) and m - 1 >= 7]   # 2 BSBC uses per symbol
    ax.plot([x for x, _ in b], [y for _, y in b], "o-", ms=3.5, lw=1,
            label="binary: hard-decision Gray QPSK = 2 x BSBC(0.01, 0.10)")
    for key, lab, st in (("QSBC(0.0199;0.19)", "quaternary: QSBC(0.0199, 0.19), same SNRs", "s-"),
                         ("QSBC(0.01;0.10)", "quaternary: QSBC(0.01, 0.10)", "s--")):
        d = [(m, g) for m, g in data[key] if np.isfinite(g) and m >= 7]
        ax.plot([x for x, _ in d], [y for _, y in d], st, ms=3.5, lw=1, label=lab)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$\log_2$ blocklength (QPSK symbols)")
    ax.set_ylabel(r"$\max_{R_2}\,[R_1^{\rm SUP}-R_1^{\rm TDM}]$ (bits/symbol)")
    ax.grid(True, lw=0.3, alpha=0.6)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "crossover_binary_vs_quaternary.pdf"))




def fig_8ary_gain():
    """Gain of the best 8-ary design over TDM with power control (solid) and equal-power TDM (dotted)."""
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, kind in zip(axs, ("8psk", "8qam")):
        for (pair, col) in (("7.33_2.15", "C0"), ("13_8", "C1"), ("16_6", "C2")):
            rows = re.findall(r"n = 2\^(\d+)\s*: ([+-][\d.]+) \| ([+-][\d.]+)",
                              open(os.path.join(HERE, f"hard8pc_{kind}_{pair}.txt")).read())
            m = [int(r[0]) for r in rows]
            ax.plot(m, [float(r[2]) for r in rows], "o-", color=col, ms=3, lw=1,
                    label=pair.replace("_", " / ") + " dB")
            ax.plot(m, [float(r[1]) for r in rows], ":", color=col, lw=1)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(kind.upper()); ax.set_xlabel(r"$\log_2$ blocklength (symbols)")
        ax.grid(True, lw=0.3, alpha=0.6)
    axs[0].set_ylabel(r"max $R_1^{\rm SUP}-R_1^{\rm TDM}$ (bits/symbol)")
    axs[1].legend(fontsize=7, frameon=False, title="vs TDM+power control\n(dotted: equal power)", title_fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "gain_8ary_vs_n.pdf"))


def fig_rate_control():
    """Advantage in R1 over TDM with power control at 13/8 dB: best design, mode mixing, biased level."""
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex="col")
    for col, kind in enumerate(("8psk", "8qam")):
        txt = open(os.path.join(HERE, f"ratectl_{kind}_13_8.txt")).read()
        for row, n in enumerate((128, 2048)):
            block = txt.split(f"n = {n}:")[1].split("\n\n")[0]
            rows = re.findall(r"^\s+([\d.]+) \|\s+(\S+)\s+(\S+)\s+(\S+) \|\s+(\S+)", block, re.M)
            val = lambda v: float(v) if v != "-" else np.nan
            r2 = [float(r[0]) for r in rows]
            ax = axs[row, col]
            tpc = np.array([val(r[4]) for r in rows])
            for k, (lab, st) in enumerate((("best design", "k-"), ("mode mixing", "C0o-"),
                                           ("biased level (paper's construction)", "C1s--"))):
                ax.plot(r2, np.array([val(r[k + 1]) for r in rows]) - tpc, st, ms=3, lw=1, label=lab)
            ax.axhline(0, color="C2", lw=1, ls=":", label="TDM + power control")
            ax.set_title(f"{kind.upper()} 13 / 8 dB, n = {n}", fontsize=8)
            ax.grid(True, lw=0.3, alpha=0.6)
            if row == 1:
                ax.set_xlabel(r"$R_2$ (bits/symbol)")
            if col == 0:
                ax.set_ylabel(r"$R_1-R_1^{\rm TDM+pc}$ (bits/symbol)")
    axs[0, 1].legend(fontsize=6.5, frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "rate_control_8ary.pdf"))



def fig_polar_8ary():
    """Polar simulations at n = 128 (13/8 dB) against the normal approximations."""
    import json
    rs = [json.loads(l) for l in open(os.path.join(HERE, "polar_8ary_results.jsonl"))]
    na = {}
    for line in open(os.path.join(HERE, "na_curves_8ary.csv")).read().split("\n")[1:]:
        if line:
            k, *v = line.split(","); na.setdefault(k, []).append([float(x) for x in v])
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    for ax, kind in zip(axs, ("8psk", "8qam")):
        R2, best, mixna, teq, tpc = np.array(na[kind]).T
        ax.plot(R2, mixna, "C0--", lw=1, label="NA: mode mixing")
        ax.plot(R2, tpc, "C2:", lw=1.2, label="NA: TDM + power control")
        def best_rm(scheme, params):
            c = [r for r in rs if r["kind"] == kind and r["scheme"] == scheme and r["params"] == params]
            return max(c, key=lambda r: (r["R1"], r["R2"]))
        mix = [best_rm("mix", [mu]) for mu in (0.0, 0.25, 0.5, 0.75, 1.0)]
        ax.plot([r["R2"] for r in mix], [r["R1"] for r in mix], "C0o-", ms=4, lw=1, label="polar: mode mixing")
        keys = sorted({tuple(r["params"]) for r in rs if r["kind"] == kind and r["scheme"] == "tdm"})
        tdm = [best_rm("tdm", list(k)) for k in keys]
        eq = [r for r in tdm if r["params"][1] == 1.0]
        pc = [r for r in tdm if r["params"][1] != 1.0]
        ax.plot([r["R2"] for r in eq], [r["R1"] for r in eq], "ks", ms=4, label="polar: TDM, equal power")
        ax.plot([r["R2"] for r in pc], [r["R1"] for r in pc], "kD", ms=3.5, mfc="none", label="polar: TDM, power control")
        gag = [best_rm("gag", [a]) for a in (0.02, 0.05, 0.1, 0.2)]
        ax.plot([r["R2"] for r in gag], [r["R1"] for r in gag], "C3^", ms=4, label="polar: paper's construction")
        ax.set_title(f"{kind.upper()}, 13 / 8 dB, n = 128", fontsize=8)
        ax.set_xlabel(r"$R_2$ (bits/symbol)")
        ax.grid(True, lw=0.3, alpha=0.6)
    axs[0].set_ylabel(r"$R_1$ (bits/symbol)")
    axs[1].legend(fontsize=6.3, frameon=False, loc="upper right")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "polar_8ary_n128.pdf"))

if __name__ == "__main__":
    import sys
    figs = {"qam_realized": fig_qam_realized, "crossover": fig_crossover,
            "gain_8ary": fig_8ary_gain, "rate_control": fig_rate_control,
            "polar_8ary": fig_polar_8ary}
    for name in (sys.argv[1:] or figs):
        figs[name]()
        print(f"wrote figure {name}", flush=True)
