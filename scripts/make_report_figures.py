"""Figures for reports/superposition_alphabets.tex.

1. figures/qam_realized_n128.pdf: polar SUP and TDM rate pairs realized at n = 128 on the 4-point AWGN
   BC at 7.33 / 2.15 dB (qam_realized_7.33dB_2.15dB_n128.txt), with normal-approximation curves.
2. figures/crossover_binary_vs_quaternary.pdf: signed SUP - TDM gap vs blocklength in QPSK symbols,
   for the binary model (hard-decision Gray QPSK = two BSBC(0.01, 0.10) uses per symbol) and the
   quaternary symmetric model, from interior_gaps.csv (interior_gaps.py).

Run from scripts/:  python3 make_report_figures.py
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


if __name__ == "__main__":
    fig_qam_realized()
    print("wrote qam_realized_n128.pdf", flush=True)
    fig_crossover()
    print("wrote crossover_binary_vs_quaternary.pdf")
