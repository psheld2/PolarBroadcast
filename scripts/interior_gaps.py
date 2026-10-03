"""Signed SUP - TDM gap over interior rate pairs, BSBC vs QSBC (normal approximation, Prop. 1 model).

Gap(n) = max over R2 of [R1_SUP(R2) - R1_TDM(R2)], using only R2 where BOTH schemes have R1 > 0.02.
(Requiring only TDM's R1 > 0.02 lets the clipped SUP rate (0) set the maximum near TDM's edge,
which floors the gap at about -0.02 regardless of SUP's real deficit.)
Writes interior_gaps.csv: model, log2n (channel uses of that model), gap in bits per channel use.
Usage: python3 interior_gaps.py
"""
import numpy as np
import normal_approx as na
import qsbc

EPS, THR = 1e-2, 0.02


def gap(s, t):
    g = np.where(np.isfinite(s) & np.isfinite(t) & (s > THR) & (t > THR), s - t, -np.inf)
    return g.max()


with open("interior_gaps.csv", "w") as f:
    f.write("model,log2n,gap\n")
    R2 = np.linspace(0.01, 1 - na.h(0.10) - 0.01, 150)
    for m in range(6, 16):
        n = 2 ** m
        g = gap(na.sup_R1(n, EPS, EPS, 0.01, 0.10, R2), na.tdm_R1(n, EPS, EPS, 0.01, 0.10, R2))
        f.write(f"BSBC(0.01;0.10),{m},{g:.5f}\n"); f.flush(); print("BSBC", m, round(g, 4), flush=True)
    cfgs = qsbc.family_configs()
    for p1, p2, tag in ((1 - 0.99 ** 2, 1 - 0.9 ** 2, "QSBC(0.0199;0.19)"), (0.01, 0.10, "QSBC(0.01;0.10)")):
        R2 = np.linspace(0.01, qsbc.qsc_capacity(p2) - 0.01, 150)
        for m in range(6, 15):
            n = 2 ** m
            g = gap(qsbc.sup_R1(n, EPS, EPS, p1, p2, R2, cfgs)[0], qsbc.tdm_R1(n, EPS, EPS, p1, p2, R2))
            f.write(f"{tag},{m},{g:.5f}\n"); f.flush(); print(tag, m, round(g, 4), flush=True)
