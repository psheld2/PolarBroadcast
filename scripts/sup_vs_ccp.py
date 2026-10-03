"""Smallest blocklength at which superposition (SUP) beats a common codeword (CCP), BSBC vs QSBC.

CCP: both messages in one point-to-point code that both receivers decode, so
R1 + R2 <= min_j [C_j - sqrt(V_j/n) Q^{-1}(eps_j)].  The gain is
max over interior R2 of R1_SUP(R2) - R1_CCP(R2) (normal approximation, eps1 = eps2 = 1e-2).
Usage: python3 sup_vs_ccp.py
"""
import math
import numpy as np
import normal_approx as na
import qsbc

EPS = 1e-2


def ccp_sum(caps_disps, n):
    return min(C - math.sqrt(V / n) * na.Qinv(EPS) for C, V in caps_disps)


def gain(sup_r1, ccp, R2):
    r1_ccp = ccp - R2
    g = np.where(np.isfinite(sup_r1) & (r1_ccp > 0.02), sup_r1 - r1_ccp, -np.inf)
    k = int(np.argmax(g))
    return g[k], R2[k]


def bsbc(p1, p2, ms):
    out = {}
    cd = [(1 - na.h(p), na.V_bsc(p)) for p in (p1, p2)]
    for m in ms:
        n = 2 ** m
        R2 = np.linspace(0.005, max(ccp_sum(cd, n) - 0.025, 0.01), 120)
        out[m] = gain(na.sup_R1(n, EPS, EPS, p1, p2, R2), ccp_sum(cd, n), R2)
    return out


def qsbc_(p1, p2, ms):
    out = {}
    cfgs = qsbc.family_configs()
    cd = [(qsbc.qsc_capacity(p), qsbc.qsc_dispersion(p)) for p in (p1, p2)]
    for m in ms:
        n = 2 ** m
        R2 = np.linspace(0.01, max(ccp_sum(cd, n) - 0.025, 0.02), 120)
        out[m] = gain(qsbc.sup_R1(n, EPS, EPS, p1, p2, R2, cfgs)[0], ccp_sum(cd, n), R2)
    return out


if __name__ == "__main__":
    ms = range(3, 11)
    rows = [("BSBC(0.01, 0.10)", bsbc(0.01, 0.10, ms)),
            ("QSBC(0.0199, 0.19)  [same QPSK SNRs, hard symbols]", qsbc_(1 - 0.99 ** 2, 1 - 0.9 ** 2, ms)),
            ("QSBC(0.01, 0.10)", qsbc_(0.01, 0.10, ms))]
    for name, res in rows:
        print(name)
        print("   " + "  ".join(f"n=2^{m}:{res[m][0]:+.3f}" for m in ms), flush=True)
