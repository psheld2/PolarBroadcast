"""Fine rate control between the users on hard-decision 8PSK / 8QAM broadcast channels.

A pure bit split (cloud = some label bits, satellite = the rest, all uniform) gives only corner points.
Two ways to move continuously between them, compared with the normal approximation:

  mix:   time-share two split modes across symbols (a fraction mu of symbols uses mode A, the rest
         mode B), with ONE code per user spanning all of its bits (a polar code over non-identical
         bit channels).  Per-symbol moments are the mu-weighted averages of the modes' moments.
  alpha: the paper's construction on one level: the satellite flips that level's bit with
         probability alpha in (0, 1/2), the other levels split as in the neighbouring modes.

Reference curves: the full design family of hard_8ary_bc.py ("best") and TDM with power control.
Output: R1 at a grid of R2 values for each n, and each strategy's largest shortfall from "best".

Usage: python3 rate_control_8ary.py {8psk|8qam} SNR1_dB SNR2_dB
"""
import math
import sys

import numpy as np

import qsbc
from hard_8ary_bc import EPS, bits_design, configs, p2p_cv, psk8_matrix, qam8_matrix
from hard_8ary_pc import pc_table, tdm_pc_R1
from normal_approx import F, Qinv

MODES = {  # (a2, a1, a0): 0 = cloud bit, 0.5 = satellite bit
    "8psk": [(0, 0, 0), (0, 0, 0.5), (0, 0.5, 0.5), (0.5, 0.5, 0.5)],
    "8qam": [(0, 0, 0), (0, 0, 0.5), (0, 0.5, 0.5), (0.5, 0.5, 0.5)],
}


def moments(design, Q1, Q2):
    PV, W = bits_design(*design)
    return qsbc.sup_moments_mat(PV, W, Q1, Q2)


def mix(mA, mB, mu):
    return {u: {k: mu * mA[u][k] + (1 - mu) * mB[u][k] for k in mA[u]} for u in (1, 2)}


def na_R1(n, m, R2):
    """Largest R1 at each R2 for one moment set (Prop. 1 normal approximation; n=None: capacity)."""
    u2, u1 = m[2], m[1]
    if n is None:
        return np.where(R2 <= min(u2["I_c"], u1["I_c"]) + 1e-12, u1["I_s"], -np.inf)
    if u2["V_c"] < 1e-12 or u1["V_c"] < 1e-12:            # no cloud information
        ok = R2 <= 1e-9
        return np.where(ok, max(u1["I_s"] - math.sqrt(u1["V_s"] / n) * Qinv(EPS), 0), -np.inf)
    ok = R2 <= u2["I_c"] - math.sqrt(u2["V_c"] / n) * Qinv(EPS)
    if u1["V_s"] < 1e-12:                                  # no satellite information
        return np.where(ok & (R2 <= u1["I_c"] - math.sqrt(u1["V_c"] / n) * Qinv(EPS)), 0.0, -np.inf)
    c_c = (u1["I_c"] - R2) / math.sqrt(u1["V_c"] / n)
    rho = np.clip(u1["V_cs"] / math.sqrt(u1["V_c"] * u1["V_s"]), -0.999999, 0.999999)
    lo, hi = np.full(len(R2), -12.0), np.full(len(R2), 12.0)
    for _ in range(45):
        mid = (lo + hi) / 2
        good = F(c_c, mid, rho) >= 1 - EPS
        hi, lo = np.where(good, mid, hi), np.where(good, lo, mid)
    ok &= hi < 11.9
    return np.where(ok, np.maximum(u1["I_s"] - math.sqrt(u1["V_s"] / n) * hi, 0), -np.inf)


def main():
    kind, s1, s2 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    mk = {"8psk": psk8_matrix, "8qam": qam8_matrix}[kind]
    S1, S2 = 10 ** (s1 / 10), 10 ** (s2 / 10)
    Q1, Q2 = mk(S1), mk(S2)
    cv2 = p2p_cv(Q2)
    P = np.logspace(np.log10(0.02), np.log10(50), 90)
    T1, T2 = pc_table(mk, S1, P), pc_table(mk, S2, P)
    modes = MODES[kind]
    mm = [moments(d, Q1, Q2) for d in modes]
    mus = np.linspace(0, 1, 41)
    alphas = np.linspace(0.0, 0.5, 41)
    R2 = np.linspace(0.0, cv2[0], 400)
    cfgs = configs()
    print(f"# {kind.upper()} {s1}/{s2} dB, eps = {EPS}. Split modes (a2,a1,a0): {modes}")
    for d, m in zip(modes, mm):
        print(f"   mode {d}: R2 <= {min(m[2]['I_c'], m[1]['I_c']):.3f}, R1 <= {m[1]['I_s']:.3f} (capacity)")
    for n in (None, 128, 512, 2048):
        mixv = np.full(len(R2), -np.inf)
        for i in range(len(mm)):
            for j in range(i + 1, len(mm)):
                for mu in mus:
                    mixv = np.maximum(mixv, na_R1(n, mix(mm[i], mm[j], mu), R2))
        alv = np.full(len(R2), -np.inf)
        for i in range(len(modes) - 1):                    # bias the one level that differs between neighbours
            A, B = np.array(modes[i], float), np.array(modes[i + 1], float)
            for al in alphas:
                alv = np.maximum(alv, na_R1(n, moments(tuple(np.where(A != B, al, A)), Q1, Q2), R2))
        best, _ = qsbc.sup_R1_mat(n, EPS, EPS, Q1, Q2, R2, cfgs, asymptotic=n is None)
        best = np.maximum(best, np.maximum(mixv, alv))
        tpc = tdm_pc_R1(n, EPS, EPS, P, T1, T2, R2)
        sel = np.isfinite(best) & (best > 0.02)
        short_mix = np.max(np.where(sel, best - mixv, -np.inf))
        short_al = np.max(np.where(sel, best - alv, -np.inf))
        tag = "capacity" if n is None else f"n = {n}"
        print(f"\n{tag}: largest shortfall from the best design: mix {short_mix:.3f}, alpha {short_al:.3f} bits/symbol")
        print("   R2    | best   mix    alpha  | TDM+pc")
        for x in np.linspace(0.1, cv2[0] - 0.1, 9):
            k = int(np.argmin(np.abs(R2 - x)))
            f = lambda v: f"{v[k]:6.3f}" if np.isfinite(v[k]) else "   -  "
            print(f"   {R2[k]:5.3f} | {f(best)} {f(mixv)} {f(alv)} | {f(tpc)}", flush=True)


if __name__ == "__main__":
    main()
