"""Follow-up to hard_8ary_bc.py: the same SUP designs against TDM *with power control*, and a
corrected check of how much general designs add beyond the families.

TDM with power control: user 1 gets a fraction lam of the symbols at energy P1, user 2 the rest at
P2 = (1 - lam P1)/(1 - lam), each slot using the same hard-decision constellation at SNR_j * P_j.
This is the fair orthogonal baseline: bit-split (hierarchical) designs give the bits unequal
protection, which acts like a power allocation that equal-power TDM cannot make.

Search check: a design (R2_d, R1_d) also achieves every R2 <= R2_d, so the family boundary at R2 is
max{R1_d : R2_d >= R2}; the excess of a searched design is R1 - (family boundary at its R2).

Usage: python3 hard_8ary_pc.py {8psk|8qam} SNR1_dB SNR2_dB > out.txt
"""
import sys

import numpy as np

import qsbc
from hard_8ary_bc import EPS, THR, configs, p2p_cv, psk8_matrix, qam8_matrix, rates_batch
from normal_approx import Qinv


def pc_table(mk, snr, P):
    return np.array([p2p_cv(mk(snr * p)) for p in P])          # (len(P), 2): C, V


def tdm_pc_R1(n, eps1, eps2, P, cv1, cv2, R2, lams=np.linspace(0, 1, 201)):
    """Largest R1 for each R2 with TDM and power control; cv_j[k] = (C, V) at energy P[k]."""
    logP = np.log(P)
    pts = []
    for lam in lams:
        for k1, P1 in enumerate(P):
            if lam == 0 and k1 > 0:
                continue
            if lam > 0 and lam * P1 > 1:
                break
            P2 = (1 - lam * P1) / (1 - lam) if lam < 1 else 0.0
            def rate(frac, C, V, eps):
                if frac <= 0:
                    return 0.0
                return max(frac * C - (0 if n is None else np.sqrt(frac * V / n) * Qinv(eps)), 0.0)
            r1 = rate(lam, *cv1[k1], eps1) if lam > 0 else 0.0
            if lam < 1 and P2 > P[0]:
                C2 = np.interp(np.log(P2), logP, cv2[:, 0]); V2 = np.interp(np.log(P2), logP, cv2[:, 1])
                r2 = rate(1 - lam, C2, V2, eps2)
            else:
                r2 = 0.0
            pts.append((r2, r1))
    pts = np.array(pts)
    return np.array([pts[pts[:, 0] >= x, 1].max(initial=-np.inf) for x in R2])


def gap(s, t):
    g = np.where(np.isfinite(s) & np.isfinite(t) & (s > THR) & (t > THR), s - t, -np.inf)
    return g.max()


def main():
    kind, s1, s2 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    mk = {"8psk": psk8_matrix, "8qam": qam8_matrix}[kind]
    S1, S2 = 10 ** (s1 / 10), 10 ** (s2 / 10)
    Q1, Q2 = mk(S1), mk(S2)
    cv1, cv2 = p2p_cv(Q1), p2p_cv(Q2)
    P = np.logspace(np.log10(0.02), np.log10(50), 90)
    T1, T2 = pc_table(mk, S1, P), pc_table(mk, S2, P)
    print(f"# {kind.upper()} {s1}/{s2} dB: SUP vs TDM with power control (eps = {EPS})", flush=True)

    cfgs = configs()
    # corrected search check (asymptotic)
    fam_R2, fam_R1 = rates_batch(np.array([c[1] for c in cfgs]), np.array([c[2] for c in cfgs]), Q1, Q2)
    def fam_env(r2):
        return np.array([fam_R1[fam_R2 >= x].max(initial=-np.inf) for x in r2])
    rng = np.random.default_rng(5)
    best_excess = 0.0
    seeds = [(c[1], c[2]) for c in cfgs]
    for it in range(30):
        PVs, Ws = [], []
        for PV, W in seeds:
            PVs.append(np.abs(PV + rng.normal(0, 0.01, 8))); Ws.append(np.abs(W + rng.normal(0, 0.03, (8, 8))))
        PVs = np.array(PVs); PVs /= PVs.sum(1, keepdims=True)
        Ws = np.array(Ws); Ws /= Ws.sum(2, keepdims=True)
        r2, r1 = rates_batch(PVs, Ws, Q1, Q2)
        ex = r1 - fam_env(r2)
        best_excess = max(best_excess, float(np.nanmax(ex)))
        improved = np.nonzero(ex > 0)[0]
        seeds = [(PVs[i], Ws[i]) for i in improved] + seeds[: max(0, len(cfgs) - len(improved))]
    print(f"Asymptotic: best excess of locally searched designs over the family boundary {best_excess:+.4f}")

    R2 = np.linspace(0.01, cv2[0] - 0.01, 150)
    s, lab = qsbc.sup_R1_mat(None, 0, 0, Q1, Q2, R2, cfgs, asymptotic=True)
    t_eq = qsbc.tdm_R1_cv(None, 0, 0, cv1, cv2, R2, asymptotic=True)
    t_pc = tdm_pc_R1(None, 0, 0, P, T1, T2, R2)
    k = int(np.argmax(s - t_pc))
    print(f"Asymptotic max R1 gain: vs equal-power TDM {np.max(s - t_eq):+.4f}, "
          f"vs TDM with power control {s[k] - t_pc[k]:+.4f} at R2 = {R2[k]:.3f} ({lab[k]})")
    print("Normal approximation, max interior R1 gain (bits/symbol):  vs equal-power TDM | vs TDM with power control")
    for m in range(5, 14):
        n = 2 ** m
        s, sl = qsbc.sup_R1_mat(n, EPS, EPS, Q1, Q2, R2, cfgs)
        ge = gap(s, qsbc.tdm_R1_cv(n, EPS, EPS, cv1, cv2, R2))
        tp = tdm_pc_R1(n, EPS, EPS, P, T1, T2, R2)
        g = np.where(np.isfinite(s) & np.isfinite(tp) & (s > THR) & (tp > THR), s - tp, -np.inf)
        j = int(np.argmax(g))
        print(f"  n = 2^{m:<2d}: {ge:+.4f} | {g[j]:+.4f}  ({sl[j]})", flush=True)


if __name__ == "__main__":
    main()
