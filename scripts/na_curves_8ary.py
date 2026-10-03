"""Normal-approximation reference curves at n = 128 for the polar simulations (hard-decision 8PSK/8QAM,
13/8 dB, eps = 1e-2): best SUP design, mode mixing, equal-power TDM, TDM with power control.

Usage: python3 na_curves_8ary.py   (writes na_curves_8ary.csv: kind,R2,best,mix,tdm_eq,tdm_pc)
"""
import numpy as np

import qsbc
from hard_8ary_bc import EPS, configs, p2p_cv, psk8_matrix, qam8_matrix
from hard_8ary_pc import pc_table, tdm_pc_R1
from rate_control_8ary import MODES, mix, moments, na_R1

N_SYM = 128


def main():
    rows = []
    for kind, mk in (("8psk", psk8_matrix), ("8qam", qam8_matrix)):
        S1, S2 = 10 ** 1.3, 10 ** 0.8
        Q1, Q2 = mk(S1), mk(S2)
        cv1, cv2 = p2p_cv(Q1), p2p_cv(Q2)
        R2 = np.linspace(0.0, cv2[0], 120)
        mm = [moments(d, Q1, Q2) for d in MODES[kind]]
        mixv = np.full(len(R2), -np.inf)
        for i in range(len(mm)):
            for j in range(i + 1, len(mm)):
                for mu in np.linspace(0, 1, 41):
                    mixv = np.maximum(mixv, na_R1(N_SYM, mix(mm[i], mm[j], mu), R2))
        best, _ = qsbc.sup_R1_mat(N_SYM, EPS, EPS, Q1, Q2, R2, configs())
        best = np.maximum(best, mixv)
        teq = qsbc.tdm_R1_cv(N_SYM, EPS, EPS, cv1, cv2, R2)
        P = np.logspace(np.log10(0.02), np.log10(50), 90)
        tpc = tdm_pc_R1(N_SYM, EPS, EPS, P, pc_table(mk, S1, P), pc_table(mk, S2, P), R2)
        rows += [(kind, *v) for v in zip(R2, best, mixv, teq, tpc)]
        print(kind, "done", flush=True)
    with open("na_curves_8ary.csv", "w") as f:
        f.write("kind,R2,best,mix,tdm_eq,tdm_pc\n")
        for r in rows:
            f.write(",".join([r[0]] + [f"{v:.5f}" if np.isfinite(v) else "nan" for v in r[1:]]) + "\n")


if __name__ == "__main__":
    main()
