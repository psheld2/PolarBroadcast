"""Hard-decision Gray-coded QPSK as a 4-ary discrete broadcast channel, compared with the QSC model.

Symbol x = 2*b_I + b_Q carries the Gray bits (b_I, b_Q): 00 (+,+), 01 (+,-), 11 (-,-), 10 (-,+), so
neighbouring constellation points differ in one bit.  With hard decisions each bit is flipped
independently with probability q = Q(sqrt(SNR)):
    P(y | x) = (1 - q)^(2 - d) q^d,   d = Hamming distance between the labels of x and y,
i.e. Y = X xor Z on Z2 x Z2 with Z = (Bern(q), Bern(q)): correct w.p. (1-q)^2, each neighbour w.p.
q(1-q), the diagonal w.p. q^2.  (The QSC with the same symbol error rate p = 1-(1-q)^2 puts p/3 on
each wrong symbol instead.)  The channel is two independent BSC(q) uses per symbol, so its
broadcast version is two BSBC(q1, q2) uses per symbol.

Superposition: cloud V uniform on the 4 symbols, X = V xor S, with S from
  product(aI, aQ): S = (Bern(aI), Bern(aQ)) -- per-bit BSBC superposition (aI = aQ: the paper's
                   construction on both bits; aI = 0, aQ = 1/2: the I/Q split, i.e. TDM at 1/2),
  symmetric(a):    P(S = 0) = 1 - a, P(S = s) = a/3,
  double-flip(a):  S in {00, 11}, P(S = 11) = a,
plus a random search over general (P_V, P_X|V) for the asymptotic boundary.

Usage: python3 gray_qpsk_bc.py   (prints the comparison; writes gray_qpsk_results.txt)
"""
import math

import numpy as np

import normal_approx as na
import qsbc

EPS, THR = 1e-2, 0.02
POP = np.array([[bin(x ^ y).count("1") for y in range(4)] for x in range(4)])


def gray_matrix(q):
    return (1 - q) ** (2 - POP) * q ** POP


def gray_cv(q):
    """Point-to-point capacity and dispersion: two independent BSC(q) per symbol."""
    return 2 * (1 - na.h(q)), 2 * na.V_bsc(q)


def xor_satellite(ps):
    """W[v, x] = P(S = v xor x) for a distribution ps over S in {0,1,2,3}."""
    return np.array([[ps[v ^ x] for x in range(4)] for v in range(4)])


def configs():
    out = []
    grid = np.concatenate([[0.0], np.linspace(0.01, 0.5, 15)])
    for aI in grid:
        for aQ in grid:
            if aI <= aQ:                               # (aI, aQ) and (aQ, aI) are equivalent
                pI, pQ = np.array([1 - aI, aI]), np.array([1 - aQ, aQ])
                ps = np.array([pI[x >> 1] * pQ[x & 1] for x in range(4)])
                out.append((f"product aI={aI:.3f} aQ={aQ:.3f}", np.full(4, 0.25), xor_satellite(ps)))
    for a in np.linspace(0.01, 0.5, 50):              # equal-alpha product, finer
        ps = np.array([(1 - a) ** 2, a * (1 - a), a * (1 - a), a * a])
        out.append((f"product aI=aQ={a:.3f}", np.full(4, 0.25), xor_satellite(ps)))
    for a in np.linspace(0.005, 0.75, 60):
        out.append((f"symmetric a={a:.3f}", np.full(4, 0.25), xor_satellite([1 - a, a / 3, a / 3, a / 3])))
    for a in np.linspace(0.005, 0.5, 40):
        out.append((f"double-flip a={a:.3f}", np.full(4, 0.25), xor_satellite([1 - a, 0, 0, a])))
    return out


def H(p, axis=-1):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=axis)


def random_search(Q1, Q2, C2, rng, samples=400_000, bins=40):
    edges = np.linspace(0, C2, bins + 1)
    best = np.full(bins, -np.inf)
    H1 = H(Q1[0])                                     # H(Y1 | X), the same for every x
    for _ in range(samples // 20000):
        conc = rng.choice([0.05, 0.3, 1.0])
        PV = rng.dirichlet(np.ones(4) * conc, 20000)
        W = rng.dirichlet(np.ones(4) * conc, (20000, 4))
        rates = []
        for Q in (Q2, Q1):
            Pyv = W @ Q
            Py = np.einsum("bk,bky->by", PV, Pyv)
            Hyv = np.einsum("bk,bk->b", PV, H(Pyv))
            rates.append((H(Py) - Hyv, Hyv))
        (I2, _), (I1c, Hy1v) = rates
        R2, R1 = np.minimum(I2, I1c), Hy1v - H1
        idx = np.digitize(R2, edges) - 1
        for b in range(bins):
            sel = idx == b
            if sel.any():
                best[b] = max(best[b], R1[sel].max())
    return edges[:-1], best


def interior_gap(s, t):
    g = np.where(np.isfinite(s) & np.isfinite(t) & (s > THR) & (t > THR), s - t, -np.inf)
    return g.max()


def main():
    q1, q2 = 0.01, 0.10                       # hard-decided Gray QPSK at 7.33 dB / 2.15 dB
    G1, G2 = gray_matrix(q1), gray_matrix(q2)
    p1, p2 = 1 - (1 - q1) ** 2, 1 - (1 - q2) ** 2
    lines = []
    say = lambda x="": (print(x, flush=True), lines.append(x))
    np.set_printoptions(precision=4, suppress=True)
    say(f"Gray-coded QPSK, hard decisions, q1 = {q1}, q2 = {q2} (labels 00, 01, 10, 11 as x = 0..3)")
    for q, G, p in ((q1, G1, p1), (q2, G2, p2)):
        say(f"\n q = {q}: Gray QPSK P(y|x)                QSC({p:.4f}) with the same symbol error rate")
        for r_g, r_q in zip(G, qsbc.qsc(p)):
            say("   " + " ".join(f"{v:.4f}" for v in r_g) + "          " + " ".join(f"{v:.4f}" for v in r_q))
    cv1, cv2 = gray_cv(q1), gray_cv(q2)
    say(f"\nP2P capacities (bits/symbol): user 1 {cv1[0]:.4f}, user 2 {cv2[0]:.4f}   "
        f"(QSC model: {qsbc.qsc_capacity(p1):.4f}, {qsbc.qsc_capacity(p2):.4f})")

    cfgs = configs()
    R2 = np.linspace(0.01, cv2[0] - 0.01, 150)
    s_inf, lab = qsbc.sup_R1_mat(None, 0, 0, G1, G2, R2, cfgs, asymptotic=True)
    t_inf = qsbc.tdm_R1_cv(None, 0, 0, cv1, cv2, R2, asymptotic=True)
    g = s_inf - t_inf; k = int(np.argmax(g))
    say(f"\nAsymptotic max R1 gain of SUP over TDM: {g[k]:+.4f} bits/symbol at R2 = {R2[k]:.3f} ({lab[k]})")
    say(f"   (2 x BSBC(0.01, 0.10): 2 x 0.0479 = {2 * 0.0479:.4f})")
    edges, rs = random_search(G1, G2, cv2[0], np.random.default_rng(11))
    fam_at, _ = qsbc.sup_R1_mat(None, 0, 0, G1, G2, edges, cfgs, asymptotic=True)
    excess = np.nanmax(np.where(np.isfinite(rs) & np.isfinite(fam_at), rs - fam_at, -np.inf))
    say(f"Random search over general (P_V, P_X|V): best excess over the families {excess:+.4f} bits/symbol")

    # equal-alpha product design must reproduce the BSBC normal approximation at 2n uses
    n = 256
    al = 0.1
    ps = np.array([(1 - al) ** 2, al * (1 - al), al * (1 - al), al * al])
    m = qsbc.sup_moments_mat(np.full(4, 0.25), xor_satellite(ps), G1, G2)
    Ic, Is, Vc, Vs, Vcs = na.strong_user_moments(q1, al)
    say(f"\nCheck (alpha = {al}): per-symbol moments vs 2 x BSBC per-use moments: "
        f"I_s {m[1]['I_s']:.6f} vs {2 * Is:.6f}, V_s {m[1]['V_s']:.6f} vs {2 * Vs:.6f}, "
        f"V_cs {m[1]['V_cs']:.6f} vs {2 * Vcs:.6f}")

    say("\nNormal approximation, eps1 = eps2 = 1e-2: max interior R1 gain of SUP over TDM (bits/symbol)")
    say(" n (symbols) |  Gray QPSK DMC (best design)            |  2 x BSBC(0.01,0.10) at 2n uses  |  QSC(0.0199,0.19)")
    bsbc = {int(r.split(",")[1]): float(r.split(",")[2]) for r in open("interior_gaps.csv").read().split()[1:]
            if r.startswith("BSBC")}
    qsc = {int(r.split(",")[1]): float(r.split(",")[2]) for r in open("interior_gaps.csv").read().split()[1:]
           if r.startswith("QSBC(0.0199")}
    for mexp in range(7, 15):
        n = 2 ** mexp
        s, sl = qsbc.sup_R1_mat(n, EPS, EPS, G1, G2, R2, cfgs)
        t = qsbc.tdm_R1_cv(n, EPS, EPS, cv1, cv2, R2)
        gg = np.where(np.isfinite(s) & np.isfinite(t) & (s > THR) & (t > THR), s - t, -np.inf)
        j = int(np.argmax(gg))
        b = 2 * bsbc[mexp + 1] if mexp + 1 in bsbc else float("nan")
        say(f"  2^{mexp:<2d}       |  {gg[j]:+.4f}  ({sl[j] if sl[j] else '-':28s}) |  {b:+.4f}"
            f"                         |  {qsc.get(mexp, float('nan')):+.4f}")
    open("gray_qpsk_results.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
