"""Hard-decision 8PSK and rectangular 8QAM as 8-ary discrete broadcast channels: SUP vs TDM.

Channel: complex AWGN, Es = 1, noise CN(0, 1/SNR), minimum-distance hard decisions, Gray labels.
  8PSK:  points e^{j 2 pi k / 8}, label gray(k) = k ^ (k >> 1); sector probabilities from an exact
         radial integral and Gauss-Legendre quadrature in angle.
  8QAM:  rectangular 4x2 grid, I in {-3,-1,1,3}, Q in {-1,1}, scaled to Es = 1; Gray 4-PAM bits
         on I (labels 00 01 11 10 from -3 to 3) and one bit on Q; label = (I bits) * 2 + Q bit.
         Exact through 1-D Gaussian integrals (I and Q decide independently).
Superposition: cloud V uniform on the 8 labels, X = V xor S, with
  bits(a2, a1, a0): the three label bits of S independent, P(bit b = 1) = a_b.  a_b = 0 keeps bit b
                    for the cloud (weak user), a_b = 1/2 gives it to the satellite (strong user),
                    in between is BSBC-style superposition on that bit;
  symmetric(a):     P(S = 0) = 1 - a, uniform over the other 7.
A local search around the best family designs (and a random search) checks the asymptotic boundary.
TDM time-shares point-to-point codes with uniform input.  Normal approximation as in the paper
(per-user eps, SIC at the strong user); interior gap = max over R2 of R1_SUP - R1_TDM over pairs where
both R1 > 0.02.

Usage: python3 hard_8ary_bc.py {8psk|8qam} SNR1_dB SNR2_dB > out.txt
"""
import itertools
import math
import sys

import numpy as np

import qsbc
from normal_approx import Phi

EPS, THR = 1e-2, 0.02
GL_X, GL_W = np.polynomial.legendre.leggauss(400)


def psk8_matrix(snr):
    N = 1 / snr
    pts = np.exp(1j * 2 * np.pi * np.arange(8) / 8)
    lab = [k ^ (k >> 1) for k in range(8)]
    W = np.zeros((8, 8))
    for k, x in enumerate(pts):
        for m in range(8):
            lo, hi = 2 * np.pi * m / 8 - np.pi / 8, 2 * np.pi * m / 8 + np.pi / 8
            th = lo + (hi - lo) * (GL_X + 1) / 2
            a = np.abs(x) * np.cos(th - np.angle(x))
            g = (0.5 * np.exp(-1 / N) + a * np.sqrt(np.pi / N) * np.exp(-(1 - a ** 2) / N)
                 * Phi(a * np.sqrt(2 / N))) / np.pi
            W[lab[k], lab[m]] = (hi - lo) / 2 * np.sum(GL_W * g)
    return W


def qam8_matrix(snr):
    s = 1 / math.sqrt(6)                                  # E|x|^2 = (5 + 1) s^2 = 1
    sig = math.sqrt(1 / (2 * snr))
    lev, thr = np.array([-3, -1, 1, 3]) * s, np.array([-np.inf, -2 * s, 0, 2 * s, np.inf])
    PI = np.array([[Phi((thr[j + 1] - lev[i]) / sig) - Phi((thr[j] - lev[i]) / sig) for j in range(4)]
                   for i in range(4)])
    qv = np.array([1, -1]) * s
    PQ = np.array([[Phi((np.inf if j == 0 else 0 - qv[i]) / sig) if j == 0 else Phi((0 - qv[i]) / sig)
                    for j in range(2)] for i in range(2)])
    PQ[:, 0] = 1 - Phi((0 - qv) / sig)                     # decide +1 when y_Q > 0
    PQ[:, 1] = Phi((0 - qv) / sig)
    gray2 = [0b00, 0b01, 0b11, 0b10]
    W = np.zeros((8, 8))
    for i, iq, j, jq in itertools.product(range(4), range(2), range(4), range(2)):
        W[(gray2[i] << 1) | iq, (gray2[j] << 1) | jq] = PI[i, j] * PQ[iq, jq]
    return W


def p2p_cv(W):
    """Capacity and dispersion (bits) of the DMC with uniform input."""
    Py = W.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        i = np.where(W > 0, np.log2(W / Py), 0.0)
    C = np.sum(W * i) / 8
    return C, np.sum(W * (i - C) ** 2) / 8


def xor_sat(ps):
    return np.array([[ps[v ^ x] for x in range(8)] for v in range(8)])


GRID = [0.0, 0.01, 0.03, 0.06, 0.1, 0.17, 0.28, 0.5]


def bits_design(a2, a1, a0):
    ps = np.array([(a2 if x >> 2 & 1 else 1 - a2) * (a1 if x >> 1 & 1 else 1 - a1) * (a0 if x & 1 else 1 - a0)
                   for x in range(8)])
    return np.full(8, 1 / 8), xor_sat(ps)


def configs():
    out = [(f"bits a=({a2},{a1},{a0})", *bits_design(a2, a1, a0)) for a2, a1, a0 in itertools.product(GRID, repeat=3)]
    out += [(f"symmetric a={a:.3f}", np.full(8, 1 / 8), xor_sat(np.r_[1 - a, np.full(7, a / 7)]))
            for a in np.linspace(0.01, 0.875, 30)]
    return out


def H(p, axis=-1):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=axis)


def rates_batch(PV, W, Q1, Q2):
    out = []
    for Q in (Q2, Q1):
        Pyv = W @ Q
        Py = np.einsum("bk,bky->by", PV, Pyv)
        Hyv = np.einsum("bk,bk->b", PV, H(Pyv))
        out.append((H(Py) - Hyv, Hyv))
    (I2, _), (I1c, Hy1v) = out
    HY1X = np.mean(H(Q1))
    # I(X;Y1|V) = H(Y1|V) - H(Y1|X), with H(Y1|X) averaged over the satellite's X distribution
    Px = np.einsum("bk,bkx->bx", PV, W)
    Hy1x = Px @ H(Q1)
    return np.minimum(I2, I1c), Hy1v - Hy1x


def boundary_search(cfgs, Q1, Q2, edges, rng, iters=25, pert=300):
    """Best R1 per R2 bin: family designs, then local hill-climbing and random samples."""
    best = np.full(len(edges) - 1, -np.inf); cfg = [None] * (len(edges) - 1)

    def consider(PV, W):
        R2, R1 = rates_batch(PV, W, Q1, Q2)
        idx = np.digitize(R2, edges) - 1
        for b in np.unique(idx):
            if 0 <= b < len(best):
                sel = np.nonzero(idx == b)[0]; j = sel[np.argmax(R1[sel])]
                if R1[j] > best[b]:
                    best[b], cfg[b] = R1[j], (PV[j].copy(), W[j].copy())
    consider(np.array([c[1] for c in cfgs]), np.array([c[2] for c in cfgs]))
    fam = best.copy()
    for _ in range(10):
        conc = rng.choice([0.05, 0.3, 1.0])
        consider(rng.dirichlet(np.ones(8) * conc, 20000), rng.dirichlet(np.ones(8) * conc, (20000, 8)))
    for _ in range(iters):
        PVs, Ws = [], []
        for c in cfg:
            if c is None:
                continue
            for _ in range(pert):
                PVs.append(np.abs(c[0] + rng.normal(0, 0.01, 8))); Ws.append(np.abs(c[1] + rng.normal(0, 0.02, (8, 8))))
        PVs = np.array(PVs); PVs /= PVs.sum(1, keepdims=True)
        Ws = np.array(Ws); Ws /= Ws.sum(2, keepdims=True)
        consider(PVs, Ws)
    return fam, best


def main():
    kind, s1, s2 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    mk = {"8psk": psk8_matrix, "8qam": qam8_matrix}[kind]
    Q1, Q2 = mk(10 ** (s1 / 10)), mk(10 ** (s2 / 10))
    assert np.allclose(Q1.sum(1), 1, atol=1e-9) and np.allclose(Q2.sum(1), 1, atol=1e-9)
    np.set_printoptions(precision=4, suppress=True, linewidth=140)
    print(f"# {kind.upper()} hard decisions, Gray labels, SNR1 = {s1} dB, SNR2 = {s2} dB, eps = {EPS}")
    for s, Q in ((s1, Q1), (s2, Q2)):
        bit_err = [sum(Q[x, y] for x in range(8) for y in range(8) if (x ^ y) >> b & 1) / 8 for b in (2, 1, 0)]
        print(f"\nP(y|x) at {s} dB (rows/cols = labels 0..7); symbol error {1 - np.trace(Q) / 8:.4f}; "
              f"bit error rates (b2, b1, b0) = {np.round(bit_err, 4)}")
        print(Q)
    try:
        T = np.linalg.solve(Q1, Q2)
        print(f"\nQ2 = Q1 T with T >= 0 (stochastically degraded)? {bool(T.min() > -1e-9)} (min entry {T.min():.2e})")
    except np.linalg.LinAlgError:
        print("\nQ1 singular; degradedness not checked")
    cv1, cv2 = p2p_cv(Q1), p2p_cv(Q2)
    print(f"P2P (uniform input): user 1 C = {cv1[0]:.4f}, V = {cv1[1]:.4f}; user 2 C = {cv2[0]:.4f}, V = {cv2[1]:.4f}")

    cfgs = configs()
    R2 = np.linspace(0.01, cv2[0] - 0.01, 150)
    s, lab = qsbc.sup_R1_mat(None, 0, 0, Q1, Q2, R2, cfgs, asymptotic=True)
    t = qsbc.tdm_R1_cv(None, 0, 0, cv1, cv2, R2, asymptotic=True)
    g = s - t; k = int(np.argmax(g))
    print(f"\nAsymptotic: max R1 gain of SUP over TDM {g[k]:+.4f} bits/symbol at R2 = {R2[k]:.3f} ({lab[k]})")
    eq = [c for c in cfgs if c[0].startswith("bits") and len(set(c[0][8:-1].split(","))) == 1]
    se, _ = qsbc.sup_R1_mat(None, 0, 0, Q1, Q2, R2, eq, asymptotic=True)
    print(f"   paper-style design (same a on every bit): max gain {np.max(se - t):+.4f}")
    edges = np.linspace(0, cv2[0], 31)
    fam, srch = boundary_search(cfgs, Q1, Q2, edges, np.random.default_rng(3))
    ok = np.isfinite(fam) & np.isfinite(srch)
    print(f"   local/random search beyond the families: up to {np.max(srch[ok] - fam[ok]):+.4f} bits/symbol")
    print("\nNormal approximation: max interior R1 gain over TDM (bits/symbol), best design")
    for m in range(5, 14):
        n = 2 ** m
        s, sl = qsbc.sup_R1_mat(n, EPS, EPS, Q1, Q2, R2, cfgs)
        t = qsbc.tdm_R1_cv(n, EPS, EPS, cv1, cv2, R2)
        gg = np.where(np.isfinite(s) & np.isfinite(t) & (s > THR) & (t > THR), s - t, -np.inf)
        j = int(np.argmax(gg))
        se, _ = qsbc.sup_R1_mat(n, EPS, EPS, Q1, Q2, R2, eq)
        ge = np.where(np.isfinite(se) & np.isfinite(t) & (se > THR) & (t > THR), se - t, -np.inf).max()
        print(f"  n = 2^{m:<2d}: best {gg[j]:+.4f} ({sl[j]})   paper-style {ge:+.4f}", flush=True)


if __name__ == "__main__":
    main()
