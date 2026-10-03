"""Realized rate pairs of polar SUP and polar TDM on the 4-point AWGN broadcast channel.

For each scheme, binary-search the largest information sizes k such that every user's simulated
block error rate is at most eps.  Rates are in bits per complex channel use.

Usage: python3 qam_realized.py SNR1_dB SNR2_dB n [eps]
"""
import math
import sys

import numpy as np

from qam_polar_sim import (SupLink, awgn, frozen_from_order, genie_order, polar_encode, qpsk_llr,
                           qpsk_mod, sc_decode)

BATCH, BLOCKS = 2000, 6000


def _largest_k(n_max, bler_of_k, eps):
    """Largest k in [0, n_max] with bler_of_k(k) <= eps, assuming BLER increases with k."""
    lo, hi = 0, n_max
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if bler_of_k(mid) <= eps:
            lo = mid
        else:
            hi = mid - 1
    return lo


def sup_pair(beta, theta, snr1, snr2, n, eps, rng):
    link = SupLink(beta, theta, snr1, snr2, rng)

    def cloud_llr_fn(user):
        def fn(xv):
            s = rng.integers(0, 2, xv.shape, dtype=np.uint8)
            return link.llr_cloud(link.channel(xv, s, user), user)
        return fn

    def sat_llr_fn(xs):
        v = rng.integers(0, 2, xs.shape, dtype=np.uint8)
        return link.llr_sat(link.channel(v, xs, 1), v)

    order_c = genie_order(cloud_llr_fn(2), n, rng)
    order_s = genie_order(sat_llr_fn, n, rng)

    def bler_weak(k2):
        fr = frozen_from_order(order_c, k2)
        e = 0
        for _ in range(BLOCKS // BATCH):
            u2 = rng.integers(0, 2, (BATCH, n), dtype=np.uint8); u2[:, fr] = 0
            v = polar_encode(u2); s = rng.integers(0, 2, v.shape, dtype=np.uint8)
            e += (sc_decode(link.llr_cloud(link.channel(v, s, 2), 2), fr) != u2).any(1).sum()
        return e / BLOCKS

    k2 = _largest_k(n, bler_weak, eps)
    fr2 = frozen_from_order(order_c, k2)

    def bler_strong(k1):
        fr1 = frozen_from_order(order_s, k1)
        e = 0
        for _ in range(BLOCKS // BATCH):
            u2 = rng.integers(0, 2, (BATCH, n), dtype=np.uint8); u2[:, fr2] = 0
            u1 = rng.integers(0, 2, (BATCH, n), dtype=np.uint8); u1[:, fr1] = 0
            v, s = polar_encode(u2), polar_encode(u1)
            y = link.channel(v, s, 1)
            u2h = sc_decode(link.llr_cloud(y, 1), fr2)
            u1h = sc_decode(link.llr_sat(y, polar_encode(u2h)), fr1)
            e += ((u1h != u1).any(1) | (u2h != u2).any(1)).sum()
        return e / BLOCKS

    k1 = _largest_k(n, bler_strong, eps)
    return k2 / n, k1 / n


def p2p_qpsk_k(n_sym, snr, eps, rng):
    """Largest k for a polar code of length 2*n_sym bits on Gray QPSK at the given SNR."""
    N = 2 * n_sym
    order = genie_order(lambda xb: qpsk_llr(awgn(qpsk_mod(xb), snr, rng), snr), N, rng)

    def bler(k):
        fr = frozen_from_order(order, k)
        e = 0
        for _ in range(BLOCKS // BATCH):
            u = rng.integers(0, 2, (BATCH, N), dtype=np.uint8); u[:, fr] = 0
            y = awgn(qpsk_mod(polar_encode(u)), snr, rng)
            e += (sc_decode(qpsk_llr(y, snr), fr) != u).any(1).sum()
        return e / BLOCKS

    return _largest_k(N, bler, eps)


def tdm_pairs(snr1, snr2, n, eps, rng, p1_list=(1.0,)):
    """lam = 1/2 with user-1 power P1 (user 2 gets 2 - P1), plus the two corner points."""
    out = [(0.0, p2p_qpsk_k(n, snr1, eps, rng) / n), (p2p_qpsk_k(n, snr2, eps, rng) / n, 0.0)]
    for P1 in p1_list:
        k1 = p2p_qpsk_k(n // 2, P1 * snr1, eps, rng)
        k2 = p2p_qpsk_k(n // 2, (2 - P1) * snr2, eps, rng)
        out.append((k2 / n, k1 / n))
    return out


if __name__ == "__main__":
    s1_db, s2_db, n = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    eps = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-2
    snr1, snr2 = 10 ** (s1_db / 10), 10 ** (s2_db / 10)
    rng = np.random.default_rng(2026)
    print(f"# SNR1={s1_db} dB SNR2={s2_db} dB n={n} eps={eps}", flush=True)
    for P1, (r2, r1) in zip([None, None, 1.0, 0.5, 0.25, 1.5],
                            tdm_pairs(snr1, snr2, n, eps, rng, p1_list=(1.0, 0.5, 0.25, 1.5))):
        print(f"TDM  P1={P1}  R2={r2:.4f}  R1={r1:.4f}", flush=True)
    for theta_deg in (0, 30, 60, 90):
        for beta in (0.05, 0.1, 0.2, 0.3, 0.5):
            r2, r1 = sup_pair(beta, math.radians(theta_deg), snr1, snr2, n, eps, rng)
            print(f"SUP  theta={theta_deg:2d} beta={beta:.2f}  R2={r2:.4f}  R1={r1:.4f}", flush=True)
