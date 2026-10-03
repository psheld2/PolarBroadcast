"""Bit-level view of hard-decision 8PSK / 8QAM (Gray labels): multilevel (chain-rule) and BICM level
capacities, Bhattacharyya parameters of the multistage levels, how a length-128 polar transform
polarizes each level (erasure-channel recursion, an upper bound used for illustration), and the
per-user rates of the bit splits.  Seconds of arithmetic.

Usage: python3 bit_levels_8ary.py > bit_levels_results.txt
"""
import itertools

import numpy as np

from hard_8ary_bc import psk8_matrix, qam8_matrix


def H(p):
    p = p[p > 0]
    return -(p * np.log2(p)).sum()


def I_bits(Q, target, cond):
    """I(target bits ; Y | cond bits) with uniform labels; bits are indexed 2, 1, 0."""
    tot = 0.0
    for cv in itertools.product((0, 1), repeat=len(cond)):
        xs = [x for x in range(8) if all((x >> b & 1) == v for b, v in zip(cond, cv))]
        groups = {}
        for x in xs:
            groups.setdefault(tuple(x >> b & 1 for b in target), []).append(x)
        tot += (H(Q[xs].mean(0)) - np.mean([H(Q[g].mean(0)) for g in groups.values()])) / 2 ** len(cond)
    return tot


def Z_level(Q, b, cond):
    z = 0.0
    for cv in itertools.product((0, 1), repeat=len(cond)):
        xs = [x for x in range(8) if all((x >> c & 1) == v for c, v in zip(cond, cv))]
        P0 = Q[[x for x in xs if not x >> b & 1]].mean(0)
        P1 = Q[[x for x in xs if x >> b & 1]].mean(0)
        z += np.sqrt(P0 * P1).sum() / 2 ** len(cond)
    return z


def polarize(z, n=128, tol=1e-3):
    zs = np.array([z])
    while len(zs) < n:
        zs = np.concatenate([2 * zs - zs ** 2, zs ** 2])
    return int((zs < tol).sum()), int((zs > 1 - tol).sum())


if __name__ == "__main__":
    for name, mk in (("8PSK", psk8_matrix), ("8QAM", qam8_matrix)):
        for sdb in (13.0, 8.0):
            Q = mk(10 ** (sdb / 10))
            msd = [I_bits(Q, [2], []), I_bits(Q, [1], [2]), I_bits(Q, [0], [2, 1])]
            bicm = [I_bits(Q, [b], []) for b in (2, 1, 0)]
            zm = [Z_level(Q, 2, []), Z_level(Q, 1, [2]), Z_level(Q, 0, [2, 1])]
            pol = [polarize(z) for z in zm]
            print(f"{name} {sdb:4.1f} dB: I(X;Y) = {sum(msd):.3f}; multistage levels (b2, b1|b2, b0|b2b1) = "
                  f"{np.round(msd, 3)}; BICM (b2, b1, b0) = {np.round(bicm, 3)} (sum {sum(bicm):.3f})")
            print(f"     Z = {np.round(zm, 4)}; n = 128 good/bad synthetic channels (tol 1e-3) = {pol}")
            if name == "8PSK":
                print(f"     split: weak user b2b1 with b0 as noise {I_bits(Q, [2, 1], []):.3f}; "
                      f"strong user b0 | b2b1 {I_bits(Q, [0], [2, 1]):.3f}")
            else:
                print(f"     split: weak user b2 with b1b0 as noise {I_bits(Q, [2], []):.3f}; "
                      f"strong user b1b0 | b2 {I_bits(Q, [1, 0], [2]):.3f}")
