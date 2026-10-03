"""Validation of the polar superposition (Goela-Abbe-Gastpar) simulator on the paper's BSBC.

BSBC(p1, p2), n binary channel uses, per-user block error rate <= eps, SC decoding.  Cloud V uniform
(an ordinary polar code for the BSC(a * p2) seen by the weak user); satellite X = V xor S, S ~ Bern(a),
with frozen satellite bits set by the deterministic rule psi (most likely value given the cloud).
Satellite positions:
  threshold  common threshold on the two genie-aided metrics (the paper's baseline, Algorithm 1)
  tau=X      the most reliable positions given (V, Y1) among those with error >= X given V alone
Prints (a, selection, R2, R1) for comparison with the thesis's realized points at n = 128.

Usage: python3 gag_bsbc_sim.py [--n 128] [--p1 0.01] [--p2 0.10]
"""
import argparse
import json
import math

import numpy as np

from polar_8ary_sim import largest_k, sc_dual
from qam_polar_sim import frozen_from_order, polar_encode, sc_decode

BATCH, BLOCKS, GENIE = 2000, 8000, 20000


def run(n, p1, p2, a, select, eps, rng):
    w = math.log((1 - a) / a)
    q1, q2 = a * (1 - p1) + (1 - a) * p1, a * (1 - p2) + (1 - a) * p2
    l1, lq = {1: math.log((1 - p1) / p1)}, {1: math.log((1 - q1) / q1), 2: math.log((1 - q2) / q2)}

    def bsc(x, p):
        return x ^ (rng.random(x.shape) < p).astype(np.uint8)

    # cloud order: genie at the weak user, satellite i.i.d.
    err = np.zeros(n)
    for _ in range(GENIE // BATCH):
        u = rng.integers(0, 2, (BATCH, n), dtype=np.uint8)
        y = bsc(bsc(polar_encode(u), a), p2)
        err += (sc_decode((1 - 2.0 * y) * lq[2], np.zeros(n, bool), genie=u) != u).sum(0)
    order2 = np.argsort(-err, kind="stable")
    # satellite metrics
    eH = np.zeros(n); eL = np.zeros(n)
    for _ in range(GENIE // BATCH):
        v = rng.integers(0, 2, (BATCH, n), dtype=np.uint8)
        x = bsc(v, a); u1 = polar_encode(x); y1 = bsc(x, p1)
        Lb = (1 - 2.0 * v) * w; La = Lb + (1 - 2.0 * y1) * l1[1]
        eH += (sc_decode(Lb, np.zeros(n, bool), genie=u1) != u1).sum(0)
        eL += (sc_decode(La, np.zeros(n, bool), genie=u1) != u1).sum(0)
    eH /= GENIE; eL /= GENIE
    score = np.maximum(0.5 - eH, eL)
    if select == "threshold":
        order1 = np.argsort(score, kind="stable")
    else:
        tau = float(select.split("=")[1]); ok = eH >= tau
        order1 = np.concatenate([np.nonzero(ok)[0][np.argsort(eL[ok], kind="stable")],
                                 np.nonzero(~ok)[0][np.argsort(score[~ok], kind="stable")]])

    def info(k):
        fr = np.ones(n, bool); fr[order1[:k]] = False; return fr

    def encode(u2, fr1):
        v = polar_encode(u2); Lb = (1 - 2.0 * v) * w
        u1 = sc_dual(Lb, Lb, ~fr1, rng.integers(0, 2, (u2.shape[0], n), dtype=np.uint8))
        return v, u1, polar_encode(u1)

    def bler_weak(k2, fr1=None):
        fr2 = frozen_from_order(order2, k2); e = 0
        for _ in range(BLOCKS // BATCH):
            u2 = rng.integers(0, 2, (BATCH, n), dtype=np.uint8); u2[:, fr2] = 0
            x = bsc(polar_encode(u2), a) if fr1 is None else encode(u2, fr1)[2]
            e += (sc_decode((1 - 2.0 * bsc(x, p2)) * lq[2], fr2) != u2).any(1).sum()
        return e / BLOCKS

    k2 = largest_k(n, bler_weak, eps)
    fr2 = frozen_from_order(order2, k2)

    def bler_strong(k1):
        fr1 = info(k1); e = 0
        for _ in range(BLOCKS // BATCH):
            u2 = rng.integers(0, 2, (BATCH, n), dtype=np.uint8); u2[:, fr2] = 0
            v, u1, x = encode(u2, fr1)
            y1 = bsc(x, p1)
            u2h = sc_decode((1 - 2.0 * y1) * lq[1], fr2)
            vh = polar_encode(u2h); Lb = (1 - 2.0 * vh) * w
            u1h = sc_dual(Lb + (1 - 2.0 * y1) * l1[1], Lb, ~fr1)
            e += ((u2h != u2).any(1) | (u1h[:, ~fr1] != u1[:, ~fr1]).any(1)).sum()
        return e / BLOCKS

    k1 = largest_k(n, bler_strong, eps)
    while k2 > 0 and bler_weak(k2, info(k1)) > eps:
        k2 -= 1
        fr2 = frozen_from_order(order2, k2)
    return dict(a=a, select=select, k2=k2, k1=k1, R2=k2 / n, R1=k1 / n, nH=int((eH >= 0.4).sum()),
                nL=int((eL <= 1e-3).sum()), nHL=int(((eH >= 0.4) & (eL <= 1e-3)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128); ap.add_argument("--p1", type=float, default=0.01)
    ap.add_argument("--p2", type=float, default=0.10); ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("a", type=float); ap.add_argument("select")
    o = ap.parse_args()
    rng = np.random.default_rng([int(o.a * 1000), sum(map(ord, o.select))])
    r = run(o.n, o.p1, o.p2, o.a, o.select, o.eps, rng)
    print(json.dumps(r), flush=True)
    with open("gag_bsbc_results.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
