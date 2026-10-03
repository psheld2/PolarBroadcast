"""Polar codes on the hard-decision 8PSK / rectangular 8QAM broadcast channels (Gray labels).

n symbols per block, per-user block error rate <= eps, successive-cancellation (SC) decoding.  For each
scheme the information sizes are the largest whose simulated block error rates meet eps (binary
search); rates are reported in bits per symbol.  Label bits: level 2 (b2), 1 (b1), 0 (b0).

Schemes
  mix MU    Bit split / mode mixing.  Mode A on round(MU*n) symbols (weak user b2 b1 | strong user b0),
            mode B on the rest (weak user b2 | strong user b1 b0).  One polar code per user over all of
            its bits.  The weak user decodes with per-bit LLRs (other bits uniform); the strong user
            decodes the weak user's code, re-encodes it, and decodes its own bits given the weak user's
            bits in each symbol.  MU = 1 and MU = 0 are the two pure bit splits.
  tdm LAM P1   User 1 gets round(LAM*n) symbols at energy P1, user 2 the rest at
            P2 = (1 - LAM*P1)/(1 - LAM); each user codes over all three bits of its symbols.
  gag A     The paper's construction on every bit: cloud V uniform, X = V xor S with the three bits of
            S i.i.d. Bern(A).  Cloud code over the 3n V bits; satellite code over the 3n X bits whose
            frozen bits are set by the deterministic rule psi (most likely value given the cloud);
            satellite positions chosen as in the paper's baseline (common threshold on the two
            genie-aided metrics).
Codes whose length is not a power of two are rate-matched by shortening (--rm short, default) or
quasi-uniform puncturing (--rm punct); the paper's construction is always punctured.  Every
information set comes from a genie-aided Monte-Carlo construction on that code's own channel.

Usage: python3 polar_8ary_sim.py KIND SCHEME PARAMS... [--rm short|punct] [--n 128] [--snr1 13] [--snr2 8]
Appends one JSON line to polar_8ary_results.jsonl.
"""
import argparse
import json
import math
import time

import numpy as np

from hard_8ary_bc import psk8_matrix, qam8_matrix
from qam_polar_sim import _boxplus, frozen_from_order, polar_encode, sc_decode

import os

BATCH = int(os.environ.get("SIM_BATCH", 2000))
BLOCKS = int(os.environ.get("SIM_BLOCKS", 8000))
GENIE_BLOCKS = int(os.environ.get("SIM_GENIE_BLOCKS", 20000))


# ---------------------------------------------------------------- channel and LLRs
def sample_y(Q, x, rng):
    cdf = np.cumsum(Q, axis=1)
    u = rng.random(x.shape)
    return np.minimum((u[..., None] > cdf[x]).sum(-1), 7)


def llr_tab(Q, lev, known):
    """T[y, kv]: LLR of bit `lev` given hard decision y and the known bits (values packed in kv)."""
    K = len(known)
    T = np.zeros((8, 2 ** K))
    for kv in range(2 ** K):
        vals = [(kv >> j) & 1 for j in range(K)]
        ok = [x for x in range(8) if all((x >> known[j]) & 1 == vals[j] for j in range(K))]
        x0 = [x for x in ok if not (x >> lev) & 1]
        x1 = [x for x in ok if (x >> lev) & 1]
        T[:, kv] = np.log(np.maximum(Q[x0].sum(0), 1e-300)) - np.log(np.maximum(Q[x1].sum(0), 1e-300))
    return np.clip(T, -60, 60)


def bitrev(i, m):
    return int(format(i, f"0{m}b")[::-1], 2) if m else 0


class Code:
    """One user's polar code: mother length N, M transmitted bits on (symbol, level) slots.

    rm = "punct": quasi-uniform puncturing (the bit-reversed first N-M positions are not sent, LLR 0).
    rm = "short": shortening.  With G = F^{(x)m}, x_j depends only on u_i with i a bitwise superset of
    j (so i >= j); the last N-M positions are therefore closed under supersets, and freezing u there
    to 0 forces those coded bits to 0, known to the receiver (LLR +inf) and not sent."""

    def __init__(self, slots, rng, rm="punct"):
        self.M = len(slots)
        self.N = 1 << max(0, math.ceil(math.log2(self.M)))
        m = int(math.log2(self.N))
        if rm == "short":
            self.short = np.arange(self.M, self.N)
            self.T = np.arange(self.M)
        else:
            punct = {bitrev(i, m) for i in range(self.N - self.M)}
            self.short = np.array([], dtype=int)
            self.T = np.array([i for i in range(self.N) if i not in punct])
        perm = rng.permutation(self.M)
        self.sym = np.array([slots[k][0] for k in perm])
        self.lev = np.array([slots[k][1] for k in perm])

    def frozen(self, order, k):
        """Frozen mask with the k most reliable non-shortened positions free (order: least reliable first)."""
        fr = np.ones(self.N, bool)
        cand = [i for i in order[::-1] if i not in set(self.short.tolist())]
        fr[np.array(cand[:k], dtype=int)] = False
        return fr

    def write(self, bits, cw):
        bits[:, self.sym, self.lev] = cw[:, self.T]

    def read(self, bits):
        return bits[:, self.sym, self.lev]

    def llrs(self, y, Q, known_of, bits_known=None):
        """Per-coded-bit LLRs; known_of[s] lists the levels known in symbol s (others uniform)."""
        L = np.zeros((y.shape[0], self.N))
        L[:, self.short] = 60.0
        groups = {}
        for k, (s, l) in enumerate(zip(self.sym, self.lev)):
            groups.setdefault((l, tuple(known_of[s])), []).append(k)
        for (l, known), ks in groups.items():
            ks = np.array(ks)
            tab = llr_tab(Q, l, list(known))
            syms = self.sym[ks]
            kv = np.zeros((y.shape[0], len(ks)), dtype=int)
            for j, kl in enumerate(known):
                kv |= bits_known[:, syms, kl].astype(int) << j
            L[:, self.T[ks]] = tab[y[:, syms], kv]
        return L


def labels(bits):
    return bits[..., 0] + 2 * bits[..., 1] + 4 * bits[..., 2]


# ---------------------------------------------------------------- generic search helpers
def genie_order(llr_fn, code, rng):
    N = code.N
    err = np.zeros(N)
    for _ in range(GENIE_BLOCKS // BATCH):
        u = rng.integers(0, 2, (BATCH, N), dtype=np.uint8)
        u[:, code.short] = 0
        uh = sc_decode(llr_fn(polar_encode(u)), np.zeros(N, bool), genie=u)
        err += (uh != u).sum(0)
    return np.argsort(-err, kind="stable")


def largest_k(kmax, bler, eps):
    lo, hi = 0, kmax
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if bler(mid) <= eps:
            lo = mid
        else:
            hi = mid - 1
    return lo


def rand_info(N, frozen, rng):
    u = rng.integers(0, 2, (BATCH, N), dtype=np.uint8)
    u[:, frozen] = 0
    return u


# ---------------------------------------------------------------- schemes
def run_mix(Q1, Q2, n, mu, eps, rng, rm):
    syms = rng.permutation(n)
    A = set(syms[: int(round(mu * n))].tolist())
    weak_slots = [(s, 2) for s in range(n)] + [(s, 1) for s in range(n) if s in A]
    strong_slots = [(s, 0) for s in range(n)] + [(s, 1) for s in range(n) if s not in A]
    weak, strong = Code(weak_slots, rng, rm), Code(strong_slots, rng, rm)
    none = [[] for _ in range(n)]
    known_strong = [[2, 1] if s in A else [2] for s in range(n)]

    def weak_llr_fn(Q):
        def fn(cw):
            bits = rng.integers(0, 2, (cw.shape[0], n, 3), dtype=np.uint8)
            weak.write(bits, cw)
            return weak.llrs(sample_y(Q, labels(bits), rng), Q, none)
        return fn

    def strong_llr_fn(cw):
        bits = rng.integers(0, 2, (cw.shape[0], n, 3), dtype=np.uint8)
        strong.write(bits, cw)
        return strong.llrs(sample_y(Q1, labels(bits), rng), Q1, known_strong, bits)

    order2 = genie_order(weak_llr_fn(Q2), weak, rng)
    order1 = genie_order(strong_llr_fn, strong, rng)

    def bler2(k2):
        fr = weak.frozen(order2, k2)
        e = 0
        for _ in range(BLOCKS // BATCH):
            u2 = rand_info(weak.N, fr, rng)
            bits = rng.integers(0, 2, (BATCH, n, 3), dtype=np.uint8)
            weak.write(bits, polar_encode(u2))
            y = sample_y(Q2, labels(bits), rng)
            e += (sc_decode(weak.llrs(y, Q2, none), fr) != u2).any(1).sum()
        return e / BLOCKS

    k2 = largest_k(weak.M, bler2, eps)
    fr2 = weak.frozen(order2, k2)

    def bler1(k1):
        fr1 = strong.frozen(order1, k1)
        e = 0
        for _ in range(BLOCKS // BATCH):
            u2, u1 = rand_info(weak.N, fr2, rng), rand_info(strong.N, fr1, rng)
            bits = np.zeros((BATCH, n, 3), dtype=np.uint8)
            weak.write(bits, polar_encode(u2)); strong.write(bits, polar_encode(u1))
            y = sample_y(Q1, labels(bits), rng)
            u2h = sc_decode(weak.llrs(y, Q1, none), fr2)
            kb = np.zeros_like(bits); weak.write(kb, polar_encode(u2h))
            u1h = sc_decode(strong.llrs(y, Q1, known_strong, kb), fr1)
            e += ((u2h != u2).any(1) | (u1h != u1).any(1)).sum()
        return e / BLOCKS

    k1 = largest_k(strong.M, bler1, eps)
    return dict(k2=k2, k1=k1, N2=weak.N, M2=weak.M, N1=strong.N, M1=strong.M)


def p2p(Q, nsym, eps, rng, rm):
    """Largest k for one user coding over all three bits of nsym symbols."""
    if nsym == 0:
        return 0, 0, 0
    code = Code([(s, l) for s in range(nsym) for l in range(3)], rng, rm)
    none = [[] for _ in range(nsym)]

    def fn(cw):
        bits = rng.integers(0, 2, (cw.shape[0], nsym, 3), dtype=np.uint8)
        code.write(bits, cw)
        return code.llrs(sample_y(Q, labels(bits), rng), Q, none)

    order = genie_order(fn, code, rng)

    def bler(k):
        fr = code.frozen(order, k)
        e = 0
        for _ in range(BLOCKS // BATCH):
            u = rand_info(code.N, fr, rng)
            bits = np.zeros((BATCH, nsym, 3), dtype=np.uint8)
            code.write(bits, polar_encode(u))
            y = sample_y(Q, labels(bits), rng)
            e += (sc_decode(code.llrs(y, Q, none), fr) != u).any(1).sum()
        return e / BLOCKS

    return largest_k(code.M, bler, eps), code.N, code.M


def run_tdm(mk, s1, s2, n, lam, P1, eps, rng, rm):
    n1 = int(round(lam * n)); n2 = n - n1
    P2 = (1 - n1 / n * P1) / (n2 / n) if n2 else 0.0
    k1, N1, M1 = p2p(mk(10 ** (s1 / 10) * P1), n1, eps, rng, rm) if n1 else (0, 0, 0)
    k2, N2, M2 = p2p(mk(10 ** (s2 / 10) * P2), n2, eps, rng, rm) if n2 else (0, 0, 0)
    return dict(k2=k2, k1=k1, N2=N2, M2=M2, N1=N1, M1=M1, P2=P2)


def sc_dual(La, Lb, info, forced=None):
    """SC over N bits with two LLR streams sharing decisions: at information positions decide from La
    (or take `forced`), at frozen positions take the most likely value from Lb (ties -> 0)."""
    B, N = La.shape
    u = np.zeros((B, N), dtype=np.uint8)

    def rec(A, Bm, lo):
        m = A.shape[1]
        if m == 1:
            if info[lo]:
                d = forced[:, lo] if forced is not None else (A[:, 0] < 0).astype(np.uint8)
            else:
                d = (Bm[:, 0] < 0).astype(np.uint8)
            u[:, lo] = d
            return d[:, None]
        h = m // 2
        a = rec(_boxplus(A[:, :h], A[:, h:]), _boxplus(Bm[:, :h], Bm[:, h:]), lo)
        sg = 1 - 2 * a.astype(np.int8)
        b = rec(A[:, h:] + sg * A[:, :h], Bm[:, h:] + sg * Bm[:, :h], lo + h)
        return np.concatenate([a ^ b, b], axis=1)

    rec(La, Lb, 0)
    return u


def run_gag(Q1, Q2, n, a, eps, rng):
    code = Code([(s, l) for s in range(n) for l in range(3)], rng)    # shared by cloud and satellite
    N, none = code.N, [[] for _ in range(n)]
    ps = np.array([np.prod([a if (s >> l) & 1 else 1 - a for l in range(3)]) for s in range(8)])
    Qv = {j: np.array([[sum(ps[s] * Q[v ^ s, y] for s in range(8)) for y in range(8)] for v in range(8)])
          for j, Q in ((1, Q1), (2, Q2))}
    w = math.log((1 - a) / a)
    # satellite LLR of x_l given (y, v-hat label): P(x | v) = prod_l (a if x_l != v_l else 1 - a)
    Px_v = np.array([[np.prod([a if ((x ^ v) >> l) & 1 else 1 - a for l in range(3)]) for x in range(8)]
                     for v in range(8)])
    sat_tab = np.zeros((3, 8, 8))                                      # [level, y, v]
    for l in range(3):
        for v in range(8):
            num = sum(Px_v[v, x] * Q1[x] for x in range(8) if not (x >> l) & 1)
            den = sum(Px_v[v, x] * Q1[x] for x in range(8) if (x >> l) & 1)
            sat_tab[l, :, v] = np.clip(np.log(num) - np.log(den), -60, 60)

    def cloud_llr(y, j):
        return code.llrs(y, Qv[j], none)

    def sat_llrs(y, v_full):
        """La (cloud + channel) and Lb (cloud only) for the satellite code, all N positions."""
        Lb = (1 - 2.0 * v_full) * w
        La = Lb.copy()
        vb = np.zeros((y.shape[0], n, 3), dtype=np.uint8); code.write(vb, v_full)
        vlab = labels(vb)
        La[:, code.T] = sat_tab[code.lev[None, :], y[:, code.sym], vlab[:, code.sym]]
        return La, Lb

    def model_tx(B):
        v = rng.integers(0, 2, (B, N), dtype=np.uint8)
        s = (rng.random((B, N)) < a).astype(np.uint8)
        return v, v ^ s

    # cloud positions: genie on the weak user's channel (satellite i.i.d. as in the model)
    def cloud_fn(vcw):
        x = vcw ^ (rng.random(vcw.shape) < a).astype(np.uint8)
        bits = np.zeros((vcw.shape[0], n, 3), dtype=np.uint8); code.write(bits, x)
        return cloud_llr(sample_y(Q2, labels(bits), rng), 2)
    order2 = genie_order(cloud_fn, code, rng)
    # satellite metrics: genie error rates given the cloud only (H) and given cloud + channel (L)
    eH = np.zeros(N); eL = np.zeros(N)
    for _ in range(GENIE_BLOCKS // BATCH):
        v, x = model_tx(BATCH)
        u1 = polar_encode(x)
        bits = np.zeros((BATCH, n, 3), dtype=np.uint8); code.write(bits, x)
        La, Lb = sat_llrs(sample_y(Q1, labels(bits), rng), v)
        eH += (sc_decode(Lb, np.zeros(N, bool), genie=u1) != u1).sum(0)
        eL += (sc_decode(La, np.zeros(N, bool), genie=u1) != u1).sum(0)
    eH /= GENIE_BLOCKS; eL /= GENIE_BLOCKS
    score = np.maximum(0.5 - eH, eL)                       # common threshold on both criteria
    order1 = np.argsort(score, kind="stable")              # best satellite positions first

    def encode(u2, fr1, B):
        v = polar_encode(u2)
        info = ~fr1
        forced = rng.integers(0, 2, (B, N), dtype=np.uint8)
        Lb = (1 - 2.0 * v) * w
        u1 = sc_dual(Lb, Lb, info, forced)
        return v, u1, polar_encode(u1)

    def info_from(k, order):
        fr = np.ones(N, bool); fr[order[:k]] = False; return fr

    def bler_weak(k2, fr1=None):
        fr2 = frozen_from_order(order2, k2)
        e = 0
        for _ in range(BLOCKS // BATCH):
            u2 = rand_info(N, fr2, rng)
            if fr1 is None:
                x = polar_encode(u2) ^ (rng.random((BATCH, N)) < a).astype(np.uint8)
            else:
                _, _, x = encode(u2, fr1, BATCH)
            bits = np.zeros((BATCH, n, 3), dtype=np.uint8); code.write(bits, x)
            e += (sc_decode(cloud_llr(sample_y(Q2, labels(bits), rng), 2), fr2) != u2).any(1).sum()
        return e / BLOCKS

    k2 = largest_k(N, bler_weak, eps)
    fr2 = frozen_from_order(order2, k2)

    def bler_strong(k1):
        fr1 = info_from(k1, order1)
        e = 0
        for _ in range(BLOCKS // BATCH):
            u2 = rand_info(N, fr2, rng)
            v, u1, x = encode(u2, fr1, BATCH)
            bits = np.zeros((BATCH, n, 3), dtype=np.uint8); code.write(bits, x)
            y = sample_y(Q1, labels(bits), rng)
            u2h = sc_decode(cloud_llr(y, 1), fr2)
            La, Lb = sat_llrs(y, polar_encode(u2h))
            u1h = sc_dual(La, Lb, ~fr1)
            e += ((u2h != u2).any(1) | (u1h[:, ~fr1] != u1[:, ~fr1]).any(1)).sum()
        return e / BLOCKS

    k1 = largest_k(N, bler_strong, eps)
    fr1 = info_from(k1, order1)
    # with the real (psi-steered) satellite, recheck the weak user; back off k2 if needed
    while k2 > 0 and bler_weak(k2, fr1) > eps:
        k2 -= 1
        fr2 = frozen_from_order(order2, k2)
    return dict(k2=k2, k1=k1, N=N, M=code.M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind"); ap.add_argument("scheme"); ap.add_argument("params", nargs="*", type=float)
    ap.add_argument("--n", type=int, default=128); ap.add_argument("--snr1", type=float, default=13.0)
    ap.add_argument("--snr2", type=float, default=8.0); ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="polar_8ary_results.jsonl")
    ap.add_argument("--rm", choices=["punct", "short"], default="short")
    a = ap.parse_args()
    mk = {"8psk": psk8_matrix, "8qam": qam8_matrix}[a.kind]
    Q1, Q2 = mk(10 ** (a.snr1 / 10)), mk(10 ** (a.snr2 / 10))
    rng = np.random.default_rng([a.seed, {"mix": 1, "tdm": 2, "gag": 3}[a.scheme],
                                 int(round(1000 * a.params[0])), int(round(1000 * (a.params[1] if len(a.params) > 1 else 0))),
                                 {"8psk": 8, "8qam": 9}[a.kind], {"punct": 0, "short": 1}[a.rm]])
    t = time.time()
    if a.scheme == "mix":
        r = run_mix(Q1, Q2, a.n, a.params[0], a.eps, rng, a.rm)
    elif a.scheme == "tdm":
        r = run_tdm(mk, a.snr1, a.snr2, a.n, a.params[0], a.params[1], a.eps, rng, a.rm)
    elif a.scheme == "gag":
        r = run_gag(Q1, Q2, a.n, a.params[0], a.eps, rng)
        a.rm = "punct"
    else:
        raise SystemExit(f"unknown scheme {a.scheme}")
    r.update(kind=a.kind, scheme=a.scheme, params=a.params, rm=a.rm, n=a.n, snr1=a.snr1, snr2=a.snr2, eps=a.eps,
             R2=r["k2"] / a.n, R1=r["k1"] / a.n, seconds=round(time.time() - t, 1))
    with open(a.out, "a") as f:
        f.write(json.dumps(r) + "\n")
    print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
