"""Satellite-position search for the paper's polar superposition construction, against the selection rules.

For a fixed cloud code, a satellite information set S is feasible if, with the actual encoder (frozen
satellite bits set by psi from the cloud), the weak user's block error rate and the strong user's
(cloud or satellite error) are both <= eps.  Every candidate set is evaluated on the same random draws
(messages, frozen-bit randomness and channel noise), so differences between sets are not noise.

Modes
  exhaustive  BSBC(p1, p2), small n: every satellite set of size 1..KMAX; reports the largest feasible
              size, the best set of each size, and where the rules' sets rank.
  local       n = 128 (BSBC or 8-ary): start from the best rule's largest feasible set, then add
              positions one at a time (best candidate first) and, when no addition is feasible, swap
              out the position that most often starts a decoding error; stop when neither helps.
Rules: "threshold" (common threshold on 1/2 - e_H and e_L) and "tau=X" (smallest e_L among e_H >= X).

Usage: python3 gag_search.py exhaustive N A KMAX
       python3 gag_search.py local bsbc|8psk|8qam A
Appends a JSON line to gag_search_results.jsonl.
"""
import itertools
import json
import math
import sys
import time

import numpy as np

from polar_8ary_sim import Code, labels, llr_tab, sample_y, sc_dual
from qam_polar_sim import frozen_from_order, polar_encode, sc_decode
from hard_8ary_bc import psk8_matrix, qam8_matrix

EPS = 1e-2
TAUS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4)


class BSBC:
    def __init__(self, n, a, p1=0.01, p2=0.10):
        self.n = self.N = n; self.a = a; self.p1, self.p2 = p1, p2
        self.w = math.log((1 - a) / a)
        q = lambda p: a * (1 - p) + (1 - a) * p
        self.lc = {1: math.log((1 - q(p1)) / q(p1)), 2: math.log((1 - q(p2)) / q(p2))}
        self.l1 = math.log((1 - p1) / p1)

    def draws(self, B, rng):
        return dict(z1=(rng.random((B, self.n)) < self.p1).astype(np.uint8),
                    z2=(rng.random((B, self.n)) < self.p2).astype(np.uint8),
                    s=(rng.random((B, self.n)) < self.a).astype(np.uint8))

    def y(self, x, d, j):
        return x ^ d["z1" if j == 1 else "z2"]

    def cloud_llr(self, y, j):
        return (1 - 2.0 * y) * self.lc[j]

    def sat_llrs(self, y1, v):
        Lb = (1 - 2.0 * v) * self.w
        return Lb + (1 - 2.0 * y1) * self.l1, Lb


class Hard8:
    def __init__(self, kind, n, a, rng, s1=13.0, s2=8.0):
        mk = {"8psk": psk8_matrix, "8qam": qam8_matrix}[kind]
        self.Q = {1: mk(10 ** (s1 / 10)), 2: mk(10 ** (s2 / 10))}
        self.n, self.a = n, a
        self.code = Code([(s, l) for s in range(n) for l in range(3)], rng)
        self.N = self.code.N
        self.w = math.log((1 - a) / a)
        ps = np.array([np.prod([a if (s >> l) & 1 else 1 - a for l in range(3)]) for s in range(8)])
        self.Qv = {j: np.array([[sum(ps[s] * Q[v ^ s, y] for s in range(8)) for y in range(8)] for v in range(8)])
                   for j, Q in self.Q.items()}
        Px_v = np.array([[np.prod([a if ((x ^ v) >> l) & 1 else 1 - a for l in range(3)]) for x in range(8)]
                         for v in range(8)])
        self.sat_tab = np.zeros((3, 8, 8))
        for l in range(3):
            for v in range(8):
                num = sum(Px_v[v, x] * self.Q[1][x] for x in range(8) if not (x >> l) & 1)
                den = sum(Px_v[v, x] * self.Q[1][x] for x in range(8) if (x >> l) & 1)
                self.sat_tab[l, :, v] = np.clip(np.log(num) - np.log(den), -60, 60)
        self.none = [[] for _ in range(n)]

    def draws(self, B, rng):
        return dict(r1=rng.random((B, self.n)), r2=rng.random((B, self.n)),
                    s=(rng.random((B, self.N)) < self.a).astype(np.uint8))

    def y(self, x, d, j):
        bits = np.zeros((x.shape[0], self.n, 3), dtype=np.uint8); self.code.write(bits, x)
        cdf = np.cumsum(self.Q[j], axis=1)
        return np.minimum((d["r1" if j == 1 else "r2"][..., None] > cdf[labels(bits)]).sum(-1), 7)

    def cloud_llr(self, y, j):
        return self.code.llrs(y, self.Qv[j], self.none)

    def sat_llrs(self, y1, v):
        Lb = (1 - 2.0 * v) * self.w
        La = Lb.copy()
        vb = np.zeros((y1.shape[0], self.n, 3), dtype=np.uint8); self.code.write(vb, v)
        vlab = labels(vb); c = self.code
        La[:, c.T] = self.sat_tab[c.lev[None, :], y1[:, c.sym], vlab[:, c.sym]]
        return La, Lb


def genie_metrics(sys_, rng, blocks=20000, batch=2000):
    N = sys_.N
    err2 = np.zeros(N); eH = np.zeros(N); eL = np.zeros(N)
    for _ in range(blocks // batch):
        d = sys_.draws(batch, rng)
        u = rng.integers(0, 2, (batch, N), dtype=np.uint8)
        v = polar_encode(u); x = v ^ d["s"][:, :N]
        err2 += (sc_decode(sys_.cloud_llr(sys_.y(x, d, 2), 2), np.zeros(N, bool), genie=u) != u).sum(0)
        v2 = rng.integers(0, 2, (batch, N), dtype=np.uint8); x2 = v2 ^ d["s"][:, :N]; u1 = polar_encode(x2)
        La, Lb = sys_.sat_llrs(sys_.y(x2, d, 1), v2)
        eH += (sc_decode(Lb, np.zeros(N, bool), genie=u1) != u1).sum(0)
        eL += (sc_decode(La, np.zeros(N, bool), genie=u1) != u1).sum(0)
    return np.argsort(-err2, kind="stable"), eH / blocks, eL / blocks


def rule_order(rule, eH, eL):
    score = np.maximum(0.5 - eH, eL)
    if rule == "threshold":
        return np.argsort(score, kind="stable")
    tau = float(rule.split("=")[1]); ok = eH >= tau
    return np.concatenate([np.nonzero(ok)[0][np.argsort(eL[ok], kind="stable")],
                           np.nonzero(~ok)[0][np.argsort(score[~ok], kind="stable")]])


class Evaluator:
    """BLERs of both users for any satellite set, on fixed random draws (common random numbers)."""

    def __init__(self, sys_, fr2, blocks, rng, batch=2000):
        self.sys, self.fr2, self.N = sys_, fr2, sys_.N
        self.batches = []
        for _ in range(max(1, blocks // batch)):
            b = min(batch, blocks)
            u2 = rng.integers(0, 2, (b, self.N), dtype=np.uint8); u2[:, fr2] = 0
            self.batches.append((u2, polar_encode(u2), rng.integers(0, 2, (b, self.N), dtype=np.uint8),
                                 sys_.draws(b, rng)))
        self.blocks = sum(b[0].shape[0] for b in self.batches)

    def __call__(self, S, first_err=False):
        info = np.zeros(self.N, bool); info[list(S)] = True
        e1 = e2 = 0; fe = np.zeros(self.N)
        for u2, v, forced, d in self.batches:
            Lb = (1 - 2.0 * v) * self.sys.w
            u1 = sc_dual(Lb, Lb, info, forced); x = polar_encode(u1)
            y2 = self.sys.y(x, d, 2)
            e2 += (sc_decode(self.sys.cloud_llr(y2, 2), self.fr2) != u2).any(1).sum()
            y1 = self.sys.y(x, d, 1)
            u2h = sc_decode(self.sys.cloud_llr(y1, 1), self.fr2)
            La, Lb1 = self.sys.sat_llrs(y1, polar_encode(u2h))
            u1h = sc_dual(La, Lb1, info)
            bad = (u1h != u1) & info
            e1 += ((u2h != u2).any(1) | bad.any(1)).sum()
            if first_err:
                rows = np.nonzero(bad.any(1))[0]
                np.add.at(fe, bad[rows].argmax(1), 1)
        return e1 / self.blocks, e2 / self.blocks, fe

    def feasible(self, S):
        b1, b2, _ = self(S)
        return b1 <= EPS and b2 <= EPS, b1, b2


def cloud_k2(sys_, order2, rng, blocks=8000):
    """Largest cloud size meeting eps at the weak user with an i.i.d. satellite."""
    def bler(k2):
        fr2 = frozen_from_order(order2, k2); e = 0
        for _ in range(blocks // 2000):
            d = sys_.draws(2000, rng)
            u2 = rng.integers(0, 2, (2000, sys_.N), dtype=np.uint8); u2[:, fr2] = 0
            x = polar_encode(u2) ^ d["s"][:, :sys_.N]
            e += (sc_decode(sys_.cloud_llr(sys_.y(x, d, 2), 2), fr2) != u2).any(1).sum()
        return e / blocks
    lo, hi = 0, sys_.N
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if bler(mid) <= EPS:
            lo = mid
        else:
            hi = mid - 1
    return lo


def rule_max(ev, order, kmax):
    """Largest k such that the rule's first k positions are feasible: binary search, then a short
    upward scan in case feasibility is not exactly monotone in k."""
    lo, hi = 0, kmax
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ev.feasible(order[:mid])[0]:
            lo = mid
        else:
            hi = mid - 1
    for k in range(lo + 1, min(kmax, lo + 3) + 1):
        if ev.feasible(order[:k])[0]:
            lo = k
    return lo


def exhaustive(n, a, kmax, rng):
    sys_ = BSBC(n, a)
    order2, eH, eL = genie_metrics(sys_, rng)
    k2 = cloud_k2(sys_, order2, rng)
    fr2 = frozen_from_order(order2, k2)
    ev = Evaluator(sys_, fr2, 20000, rng)
    rules = {r: rule_order(r, eH, eL) for r in ["threshold"] + [f"tau={t}" for t in TAUS]}
    out = dict(mode="exhaustive", n=n, a=a, k2=k2, sizes={})
    best_feasible = 0
    for k in range(1, kmax + 1):
        results = []
        for S in itertools.combinations(range(n), k):
            b1, b2, _ = ev(S)
            results.append((max(b1, b2), b1, b2, S))
        results.sort()
        feas = [r for r in results if r[0] <= EPS]
        if feas:
            best_feasible = k
        rank = {}
        for rname, order in rules.items():
            Sr = tuple(sorted(order[:k].tolist()))
            pos = next(i for i, r in enumerate(results) if tuple(sorted(r[3])) == Sr)
            rank[rname] = dict(rank=pos + 1, worst_bler=results[pos][0])
        out["sizes"][k] = dict(n_sets=len(results), n_feasible=len(feas), best_set=list(results[0][3]),
                               best_worst_bler=results[0][0], rules=rank)
        print(f"k1={k}: {len(results)} sets, {len(feas)} feasible, best {results[0][3]} "
              f"({results[0][0]:.4f}); rules: " + ", ".join(f"{r}#{v['rank']}" for r, v in rank.items()), flush=True)
    out["exhaustive_max_k1"] = best_feasible
    out["rule_max_k1"] = {r: rule_max(ev, o, kmax + 2) for r, o in rules.items()}
    fresh = Evaluator(sys_, fr2, 20000, np.random.default_rng(999))      # independent draws
    out["fresh"] = {}
    for k, v in out["sizes"].items():
        out["fresh"][f"best_k{k}"] = list(fresh(v["best_set"])[:2])
    for r, o in rules.items():
        km = out["rule_max_k1"][r]
        if km:
            out["fresh"][f"{r}_k{km}"] = list(fresh(o[:km])[:2])
    print("fresh-draw BLERs (strong, weak):", out["fresh"], flush=True)
    return out


def local(kind, a, rng, n=128, max_steps=40, pool=40):
    sys_ = BSBC(n, a) if kind == "bsbc" else Hard8(kind, n, a, rng)
    N = sys_.N
    order2, eH, eL = genie_metrics(sys_, rng)
    k2 = cloud_k2(sys_, order2, rng)
    fr2 = frozen_from_order(order2, k2)
    ev = Evaluator(sys_, fr2, 8000, rng)
    screen = Evaluator(sys_, fr2, 4000, rng)          # cheaper ranking of candidates (own draws)
    rules = {r: rule_order(r, eH, eL) for r in ["threshold"] + [f"tau={t}" for t in TAUS]}
    kmax = N
    rmax = {r: rule_max(ev, o, kmax) for r, o in rules.items()}
    best_rule = max(rmax, key=rmax.get)
    S = set(rules[best_rule][:rmax[best_rule]].tolist())
    print(f"{kind} a={a}: k2={k2}, rule maxima {rmax}; local search from {best_rule} (k1={len(S)})", flush=True)
    t0 = time.time()
    for step in range(max_steps):
        cand = [i for i in np.argsort(eL) if i not in S][:pool]          # most promising additions first
        scored = sorted((max(screen(S | {c})[:2]), c) for c in cand)
        added = False
        for _, c in scored[:3]:                        # confirm the best few on the full draws
            ok, b1, b2 = ev.feasible(S | {c})
            if ok:
                S.add(c); added = True
                print(f"  step {step}: add {c} -> k1={len(S)} ({max(b1, b2):.4f})", flush=True)
                break
        if added:
            continue
        # no feasible addition: swap out the position that most often starts a satellite error
        _, _, fe = ev(S, first_err=True)
        worst = int(np.argmax(np.where(np.isin(np.arange(N), list(S)), fe, -1)))
        swapped = False
        cur = max(ev(S)[:2])
        for c in [x for x in cand if x != worst][:pool]:
            T = (S - {worst}) | {c}
            b = max(ev(T)[:2])
            if b < cur:
                S = T; swapped = True
                print(f"  step {step}: swap {worst} -> {c} ({cur:.4f} -> {b:.4f})", flush=True)
                break
        if not swapped:
            break
    fresh = Evaluator(sys_, fr2, 20000, np.random.default_rng(999))      # independent draws
    rule_set = rules[best_rule][:rmax[best_rule]].tolist()
    fr_rule, fr_local = fresh(rule_set)[:2], fresh(sorted(S))[:2]
    print(f"  fresh draws: rule set BLER (strong, weak) = {fr_rule}, local set = {fr_local}", flush=True)
    return dict(mode="local", kind=kind, n=n, a=a, k2=k2, rule_max_k1=rmax, best_rule=best_rule,
                local_k1=len(S), R2=k2 / n, R1_rule=rmax[best_rule] / n, R1_local=len(S) / n,
                fresh_rule=list(fr_rule), fresh_local=list(fr_local), local_set=sorted(int(i) for i in S),
                rule_set=[int(i) for i in rule_set], minutes=round((time.time() - t0) / 60, 1))


def main():
    mode = sys.argv[1]
    rng = np.random.default_rng(12345)
    if mode == "exhaustive":
        r = exhaustive(int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), rng)
        r["sizes"] = {k: {kk: (list(map(int, vv)) if kk == "best_set" else vv) for kk, vv in v.items()}
                      for k, v in r["sizes"].items()}
    else:
        r = local(sys.argv[2], float(sys.argv[3]), rng)
    print(json.dumps(r), flush=True)
    with open(sys.argv[-1] if sys.argv[-1].endswith(".jsonl") else "gag_search_results.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
