"""QSBC(p1, p2): superposition vs TDM, asymptotically and with the normal approximation.

1) Asymptotic boundary from three structured families, checked against a random search over
   general (P_V, P_{X|V}) with |V| = 4.
2) Max R1 advantage of SUP over TDM at equal R2, vs blocklength (eps1 = eps2 = 1e-2).
Usage: python3 qsbc_sweep.py
"""
import math
import numpy as np
from qsbc import (FAMILIES, family_configs, qsc, qsc_capacity, sup_R1, tdm_R1)


def H(p, axis=-1):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=axis)


def rates_batch(PV, W, p1, p2):
    """PV (B,k), W (B,k,4) -> I(V;Y2), I(V;Y1), I(X;Y1|V)."""
    out = []
    for p in (p2, p1):
        Pyv = W @ qsc(p)
        Py = np.einsum("bk,bky->by", PV, Pyv)
        Hyv = np.einsum("bk,bk->b", PV, H(Pyv))
        out.append((H(Py) - Hyv, Hyv))
    (I2, _), (I1c, Hy1v) = out
    return I2, I1c, Hy1v - H(qsc(p1)[0])


def random_search(p1, p2, R2bins, rng, samples=400_000, refine=40):
    best = np.full(len(R2bins) - 1, -np.inf); best_cfg = [None] * (len(R2bins) - 1)
    def consider(PV, W):
        I2, I1c, Is = rates_batch(PV, W, p1, p2)
        R2 = np.minimum(I2, I1c)
        idx = np.digitize(R2, R2bins) - 1
        for b in range(len(best)):
            sel = np.nonzero(idx == b)[0]
            if sel.size:
                j = sel[np.argmax(Is[sel])]
                if Is[j] > best[b]:
                    best[b], best_cfg[b] = Is[j], (PV[j].copy(), W[j].copy())
    for _ in range(samples // 20000):
        conc = rng.choice([0.05, 0.3, 1.0])
        consider(rng.dirichlet(np.ones(4) * conc, 20000), rng.dirichlet(np.ones(4) * conc, (20000, 4)))
    for _ in range(refine):                                   # local refinement around the incumbents
        PVs, Ws = [], []
        for cfg in best_cfg:
            if cfg is None: continue
            PV, W = cfg
            for _ in range(200):
                PVs.append(np.abs(PV + rng.normal(0, 0.02, 4))); Ws.append(np.abs(W + rng.normal(0, 0.02, (4, 4))))
        PVs = np.array(PVs); PVs /= PVs.sum(1, keepdims=True)
        Ws = np.array(Ws); Ws /= Ws.sum(2, keepdims=True)
        consider(PVs, Ws)
    return best, best_cfg


def main():
    rng = np.random.default_rng(7)
    pairs = [(1 - 0.99 ** 2, 1 - 0.9 ** 2, "symbol-error-matched to hard QPSK at the BSBC(0.01,0.10) SNRs"),
             (0.01, 0.10, "same numbers as the paper's BSBC"),
             (0.05, 0.30, "")]
    cfgs = family_configs()
    for p1, p2, tag in pairs:
        C1, C2 = qsc_capacity(p1), qsc_capacity(p2)
        print(f"== QSBC({p1:.4f}, {p2:.4f})  {tag}\n   P2P capacities: user1 {C1:.4f}, user2 {C2:.4f} bits/symbol")
        R2 = np.linspace(0.01, C2 - 0.01, 150)
        fam, lab = sup_R1(None, 0, 0, p1, p2, R2, cfgs, asymptotic=True)
        ts = C1 * (1 - R2 / C2)
        g = fam - ts; k = int(np.argmax(g))
        # random search over general distributions, per R2 bin
        bins = np.linspace(0, C2, 41)
        rs, rs_cfg = random_search(p1, p2, bins, rng)
        mid = (bins[:-1] + bins[1:]) / 2
        fam_at_bins, _ = sup_R1(None, 0, 0, p1, p2, bins[:-1], cfgs, asymptotic=True)  # family value at bin's left edge
        excess = np.nanmax(np.where(np.isfinite(rs) & np.isfinite(fam_at_bins), rs - fam_at_bins, -np.inf))
        print(f"   asymptotic: max R1 gain of SUP over time sharing {g[k]:+.4f} at R2={R2[k]:.3f} ({lab[k]})")
        print(f"   random search over general P_V, P_X|V (|V|=4): best excess over the families {excess:+.4f}")
        fams_used = sorted(set(l.split()[0] for l in lab if l))
        print(f"   families on the asymptotic boundary: {fams_used}")
        row = []
        for m in range(5, 14):
            n = 2 ** m
            s, sl = sup_R1(n, 1e-2, 1e-2, p1, p2, R2, cfgs)
            t = tdm_R1(n, 1e-2, 1e-2, p1, p2, R2)
            gg = np.where(np.isfinite(s) & np.isfinite(t) & (s > 0.02) & (t > 0.02), s - t, -np.inf)  # both interior
            j = int(np.argmax(gg))
            row.append(f"2^{m}:{gg[j]:+.3f}({sl[j].split()[0] if sl[j] else '-'})")
        print("   NA, eps=1e-2, max R1 gain over TDM: " + "  ".join(row), flush=True)


if __name__ == "__main__":
    main()
