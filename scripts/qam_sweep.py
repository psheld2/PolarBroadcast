"""Normal-approximation sweep: best 4-point SUP vs TDM (equal power / power control).
Usage: python3 qam_sweep.py a|b   (each runs two SNR pairs)"""
import sys
import numpy as np, math
import qam_bc
from qam_bc import *
db = lambda x: 10**(x/10)
_cache = {}
_orig = qam_bc.sup_moments
def cached(beta, theta, s1, s2):
    k = (round(beta, 6), round(theta, 6), s1, s2)
    if k not in _cache: _cache[k] = _orig(beta, theta, s1, s2)
    return _cache[k]
qam_bc.sup_moments = cached
betas = np.concatenate([np.linspace(0.01, 0.3, 30), np.linspace(0.32, 0.98, 34)])
thetas = np.deg2rad([0, 15, 30, 45, 60, 75, 90])
pairs = {"a": [(7.33, 2.15, "BSBC-matched"), (10, 0, "")], "b": [(15, 0, ""), (20, 5, "")]}[sys.argv[1]]
for s1, s2, tag in pairs:
    S1, S2 = db(s1), db(s2)
    Cq2 = qpsk_moments(S2)[0]
    R2 = np.linspace(0.02, Cq2 - 0.02, 120)
    lines = {False: [], True: []}
    for n in [None] + [2**m for m in range(7, 13)]:
        s, arg = sup_region_R1(n, 1e-2, 1e-2, S1, S2, R2, betas, thetas, asymptotic=n is None)
        for pc in (False, True):
            t = tdm_region_R1(n, 1e-2, 1e-2, S1, S2, R2, power_control=pc, asymptotic=n is None)
            g = np.where(np.isfinite(s) & np.isfinite(t) & (t > 0.02), s - t, -np.inf)
            k = int(np.argmax(g)); b, th = arg[k] if arg[k] else (float('nan'), float('nan'))
            lines[pc].append(f"{'inf' if n is None else 'n=2^%d' % int(math.log2(n))}: {g[k]:+.3f} @R2={R2[k]:.2f} (beta={b:.2f},th={np.rad2deg(th):.0f})")
    print(f"== SNR1={s1} dB, SNR2={s2} dB {tag}  QPSK caps: u1 {qpsk_moments(S1)[0]:.3f}, u2 {Cq2:.3f}")
    for pc in (False, True):
        print(f"   max R1 gain of best 4-point SUP vs TDM {'+power ctrl' if pc else 'equal power'}:")
        for l in lines[pc]: print("      " + l)
    sys.stdout.flush()
