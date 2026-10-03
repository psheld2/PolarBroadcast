"""Summarize polar_8ary_results.jsonl against the TDM hull and the n = 128 normal approximations.

Usage: python3 polar_8ary_summary.py > polar_8ary_summary.txt
"""
import json

import numpy as np

rs = [json.loads(l) for l in open("polar_8ary_results.jsonl")]
na = {}
for line in open("na_curves_8ary.csv").read().split("\n")[1:]:
    if line:
        k, *v = line.split(",")
        na.setdefault(k, []).append([float(x) for x in v])


def hull(pts):
    """Upper concave envelope through (R2, R1) points."""
    H = []
    for p in sorted(set(pts)):
        while len(H) >= 2 and (H[-1][0] - H[-2][0]) * (p[1] - H[-2][1]) - (H[-1][1] - H[-2][1]) * (p[0] - H[-2][0]) >= 0:
            H.pop()
        H.append(p)
    xs, ys = np.array([h[0] for h in H]), np.array([h[1] for h in H])
    return lambda x: np.interp(x, xs, ys, left=np.nan, right=np.nan)


def best_rm(kind, scheme, params):
    c = [r for r in rs if r["kind"] == kind and r["scheme"] == scheme and r["params"] == list(params)]
    return max(c, key=lambda r: (r["R1"], r["R2"]))


print("Polar simulations, n = 128 symbols, 13/8 dB, per-user BLER <= 1e-2, SC decoding.")
print("Best of shortening/puncturing per configuration. TDM reference = upper convex hull of all TDM points.\n")
for kind in ("8psk", "8qam"):
    R2g, best, mixna, teq, tpc = np.array(na[kind]).T
    keys = sorted({tuple(r["params"]) for r in rs if r["kind"] == kind and r["scheme"] == "tdm"})
    tdm = [best_rm(kind, "tdm", p) for p in keys]
    T = hull([(r["R2"], r["R1"]) for r in tdm])
    print(f"== {kind.upper()}")
    print("  mode mixing:    mu     R2     R1   | vs TDM hull | NA mixing at R2 | polar loss to NA")
    for mu in (0.0, 0.25, 0.5, 0.75, 1.0):
        r = best_rm(kind, "mix", [mu]); nm = np.interp(r["R2"], R2g, mixna)
        print(f"                {mu:4.2f}  {r['R2']:.3f}  {r['R1']:.3f}  |  {r['R1'] - T(r['R2']):+.3f}     |  {nm:.3f}          | {nm - r['R1']:.3f} ({r['rm']})")
    print("  TDM:          lam   P1    R2     R1   | NA TDM+pc at R2 | polar loss to NA")
    for r in tdm:
        nt = np.interp(r["R2"], R2g, tpc)
        print(f"                {r['params'][0]:4.2f}  {r['params'][1]:3.1f}  {r['R2']:.3f}  {r['R1']:.3f}  |  {nt:.3f}          | {nt - r['R1']:.3f} ({r['rm']})")
    print("  paper's construction (gag):   a     R2     R1   | vs TDM hull")
    for a in (0.02, 0.05, 0.1, 0.2):
        r = best_rm(kind, "gag", [a])
        print(f"                              {a:4.2f}  {r['R2']:.3f}  {r['R1']:.3f}  |  {r['R1'] - T(r['R2']):+.3f}")
    sel = np.isfinite(mixna) & np.isfinite(tpc) & (mixna > 0.02) & (tpc > 0.02)
    print(f"  NA prediction at n = 128: max (mixing - TDM+pc) = {np.max((mixna - tpc)[sel]):+.3f}\n")
