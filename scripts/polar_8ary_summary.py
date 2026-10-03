"""Summarize polar_8ary_results.jsonl (and the selection-rule runs in gag_select_results.jsonl and
gag_bsbc_results.jsonl) against the TDM hull and the n = 128 normal approximations.

Usage: python3 polar_8ary_summary.py > polar_8ary_summary.txt
"""
import json

import numpy as np

rs = [json.loads(l) for l in open("polar_8ary_results.jsonl")]
# the paper's construction: threshold rule from the main run, plus the selection-rule runs
gs = [dict(r, select=r.get("select", "threshold")) for r in rs if r["scheme"] == "gag"]
try:
    gs += [json.loads(l) for l in open("gag_select_results.jsonl")]
except FileNotFoundError:
    pass
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
    print("  paper's construction:  a   rule        R2     R1   | vs TDM hull   (threshold rule, then the best reliability rule)")
    for a in sorted({r["params"][0] for r in gs if r["kind"] == kind}):
        cand = [r for r in gs if r["kind"] == kind and r["params"][0] == a]
        thr = [r for r in cand if r["select"] == "threshold"][:1]
        rel = [r for r in cand if r["select"] != "threshold"]
        for r in thr + ([max(rel, key=lambda r: r["R1"] - T(r["R2"]))] if rel else []):
            print(f"                      {a:4.2f}  {r['select']:10s} {r['R2']:.3f}  {r['R1']:.3f}  |  {r['R1'] - T(r['R2']):+.3f}")
    sel = np.isfinite(mixna) & np.isfinite(tpc) & (mixna > 0.02) & (tpc > 0.02)
    print(f"  NA prediction at n = 128: max (mixing - TDM+pc) = {np.max((mixna - tpc)[sel]):+.3f}\n")

try:
    bs = [json.loads(l) for l in open("gag_bsbc_results.jsonl")]
    thesis = [(0.008, 0.375), (0.055, 0.281), (0.117, 0.180), (0.180, 0.055)]
    print("== Validation on the paper's BSBC(0.01, 0.10), n = 128 binary uses: this simulator's best R1 with")
    print("   R2 at least the thesis point's R2 (staircase over all a and selection rules) vs the thesis's")
    print("   realized polar SUP points (read from its Fig. 4)")
    print("   thesis (R2, R1)   | this simulator: threshold rule | best rule")
    for r2, r1 in thesis:
        thr = max([r["R1"] for r in bs if r["select"] == "threshold" and r["R2"] >= r2 - 1e-9], default=float("nan"))
        best = max([r["R1"] for r in bs if r["R2"] >= r2 - 1e-9], default=float("nan"))
        print(f"   ({r2:.3f}, {r1:.3f})   |  {thr:.3f}                         | {best:.3f}")
except FileNotFoundError:
    pass
