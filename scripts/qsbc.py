"""Quaternary symmetric broadcast channel QSBC(p1, p2): Y_j = X + Z_j (mod 4) with
P(Z_j = 0) = 1 - p_j and P(Z_j = z) = p_j / 3 for z != 0.  The 4-ary analogue of the BSBC.

Superposition (SUP) uses a cloud V ~ P_V and a satellite channel P_{X|V}; user 2 decodes V treating
the satellite as noise, user 1 uses SIC.  Exact information-density moments come from enumeration;
the normal approximation is that of Proposition 1 (per-user reliability).  The TDM baseline time
shares point-to-point QSC codes with uniform input.  Rates are in bits per channel use (per symbol).
"""
import math

import numpy as np

from normal_approx import F, Qinv


def qsc(p):
    return np.full((4, 4), p / 3) + np.eye(4) * (1 - p - p / 3)


def qsc_capacity(p):
    h = -(1 - p) * math.log2(1 - p) - p * math.log2(p / 3) if p > 0 else 0.0
    return 2 - h


def qsc_dispersion(p):
    return p * (1 - p) * math.log2((1 - p) * 3 / p) ** 2


def sup_moments(PV, W, p1, p2):
    """PV: (k,), W: (k, 4) = P_{X|V}.  Returns means/variances/covariance of the densities (bits)."""
    return sup_moments_mat(PV, W, qsc(p1), qsc(p2))


def sup_moments_mat(PV, W, Q1, Q2):
    """As sup_moments, for arbitrary channel matrices Q_j[x, y] = P(y | x) of the two users."""
    out = {}
    for user, Q in ((2, Q2), (1, Q1)):
        Pyv = W @ Q                         # (k, 4)  P(y|v)
        Py = PV @ Pyv                       # (4,)
        w = PV[:, None, None] * W[:, :, None] * Q[None, :, :]          # P(v, x, y)
        with np.errstate(divide="ignore", invalid="ignore"):
            ic = np.log2(Pyv / Py)[:, None, :] * np.ones_like(w)
            is_ = np.log2(Q[None, :, :] / Pyv[:, None, :]) * np.ones_like(w)
        mask = w > 0
        ic, is_, w = ic[mask], is_[mask], w[mask]
        mc, ms = w @ ic, w @ is_
        out[user] = dict(I_c=mc, V_c=w @ (ic - mc) ** 2, I_s=ms, V_s=w @ (is_ - ms) ** 2,
                         V_cs=w @ ((ic - mc) * (is_ - ms)))
    return out


# ---------------------------------------------------------------- superposition families
def fam_symmetric(a):
    """V uniform on Z4, X = V + S with S q-ary symmetric: P(S=0) = 1-a, P(S=s) = a/3."""
    return np.full(4, 0.25), qsc(a)


def fam_subgroup(a):
    """V uniform on Z4, X = V + S with S in {0, 2}, P(S = 2) = a."""
    return np.full(4, 0.25), (1 - a) * np.eye(4) + a * np.roll(np.eye(4), 2, axis=1)


def fam_bitlayer(a):
    """V binary uniform selects the pair {V, V+2}; the satellite picks V+2 with probability a."""
    W = np.zeros((2, 4)); W[0, 0], W[0, 2], W[1, 1], W[1, 3] = 1 - a, a, 1 - a, a
    return np.full(2, 0.5), W


FAMILIES = {"symmetric": (fam_symmetric, np.linspace(0.002, 0.75, 120)),
            "subgroup": (fam_subgroup, np.linspace(0.002, 0.5, 80)),
            "bitlayer": (fam_bitlayer, np.linspace(0.002, 0.5, 80))}


def sup_R1(n, eps1, eps2, p1, p2, R2, configs, asymptotic=False):
    """Largest R1 for each R2 over a list of (label, PV, W) configurations."""
    return sup_R1_mat(n, eps1, eps2, qsc(p1), qsc(p2), R2, configs, asymptotic)


def sup_R1_mat(n, eps1, eps2, Q1, Q2, R2, configs, asymptotic=False):
    """As sup_R1, for arbitrary 4x4 channel matrices."""
    best = np.full(len(R2), -np.inf); arg = [None] * len(R2)
    for label, PV, W in configs:
        m = sup_moments_mat(PV, W, Q1, Q2); u2, u1 = m[2], m[1]
        if asymptotic:
            R1 = np.where(R2 <= min(u2["I_c"], u1["I_c"]) + 1e-12, u1["I_s"], -np.inf)
        else:
            ok = R2 <= u2["I_c"] - math.sqrt(u2["V_c"] / n) * Qinv(eps2)
            if u1["V_c"] < 1e-12 or u1["V_s"] < 1e-12:
                continue
            c_c = (u1["I_c"] - R2) / math.sqrt(u1["V_c"] / n)
            rho = np.clip(u1["V_cs"] / math.sqrt(u1["V_c"] * u1["V_s"]), -0.999999, 0.999999)
            lo, hi = np.full(len(R2), -12.0), np.full(len(R2), 12.0)
            for _ in range(45):
                mid = (lo + hi) / 2
                good = F(c_c, mid, rho) >= 1 - eps1
                hi, lo = np.where(good, mid, hi), np.where(good, lo, mid)
            ok &= hi < 11.9
            R1 = np.where(ok, np.maximum(u1["I_s"] - math.sqrt(u1["V_s"] / n) * hi, 0), -np.inf)
        for k in np.nonzero(R1 > best)[0]:
            best[k], arg[k] = R1[k], label
    return best, arg


def tdm_R1(n, eps1, eps2, p1, p2, R2, asymptotic=False, lams=np.linspace(0, 1, 4001)):
    return tdm_R1_cv(n, eps1, eps2, (qsc_capacity(p1), qsc_dispersion(p1)),
                     (qsc_capacity(p2), qsc_dispersion(p2)), R2, asymptotic, lams)


def tdm_R1_cv(n, eps1, eps2, cv1, cv2, R2, asymptotic=False, lams=np.linspace(0, 1, 4001)):
    """TDM from each user's point-to-point (capacity, dispersion)."""
    def r(frac, cv, eps):
        C, V = cv
        val = frac * C - (0 if asymptotic else np.sqrt(frac * V / n) * Qinv(eps))
        return np.where(frac > 0, np.maximum(val, 0), 0.0)
    r1, r2 = r(lams, cv1, eps1), r(1 - lams, cv2, eps2)
    return np.array([np.max(np.where(r2 >= x, r1, -np.inf)) for x in R2])


def family_configs():
    return [(f"{name} a={a:.3f}", *fn(a)) for name, (fn, grid) in FAMILIES.items() for a in grid]
