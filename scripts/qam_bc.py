"""Two-user complex AWGN broadcast channel with a 4-point (2-bit) constellation.

Superposition (SUP): the cloud bit v (weak user 2) and the satellite bit s (strong user 1) select
    x(v, s) = sqrt(1-beta) (-1)^v + sqrt(beta) e^{j theta} (-1)^s,      E|x|^2 = 1.
theta = 90 deg gives a rectangular 4QAM (I/Q split), theta = 0 a one-dimensional, 4-PAM-like
hierarchical constellation.  Baselines: TDM with Gray QPSK at equal power, and TDM with power
control (lambda P1 + (1-lambda) P2 = 1).

Noise is CN(0, 1/snr).  Expectations over the noise use 2-D Gauss-Hermite quadrature.
All rates are in bits per complex channel use.
"""
import math

import numpy as np

from normal_approx import F, Qinv

_GH_X, _GH_W = np.polynomial.hermite.hermgauss(48)
_ZX, _ZY = np.meshgrid(math.sqrt(2) * _GH_X, math.sqrt(2) * _GH_X, indexing="ij")
_ZW = np.outer(_GH_W, _GH_W).ravel() / math.pi           # weights for (xi1, xi2) ~ N(0, I)
_Z = (_ZX + 1j * _ZY).ravel()                              # unit-variance-per-dimension samples


def _logsumexp(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def sup_points(beta, theta):
    """points[v, s]"""
    a, b = math.sqrt(1 - beta), math.sqrt(beta) * np.exp(1j * theta)
    return np.array([[a + b, a - b], [-a + b, -a - b]])


def _loglik(y, pts, snr):
    # natural-log likelihood up to a constant common to all hypotheses
    return -np.abs(y[..., None] - pts.ravel()) ** 2 * snr


def sup_moments(beta, theta, snr1, snr2):
    """Means/variances of i(V;Y2), i(V;Y1), i(S;Y1|V) and Cov[i(V;Y1), i(S;Y1|V)] in bits."""
    pts = sup_points(beta, theta)
    out = {}
    for user, snr in ((2, snr2), (1, snr1)):
        sig = math.sqrt(1 / (2 * snr))
        ic_all, is_all, w_all = [], [], []
        for v in (0, 1):
            for s in (0, 1):
                y = pts[v, s] + sig * _Z
                ll = _loglik(y, pts, snr).reshape(-1, 2, 2)          # [node, v', s']
                l_y = _logsumexp(ll.reshape(-1, 4), 1) - math.log(4)
                l_yv = _logsumexp(ll[:, v, :], 1) - math.log(2)
                l_yvs = ll[:, v, s]
                ic_all.append((l_yv - l_y) / math.log(2))
                is_all.append((l_yvs - l_yv) / math.log(2))
                w_all.append(_ZW / 4)
        ic, is_, w = np.concatenate(ic_all), np.concatenate(is_all), np.concatenate(w_all)
        mc, ms = w @ ic, w @ is_
        out[user] = dict(I_c=mc, V_c=w @ (ic - mc) ** 2, I_s=ms, V_s=w @ (is_ - ms) ** 2,
                         V_cs=w @ ((ic - mc) * (is_ - ms)))
    return out


def qpsk_moments(snr):
    """Capacity and dispersion of Gray QPSK (two independent BPSK dimensions)."""
    pts = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / math.sqrt(2)
    sig = math.sqrt(1 / (2 * snr))
    vals, w_all = [], []
    for x in pts:
        y = x + sig * _Z
        ll = _loglik(y, pts, snr)
        vals.append((ll[:, list(pts).index(x)] - (_logsumexp(ll, 1) - math.log(4))) / math.log(2))
        w_all.append(_ZW / 4)
    i, w = np.concatenate(vals), np.concatenate(w_all)
    m = w @ i
    return m, w @ (i - m) ** 2


# ---------------------------------------------------------------- regions
def sup_region_R1(n, eps1, eps2, snr1, snr2, R2, betas, thetas, asymptotic=False):
    """Largest R1 for each R2 (normal approximation, per-user reliability, SIC at user 1)."""
    best = np.full(len(R2), -np.inf)
    arg = [None] * len(R2)
    for th in thetas:
        for be in betas:
            m = sup_moments(be, th, snr1, snr2)
            u2, u1 = m[2], m[1]
            if asymptotic:
                ok = R2 <= min(u2["I_c"], u1["I_c"])
                R1 = np.where(ok, u1["I_s"], -np.inf)
            else:
                ok = R2 <= u2["I_c"] - math.sqrt(u2["V_c"] / n) * Qinv(eps2)
                c_c = (u1["I_c"] - R2) / math.sqrt(u1["V_c"] / n)
                rho = u1["V_cs"] / math.sqrt(u1["V_c"] * u1["V_s"])
                lo, hi = np.full(len(R2), -12.0), np.full(len(R2), 12.0)
                for _ in range(45):
                    mid = (lo + hi) / 2
                    good = F(c_c, mid, rho) >= 1 - eps1
                    hi, lo = np.where(good, mid, hi), np.where(good, lo, mid)
                ok &= hi < 11.9
                R1 = np.where(ok, np.maximum(u1["I_s"] - math.sqrt(u1["V_s"] / n) * hi, 0), -np.inf)
            for k in np.nonzero(R1 > best)[0]:
                best[k], arg[k] = R1[k], (be, th)
    return best, arg


_QPSK_CACHE = {}


def _qpsk(snr):
    key = round(float(snr), 12)
    if key not in _QPSK_CACHE:
        _QPSK_CACHE[key] = qpsk_moments(snr)
    return _QPSK_CACHE[key]


def tdm_region_R1(n, eps1, eps2, snr1, snr2, R2, power_control=False, asymptotic=False,
                  lams=np.linspace(0.0, 1.0, 201), p1s=np.linspace(0.05, 4.0, 80)):
    """Largest R1 for each R2 with TDM (Gray QPSK in each slot)."""
    def rate(frac, snr, eps):
        if frac <= 0:
            return 0.0
        C, V = _qpsk(snr)
        if asymptotic:
            return frac * C
        return max(frac * C - math.sqrt(frac * V / n) * Qinv(eps), 0.0)

    pts = []
    for lam in lams:
        powers = [(1.0, 1.0)]
        if power_control and 0 < lam < 1:
            powers = [(p1, (1 - lam * p1) / (1 - lam)) for p1 in p1s if lam * p1 < 1]
        for P1, P2 in powers:
            pts.append((rate(1 - lam, P2 * snr2, eps2), rate(lam, P1 * snr1, eps1)))
    pts = np.array(pts)
    return np.array([pts[pts[:, 0] >= x, 1].max(initial=-np.inf) for x in R2])
