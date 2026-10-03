"""Polar codes on the two-user 4-point AWGN broadcast channel of qam_bc.py.

* SUP: a cloud polar code (user 2) and a satellite polar code (user 1), each of length n, map one bit
  each to x(v, s) = sqrt(1-beta)(-1)^v + sqrt(beta) e^{j theta}(-1)^s.  User 2 decodes the cloud
  treating the satellite as noise; user 1 decodes the cloud, re-encodes it, then decodes the satellite.
* TDM: user 1 gets lam*n symbols of Gray QPSK (2 coded bits per symbol), user 2 the rest.

Codes use G_n = F^{(x)m} without bit reversal and exact successive-cancellation (SC) decoding.  Each
layer's information set comes from a genie-aided Monte-Carlo construction on that layer's channel.
"""
import math

import numpy as np

LN2 = math.log(2)


def polar_encode(u):
    x = u.copy()
    n = x.shape[-1]
    step = 1
    while step < n:
        for i in range(0, n, 2 * step):
            x[..., i:i + step] ^= x[..., i + step:i + 2 * step]
        step *= 2
    return x


def _boxplus(a, b):
    return (np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))
            + np.log1p(np.exp(-np.abs(a + b))) - np.log1p(np.exp(-np.abs(a - b))))


def sc_decode(llr, frozen, genie=None):
    """llr: (B, n) channel LLRs log P(x=0)/P(x=1); frozen: bool (n,) (frozen bits are 0).
    With genie=u (B, n), previous decisions are replaced by the true bits and the hard decision
    of every bit is returned, which gives the genie-aided bit-channel error rates.
    Returns u_hat (B, n)."""
    B, n = llr.shape
    u_hat = np.zeros((B, n), dtype=np.uint8)

    def rec(L, lo):
        N = L.shape[1]
        if N == 1:
            i = lo
            dec = (L[:, 0] < 0).astype(np.uint8)
            if genie is not None:
                u_hat[:, i] = dec
                return genie[:, i:i + 1].copy()
            dec = np.zeros(B, np.uint8) if frozen[i] else dec
            u_hat[:, i] = dec
            return dec[:, None]
        h = N // 2
        a = rec(_boxplus(L[:, :h], L[:, h:]), lo)
        b = rec(L[:, h:] + (1 - 2 * a.astype(np.int8)) * L[:, :h], lo + h)
        return np.concatenate([a ^ b, b], axis=1)

    rec(llr, 0)
    return u_hat


def _lse(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def sup_points(beta, theta):
    a, b = math.sqrt(1 - beta), math.sqrt(beta) * np.exp(1j * theta)
    return np.array([[a + b, a - b], [-a + b, -a - b]])          # [v, s]


class SupLink:
    def __init__(self, beta, theta, snr1, snr2, rng):
        self.pts, self.snr = sup_points(beta, theta), {1: snr1, 2: snr2}
        self.rng = rng

    def channel(self, v, s, user):
        x = self.pts[v, s]
        sig = math.sqrt(1 / (2 * self.snr[user]))
        return x + sig * (self.rng.standard_normal(x.shape) + 1j * self.rng.standard_normal(x.shape))

    def llr_cloud(self, y, user):
        ll = -np.abs(y[..., None, None] - self.pts) ** 2 * self.snr[user]   # [..., v, s]
        return _lse(ll[..., 0, :], -1) - _lse(ll[..., 1, :], -1)

    def llr_sat(self, y, v, user=1):
        d0 = np.abs(y - self.pts[v, 0]) ** 2
        d1 = np.abs(y - self.pts[v, 1]) ** 2
        return (d1 - d0) * self.snr[user]


def qpsk_llr(y, snr):
    """Gray QPSK, unit energy: two independent BPSK dimensions, interleaved as (I, Q) per symbol."""
    s = 2 * math.sqrt(2) * snr
    return np.stack([y.real * s, y.imag * s], axis=-1).reshape(y.shape[0], -1)


def qpsk_mod(bits):
    b = bits.reshape(bits.shape[0], -1, 2)
    return ((1 - 2.0 * b[..., 0]) + 1j * (1 - 2.0 * b[..., 1])) / math.sqrt(2)


def awgn(x, snr, rng):
    sig = math.sqrt(1 / (2 * snr))
    return x + sig * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))


def genie_order(llr_fn, n, rng, blocks=20000, batch=2000):
    """Bit-channel reliability order (most reliable last) by genie-aided SC on a given channel.
    llr_fn(x_bits) -> channel LLRs for coded bits x_bits (B, n)."""
    err = np.zeros(n)
    for _ in range(blocks // batch):
        u = rng.integers(0, 2, (batch, n), dtype=np.uint8)
        x = polar_encode(u)
        uh = sc_decode(llr_fn(x), np.zeros(n, bool), genie=u)
        err += (uh != u).sum(0)
    return np.argsort(-err, kind="stable")                      # least reliable first


def frozen_from_order(order, k):
    fr = np.ones(len(order), bool)
    fr[order[len(order) - k:]] = False
    return fr
