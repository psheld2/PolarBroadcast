"""Normal approximations for the two-user binary symmetric broadcast channel (BSBC).

Reproduces the analytical quantities used in the paper:
  * the information-density moments of Proposition 1 (checked against exact enumeration),
  * the per-user-reliability normal approximations of superposition coding (SUP),
    time-division multiplexing (TDM) and a common codeword (CCP),
  * Fig. 5: the largest R1 advantage of SUP over TDM as a function of blocklength.

All logarithms are base 2; third-order (log n / n) terms are omitted for every scheme.

Usage:  python3 scripts/normal_approx.py      (writes figures/crossover.pdf and figures/crossover.csv)
"""
import itertools
import math
import os
from statistics import NormalDist

import numpy as np

_N = NormalDist()
LOG2 = np.log2


def h(x):
    x = np.clip(x, 1e-300, 1 - 1e-16)
    return -x * LOG2(x) - (1 - x) * LOG2(1 - x)


def star(a, b):
    return a * (1 - b) + (1 - a) * b


def V_bsc(q):
    """Dispersion of a BSC(q) with uniform input."""
    return q * (1 - q) * LOG2((1 - q) / q) ** 2


def Qinv(eps):
    return _N.inv_cdf(1 - eps)


def _erfc(x):
    # Numerical Recipes erfcc: fractional error < 1.2e-7 everywhere.
    z = np.abs(x)
    t = 1 / (1 + 0.5 * z)
    r = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (
        -0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (
            -0.82215223 + t * 0.17087277)))))))))
    return np.where(x >= 0, r, 2 - r)


def Phi(x):
    return 0.5 * _erfc(-x / math.sqrt(2))


_GL_X, _GL_W = np.polynomial.legendre.leggauss(96)


def F(c_a, c_b, rho, lo=-10.0):
    """P[G1 <= c_a, rho*G1 + sqrt(1-rho^2)*G2 <= c_b] for i.i.d. standard Gaussians G1, G2.

    Gauss-Legendre quadrature over G1 in [lo, c_a]; broadcasts over array arguments.
    """
    c_a, c_b, rho = np.broadcast_arrays(*(np.asarray(v, float) for v in (c_a, c_b, rho)))
    a = np.clip(c_a, lo, 10.0)[..., None]
    x = lo + (a - lo) * (_GL_X + 1) / 2
    w = (a - lo) / 2 * _GL_W
    pdf = np.exp(-x ** 2 / 2) / math.sqrt(2 * math.pi)
    inner = Phi((c_b[..., None] - rho[..., None] * x) / np.sqrt(1 - rho[..., None] ** 2))
    return np.sum(w * pdf * inner, axis=-1)


# ---------------------------------------------------------------- Proposition 1 moments
def strong_user_moments(p1, alpha):
    """Means, variances and covariance of i(V;Y1) (cloud) and i(X;Y1|V) (satellite)."""
    q = star(alpha, p1)
    Lp, Lq = LOG2((1 - p1) / p1), LOG2((1 - q) / q)
    I_c, I_s = 1 - h(q), h(q) - h(p1)
    V_c = q * (1 - q) * Lq ** 2
    V_s = p1 * (1 - p1) * Lp ** 2 + V_c - 2 * (1 - 2 * alpha) * p1 * (1 - p1) * Lp * Lq
    V_cs = (1 - 2 * alpha) * p1 * (1 - p1) * Lp * Lq - V_c
    return I_c, I_s, V_c, V_s, V_cs


def check_moments():
    """Compare the closed forms with exact enumeration over (V, S, Z1)."""
    worst = 0.0
    for p1, alpha in [(0.01, 0.1), (0.05, 0.3), (0.11, 0.02), (0.2, 0.45), (0.01, 0.25)]:
        q = star(alpha, p1)
        rows = []
        for v, s, z in itertools.product([0, 1], repeat=3):
            pr = 0.5 * (alpha if s else 1 - alpha) * (p1 if z else 1 - p1)
            x = v ^ s
            y = x ^ z
            i_c = 1 + math.log2(1 - q if y == v else q)
            i_s = math.log2(1 - p1 if y == x else p1) - math.log2(1 - q if y == v else q)
            rows.append((pr, i_c, i_s))
        E = lambda f: sum(r[0] * f(r) for r in rows)
        m_c, m_s = E(lambda r: r[1]), E(lambda r: r[2])
        exact = (m_c, m_s, E(lambda r: (r[1] - m_c) ** 2), E(lambda r: (r[2] - m_s) ** 2),
                 E(lambda r: (r[1] - m_c) * (r[2] - m_s)))
        worst = max(worst, max(abs(a - b) for a, b in zip(exact, strong_user_moments(p1, alpha))))
    assert worst < 1e-12, worst
    print(f"moment closed forms match enumeration (max abs error {worst:.1e})")


# ---------------------------------------------------------------- rate regions
def sup_R1(n, eps1, eps2, p1, p2, R2, alphas=np.linspace(0.0005, 0.4995, 200)):
    """Largest R1 for each R2 in the SUP normal approximation (Proposition 1)."""
    A, RR = np.meshgrid(alphas, R2, indexing="ij")
    I_c, I_s, V_c, V_s, V_cs = strong_user_moments(p1, A)
    rho = V_cs / np.sqrt(V_c * V_s)
    q2 = star(A, p2)
    feasible = RR <= 1 - h(q2) - np.sqrt(V_bsc(q2) / n) * Qinv(eps2)
    c_c = (I_c - RR) / np.sqrt(V_c / n)          # Q^{-1}(eps_c) at equality in the cloud constraint
    lo, hi = np.full(A.shape, -12.0), np.full(A.shape, 12.0)
    for _ in range(45):                          # bisection on c_s = Q^{-1}(eps_s)
        mid = (lo + hi) / 2
        ok = F(c_c, mid, rho) >= 1 - eps1
        hi, lo = np.where(ok, mid, hi), np.where(ok, lo, mid)
    feasible &= hi < 11.9
    R1 = np.where(feasible, np.maximum(I_s - np.sqrt(V_s / n) * hi, 0), -np.inf)
    return R1.max(axis=0)


def tdm_R1(n, eps1, eps2, p1, p2, R2, lams=np.linspace(0, 1, 4001)):
    """Largest R1 for each R2 when user 1 gets a fraction lam of the n channel uses."""
    r1 = lams * (1 - h(p1)) - np.sqrt(lams * V_bsc(p1) / n) * Qinv(eps1)
    r2 = (1 - lams) * (1 - h(p2)) - np.sqrt((1 - lams) * V_bsc(p2) / n) * Qinv(eps2)
    r2 = np.where(lams == 1, 0.0, r2)            # lam = 1: no message for user 2
    return np.array([np.max(np.where(r2 >= x, np.maximum(r1, 0), -np.inf)) for x in R2])


def ccp_sum_rate(n, eps1, eps2, p1, p2):
    """Both messages in one codeword that both receivers decode."""
    return min(1 - h(p) - math.sqrt(V_bsc(p) / n) * Qinv(e) for p, e in ((p1, eps1), (p2, eps2)))


def sup_advantage(n, eps, p1, p2):
    """max over R2 of R1_SUP(R2) - R1_TDM(R2); 0 means the SUP region lies inside the TDM region."""
    R2 = np.linspace(0, 1 - h(p2), 150)
    gap = sup_R1(n, eps, eps, p1, p2, R2) - tdm_R1(n, eps, eps, p1, p2, R2)
    i = np.argmax(np.where(np.isfinite(gap), gap, -np.inf))
    return max(gap[i], 0.0), R2[i]


def main():
    check_moments()
    p1, p2 = 0.01, 0.10
    R2 = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
    print("n=128, eps1=eps2=1e-2 (compare with Fig. 4):")
    print("  R2      ", R2)
    print("  SUP R1  ", np.round(sup_R1(128, 1e-2, 1e-2, p1, p2, R2), 3))
    print("  TDM R1  ", np.round(tdm_R1(128, 1e-2, 1e-2, p1, p2, R2), 3))
    print("  CCP R1+R2", round(ccp_sum_rate(128, 1e-2, 1e-2, p1, p2), 3))

    ms = np.arange(7, 21)
    epss = [1e-2, 1e-3]
    table = {e: [sup_advantage(2 ** m, e, p1, p2) for m in ms] for e in epss}
    here = os.path.dirname(os.path.abspath(__file__))
    figdir = os.path.join(here, "..", "figures")
    with open(os.path.join(figdir, "crossover.csv"), "w") as f:
        f.write("log2n," + ",".join(f"gain_eps{e:g},R2_eps{e:g}" for e in epss) + "\n")
        for k, m in enumerate(ms):
            f.write(f"{m}," + ",".join(f"{table[e][k][0]:.5f},{table[e][k][1]:.4f}" for e in epss) + "\n")
    for e in epss:
        print(f"eps={e:g}: " + "  ".join(f"2^{m}:{g:.4f}" for m, (g, _) in zip(ms, table[e])))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 8})
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    for e, mk in zip(epss, ["o", "s"]):
        ax.plot(ms, [g for g, _ in table[e]], marker=mk, ms=3.5, lw=1,
                label=rf"$\varepsilon_1=\varepsilon_2=10^{{{int(round(math.log10(e)))}}}$")
    ax.set_xlabel(r"$\log_2 n$")
    ax.set_ylabel(r"$\max_{R_2}\,[R_1^{\rm SUP}-R_1^{\rm TDM}]$ (bits/ch. use)")
    ax.set_xticks(ms[::2])
    ax.grid(True, lw=0.4, alpha=0.6)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(figdir, "crossover.pdf"))
    print("wrote figures/crossover.pdf")


if __name__ == "__main__":
    main()
