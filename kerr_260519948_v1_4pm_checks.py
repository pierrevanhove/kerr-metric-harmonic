#!/usr/bin/env python3
"""
Comparison of the fourth-order metric with the gauge-term-free recursion of

    P. H. Damgaard, H. Lee, K. Lee and T. Rahnuma, "Iterative Solution of the Kerr
    Black Hole Metric", arXiv:2605.19948 (v1),

for A. Sadu and P. Vanhove,
"The Kerr Metric in Canonical Harmonic Coordinates to All Orders"
(section 5.5 and table of checks).

Usage
-----
    python kerr_260519948_v1_4pm_checks.py [PMAX=20]

Damgaard et al. solve the harmonic-gauge recursion without the gauge term of order
G^2 in the spatial components, and give the term of order G^4 a^p of the result,
h'^{mu nu}, as finite sums of Jacobi polynomials P_k^{(0,beta)}(u) of
u = 1 - 2 z^2/r^2: eq. (5.19), repeated as the first line of (D.19), for h'^{00}, the
second line of (D.19) for h'^{0i}, and (D.20) for the spatial components.  Their
field is defined by sqrt(-g) g^{mu nu} = eta^{mu nu} - h^{mu nu} (their eq. (2.13)),
as hhat here, in harmonic coordinates (t, x, y, z) with the spin along z, and the
comparison is made in units G_N = M = 1.

The formulas are transcribed in their_fourth_order() below.  Eq. (D.20) contains
misprints, and it is read as in the footnote of section 5.5 of the paper:
  * h'^{12} is the coefficient of a^n with n even: for the coefficient of a^{2N} the
    prefactor is x y/r^{2N+6} and the bracket contains P_{N-1}^{(0,3)} and the sum
    over j = 2, ..., N;
  * in h'^{11}, m = N - 1 for the coefficient of a^{2N}, the prefactor is 1/r^{2N+6}
    (printed: 1/r^{2n+8}), and the sum of h'^{12} is added to the coefficient of
    x^2, as required by axial symmetry, h^{11} - h^{22} = (x^2 - y^2) h^{12}/(x y).
The other components are used as printed.

Checks
------
 1. their h'^{mu nu} at order G^4 a^p, p <= PMAX, against the spin expansion of the
    closed forms (h004)-(hij4), all ten components, in exact rational arithmetic;
 2. the same against the output of the cubic recursion at order G^4;
 3. the reading of (D.20): without the sum of h'^{12} in h'^{11} and h'^{22}, the
    comparison fails from order a^4 on;
 4. control: a deliberately wrong coefficient, 59n+127 instead of 59n+126 in (5.19),
    is rejected.

Names such as (h004) are the LaTeX labels of the equations in the source of the
paper, without their prefix "eq:".  Requires kerr_harmonic_checks.py in the same
directory, and python-flint, sympy.
"""
import os
import sys
from functools import lru_cache
from math import factorial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
from flint import fmpq
import kerr_harmonic_checks as khc
from kerr_harmonic_checks import x, y, z, r2, r2pow, sadd, smul, sscale, ssub, ONE, IDX2

def _dfact(n):
    """double factorial n!!"""
    out = 1
    while n > 1:
        out *= n
        n -= 2
    return out

@lru_cache(maxsize=None)
def jacobi(k, beta):
    """P_k^{(0,beta)}(u) with u = 1 - 2 z^2/r^2, standard normalisation, as a scalar (N, 2k) of
    kerr_harmonic_checks, i.e. N(x, y, z)/|x|^{2k}; P_k = 0 for k < 0"""
    if k < 0:
        return None
    u = sp.Symbol('u')
    w = r2 - 2*z*z                                   # r^2 u
    N = None
    for (i,), c in sp.Poly(sp.expand(sp.jacobi(k, 0, beta, u)), u).terms():
        t = (w**i)*r2pow(k - i)*fmpq(int(c.p), int(c.q))
        N = t if N is None else N + t
    return None if N is None else (N, 2*k)

def _combination(terms):
    """sum of c*f over the pairs (c, f)"""
    out = None
    for c, f in terms:
        if f is not None:
            out = sadd(out, sscale(f, c))
    return out

def _times(f, P, m):
    """f * P/|x|^m for a polynomial P"""
    return None if f is None else smul(f, (P, m))

def their_fourth_order(PMAX, read_D20=True, wrong_coefficient=False):
    """The term of order G^4 a^p, p <= PMAX, of the gauge-term-free metric of Damgaard et al.:
    {p: {(mu, nu): scalar}} in the format of kerr_harmonic_checks.run (units G_N = M = 1).

    read_D20=False omits the sum of h'^{12} in h'^{11} and h'^{22} (see the module docstring).
    wrong_coefficient=True replaces 59n+126 by 59n+127 in (5.19) (negative control)."""
    out = {}
    for p in range(PMAX + 1):
        comp = {}
        if p % 2 == 1:
            n = (p - 1)//2
            # (D.19): h'^{0i} = -2 eps^{ij3} x_j P_n^{(0,2)}(u)/r^{2n+6}
            P = jacobi(n, 2)
            comp[(0, 1)] = _times(sscale(P, -2), y, 2*n + 6)
            comp[(0, 2)] = _times(sscale(P, 2), x, 2*n + 6)
        else:
            n = p//2
            # (5.19) = (D.19): h'^{00}
            c = 127 if wrong_coefficient else 126
            terms = [(fmpq(23*n + 48, 3*(n + 2)), jacobi(n, 1)), (fmpq(-(59*n + c), 15*(n + 2)), jacobi(n - 1, 2))]
            terms += [(fmpq(2*n + 3, n + 2)*fmpq(2**(j - 2)*factorial(j), _dfact(2*j + 3)), jacobi(n - j, j + 1))
                      for j in range(2, n + 1)]
            comp[(0, 0)] = _times(_combination(terms), ONE, 2*n + 4)
            if n >= 1:                                   # (D.20); h'^{ij} = 0 at order a^0
                m = n - 1
                Pm = jacobi(m, 3)
                Fm = _combination([(fmpq(2**(j - 1)*factorial(j), _dfact(2*j + 3)), jacobi(m - j, j + 3))
                                   for j in range(1, m + 1)])
                sum12 = _combination([(fmpq(2**(j - 1)*factorial(j - 1), _dfact(2*j + 1)), jacobi(n - j, j + 2))
                                      for j in range(2, n + 1)])
                cx = _combination([(fmpq(1, 3), Pm), (1, Fm)])          # coefficient of x^2 in h'^{11}
                cy = _combination([(fmpq(4, 3), Pm), (-1, Fm)])         # minus coefficient of y^2
                for key, a1, a2 in (((1, 1), x, y), ((2, 2), y, x)):   # h'^{22} = h'^{11} with x <-> y
                    t = _combination([(-1, _times(cx, a1*a1, 0)), (1, _times(cy, a2*a2, 0))])
                    if read_D20:
                        t = sadd(t, _times(sum12, a1*a1, 0))
                    comp[key] = _times(t, ONE, 2*n + 6)
                comp[(3, 3)] = _times(_combination(
                    [(fmpq(n, 3*n + 6), jacobi(n, 1)), (fmpq(6 - n, 15*(n + 2)), jacobi(n - 1, 2))]
                    + [(-fmpq(2*n + 3, n + 2)*fmpq(2**(j - 2)*factorial(j), _dfact(2*j + 3)), jacobi(n - j, j + 1))
                       for j in range(2, n + 1)]), ONE, 2*n + 4)
                comp[(1, 2)] = _times(_combination([(fmpq(-5, 3), jacobi(n - 1, 3)), (1, sum12)]), x*y, 2*n + 6)
                b13 = _combination([(fmpq(2**(j - 1)*factorial(j), _dfact(2*j + 1)), jacobi(n - j, j + 2))
                                    for j in range(1, n + 1)])
                comp[(1, 3)] = _times(sscale(b13, -1), x*z, 2*n + 6)
                comp[(2, 3)] = _times(sscale(b13, -1), y*z, 2*n + 6)    # h'^{23} = h'^{13} with x -> y
        out[p] = {k: khc.sreduce(v) for k, v in comp.items() if v is not None}
    return out

def fourth_order_references(PMAX):
    """the term of order G^4 a^p, p <= PMAX, of the closed forms (h004)-(hij4) (spin expansion) and of the
    cubic recursion, both as {p: {(mu, nu): scalar}}"""
    closed = khc.closed_fields(khc.structure_series(PMAX), 4, PMAX)
    h = khc.run(4, PMAX)[0]
    return {p: closed[(4, p)] for p in range(PMAX + 1)}, {p: h[4][p] for p in range(PMAX + 1)}

def mismatches(theirs, ref, PMAX):
    """list of (p, component) where the two fields differ"""
    return [(p, k) for p in range(PMAX + 1) for k in IDX2 if ssub(theirs[p].get(k), ref[p].get(k)) is not None]

if __name__ == "__main__":
    PMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    closed, recursion = fourth_order_references(PMAX)
    theirs = their_fourth_order(PMAX)
    ncomp = sum(len(v) for v in theirs.values())
    bad1 = mismatches(theirs, closed, PMAX)
    print("1. (5.19), (D.19), (D.20) of Damgaard et al. vs the closed forms (h004)-(hij4), order G^4,")
    print("   p <= %d, all components (%d non-vanishing): %s"
          % (PMAX, ncomp, "exact agreement" if not bad1 else "MISMATCH at %s" % bad1[:6]))
    bad2 = mismatches(theirs, recursion, PMAX)
    print("2. the same vs the cubic recursion: %s" % ("exact agreement" if not bad2 else "MISMATCH at %s" % bad2[:6]))
    lit = mismatches(their_fourth_order(PMAX, read_D20=False), closed, PMAX)
    orders = sorted({p for p, _ in lit})
    needed = PMAX < 4 or (orders[:1] == [4] and sorted({k for _, k in lit}) == [(1, 1), (2, 2)])
    print("3. reading of (D.20): without the sum of h'^{12} in h'^{11}, h'^{22} the comparison fails at %d"
          " orders,\n   in h'^{11} and h'^{22} only, from a^4 on: %s" % (len(orders), needed))
    rejected = bool(mismatches(their_fourth_order(PMAX, wrong_coefficient=True), closed, PMAX))
    print("   " + khc.control_line("59n+127 instead of 59n+126 in (5.19)", rejected))
    sys.exit(0 if not bad1 and not bad2 and needed and rejected else 1)
