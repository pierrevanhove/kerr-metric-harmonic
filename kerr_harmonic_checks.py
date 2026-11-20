#!/usr/bin/env python3
"""
Recursion and checks for

    A. Sadu and P. Vanhove,
    "The Kerr Metric in Canonical Harmonic Coordinates to All Orders".

Usage
-----
    python kerr_harmonic_checks.py [NMAX=5] [PMAX=12]

or, from Python / the notebook checks.ipynb,

    import kerr_harmonic_checks as khc
    h, A, Y, slots = khc.run(5, 12)

Contents
--------
 1. Exact rational arithmetic for functions N(x,y,z)/|x|^m, and the inverse
    Laplacian (invlap) acting on H_l(x)/|x|^mu, with an explicit resonance check.
 2. The cubic recursion (recY)-(recJ) in position space, started from the Kerr
    multipole seed (h1), with no homogeneous solution added. At every order
    G^n a^p it asserts the equation of the auxiliary field, the Poisson
    equation of the metric, the harmonic gauge condition and the absence of
    resonances.
 3. The spin expansion of the closed forms (h1), (h00PM)-(hijPM), (h004)-(hij4)
    and (h005), compared with the output of the recursion as rational functions.
 4. The exact gothic metric: direct tensor-density transformation of the
    Boyer-Lindquist metric to the chart (chart) or to the Cook-Scheel chart (CS),
    compared with the formulas (gexact1)-(gexact3), (CSframe); the closed forms as
    the expansion of (gexact1)-(gexact3) in G_N M at fixed harmonic point, both
    symbolically and through Taylor coefficients of the transformed metric.
 5. The spin-expansion coefficients of appendix C, which are those of the file
    kerr-harmonic-coefficients.m (with other values of NMAX, PMAX a new file is
    written).

Names in parentheses, such as (recY), (invlap) or (chart), are the LaTeX labels
of the corresponding equations in the source of the paper, without the prefix "eq:".

Conventions
-----------
Signature (-,+,+,+), D = 4, spin along z.  hhat^{mu nu} = eta^{mu nu} - sqrt(-g) g^{mu nu}.
The recursion uses units G_N = M = 1: the term of order G_N^n M^n a^p of a field
is stored as h[n][p], a dictionary {(mu, nu): (N, m)} with mu <= nu, where the
pair (N, m) stands for N(x,y,z)/|x|^m and N is a polynomial with rational
coefficients.  A component that vanishes is absent from the dictionary.

Requires python-flint, sympy and mpmath.
"""
import os
import sys
import time
from math import factorial
from functools import reduce

from flint import fmpq_mpoly_ctx, Ordering, fmpq
from mpmath import (mp, mpf, sqrt, sin, cos, atan, atan2, acos, log, matrix, det,
                    diff, findroot, taylor, quad, inf)

# =====================================================================================
# 1. Rational functions:  a scalar is (N, m) = N(x,y,z)/|x|^m ; None is zero
# =====================================================================================
ctx = fmpq_mpoly_ctx.get(("x", "y", "z"), Ordering.degrevlex)
X = ctx.gens()
x, y, z = X
r2 = x*x + y*y + z*z
ONE = ctx.from_dict({(0, 0, 0): 1})
SGN = (-1, 1, 1, 1)                                             # eta = diag(-1,1,1,1)
IDX2 = [(m, n) for m in range(4) for n in range(m, 4)]          # symmetric pairs (mu <= nu)
IDX3 = [(l, m, n) for l in range(4) for (m, n) in IDX2]         # A^l_{mn}

_r2cache = {0: ONE}
def r2pow(k):
    if k not in _r2cache:
        _r2cache[k] = r2pow(k - 1)*r2
    return _r2cache[k]

def sadd(f, g):
    if f is None:
        return g
    if g is None:
        return f
    N1, m1 = f
    N2, m2 = g
    if m1 != m2:
        assert (m1 - m2) % 2 == 0, "parity mismatch in sum"
        if m1 < m2:
            N1 = N1*r2pow((m2 - m1)//2); m1 = m2
        else:
            N2 = N2*r2pow((m1 - m2)//2)
    N = N1 + N2
    return None if N.is_zero() else (N, m1)

def sscale(f, c):
    if f is None or c == 0:
        return None
    return (f[0]*c, f[1])

def ssub(f, g):
    return sadd(f, sscale(g, -1))

def smul(f, g):
    if f is None or g is None:
        return None
    return (f[0]*g[0], f[1] + g[1])

def sdiff(f, i):
    """d/dx^i, i = 1,2,3"""
    if f is None:
        return None
    N, m = f
    Nn = r2*N.derivative(i - 1) - m*X[i - 1]*N
    return None if Nn.is_zero() else (Nn, m + 2)

def sreduce(f):
    """cancel common factors of |x|^2"""
    if f is None:
        return None
    N, m = f
    while True:
        q, rem = divmod(N, r2)
        if rem.is_zero() and not N.is_zero():
            N, m = q, m - 2
        else:
            break
    return (N, m)

def slap(f):
    """Laplacian of N/|x|^m"""
    t = None
    for i in (1, 2, 3):
        t = sadd(t, sdiff(sdiff(f, i), i))
    return sreduce(t)

def sstr(f):
    """readable form of a scalar"""
    if f is None:
        return "0"
    f = sreduce(f)
    N, m = f
    if m == 0:
        return str(N)
    num = str(N)
    if len(N) > 1:
        num = "(" + num + ")"
    return "%s/|x|^%d" % (num, m)

class Acc:
    """accumulator for sums of N/|x|^m, grouped by m"""
    def __init__(self):
        self.d = {}
    def add(self, key, f, c=1):
        if f is None:
            return
        N, m = f
        dd = self.d.setdefault(key, {})
        t = N if c == 1 else N*c
        dd[m] = dd[m] + t if m in dd else t
    def get(self, key):
        dd = self.d.get(key)
        if not dd:
            return None
        mmax = max(dd)
        tot = None
        for m, N in dd.items():
            assert (mmax - m) % 2 == 0, "parity mismatch in accumulator"
            t = N if m == mmax else N*r2pow((mmax - m)//2)
            tot = t if tot is None else tot + t
        return None if tot.is_zero() else sreduce((tot, mmax))

# ----------------------------------------------------------------- inverse Laplacian
def lap(N):
    return N.derivative(0).derivative(0) + N.derivative(1).derivative(1) + N.derivative(2).derivative(2)

def harmonic_decomp(N):
    """homogeneous N -> [(j, l, H_l)], N = sum_j |x|^{2j} H_l, l = deg N - 2j, H_l harmonic and non-zero"""
    comps = []
    j = 0
    cur = N
    while not cur.is_zero():
        d = cur.total_degree()
        H = cur
        L = cur
        coef = fmpq(1)
        k = 0
        while True:
            L = lap(L)
            k += 1
            if L.is_zero():
                break
            coef = coef*fmpq(-1, 2*k*(2*d - 2*k + 1))
            H = H + r2pow(k)*L*coef
        if not H.is_zero():
            comps.append((j, d, H))
        rem = cur - H
        if rem.is_zero():
            break
        q, rr = divmod(rem, r2)
        assert rr.is_zero()
        cur = q
        j += 1
    return comps

class Resonance(Exception):
    """raised when a source term H_l/|x|^(2l+3) is met: its inverse Laplacian needs a logarithm"""

def inv_lap(f, tag=None):
    """eq. (invlap): Delta^{-1} H_l/|x|^mu = H_l/|x|^{mu-2} / ((mu-2)(mu-2l-3)); no homogeneous solution added"""
    if f is None:
        return None
    N, m = f
    acc = Acc()
    for (j, l, H) in harmonic_decomp(N):
        mu = m - 2*j
        den = (mu - 2)*(mu - 2*l - 3)
        if den == 0:
            raise Resonance("resonant source: l=%d, mu=%d, at %s" % (l, mu, tag))
        acc.add(0, (H*fmpq(1, den), mu - 2))
    return acc.get(0)

def resonant_slot(f):
    """(l, True/False) if the harmonic degree l that would be resonant lies within the range of degrees
    of the source N/|x|^m (True if its component is non-zero); (None, False) otherwise"""
    if f is None:
        return (None, False)
    N, m = f
    d = N.total_degree()
    num = 2*d + 3 - m                      # mu = 2l+3 with mu = m-2j, l = d-2j  ->  j = (2d+3-m)/2
    if num % 2 or num < 0 or num//2 > d//2:
        return (None, False)
    j0 = num//2
    for (j, l, H) in harmonic_decomp(N):
        if j == j0:
            return (l, True)
    return (d - 2*j0, False)

# =====================================================================================
# 2. Tensors of the cubic formulation and the recursion (recY)-(recJ)
# =====================================================================================
def hget(h, m, n):
    return h.get((m, n) if m <= n else (n, m))

def aget(A, l, m, n):
    return A.get((l, m, n) if m <= n else (l, n, m))

def A_lin(h):
    """eq. (Alin) in D=4: A^l_{mn} = -1/2 eta^{ls}(d_m h_{ns} + d_n h_{ms} - d_s h_{mn} + 1/2 eta_{mn} d_s h)"""
    hl = {k: sscale(v, SGN[k[0]]*SGN[k[1]]) for k, v in h.items()}
    tr = None
    for mu in range(4):
        tr = sadd(tr, sscale(hget(h, mu, mu), SGN[mu]))
    dh = {}
    for k, v in hl.items():
        for s in (1, 2, 3):
            t = sdiff(v, s)
            if t is not None:
                dh[(s,) + k] = t
    dtr = {s: sdiff(tr, s) for s in (1, 2, 3)}
    def D(s, m, n):
        if s == 0:
            return None
        return dh.get((s, m, n) if m <= n else (s, n, m))
    A = {}
    for (l, m, n) in IDX3:
        t = sadd(D(m, n, l), D(n, m, l))
        t = ssub(t, D(l, m, n))
        if m == n and l != 0:
            t = sadd(t, sscale(dtr[l], fmpq(SGN[m], 2)))
        t = sscale(t, fmpq(-SGN[l], 2))
        if t is not None:
            A[(l, m, n)] = sreduce(t)
    return A

def Atrace(A):
    out = {}
    for s in range(4):
        t = None
        for l in range(4):
            t = sadd(t, aget(A, l, l, s))
        if t is not None:
            out[s] = t
    return out

def accum_E(acc, A, h):
    """acc += E(A;h)_l^{mn}, eq. (Aeq) with D=4"""
    At = Atrace(A)
    tv = {}
    for n in range(4):
        t = None
        for s in range(4):
            if s in At:
                t = sadd(t, smul(At[s], hget(h, n, s)))
        if t is not None:
            tv[n] = t
    for (l, m, n) in IDX3:
        for s in range(4):
            acc.add((l, m, n), smul(aget(A, m, l, s), hget(h, n, s)))
            acc.add((l, m, n), smul(aget(A, n, l, s), hget(h, m, s)))
        if m == l and n in tv:
            acc.add((l, m, n), tv[n], fmpq(-1, 3))
        if n == l and m in tv:
            acc.add((l, m, n), tv[m], fmpq(-1, 3))

def accum_AA(acc, A1, A2):
    """acc += A1^r_{sm} A2^s_{rn} - 1/3 A1_m A2_n  (m <= n; the sum over ordered pairs symmetrises)"""
    t1 = Atrace(A1); t2 = Atrace(A2)
    for (m, n) in IDX2:
        for r in range(4):
            for s in range(4):
                acc.add((m, n), smul(aget(A1, r, s, m), aget(A2, s, r, n)))
        if m in t1 and n in t2:
            acc.add((m, n), smul(t1[m], t2[n]), fmpq(-1, 3))

def E0(A):
    eta = {(m, m): (ONE*SGN[m], 0) for m in range(4)}
    acc = Acc()
    accum_E(acc, A, eta)
    return {k: acc.get(k) for k in IDX3 if acc.get(k) is not None}

def E0inv(Xd):
    """eq. (E0inv) in D=4"""
    def Xl(l, m, n):
        return sscale(aget(Xd, l, m, n), SGN[m]*SGN[n])
    Xt = {}
    for l in range(4):
        t = None
        for r in range(4):
            t = sadd(t, sscale(aget(Xd, l, r, r), SGN[r]))
        Xt[l] = t
    Y = {}
    for (l, m, n) in IDX3:
        t = sadd(Xl(m, l, n), Xl(n, l, m))
        t = ssub(t, Xl(l, m, n))
        t = sscale(t, fmpq(1, 2))
        if m == n:
            t = sadd(t, sscale(Xt[l], fmpq(SGN[m], 4)))
        t = sscale(t, SGN[l])
        if t is not None:
            Y[(l, m, n)] = sreduce(t)
    return Y

def tadd(A, B):
    out = dict(A)
    for k, v in B.items():
        t = sadd(out.get(k), v)
        if t is None:
            out.pop(k, None)
        else:
            out[k] = sreduce(t)
    return out

# ----------------------------------------------------------------- seed and recursion
def seed(PMAX, flip=False):
    """h^(1) of eq. (h1) from the Kerr multipoles: T^{00} = M cos(a.grad) delta,
    T^{0i} = 1/2 [curl(S sin(a.grad)/(a.grad) delta)]^i.   flip=True replaces cos -> cosh in
    position space (i.e. cosh -> cos in momentum space): negative control."""
    h1 = {}
    dz = [(ONE, 1)]
    for k in range(1, PMAX + 1):
        dz.append(sreduce(sdiff(dz[-1], 3)))
    for p in range(PMAX + 1):
        comp = {}
        if p % 2 == 0:
            sg = 1 if flip else (-1)**(p//2)
            comp[(0, 0)] = sscale(dz[p], fmpq(4*sg, factorial(p)))
        else:
            psi = sscale(dz[p - 1], fmpq((-1)**((p - 1)//2), factorial(p)))
            comp[(0, 1)] = sreduce(sscale(sdiff(psi, 2), 2))        # h^{0i} = 2a (grad psi x z)^i
            comp[(0, 2)] = sreduce(sscale(sdiff(psi, 1), -2))
        h1[p] = comp
    return h1

def run(NMAX, PMAX, flip=False, hom=None):
    """The recursion (recY)-(recJ) through order G^NMAX a^PMAX.

    Returns (h, A, Y, slots): h[n][p], A[n][p], Y[n][p] are the terms of order G^n a^p of the
    metric perturbation, of the auxiliary field and of Y = A - A_lin[h]; slots lists the sources
    whose range of harmonic degrees contains the degree that would be resonant, with a flag
    telling whether that component is actually present (a present one raises Resonance).

    At every order the function asserts
      * E0(Y) = sum E(A;h), the equation (Yeq) of the auxiliary field;
      * that the Laplacian of the solution gives back the source of (recJ);
      * the harmonic gauge condition d_mu hhat^{mu nu} = 0.

    hom: optional {(n, p): {(mu, nu): scalar}}, homogeneous solutions added by hand at order (n, p).
    flip: negative control, see seed()."""
    h = {1: seed(PMAX, flip)}
    A = {1: {p: A_lin(h[1][p]) for p in range(PMAX + 1)}}
    Y = {1: {p: {} for p in range(PMAX + 1)}}
    slots = []
    for n in range(2, NMAX + 1):
        h[n] = {}; A[n] = {}; Y[n] = {}
        for p in range(PMAX + 1):
            accE = Acc(); accA = Acc()
            for m in range(1, n):
                for p1 in range(p + 1):
                    A1 = A[m][p1]
                    if not A1:
                        continue
                    h2 = h[n - m][p - p1]
                    if h2:
                        accum_E(accE, A1, h2)
                    A2 = A[n - m][p - p1]
                    if A2:
                        accum_AA(accA, A1, A2)
            Esum = {k: accE.get(k) for k in IDX3}
            Esum = {k: v for k, v in Esum.items() if v is not None}
            Yn = E0inv(Esum)                                           # eq. (recY)
            back = E0(Yn)
            for k in IDX3:
                assert ssub(back.get(k), Esum.get(k)) is None, "E0(Y) != sum E(A;h)"
            Xs = {}
            for (mu, nu) in IDX2:                                      # eq. (recJ)
                t = sscale(accA.get((mu, nu)), -1)
                for l in (1, 2, 3):
                    t = ssub(t, sdiff(aget(Yn, l, mu, nu), l))
                t = sreduce(t) if t is not None else None
                lres, present = resonant_slot(t)
                if lres is not None:
                    slots.append(((n, p), (mu, nu), lres, present))
                sol = inv_lap(t, tag=(n, p, mu, nu))
                assert ssub(slap(sol), t) is None, "Poisson equation violated at G^%d a^%d" % (n, p)
                Xs[(mu, nu)] = sscale(sol, 2)
            trX = None
            for mu in range(4):
                trX = sadd(trX, sscale(Xs[(mu, mu)], SGN[mu]))
            hn = {}
            for (mu, nu) in IDX2:
                t = Xs[(mu, nu)]
                if mu == nu:
                    t = ssub(t, sscale(trX, fmpq(SGN[mu], 2)))
                t = sscale(t, SGN[mu]*SGN[nu])
                if hom and (n, p) in hom:
                    t = sadd(t, hom[(n, p)].get((mu, nu)))
                if t is not None:
                    hn[(mu, nu)] = sreduce(t)
            for nu in range(4):                                        # harmonic gauge
                t = None
                for i in (1, 2, 3):
                    t = sadd(t, sdiff(hget(hn, i, nu), i))
                assert t is None, "harmonic gauge violated at G^%d a^%d" % (n, p)
            h[n][p] = hn
            Y[n][p] = Yn
            A[n][p] = tadd(A_lin(hn), Yn)
    return h, A, Y, slots

# =====================================================================================
# 3. The closed forms of section 6 and their spin expansion
# =====================================================================================
def closed_form_functions(rho, s2, c, a, d2):
    """The closed forms (h1), (h00PM)-(hijPM), (h004)-(hij4), (h005) as functions of the flat oblate
    spheroidal coordinates of the harmonic point: rho = varrho, s2 = sin^2(vartheta), c = cos(vartheta),
    and d2 = delta_2(varrho) = [arctan(a/varrho) - a varrho/(varrho^2 + a^2)]/(2 a^2).  The arguments
    can be numbers or symbols.  Returns {name: {n: coefficient of (G_N M)^n}} for the functions
        hhat^{00} = h00,        hhat^{0i} = W (zhat x x)^i,
        hhat^{ij} = alp deltaPerp^{ij} + alz zhat^i zhat^j + beta xPerp^i xPerp^j
                    + kappa (xPerp^i zhat^j + zhat^i xPerp^j)                           (hijdecomp)."""
    A = rho**2 + a**2
    Sig = rho**2 + a**2*c**2
    f = {}
    f['h00'] = {1: 4*rho/Sig,
                2: (7*rho**2 + 3*a**2)/(A*Sig),
                3: 8*rho/(A*Sig),
                4: 4*(2*rho**2 + a**2)/(A**2*Sig) - a*rho*s2/(A*Sig)*d2 + d2**2,
                5: 8*rho/(A**2*Sig) - 4*a*rho**2*s2/(A*Sig**2)*d2
                   + (4*rho/Sig + 2*rho*A*s2*(3*a**2*c**2 - rho**2)/Sig**3)*d2**2}
    f['W'] = {1: 2*a*rho/(A*Sig),
              2: 2*a/(A*Sig),
              3: 2*a*rho/(A**2*Sig),
              4: 2*a/(A**2*Sig),
              5: a*rho*(2/(A**3*Sig) - 2*a*rho*s2/(A**2*Sig**2)*d2
                        + (3*A*c**2 - rho**2 - a**2*c**4)/Sig**3*d2**2)}
    f['alp'] = {2: a**2*s2/(A*Sig), 4: a*s2*(rho*A*d2 + a)/(A**2*Sig)}
    f['alz'] = {2: c**2/Sig, 4: a*rho*s2/(A*Sig)*d2 - d2**2}
    f['beta'] = {2: (rho**2 - a**2)/(A**2*Sig), 4: -a*(2*rho*A*d2 + a)/(A**3*Sig)}
    f['kappa'] = {2: rho*c/(A*Sig), 4: -a*c*d2/(A*Sig)}
    return f

class _SpinSeries:
    """power series in a, truncated at a^prec, with coefficients in Q[c]: the arithmetic needed to
    evaluate closed_form_functions"""
    def __init__(self, pol, env):
        self.pol, self.env = pol, env
    def _pol(self, other):
        return other.pol if isinstance(other, _SpinSeries) else self.env['ring'](other)
    def _new(self, pol):
        return _SpinSeries(self.env['trunc'](pol), self.env)
    def __add__(self, other):
        return self._new(self.pol + self._pol(other))
    __radd__ = __add__
    def __neg__(self):
        return self._new(-self.pol)
    def __sub__(self, other):
        return self._new(self.pol - self._pol(other))
    def __rsub__(self, other):
        return self._new(self._pol(other) - self.pol)
    def __mul__(self, other):
        return self._new(self.env['mul'](self.pol, self._pol(other)))
    __rmul__ = __mul__
    def inverse(self):
        return self._new(self.env['inv'](self.pol))
    def __truediv__(self, other):
        if isinstance(other, _SpinSeries):
            return self*other.inverse()
        return self._new(self.pol/other)
    def __rtruediv__(self, other):
        return self.inverse()*other
    def __pow__(self, n):
        assert isinstance(n, int) and n >= 1
        return reduce(lambda u, v: u*v, [self]*n)

def structure_series(PMAX):
    """The functions of closed_form_functions at |x| = 1, as power series in a through a^PMAX with
    coefficients in Q[c], c = z/|x|.  Returns S[name][n] for the names h00, W, alp, alz, beta, kappa."""
    from sympy.polys.rings import ring
    from sympy.polys.domains import QQ
    from sympy.polys.ring_series import rs_mul, rs_nth_root, rs_series_inversion, rs_atan, rs_trunc
    prec = PMAX + 8
    R, a, c = ring('a,c', QQ)
    env = {'ring': R, 'trunc': lambda u: rs_trunc(u, a, prec), 'mul': lambda u, v: rs_mul(u, v, a, prec),
           'inv': lambda u: rs_series_inversion(u, a, prec)}
    ser = lambda pol: _SpinSeries(rs_trunc(pol, a, prec), env)
    # flat oblate coordinates of the point (|x| = 1, z = c): varrho^2 and cos(vartheta) = c/varrho
    rho2 = (1 - a**2 + rs_nth_root((1 - a**2)**2 + 4*a**2*c**2, 2, a, prec))/2
    rho = ser(rs_nth_root(rho2, 2, a, prec))
    cth = ser(R(c))/rho
    sth2 = 1 - cth*cth
    # delta_2 = [arctan(a/varrho) - a varrho/(varrho^2 + a^2)]/(2 a^2): divide the series by 2 a^2
    aS = ser(R(a))
    num = (ser(rs_atan((aS/rho).pol, a, prec)) - aS*rho/(rho*rho + aS*aS)).pol
    d2 = R(0)
    for mon, co in num.terms():
        assert mon[0] >= 2
        d2 += R({(mon[0] - 2, mon[1]): co/2})
    f = closed_form_functions(rho, sth2, cth, aS, ser(d2))
    return {name: {n: v.pol for n, v in orders.items()} for name, orders in f.items()}

def coeff(S, name, n, p):
    """coefficient of a^p as {power of c: rational}"""
    pol = S[name].get(n)
    out = {}
    if pol is not None:
        for mon, co in pol.terms():
            if mon[0] == p:
                out[mon[1]] = co
    return out

def to_scalar(poly_c, n, p, w):
    """structure-function coefficient with radial weight |x|^{-(n+p+w)} -> (N, m)"""
    if not poly_c:
        return None
    K = max(poly_c)
    N = None
    for k, co in poly_c.items():
        assert (K - k) % 2 == 0
        t = z**k*r2pow((K - k)//2)*fmpq(int(co.numerator), int(co.denominator))
        N = t if N is None else N + t
    return sreduce((N, n + p + w + K))

def closed_fields(S, NMAX, PMAX):
    """the ten Cartesian components of the closed forms at each order (n, p), in the format of run()"""
    out = {}
    for n in range(1, NMAX + 1):
        for p in range(PMAX + 1):
            comp = {}
            g = lambda name, w: to_scalar(coeff(S, name, n, p), n, p, w)
            h00, W, alp, alz, beta, kap = g('h00', 0), g('W', 1), g('alp', 0), g('alz', 0), g('beta', 2), g('kappa', 1)
            def put(k, f):
                f = sreduce(f) if f is not None else None
                if f is not None:
                    comp[k] = f
            put((0, 0), h00)
            put((0, 1), sscale(smul(W, (y, 0)), -1))
            put((0, 2), smul(W, (x, 0)))
            put((1, 1), sadd(alp, smul(beta, (x*x, 0))))
            put((2, 2), sadd(alp, smul(beta, (y*y, 0))))
            put((1, 2), smul(beta, (x*y, 0)))
            put((1, 3), smul(kap, (x, 0)))
            put((2, 3), smul(kap, (y, 0)))
            put((3, 3), alz)
            out[(n, p)] = comp
    return out

def compare(h, cl, NMAX, PMAX):
    """list of (n, p, component) where the recursion and the closed forms differ"""
    bad = []
    for n in range(1, NMAX + 1):
        for p in range(PMAX + 1):
            for k in IDX2:
                if ssub(h[n][p].get(k), cl[(n, p)].get(k)) is not None:
                    bad.append((n, p, k))
    return bad

def control_line(wrong_input, rejected, detail=""):
    """One line reporting a control: a check run on a deliberately wrong input, which it must reject.
    The control succeeds when the input is rejected; the mismatches it reports are then expected."""
    if rejected:
        return "control OK, wrong input rejected: " + wrong_input + (" (%s)" % detail if detail else "")
    return "CONTROL FAILED, wrong input not rejected: " + wrong_input

def negative_controls(h, cl, NMAX, PMAX):
    """Three deliberately wrong inputs, each of which the comparison between the recursion and the
    closed forms must reject.  Returns {description of the wrong input: True if it is rejected}.
      (i)   one coefficient of the closed forms altered, at order G^NMAX a^2;
      (ii)  the leading term of (CSdiff), -2 x^(i (zhat x x)^j)/|x|^5 at order G^2 a, by which the
            Cook-Scheel chart differs from the chart (chart), added to the closed forms;
      (iii) cos instead of cosh in the momentum-space seed (i.e. cosh instead of cos in position
            space): the recursion started from it must disagree with the closed forms at G^2 a^2."""
    out = {}
    c2 = dict(cl); comp = dict(c2[(NMAX, 2)]); N, m = comp[(0, 0)]
    comp[(0, 0)] = (N + z**N.total_degree(), m); c2[(NMAX, 2)] = comp
    out["one coefficient of the closed forms changed at order G^%d a^2" % NMAX] = bool(compare(h, c2, NMAX, PMAX))
    c3 = dict(cl); comp = dict(c3[(2, 1)])
    for k, N in {(1, 1): 2*x*y, (1, 2): y*y - x*x, (1, 3): y*z, (2, 2): -2*x*y, (2, 3): -x*z}.items():
        comp[k] = sadd(comp.get(k), (N, 5))
    c3[(2, 1)] = comp
    out["the time-odd Cook-Scheel term at order G^2 a added to the closed forms"] = bool(compare(h, c3, 2, 1))
    hf = run(2, 2, flip=True)[0]
    out["cos instead of cosh in the momentum-space seed"] = any(b[0] == 2 and b[1] == 2 for b in compare(hf, cl, 2, 2))
    return out

# =====================================================================================
# 4. Exact metric: direct transformation of Boyer-Lindquist (multiprecision, numerical)
# =====================================================================================
def Theta(r, m, a):
    """eq. (Theta): int_r^oo a du/(u^2 - 2 m u + a^2), with m = G_N M"""
    R = r - m
    if abs(a) < m:
        q = sqrt(m*m - a*a)
        return a/(2*q)*log((R + q)/(R - q))
    if abs(a) == m:
        return a/R
    chi = sqrt(a*a - m*m)
    return a/chi*atan(chi/R)

def chart_map(r, th, ph, m, a, cs=False):
    """harmonic coordinates (x, y, z) of the Boyer-Lindquist point (r, theta, phi): the chart (chart),
    or the Cook-Scheel chart (CS) if cs=True"""
    R = r - m
    if cs:
        Th = Theta(r, m, a)
        c, s = cos(ph - Th), sin(ph - Th)
        return (R*c - a*s)*sin(th), (R*s + a*c)*sin(th), R*cos(th)
    d = Theta(r, m, a) - atan(a/R)
    F = sqrt(R*R + a*a)*cos(d)
    return F*sin(th)*cos(ph), F*sin(th)*sin(ph), R*cos(th)

def bl_gothic(r, th, m, a):
    """sqrt(-g) g^{mu nu} of the Kerr metric in Boyer-Lindquist coordinates (t, r, theta, phi)"""
    Sig = r*r + a*a*cos(th)**2
    Del = r*r - 2*m*r + a*a
    g = matrix(4, 4)
    g[0, 0] = -((r*r + a*a)**2 - a*a*Del*sin(th)**2)/(Sig*Del)
    g[0, 3] = g[3, 0] = -2*m*a*r/(Sig*Del)
    g[1, 1] = Del/Sig
    g[2, 2] = 1/Sig
    g[3, 3] = (Del - a*a*sin(th)**2)/(Sig*Del*sin(th)**2)
    return g*(Sig*sin(th))

def gothic_direct(r, th, m, a, ph=0, cs=False, frame=False):
    """Gothic inverse metric in (T, x, y, z), from the tensor-density transformation of bl_gothic with
    the Jacobian of chart_map obtained by numerical differentiation.  With frame=True the spatial
    indices are rotated to the cylindrical frame (e_varpi, e_varphi, zhat) at the harmonic point."""
    ph = mpf(ph)
    J = matrix(4, 4)
    J[0, 0] = 1
    for i in range(3):
        J[i + 1, 1] = diff(lambda rr: chart_map(rr, th, ph, m, a, cs)[i], r)
        J[i + 1, 2] = diff(lambda tt: chart_map(r, tt, ph, m, a, cs)[i], th)
        J[i + 1, 3] = diff(lambda pp: chart_map(r, th, pp, m, a, cs)[i], ph)
    g = J*bl_gothic(r, th, m, a)*J.T/abs(det(J))
    if frame:
        xx, yy, _ = chart_map(r, th, ph, m, a, cs)
        al = atan2(yy, xx)
        rot = matrix(4, 4)
        rot[0, 0] = 1; rot[3, 3] = 1
        rot[1, 1] = cos(al); rot[1, 2] = sin(al); rot[2, 1] = -sin(al); rot[2, 2] = cos(al)
        g = rot*g*rot.T
    return g

def gothic_formula(r, th, m, a, cs=False):
    """eqs. (gexact1)-(gexact3) in the frame (T, varpi, varphi, z); with cs=True the Cook-Scheel
    components: the same expressions with delta = 0, except for the three components (CSframe)"""
    R = r - m
    Del = R*R + a*a - m*m
    A = R*R + a*a
    if cs:
        d = dp = mpf(0)
    else:
        d = Theta(r, m, a) - atan(a/R)
        dp = -a*m*m/(A*Del)
    F = sqrt(A)*cos(d)
    Fp = R*cos(d)/sqrt(A) - sqrt(A)*dp*sin(d)
    SigR = R*R + a*a*cos(th)**2
    Dc = cos(d)**2*SigR + a*m*m*R/Del*sin(th)**2*sin(d)*cos(d)
    g = matrix(4, 4)
    g[0, 0] = -((r*r + a*a)**2 - a*a*Del*sin(th)**2)/(Del*Dc)
    g[0, 2] = g[2, 0] = -2*m*a*r*F*sin(th)/(Del*Dc)
    g[1, 1] = (Fp**2*Del*sin(th)**2 + F**2*cos(th)**2)/Dc
    g[2, 2] = F**2*(Del - a*a*sin(th)**2)/(Del*Dc)
    g[3, 3] = (Del*cos(th)**2 + R*R*sin(th)**2)/Dc
    g[1, 3] = g[3, 1] = sin(th)*cos(th)*(Fp*Del - R*F)/Dc
    if cs:
        g[2, 2] = 1 - a*a*m*m*sin(th)**2/(A*SigR)
        g[1, 2] = g[2, 1] = a*m*m*R*sin(th)**2/(A*SigR)
        g[2, 3] = g[3, 2] = a*m*m*sin(th)*cos(th)/(sqrt(A)*SigR)
    return g

def bl_of_harmonic_point(eps, rho, vth, a):
    """(R, theta) of the Boyer-Lindquist point whose image by (chart), with G_N M = eps, is the harmonic
    point with flat oblate spheroidal coordinates (rho, vth)"""
    if eps == 0:
        return rho, vth
    varpi = sqrt(rho*rho + a*a)*sin(vth)
    zz = rho*cos(vth)
    def Ffun(R):
        return sqrt(R*R + a*a)*cos(Theta(R + eps, eps, a) - atan(a/R))
    R = findroot(lambda R: Ffun(R)**2*(1 - zz*zz/(R*R)) - varpi**2, rho,
                 tol=mpf(10)**(-mp.dps + 5), maxsteps=200)
    return R, acos(zz/R)

def hhat_at_x(eps, rho, vth, a):
    """eta - gothic of the exact metric with G_N M = eps, at the harmonic point with flat oblate
    coordinates (rho, vth) and phi = 0 (there the indices 1, 2, 3 are varpi, varphi, z)"""
    if eps == 0:
        return matrix(4, 4)
    R, th = bl_of_harmonic_point(eps, rho, vth, a)
    eta = matrix(4, 4)
    eta[0, 0] = -1
    for i in (1, 2, 3):
        eta[i, i] = 1
    return eta - gothic_direct(R + eps, th, eps, a)

def closed(rho, vth, a):
    """the closed forms of section 6 evaluated numerically: {frame component: [orders 0..5 in G_N M]},
    at the harmonic point with flat oblate coordinates (rho, vth) and phi = 0"""
    A = rho*rho + a*a
    s, c = sin(vth), cos(vth)
    d2 = (atan(a/rho) - a*rho/A)/(2*a*a)
    f = closed_form_functions(rho, s*s, c, a, d2)
    varpi = sqrt(A)*s
    col = lambda name: [0] + [f[name].get(n, 0) for n in range(1, 6)]
    al_p, beta = col('alp'), col('beta')
    o = {(0, 0): col('h00'),
         (0, 2): [w*varpi for w in col('W')],
         (1, 1): [al_p[n] + beta[n]*varpi**2 for n in range(6)],
         (2, 2): al_p,
         (3, 3): col('alz'),
         (1, 3): [k*varpi for k in col('kappa')]}
    for k in [(0, 1), (0, 3), (1, 2), (2, 3)]:
        o[k] = [0]*6
    return o

def closed_forms_vs_exact_formulas(control=False):
    """The closed forms of section 6 are the expansion in G_N M, through fifth order and at fixed
    harmonic point, of the exact components (gexact1)-(gexact3): symbolic check, exact in the spin.

    The exact components are expanded at fixed (R, theta), with delta = eps^2 delta_2 + eps^4 delta_4
    and its derivative (delta') given exactly.  The Boyer-Lindquist variables (R, theta) of the
    harmonic point are then expressed by (Rtheta4), which is verified here as well.  Through fifth
    order only the first-order term is affected by this change of variables.

    Returns the list of (component, n) that fail; the components are 00, W, the three diagonal
    frame components and hhat^{varpi z}/varpi.  With control=True the coefficient 4 of
    varrho delta_2^2/Sigma in the fifth-order closed form of hhat^{00} is changed to 5, and the list
    must be [('00', 5)]."""
    import sympy as sp
    eps, a = sp.symbols('epsilon a', positive=True)
    R, rho = sp.symbols('R varrho', positive=True)
    s, c = sp.symbols('s c', real=True)                       # sin and cos of the polar angle
    d2, d4 = sp.symbols('delta_2 delta_4')
    on_circle = lambda e: sp.expand(sp.rem(sp.expand(sp.fraction(sp.together(e))[0]), s**2 + c**2 - 1, s))
    # (gexact1)-(gexact3) as eta - gothic, with F = sqrt(A) cos(delta)
    A = R**2 + a**2
    Del = A - eps**2
    r = R + eps
    delta = eps**2*d2 + eps**4*d4
    dprime = -a*eps**2/(A*Del)
    cd, sd = sp.cos(delta), sp.sin(delta)
    F2 = A*cd**2                                              # F^2
    FFp = R*cd**2 - A*dprime*sd*cd                            # F F'
    Fp2 = (R*cd - A*dprime*sd)**2/A                           # F'^2
    Dc = cd**2*(R**2 + a**2*c**2) + a*eps**2*R/Del*s**2*sd*cd
    exact = {'00': -1 + ((r**2 + a**2)**2 - a**2*Del*s**2)/(Del*Dc),
             'W': 2*eps*a*r/(Del*Dc),                         # hhat^{0 varphi} = W varpi
             'varpi varpi': 1 - (Fp2*Del*s**2 + F2*c**2)/Dc,
             'varphi varphi': 1 - F2*(Del - a**2*s**2)/(Del*Dc),
             'zz': 1 - (Del*c**2 + R**2*s**2)/Dc,
             'varpi z/varpi': -c*(FFp*Del - R*F2)/(Dc*F2)}    # varpi = F sin(theta)
    # (Rtheta4): R = rho + eps^4 R4, theta = vartheta + eps^4 T4, with (s, c) those of vartheta
    A0 = rho**2 + a**2
    Sig0 = rho**2 + a**2*c**2
    R4 = d2**2*rho*A0*s**2/(2*Sig0)
    T4 = d2**2*A0*s*c/(2*Sig0)
    # the harmonic point is unchanged at order eps^4: varpi = sqrt(A) cos(delta) sin(theta), z = R cos(theta)
    assert on_circle(rho*R4/A0 - d2**2/2 + T4*c/s) == 0 and on_circle(R4*c - rho*T4*s) == 0, "(Rtheta4)"
    dtheta = lambda e: c*sp.diff(e, s) - s*sp.diff(e, c)
    f = closed_form_functions(rho, s**2, c, a, d2)
    varpi2 = A0*s**2
    ref = {'00': f['h00'], 'W': f['W'],
           'varpi varpi': {n: f['alp'].get(n, 0) + f['beta'].get(n, 0)*varpi2 for n in range(1, 6)},
           'varphi varphi': f['alp'], 'zz': f['alz'], 'varpi z/varpi': f['kappa']}
    if control:
        ref['00'] = dict(ref['00'])
        ref['00'][5] += rho/Sig0*d2**2
    bad = []
    for name, e in exact.items():
        ser = sp.expand(sp.series(e, eps, 0, 6).removeO())
        for n in range(1, 6):
            cn = ser.coeff(eps, n).subs(R, rho)
            if n == 5:
                c1 = ser.coeff(eps, 1)
                cn += (sp.diff(c1, R)*R4 + dtheta(c1)*T4).subs(R, rho)
            if on_circle(cn - ref[name].get(n, 0)) != 0 or d4 in sp.sympify(cn).free_symbols:
                bad.append((name, n))
    return bad

# ----------------------------------------------------------------- checks of the exact metric
# (r, theta, G_N M, a): sub-extremal, near-extremal, super-extremal, extremal, negative spin
EXACT_POINTS = [('7.3', '0.9', '1', '0.6'), ('3.1', '2.2', '1', '0.95'), ('5.7', '0.4', '0.3', '1.7'),
                ('12.5', '1.3', '2', '2'), ('4.2', '1.1', '1', '-0.8')]

def theta_check(dps=40):
    """The three branches of (Theta) against the defining integral, and delta = Theta - arctan(a/R)
    against its integral representation (delta).  Returns the largest deviation."""
    with mp.workdps(dps):
        worst = mpf(0)
        for pt in EXACT_POINTS:
            r, _, m, a = map(mpf, pt)
            R = r - m
            worst = max(worst, abs(Theta(r, m, a) - quad(lambda u: a/(u*u - 2*m*u + a*a), [r, inf])))
            d = Theta(r, m, a) - atan(a/R)
            worst = max(worst, abs(d - quad(lambda u: a*m*m/((u*u + a*a)*(u*u + a*a - m*m)), [R, inf])))
        return worst

def harmonicity_check(dps=50):
    """Box_g x^A = 0 for the spatial coordinates of the chart (chart) and of the Cook-Scheel chart,
    evaluated in Boyer-Lindquist coordinates:
       d_r(Delta d_r f) + (1/sin) d_theta(sin d_theta f) + (1/sin^2 - a^2/Delta) d_phi^2 f = 0.
    Returns the largest residual, relative to the sum of the absolute values of the terms."""
    with mp.workdps(dps):
        worst = {False: mpf(0), True: mpf(0)}
        ph = mpf('0.37')
        for pt in EXACT_POINTS:
            r, th, m, a = map(mpf, pt)
            Del = r*r - 2*m*r + a*a
            for cs in (False, True):
                for i in range(3):
                    f = lambda rr, tt, pp: chart_map(rr, tt, pp, m, a, cs)[i]
                    terms = [Del*diff(lambda rr: f(rr, th, ph), r, 2),
                             (2*r - 2*m)*diff(lambda rr: f(rr, th, ph), r),
                             diff(lambda tt: f(r, tt, ph), th, 2),
                             cos(th)/sin(th)*diff(lambda tt: f(r, tt, ph), th),
                             (1/sin(th)**2 - a*a/Del)*diff(lambda pp: f(r, th, pp), ph, 2)]
                    worst[cs] = max(worst[cs], abs(sum(terms))/sum(abs(t) for t in terms))
        return {"chart": worst[False], "Cook-Scheel": worst[True]}

def exact_formula_check(dps=50):
    """(gexact1)-(gexact3), and their Cook-Scheel counterparts with (CSframe), against the direct
    transformation of the Boyer-Lindquist metric.  Returns the largest deviation for each chart."""
    with mp.workdps(dps):
        worst = {False: mpf(0), True: mpf(0)}
        for pt in EXACT_POINTS:
            r, th, m, a = map(mpf, pt)
            for cs in (False, True):
                gd = gothic_direct(r, th, m, a, ph=mpf('0.37'), cs=cs, frame=True)
                gf = gothic_formula(r, th, m, a, cs=cs)
                worst[cs] = max(worst[cs], max(abs(gd[i, j] - gf[i, j]) for i in range(4) for j in range(4)))
        return {"chart": worst[False], "Cook-Scheel": worst[True]}

HARMONIC_POINTS = [('6.5', '0.8', '1.3'), ('4', '1.9', '2.5'), ('9', '0.3', '0.4'), ('5.5', '1.1', '-1.7')]

def closed_vs_exact(dps=50):
    """The closed forms of orders n <= 5 against the Taylor coefficients in G_N M of the exact metric
    at fixed harmonic point (four points, ten components).  Returns the largest deviation."""
    with mp.workdps(dps):
        worst = mpf(0)
        for pt in HARMONIC_POINTS:
            rho, vth, a = map(mpf, pt)
            cl = closed(rho, vth, a)
            for (i, j), ref in cl.items():
                tc = taylor(lambda e: hhat_at_x(e, rho, vth, a)[i, j], 0, 5)
                worst = max(worst, max(abs(tc[n] - ref[n]) for n in range(6)))
        return worst

def inversion_check(dps=40):
    """(Rtheta4): the Boyer-Lindquist variables (R, theta) of a fixed harmonic point differ from its
    flat oblate coordinates at order G^4.  Returns (largest deviation of the G^4 coefficients from
    the formula, largest coefficient of G^1, G^2, G^3, G^5)."""
    with mp.workdps(dps):
        dev = low = mpf(0)
        for pt in HARMONIC_POINTS[:2]:
            rho, vth, a = map(mpf, pt)
            A = rho*rho + a*a
            Sig = rho*rho + a*a*cos(vth)**2
            d2 = (atan(a/rho) - a*rho/A)/(2*a*a)
            pred = (rho*A*sin(vth)**2/(2*Sig)*d2**2, A*sin(vth)*cos(vth)/(2*Sig)*d2**2)
            for k in (0, 1):
                tc = taylor(lambda e: bl_of_harmonic_point(e, rho, vth, a)[k], 0, 5)
                dev = max(dev, abs(tc[4] - pred[k]))
                low = max(low, max(abs(tc[n]) for n in (1, 2, 3, 5)))
        return dev, low

def recursion_vs_exact(h, NMAX, PMAX, spin=(1, 1000), dps=70):
    """The recursion, summed over the spin orders p <= PMAX, against the Taylor coefficients in G_N M
    of the exact metric at the point x = (3/5, 0, 4/5) and small spin a = spin[0]/spin[1].
    Returns ({n: largest deviation over the ten components}, size a^(PMAX+1) of the neglected terms)."""
    with mp.workdps(dps):
        a = mpf(spin[0])/spin[1]
        zz = mpf(4)/5
        rho = sqrt(((1 - a*a) + sqrt((1 - a*a)**2 + 4*a*a*zz*zz))/2)
        vth = acos(zz/rho)
        def ev(f):
            if f is None:
                return mpf(0)
            v = f[0](fmpq(3, 5), fmpq(0), fmpq(4, 5))
            return mpf(int(v.p))/mpf(int(v.q))
        worst = {}
        for (i, j) in IDX2:
            tc = taylor(lambda e: hhat_at_x(e, rho, vth, a)[i, j], 0, NMAX)
            for n in range(1, NMAX + 1):
                rec = sum(a**p*ev(h[n][p].get((i, j))) for p in range(PMAX + 1))
                worst[n] = max(worst.get(n, 0), abs(rec - tc[n]))
        return worst, abs(a)**(PMAX + 1)

# =====================================================================================
# 5. Output of the coefficients (appendix C)
# =====================================================================================
COEFFICIENT_NAMES = [('h00', 'h00'), ('W', 'w'), ('alp', 'aPerp'), ('alz', 'aZ'), ('beta', 'b'), ('kappa', 'k')]

def poly_str(pc, var='xi'):
    if not pc:
        return '0'
    terms = []
    for k in sorted(pc, reverse=True):
        co = pc[k]
        num, den = int(co.numerator), int(co.denominator)
        s = '%d' % abs(num) if den == 1 else '%d/%d' % (abs(num), den)
        if k == 1:
            s += '*' + var
        elif k > 1:
            s += '*%s^%d' % (var, k)
        terms.append((' - ' if num < 0 else ' + ') + s)
    out = ''.join(terms)
    return out[3:] if out.startswith(' + ') else '-' + out[3:]

HEADER = """(* Kerr metric in harmonic coordinates, chart of eq. (mainchart):
   spin expansion of the post-Minkowskian coefficients through order G_N^%d and a^%d.

   Conventions (appendix "Spin expansion of the post-Minkowskian coefficients"):
     n = x/|x|,  xi = zhat.n,  rho = G_N M/|x|,  hhat^{mu nu} = eta^{mu nu} - sqrt(-g) g^{mu nu},
     hhat^{00} = Sum_{n,p} rho^n (a/|x|)^p h00[n,p],
     hhat^{0i} = (zhat x n)^i Sum_{n,p} rho^n (a/|x|)^p w[n,p],
     hhat^{ij} = alphaPerp deltaPerp^{ij} + alphaZ zhat^i zhat^j + beta xPerp^i xPerp^j
                 + kappa (xPerp^i zhat^j + zhat^i xPerp^j),
     alphaPerp = Sum rho^n (a/|x|)^p aPerp[n,p],   alphaZ = Sum rho^n (a/|x|)^p aZ[n,p],
     |x|^2 beta = Sum rho^n (a/|x|)^p b[n,p],      |x| kappa = Sum rho^n (a/|x|)^p k[n,p].
   Coefficients that are not listed vanish.  Each entry was obtained from the recursion
   (recY)-(recJ) and agrees with the spin expansion of the closed forms.
   Generated by kerr_harmonic_checks.py. *)
"""

def coefficient_lines(S, NMAX, PMAX):
    """the non-vanishing coefficients as lines 'name[n, p] = polynomial in xi;'"""
    blocks = []
    for key, nm in COEFFICIENT_NAMES:
        lines = []
        for n in range(1, NMAX + 1):
            for p in range(PMAX + 1):
                pc = coeff(S, key, n, p)
                if pc:
                    lines.append("%s[%d, %d] = %s;" % (nm, n, p, poly_str(pc)))
        blocks.append(lines)
    return blocks

def coefficients_text(S, NMAX, PMAX):
    return HEADER % (NMAX, PMAX) + "".join("\n" + "\n".join(b) + "\n" for b in coefficient_lines(S, NMAX, PMAX))

def write_coefficients(S, NMAX, PMAX, fname):
    """write the coefficient file; returns the number of coefficients"""
    with open(fname, 'w') as f:
        f.write(coefficients_text(S, NMAX, PMAX))
    return sum(len(b) for b in coefficient_lines(S, NMAX, PMAX))

# =====================================================================================
if __name__ == "__main__":
    NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    PMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    NC = min(NMAX, 5)
    TOL = 1e-30                                   # for the numerical checks (40 to 50 digits)
    ok = True
    t0 = time.time()
    h, A, Y, slots = run(NMAX, PMAX)
    print("1. recursion through G^%d, a^%d: done in %.1fs" % (NMAX, PMAX, time.time() - t0))
    print("   auxiliary-field equation, Poisson equation and harmonic gauge hold at every order (asserted)")
    print("   resonances: none; sources containing the would-be resonant degree: %d" % len(slots))
    S = structure_series(PMAX)
    cl = closed_fields(S, NC, PMAX)
    bad = compare(h, cl, NC, PMAX)
    ok = ok and not bad
    print("2. recursion vs spin expansion of the closed forms, n <= %d, p <= %d, 10 components: %s"
          % (NC, PMAX, "exact agreement" if not bad else "MISMATCH at %s" % bad[:5]))
    if PMAX >= 2:
        for name, rejected in negative_controls(h, cl, NC, PMAX).items():
            ok = ok and rejected
            print("   " + control_line(name, rejected))
    print("3. exact metric")
    dev = theta_check()
    ok = ok and dev < TOL
    print("   (Theta), (delta) vs their integral representations: max deviation %.1e" % float(dev))
    res = harmonicity_check()
    ok = ok and max(res.values()) < TOL
    print("   Box x^A = 0: relative residual %.1e (chart), %.1e (Cook-Scheel)"
          % (float(res["chart"]), float(res["Cook-Scheel"])))
    res = exact_formula_check()
    ok = ok and max(res.values()) < TOL
    print("   (gexact1)-(gexact3) vs direct transformation of Boyer-Lindquist: max deviation %.1e"
          % float(res["chart"]))
    print("   Cook-Scheel components (CSframe) vs direct transformation: max deviation %.1e"
          % float(res["Cook-Scheel"]))
    bad_cf = closed_forms_vs_exact_formulas()
    ok = ok and not bad_cf
    print("   closed forms vs expansion of (gexact1)-(gexact3) in G_N M, n <= 5, symbolic: %s"
          % ("exact agreement" if not bad_cf else "MISMATCH at %s" % bad_cf))
    rejected = closed_forms_vs_exact_formulas(control=True) == [('00', 5)]
    ok = ok and rejected
    print("   " + control_line("5 instead of 4 as coefficient of varrho delta_2^2/Sigma in the fifth-order hhat^00",
                               rejected, "flagged at that component only"))
    dev = closed_vs_exact()
    ok = ok and dev < TOL
    print("   closed forms vs Taylor coefficients of the directly transformed metric, n <= 5, 4 points,")
    print("   10 components: max deviation %.1e" % float(dev))
    dev, low = inversion_check()
    ok = ok and dev < TOL and low < TOL
    print("   (Rtheta4): G^4 coefficients, max deviation %.1e; orders G^1, G^2, G^3, G^5: %.1e"
          % (float(dev), float(low)))
    worst, trunc = recursion_vs_exact(h, NMAX, PMAX)
    ok = ok and max(worst.values()) < 100*trunc
    print("4. recursion vs exact metric at a/|x| = 1/1000 (neglected spin orders ~ %.0e):" % float(trunc))
    print("   " + ", ".join("G^%d: %.0e" % (n, float(worst[n])) for n in range(1, NMAX + 1)))
    if not bad:
        fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kerr-harmonic-coefficients.m")
        nent = sum(len(b) for b in coefficient_lines(S, NC, PMAX))
        if (NC, PMAX) == (5, 12) and os.path.exists(fname):
            with open(fname) as f:
                same = f.read() == coefficients_text(S, NC, PMAX)
            ok = ok and same
            print("5. the %d coefficients of appendix C are %s kerr-harmonic-coefficients.m"
                  % (nent, "identical to those of" if same else "DIFFERENT from those of"))
        else:
            out = "kerr-harmonic-coefficients-G%d-a%d.m" % (NC, PMAX)
            write_coefficients(S, NC, PMAX, out)
            print("5. wrote %d coefficients to %s" % (nent, out))
    print("total time %.1fs" % (time.time() - t0))
    sys.exit(0 if ok else 1)
