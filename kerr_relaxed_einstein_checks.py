#!/usr/bin/env python3
"""
The relaxed Einstein equations in Landau-Lifshitz form, and the on-shell value of the
auxiliary field, for

    A. Sadu and P. Vanhove,
    "The Kerr Metric in Canonical Harmonic Coordinates to All Orders".

Usage
-----
    python kerr_relaxed_einstein_checks.py [NMAX=5] [PMAX=12]

 1. landau_lifshitz_iteration: the source Lambda of the relaxed equations (LLwave) of
    appendix A is evaluated on the output of the cubic recursion, order by order in
    G^n a^p.  For every n >= 2 the term u_(n) = Delta^{-1} Lambda_(n), computed with
    the inverse Laplacian (invlap) and no homogeneous solution, is compared with
    hhat^(n) of the cubic recursion.  Since Lambda_(n) depends only on the orders
    below n, this shows by induction that the two recursions have the same solution.
    It is also the iteration (MPMalgorithm) of the multipolar post-Minkowskian
    formalism for the canonical metric: the check establishes, at the orders
    computed, that no resonant term occurs (no finite part is needed) and that
    d_mu u^{mu nu}_(n) = 0 (the homogeneous term v_(n) vanishes).
 2. canonical_iteration: the same iteration started from the linear field (h1)
    alone, each order being built from the orders obtained before it, without
    reference to the cubic recursion; its output is then compared with that of
    the cubic recursion.
 3. onshell_auxiliary_field: the on-shell expression (Aonshell) of the auxiliary
    field, written with the gothic metric and its inverse, against the auxiliary
    field A = A_lin[hhat] + Y of the cubic recursion.

Names such as (LLwave) are the LaTeX labels of the equations in the source of the
paper.  Requires kerr_harmonic_checks.py in the same directory.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flint import fmpq
import kerr_harmonic_checks as khc
from kerr_harmonic_checks import Acc, smul, sdiff, sadd, ssub, sscale, ONE, SGN, IDX2

SP = (1, 2, 3)                                                  # spatial indices

# =====================================================================================
# Truncated double series:  {(n, p): scalar} is sum_{n,p} G^n a^p N_{n,p}(x)/|x|^m
# =====================================================================================
class Series:
    """algebra of double series truncated at G^NMAX a^PMAX"""
    def __init__(self, NMAX, PMAX):
        self.NMAX, self.PMAX = NMAX, PMAX

    def const(self, c):
        return {(0, 0): (ONE*c, 0)} if c else {}

    def lincomb(self, terms):
        """sum of c*u over the pairs (c, u)"""
        acc = Acc()
        for cf, u in terms:
            for k, f in u.items():
                acc.add(k, f, cf)
        return self._collect(acc)

    def contract(self, pairs, c=1):
        """c * sum of the products u*v over the pairs (u, v), truncated"""
        acc = Acc()
        NMAX, PMAX = self.NMAX, self.PMAX
        for u, v in pairs:
            if not u or not v:
                continue
            for (n1, p1), f in u.items():
                for (n2, p2), g in v.items():
                    if n1 + n2 <= NMAX and p1 + p2 <= PMAX:
                        acc.add((n1 + n2, p1 + p2), smul(f, g))
        out = self._collect(acc)
        return out if c == 1 else {k: sscale(f, c) for k, f in out.items()}

    def diff(self, u, i):
        out = {}
        for k, f in u.items():
            t = sdiff(f, i)
            if t is not None:
                out[k] = khc.sreduce(t)
        return out

    @staticmethod
    def _collect(acc):
        out = {}
        for k in acc.d:
            t = acc.get(k)
            if t is not None:
                out[k] = t
        return out

def sym(T, m, n):
    """component of a symmetric tensor stored with m <= n"""
    return T.get((m, n) if m <= n else (n, m), {})

def field_series(h, NMAX, PMAX):
    """hhat^{mu nu} of the recursion as series: {(mu, nu): {(n, p): scalar}}"""
    out = {k: {} for k in IDX2}
    for n in range(1, NMAX + 1):
        for p in range(PMAX + 1):
            for k, f in h[n][p].items():
                out[k][(n, p)] = f
    return out

def gothic_and_inverse(hs, S):
    """gothic^{mu nu} = eta^{mu nu} - hhat^{mu nu} and its matrix inverse K_{mu nu}, with a check"""
    g = {(m, n): S.lincomb([(1, S.const(SGN[m] if m == n else 0)), (-1, hs[(m, n)])]) for (m, n) in IDX2}
    eta = {(m, n): S.const(SGN[m] if m == n else 0) for (m, n) in IDX2}
    K = dict(eta)
    for _ in range(S.NMAX):                                      # K = eta + eta hhat K, one more order per step
        K = {(m, n): S.lincomb([(1, eta[(m, n)]),
                                (SGN[m], S.contract([(sym(hs, m, b), sym(K, b, n)) for b in range(4)]))])
             for (m, n) in IDX2}
    for m in range(4):
        for n in range(4):
            prod = S.contract([(sym(g, m, b), sym(K, b, n)) for b in range(4)])
            assert prod == (S.const(1) if m == n else {}), "inverse of the gothic metric"
    return g, K

# =====================================================================================
# 1. Landau-Lifshitz source and the iteration
# =====================================================================================
def landau_lifshitz_source(h, NMAX, PMAX, control=False):
    """Lambda^{alpha beta} of eq. (LLwave), Box hhat = Lambda, as series through G^NMAX a^PMAX:
         Lambda = hhat^{mu nu} d_mu d_nu hhat^{ab} - d_mu hhat^{a nu} d_nu hhat^{b mu} - 16 pi G (-g) t_LL^{ab}
    with (-g) t_LL written with the gothic metric g and its inverse K (products g^{..} g_{..} of the
    metric and its inverse are the same for the metric and for the gothic metric).  All fields are
    stationary, so that derivative indices are spatial.
    control=True replaces the coefficient 1/8 of the last term of t_LL by 1/4 (negative control)."""
    S = Series(NMAX, PMAX)
    hs = field_series(h, NMAX, PMAX)
    g, K = gothic_and_inverse(hs, S)
    dh = {(l, m, n): S.diff(hs[(m, n)], l) for l in SP for (m, n) in IDX2}
    d = lambda l, m, n: dh[(l, m, n) if m <= n else (l, n, m)]
    # B_l[m][t] = K_{m nu} d_l hhat^{nu t}
    B = {(l, m, t): S.contract([(sym(K, m, nu), d(l, nu, t)) for nu in range(4)])
         for l in SP for m in range(4) for t in range(4)}
    # S1 = K_{lam mu} d_rho hhat^{lam nu} d_nu hhat^{mu rho}
    S1 = S.contract([(B[(rho, mu, nu)], d(nu, mu, rho)) for rho in SP for nu in SP for mu in range(4)])
    # N^b_lam = K_{mu nu} d_rho hhat^{b nu} d_lam hhat^{mu rho}
    N = {(b, lam): S.contract([(B[(rho, mu, b)], d(lam, mu, rho)) for rho in SP for mu in range(4)])
         for b in range(4) for lam in SP}
    # Q_{lam mu} = (2 K K - K K) d_lam hhat d_mu hhat = 2 tr(B_lam B_mu) - tr(B_lam) tr(B_mu)
    trB = {l: S.lincomb([(1, B[(l, t, t)]) for t in range(4)]) for l in SP}
    Q = {}
    for lam in SP:
        for mu in SP:
            if lam <= mu:
                t1 = S.contract([(B[(lam, r, t)], B[(mu, t, r)]) for r in range(4) for t in range(4)])
                Q[(lam, mu)] = S.lincomb([(2, t1), (-1, S.contract([(trB[lam], trB[mu])]))])
    q = S.contract([(sym(g, lam, mu), sym(Q, lam, mu)) for lam in SP for mu in SP])
    V = {(al, mu): S.contract([(sym(g, al, lam), sym(Q, lam, mu)) for lam in SP]) for al in range(4) for mu in SP}
    Lam = {}
    for (al, be) in IDX2:
        ddh = lambda mu, nu: S.diff(d(mu, al, be), nu)
        first = S.contract([(sym(hs, mu, nu), ddh(mu, nu)) for mu in SP for nu in SP])
        second = S.contract([(d(mu, al, nu), d(nu, be, mu)) for mu in SP for nu in SP])
        t1 = S.contract([(sym(g, al, be), S1)])
        t2 = S.contract([(sym(g, al, lam), N[(be, lam)]) for lam in SP]
                        + [(sym(g, be, lam), N[(al, lam)]) for lam in SP])
        # K_{lam mu} g^{nu rho} d_nu hhat^{a lam} d_rho hhat^{b mu}
        C = {(nu, rho): S.contract([(B[(nu, mu, al)], d(rho, mu, be)) for mu in range(4)]) for nu in SP for rho in SP}
        t3 = S.contract([(sym(g, nu, rho), C[(nu, rho)]) for nu in SP for rho in SP])
        t4 = S.lincomb([(2, S.contract([(V[(al, mu)], sym(g, be, mu)) for mu in SP])),
                        (-1, S.contract([(sym(g, al, be), q)]))])
        tau = S.lincomb([(fmpq(1, 2), t1), (-1, t2), (1, t3), (fmpq(1, 4 if control else 8), t4)])
        Lam[(al, be)] = S.lincomb([(1, first), (-1, second), (-1, tau)])
    return Lam

def landau_lifshitz_iteration(h, NMAX, PMAX, control=False):
    """For 2 <= n <= NMAX and p <= PMAX: u = Delta^{-1} Lambda with (invlap).  Returns a dictionary with
    the number of orders (n, p) treated, the number of non-vanishing source components, and the lists
    of failures: resonant sources, components where u differs from hhat of the cubic recursion, and
    orders where d_i u^{i nu} does not vanish.  The three lists are empty when the check passes.
    control: see landau_lifshitz_source; the lists are then not empty."""
    Lam = landau_lifshitz_source(h, NMAX, PMAX, control)
    res = {"orders": 0, "source components": 0, "resonances": [], "mismatches": [], "divergence": []}
    for n in range(2, NMAX + 1):
        for p in range(PMAX + 1):
            res["orders"] += 1
            u = {}
            for k in IDX2:
                src = Lam[k].get((n, p))
                if src is None:
                    continue
                res["source components"] += 1
                try:
                    u[k] = khc.inv_lap(src, tag=(n, p) + k)
                except khc.Resonance:
                    res["resonances"].append((n, p, k))
            for k in IDX2:
                if ssub(u.get(k), h[n][p].get(k)) is not None:
                    res["mismatches"].append((n, p, k))
            for nu in range(4):
                t = None
                for i in SP:
                    t = sadd(t, sdiff(khc.hget(u, i, nu), i))
                if t is not None:
                    res["divergence"].append((n, p, nu))
    return res

def canonical_iteration(NMAX, PMAX):
    """The iteration (MPMalgorithm) carried out from the linear field alone: starting from (h1), the
    term of order n >= 2 is u_(n) = Delta^{-1} Lambda_(n), with (invlap) and the source Lambda_(n) built
    from the terms of order < n obtained in the previous steps.  The cubic recursion is not used.

    Returns (h, res): h[n][p] in the format of the cubic recursion (hhat = -h_can), and res with the
    lists of resonant source components and of orders where d_i u^{i nu} does not vanish.  Both lists
    are empty when no finite part is needed and the homogeneous term v_(n) vanishes."""
    h = {1: khc.seed(PMAX)}
    res = {"resonances": [], "divergence": []}
    for n in range(2, NMAX + 1):
        h[n] = {p: {} for p in range(PMAX + 1)}
        Lam = landau_lifshitz_source(h, n, PMAX)
        for p in range(PMAX + 1):
            u = {}
            for k in IDX2:
                src = Lam[k].get((n, p))
                if src is None:
                    continue
                try:
                    u[k] = khc.inv_lap(src, tag=(n, p) + k)
                except khc.Resonance:
                    res["resonances"].append((n, p, k))
            for nu in range(4):
                t = None
                for i in SP:
                    t = sadd(t, sdiff(khc.hget(u, i, nu), i))
                if t is not None:
                    res["divergence"].append((n, p, nu))
            h[n][p] = u
    return h, res

def same_fields(h1, h2, NMAX, PMAX):
    """list of (n, p, component) where two outputs in the format of the recursion differ"""
    return [(n, p, k) for n in range(1, NMAX + 1) for p in range(PMAX + 1) for k in IDX2
            if ssub(h1[n][p].get(k), h2[n][p].get(k)) is not None]

# =====================================================================================
# 2. On-shell auxiliary field
# =====================================================================================
def onshell_auxiliary_field(h, A, NMAX, PMAX, control=False):
    """(Aonshell), A^l_{mn} = -Gamma^l_{mn} + 1/2 (delta^l_m d_n + delta^l_n d_m) log sqrt(-g), in D = 4.
    With the gothic metric g and its inverse K this is
         A^l_{mn} = -1/2 g^{ls} (d_m K_{sn} + d_n K_{sm} - d_s K_{mn}) - 1/4 g^{ls} K_{mn} K_{ab} d_s hhat^{ab}.
    Returns the list of (n, p, component) where it differs from the auxiliary field of the recursion,
    for 1 <= n <= NMAX and p <= PMAX.
    control=True replaces the coefficient 1/4 of the last term by 1/2 (negative control)."""
    S = Series(NMAX, PMAX)
    hs = field_series(h, NMAX, PMAX)
    g, K = gothic_and_inverse(hs, S)
    dK = {(l, m, n): S.diff(K[(m, n)], l) for l in SP for (m, n) in IDX2}
    dk = lambda l, m, n: dK[(l, m, n) if m <= n else (l, n, m)] if l != 0 else {}
    # w_s = K_{ab} d_s hhat^{ab}
    w = {s: S.contract([(sym(K, al, be), S.diff(sym(hs, al, be), s)) for al in range(4) for be in range(4)])
         for s in SP}
    bad = []
    for (l, m, n) in khc.IDX3:
        chris = {s: S.lincomb([(1, dk(m, s, n)), (1, dk(n, s, m)), (-1, dk(s, m, n))]) for s in range(4)}
        t1 = S.contract([(sym(g, l, s), chris[s]) for s in range(4)], fmpq(-1, 2))
        Kw = {s: S.contract([(sym(K, m, n), w[s])]) for s in SP}
        t2 = S.contract([(sym(g, l, s), Kw[s]) for s in SP], fmpq(-1, 2 if control else 4))
        Aon = S.lincomb([(1, t1), (1, t2)])
        for nn in range(1, NMAX + 1):
            for p in range(PMAX + 1):
                if ssub(Aon.get((nn, p)), khc.aget(A[nn][p], l, m, n)) is not None:
                    bad.append((nn, p, (l, m, n)))
    return bad

# =====================================================================================
if __name__ == "__main__":
    NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    PMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    t0 = time.time()
    h, A, Y, slots = khc.run(NMAX, PMAX)
    res = landau_lifshitz_iteration(h, NMAX, PMAX)
    ok = not (res["resonances"] or res["mismatches"] or res["divergence"])
    print("1. Landau-Lifshitz iteration, 2 <= n <= %d, p <= %d (%d orders, %d source components), %.0fs"
          % (NMAX, PMAX, res["orders"], res["source components"], time.time() - t0))
    print("   resonant source terms: %d" % len(res["resonances"]))
    print("   Delta^{-1} Lambda vs cubic recursion: %s"
          % ("exact agreement" if not res["mismatches"] else "MISMATCH at %s" % res["mismatches"][:5]))
    print("   d_mu u^{mu nu} = 0: %s" % ("yes" if not res["divergence"] else "NO at %s" % res["divergence"][:5]))
    ctl = landau_lifshitz_iteration(h, min(NMAX, 3), min(PMAX, 2), control=True)
    rejected = bool(ctl["mismatches"])
    ok = ok and rejected
    print("   " + khc.control_line("1/4 instead of 1/8 in the last term of t_LL", rejected,
                                   "%d mismatches for n <= 3, p <= 2, as expected" % len(ctl["mismatches"])))
    t1 = time.time()
    N2 = max(NMAX, 8)
    hc, resc = canonical_iteration(N2, PMAX)
    diff = same_fields(hc, khc.run(N2, PMAX)[0], N2, PMAX)
    ok = ok and not (resc["resonances"] or resc["divergence"] or diff)
    print("2. iteration (MPMalgorithm) from the linear field alone, n <= %d, p <= %d, %.0fs" % (N2, PMAX, time.time() - t1))
    print("   resonant source terms: %d; orders with non-zero divergence: %d"
          % (len(resc["resonances"]), len(resc["divergence"])))
    print("   output vs cubic recursion: %s" % ("exact agreement" if not diff else "MISMATCH at %s" % diff[:5]))
    t1 = time.time()
    bad = onshell_auxiliary_field(h, A, NMAX, PMAX)
    print("3. on-shell auxiliary field (Aonshell) vs recursion, n <= %d, p <= %d: %s, %.0fs"
          % (NMAX, PMAX, "exact agreement" if not bad else "MISMATCH at %s" % bad[:5], time.time() - t1))
    rejected = bool(onshell_auxiliary_field(h, A, min(NMAX, 2), min(PMAX, 2), control=True))
    ok = ok and rejected
    print("   " + khc.control_line("1/2 instead of 1/4 in the last term of (Aonshell)", rejected))
    sys.exit(0 if ok and not bad else 1)
