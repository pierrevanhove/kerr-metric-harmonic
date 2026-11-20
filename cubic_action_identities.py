#!/usr/bin/env python3
"""
Algebraic identities of the cubic formulation of the Einstein-Hilbert action in D
dimensions (section 2 and appendix D of A. Sadu and P. Vanhove,
"The Kerr Metric in Canonical Harmonic Coordinates to All Orders").

Usage
-----
    python cubic_action_identities.py

Each identity is a polynomial identity in the components of the tensors involved.
It is tested in exact rational arithmetic on tensors with random integer components,
for D = 3, 4, 5, 6.  The derivatives d_l hhat^{mn} and d_a d_b hhat^{mn} are replaced by
random tensors with the same symmetries, which is all that the identities use.

Notation: eta = diag(-1, 1, ..., 1); A^l_{mn} = A^l_{nm}; A_s = A^l_{ls};
    Q(A,B;g)       = (A^l_{mr} B^m_{ls} - A_r B_s/(D-1)) g^{rs}                       (Qdef)
    E(A;g)_l^{mn}  = A^m_{ls} g^{ns} + A^n_{ls} g^{ms}
                     - (delta^m_l A_s g^{ns} + delta^n_l A_s g^{ms})/(D-1)            (Aeq)
    A.X            = A^l_{mn} X_l^{mn},     E0 = E( . ; eta).
Names such as (Qdef) are the LaTeX labels of the equations in the source of the paper.
"""
import random
from fractions import Fraction
from itertools import product

def identities(D, seed=1, control=False):
    """returns {name: True/False} for the identities of section 2 and appendix D in dimension D.
    control=True replaces the coefficient 1/(2(D-2)) of the trace term of (E0inv) by 1/(2(D-1)):
    the identities that involve the inverse of E0 then fail (negative control)."""
    rng = random.Random(seed)
    I = range(D)
    eta = [-1] + [1]*(D - 1)
    rnd = lambda: Fraction(rng.randint(-9, 9))
    def sym2():
        g = {}
        for m in I:
            for n in range(m, D):
                g[(m, n)] = g[(n, m)] = rnd()
        return g
    def conn():                                  # A^l_{mn} or X_l^{mn}, symmetric in the last two indices
        A = {}
        for l in I:
            for m in I:
                for n in range(m, D):
                    A[(l, m, n)] = A[(l, n, m)] = rnd()
        return A
    def second():                                # d_a d_b hhat^{mn}, symmetric in (ab) and in (mn)
        H = {}
        for a in I:
            for b in range(a, D):
                for m in I:
                    for n in range(m, D):
                        v = rnd()
                        for k in ((a, b, m, n), (b, a, m, n), (a, b, n, m), (b, a, n, m)):
                            H[k] = v
        return H
    add = lambda U, V, c=1: {k: U[k] + c*V[k] for k in U}
    tr = lambda A: [sum(A[(l, l, s)] for l in I) for s in I]
    dot = lambda A, Xd: sum(A[k]*Xd[k] for k in A)
    def Q(A, B, g):
        tA, tB = tr(A), tr(B)
        return (sum(A[(l, m, r)]*B[(m, l, s)]*g[(r, s)] for l, m, r, s in product(I, repeat=4))
                - sum(tA[r]*tB[s]*g[(r, s)] for r in I for s in I)/(D - 1))
    def E(A, g):
        tA = tr(A)
        v = [sum(tA[s]*g[(n, s)] for s in I) for n in I]
        out = {}
        for l, m, n in product(I, repeat=3):
            t = sum(A[(m, l, s)]*g[(n, s)] + A[(n, l, s)]*g[(m, s)] for s in I)
            if m == l:
                t -= v[n]/(D - 1)
            if n == l:
                t -= v[m]/(D - 1)
            out[(l, m, n)] = t
        return out
    geta = {(m, n): Fraction(eta[m] if m == n else 0) for m in I for n in I}
    E0 = lambda A: E(A, geta)
    def E0inv(Xd):                               # eq. (E0inv); Xd = X_l^{mn}, result A^l_{mn}
        low = lambda l, m, n: eta[m]*eta[n]*Xd[(l, m, n)]
        out = {}
        for l, m, n in product(I, repeat=3):
            t = Fraction(1, 2)*(low(m, l, n) + low(n, l, m) - low(l, m, n))
            if m == n:
                t += Fraction(eta[m], 2*(D - 1 if control else D - 2))*sum(eta[r]*Xd[(l, r, r)] for r in I)
            out[(l, m, n)] = eta[l]*t
        return out
    def Alin(Dh):                                # eq. (Alin); Dh = d_l hhat^{mn}
        low = lambda s, m, n: eta[m]*eta[n]*Dh[(s, m, n)]          # d_s hhat_{mn}
        trh = [sum(eta[r]*Dh[(s, r, r)] for r in I) for s in I]    # d_s hhat
        out = {}
        for l, m, n in product(I, repeat=3):
            t = low(m, n, l) + low(n, m, l) - low(l, m, n)
            if m == n:
                t += Fraction(eta[m], D - 2)*trh[l]
            out[(l, m, n)] = Fraction(-eta[l], 2)*t
        return out
    P = lambda Xs: {(m, n): Xs[(m, n)] - (Fraction(eta[m], D - 2)*sum(eta[r]*Xs[(r, r)] for r in I) if m == n else 0)
                    for m in I for n in I}
    Pinv = lambda Xs: {(m, n): Xs[(m, n)] - (Fraction(eta[m], 2)*sum(eta[r]*Xs[(r, r)] for r in I) if m == n else 0)
                       for m in I for n in I}

    g, hh = sym2(), sym2()
    A, B, Y, Xd, Xe, Dh, dDh = conn(), conn(), conn(), conn(), conn(), conn(), conn()
    res = {}
    # (QE): A.E(B;g) = 2 Q(A,B;g) for any symmetric g; symmetry of E0
    res["A.E(B;g) = 2Q(A,B;g)  (QE)"] = dot(A, E(B, g)) == 2*Q(A, B, g)
    res["A.E0(B) = B.E0(A)"] = dot(A, E0(B)) == dot(B, E0(A))
    # (E0inv): inverse of E0, and its symmetry
    res["E0inv is the inverse of E0  (E0inv)"] = E0(E0inv(Xd)) == Xd and E0inv(E0(A)) == A
    res["X'.E0inv(X) = X.E0inv(X')"] = dot(E0inv(Xd), Xe) == dot(E0inv(Xe), Xd)
    # (Alin): E0(A_lin[hhat]) = -d hhat, and Q(A_lin, A_lin; eta) = -1/2 A_lin.d hhat
    AL = Alin(Dh)
    res["E0(A_lin) = -d hhat  (Alin)"] = E0(AL) == {k: -v for k, v in Dh.items()}
    res["Q(A_lin,A_lin;eta) = -1/2 A_lin.d hhat"] = Q(AL, AL, geta) == -Fraction(1, 2)*dot(AL, Dh)
    # bulk action: -Q(A,A;gothic) + A.d gothic = -Q(Y,Y;eta) + Q(A_lin,A_lin;eta) + Q(A,A;hhat),
    # with gothic = eta - hhat and A = A_lin + Y; this gives (Lbulk) with (kinetic) below
    Atot = add(AL, Y)
    goth = add(geta, hh, -1)
    res["decomposition of the bulk action  (Lbulk)"] = (
        -Q(Atot, Atot, goth) - dot(Atot, Dh) == -Q(Y, Y, geta) + Q(AL, AL, geta) + Q(Atot, Atot, hh))
    # (Yeq): the variation of -Q(Y,Y;eta) + Q(A,A;hhat) along dY is dY.(-E0(Y) + E(A;hhat))
    dY = conn()
    res["equation of Y  (Yeq)"] = (-2*Q(dY, Y, geta) + 2*Q(dY, Atot, hh) == dot(dY, add(E(Atot, hh), E0(Y), -1)))
    # identity used for (heq): A_lin[d hhat'] . X = - d hhat' . E0inv(X)
    res["A_lin[dh'].X = -dh'.E0inv(X)  (heq)"] = dot(Alin(dDh), Xd) == -dot(dDh, E0inv(Xd))
    # (kinetic): d_l A_lin^l_{mn} - eta_{s(m} d_{n)} d_r gothic^{rs} = 1/2 Box (P hhat)_{mn}
    H = second()
    low = lambda a, b, m, n: eta[m]*eta[n]*H[(a, b, m, n)]                       # d_a d_b hhat_{mn}
    trH = {(a, b): sum(eta[r]*H[(a, b, r, r)] for r in I) for a in I for b in I}  # d_a d_b hhat
    box = {(m, n): sum(eta[l]*low(l, l, m, n) for l in I) for m in I for n in I}  # Box hhat_{mn}
    boxtr = sum(eta[l]*trH[(l, l)] for l in I)
    ok = True
    for m in I:
        for n in I:
            dA = -Fraction(1, 2)*sum(eta[l]*(low(l, m, n, l) + low(l, n, m, l)) for l in I) \
                 + Fraction(1, 2)*box[(m, n)] - (Fraction(eta[m], 2*(D - 2))*boxtr if m == n else 0)
            # -eta_{s(m} d_{n)} d_r gothic^{rs} = +1/2 (d_n d_r hhat^{r s} eta_{sm} + d_m d_r hhat^{r s} eta_{sn})
            gf = Fraction(1, 2)*sum(eta[m]*H[(n, r, r, m)] + eta[n]*H[(m, r, r, n)] for r in I)
            rhs = Fraction(1, 2)*(box[(m, n)] - (Fraction(eta[m], D - 2)*boxtr if m == n else 0))
            ok = ok and dA + gf == rhs
    res["kinetic operator  (kinetic)"] = ok
    # (Pdef), (props): P^{-1} P = 1, with P^{-1} independent of D
    res["Pinv is the inverse of P  (Pdef)"] = Pinv(P(hh)) == hh and P(Pinv(hh)) == hh
    # source term (Swlexp): hhat^{mn} (P T)_{mn} = (P hhat)_{mn} T^{mn}, with hh = hhat^{mn}, Tm = T_{mn}
    Tm = sym2()
    flip = lambda Xs: {(m, n): eta[m]*eta[n]*Xs[(m, n)] for m in I for n in I}     # raise or lower both indices
    res["hhat.(PT) = (P hhat).T  (Swlexp)"] = (sum(hh[k]*P(Tm)[k] for k in hh)
                                                == sum(P(flip(hh))[k]*flip(Tm)[k] for k in hh))
    return res

def all_dimensions(dims=(3, 4, 5, 6), seeds=(1, 2, 3)):
    """{name: {D: True/False}}; an identity is True in dimension D if it holds for every seed"""
    table = {}
    for D in dims:
        for seed in seeds:
            for name, ok in identities(D, seed).items():
                table.setdefault(name, {})
                table[name][D] = table[name].get(D, True) and ok
    return table

if __name__ == "__main__":
    table = all_dimensions()
    dims = sorted(next(iter(table.values())))
    width = max(len(n) for n in table)
    print("%-*s  %s" % (width, "identity", "  ".join("D=%d" % D for D in dims)))
    for name, row in table.items():
        print("%-*s  %s" % (width, name, "  ".join("ok " if row[D] else "NO " for D in dims)))
    failing = [name for name, ok in identities(4, control=True).items() if not ok]
    print(("control OK, wrong input rejected: " if failing else "CONTROL FAILED, wrong input not rejected: ")
          + "1/(2(D-1)) instead of 1/(2(D-2)) in (E0inv), D = 4"
          + (" (%d identities fail, as expected)" % len(failing) if failing else ""))
    raise SystemExit(0 if all(all(r.values()) for r in table.values()) and failing else 1)
