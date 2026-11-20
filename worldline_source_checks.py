#!/usr/bin/env python3
"""
The linear coupling of the spinning worldline action and the Kerr stress tensor
(section 3 of A. Sadu and P. Vanhove,
"The Kerr Metric in Canonical Harmonic Coordinates to All Orders").

Usage
-----
    python worldline_source_checks.py

 1. curvature_linearisation: for a static perturbation delta g_{mu nu}(x) of flat
    space and a body at rest, the electric and magnetic parts of the curvature (EB),
        E_{mn} = R_{m a n b} u^a u^b,   B_{mn} = 1/2 eps_{m a r s} R^{rs}_{n b} u^a u^b,
    with eps_{0123} = +1 and R^r_{smn} = d_m Gamma^r_{ns} - d_n Gamma^r_{ms} + ...,
    are computed from the Christoffel symbols and compared with (EBlin),
        E_{ij} = -1/2 d_i d_j delta g_{00},   B_{ij} = 1/2 d_j (curl gamma)_i,
    gamma_i = delta g_{0i}; also Gamma^k_{0i} = 1/2 (d_i gamma_k - d_k gamma_i) and
    1/2 S^{ik} Gamma^k_{0i} = 1/2 S.(curl gamma) of (spinconnectionlin).
 2. stress_tensor_resummation: with these expressions and all Wilson coefficients
    equal to one, the couplings of (Swl) acting on a plane wave delta g ~ exp(i q.x)
    are summed through a given order in the spin and compared with the expansion of
    1/2 int d^3x T^{mu nu} delta g_{mu nu} for the Kerr stress tensor (Trest).

The linearised spin kinetic term is taken in the form (spinconnectionlin).  Names
such as (EB) are the LaTeX labels of the equations in the source of the paper.
Requires sympy.
"""
import sympy as sp

def curvature_linearisation():
    """returns {name: True/False} for the statements of (EBlin) and (spinconnectionlin)"""
    t, x, y, z = co = sp.symbols('t x y z', real=True)
    eps = sp.Symbol('epsilon')
    eta = sp.diag(-1, 1, 1, 1)
    # static perturbation: ten arbitrary functions of (x, y, z)
    hfun = {}
    for m in range(4):
        for n in range(m, 4):
            hfun[(m, n)] = hfun[(n, m)] = sp.Function('h%d%d' % (m, n))(x, y, z)
    h = sp.Matrix(4, 4, lambda m, n: hfun[(m, n)])
    d = lambda f, m: sp.diff(f, co[m])
    # linearised Christoffel symbols and Riemann tensor, R^r_{smn} = d_m Gamma^r_{ns} - d_n Gamma^r_{ms}
    Gam = lambda r, m, n: sp.Rational(1, 2)*eta[r, r]*(d(h[r, n], m) + d(h[r, m], n) - d(h[m, n], r))
    Riem_low = lambda r, s, m, n: eta[r, r]*(d(Gam(r, n, s), m) - d(Gam(r, m, s), n))        # R_{rsmn}
    u = [1, 0, 0, 0]
    E = sp.Matrix(4, 4, lambda m, n: sum(Riem_low(m, a, n, b)*u[a]*u[b] for a in range(4) for b in range(4)))
    lc = lambda *idx: sp.LeviCivita(*idx)                                                      # eps_{0123} = +1
    B = sp.Matrix(4, 4, lambda m, n: sp.Rational(1, 2)*sum(
        lc(m, a, r, s)*eta[r, r]*eta[s, s]*Riem_low(r, s, n, b)*u[a]*u[b]
        for a in range(4) for r in range(4) for s in range(4) for b in range(4)))
    gam = [h[0, i] for i in (1, 2, 3)]
    curl = [sum(sp.LeviCivita(i, j, k)*d(gam[k], j + 1) for j in range(3) for k in range(3)) for i in range(3)]
    res = {}
    res["E_ij = -1/2 d_i d_j dg_00"] = all(
        sp.simplify(E[i, j] + sp.Rational(1, 2)*d(d(h[0, 0], i), j)) == 0 for i in (1, 2, 3) for j in (1, 2, 3))
    res["B_ij = 1/2 d_j (curl gamma)_i"] = all(
        sp.simplify(B[i, j] - sp.Rational(1, 2)*d(curl[i - 1], j)) == 0 for i in (1, 2, 3) for j in (1, 2, 3))
    res["E_0m = B_0m = 0"] = all(sp.simplify(E[0, m]) == 0 and sp.simplify(B[0, m]) == 0 and sp.simplify(B[m, 0]) == 0
                                for m in range(4))
    res["Gamma^k_0i = 1/2 (d_i gamma_k - d_k gamma_i)"] = all(
        sp.simplify(Gam(k, 0, i) - sp.Rational(1, 2)*(d(gam[k - 1], i) - d(gam[i - 1], k))) == 0
        for i in (1, 2, 3) for k in (1, 2, 3))
    S = sp.symbols('S1:4', real=True)
    spin = sum(sp.Rational(1, 2)*sp.LeviCivita(i, k, mm)*S[mm]*Gam(k + 1, 0, i + 1)
               for i in range(3) for k in range(3) for mm in range(3))                         # S^{ik} = eps^{ikm} S^m
    res["1/2 S^ik Gamma^k_0i = 1/2 S.(curl gamma)"] = sp.simplify(
        spin - sp.Rational(1, 2)*sum(S[i]*curl[i] for i in range(3))) == 0
    return res

def stress_tensor_resummation(order=13, wilson=None):
    """Returns (ok, order): ok is True if the couplings of (Swl), with C_ES = C_BS = 1 and the
    linearised curvature (EBlin), reproduce 1/2 int T^{mu nu} delta g_{mu nu} for the stress tensor
    (Trest) through the given order in the spin.  The perturbation is the plane wave
    delta g_{00} = c0 exp(i q.x), delta g_{0i} = gamma_i exp(i q.x).
    wilson: optional {spin order: value} changing the Wilson coefficient C_ES or C_BS of that order
    (negative control: the comparison then fails)."""
    wilson = wilson or {}
    lam = sp.Symbol('lambda')                                   # counts powers of the spin: a -> lam a
    Mass = sp.Symbol('M', positive=True)
    a = sp.Matrix(sp.symbols('a1:4', real=True))
    q = sp.Matrix(sp.symbols('q1:4', real=True))
    gam = sp.Matrix(sp.symbols('gamma1:4'))
    c0 = sp.Symbol('c0')
    S = Mass*a                                                  # S = M a
    aq = (a.T*q)[0]
    D = sp.I*lam*aq                                             # a.grad on the plane wave
    curl = sp.I*q.cross(gam)                                    # curl gamma
    # (EBlin) contracted with a^i a^j:
    Eaa = -sp.Rational(1, 2)*D**2*c0                            # E_ij a^i a^j = -1/2 (a.grad)^2 dg_00
    Baa = sp.Rational(1, 2)*D*lam*(a.T*curl)[0]                 # B_ij a^i a^j = 1/2 (a.grad) a.(curl gamma)
    # worldline action (Swl) at linear order, evaluated at x = 0
    wl = sp.Rational(1, 2)*Mass*c0 + sp.Rational(1, 2)*lam*(S.T*curl)[0]      # -M sqrt(-g_00) and spin kinetic term
    for n in range(1, order//2 + 2):
        wl += wilson.get(2*n, 1)*Mass*sp.Rational((-1)**(n + 1), sp.factorial(2*n))*D**(2*n - 2)*Eaa
        wl += wilson.get(2*n + 1, 1)*Mass*sp.Rational((-1)**n, sp.factorial(2*n + 1))*D**(2*n - 1)*Baa
    # 1/2 int d^3x T^{mu nu}(x) dg_{mu nu}(x) = 1/2 T^{00}(-q) c0 + T^{0i}(-q) gamma_i, with (Trest)
    k = -q
    w = sp.Symbol('w')
    cosh_w = sp.series(sp.cosh(w), w, 0, order + 2).removeO()                 # cosh(a.k)
    sinhc_w = sp.series(sp.sinh(w)/w, w, 0, order + 2).removeO()              # sinh(a.k)/(a.k)
    ak = lam*(a.T*k)[0]
    T00 = Mass*cosh_w.subs(w, ak)
    T0i = sp.I/2*k.cross(lam*S)*sinhc_w.subs(w, ak)
    ref = sp.expand(sp.Rational(1, 2)*T00*c0 + (T0i.T*gam)[0])
    wl = sp.expand(wl)
    return all(sp.expand(wl.coeff(lam, j) - ref.coeff(lam, j)) == 0 for j in range(order + 1)), order

if __name__ == "__main__":
    res = curvature_linearisation()
    for name, ok in res.items():
        print("1. %-45s %s" % (name, "ok" if ok else "NO"))
    ok, order = stress_tensor_resummation()
    print("2. linear worldline coupling (Swl) vs Kerr stress tensor (Trest), through a^%d: %s"
          % (order, "exact agreement" if ok else "MISMATCH"))
    rejected = not stress_tensor_resummation(wilson={4: 2})[0]
    print("   " + ("control OK, wrong input rejected: " if rejected else "CONTROL FAILED, wrong input not rejected: ")
          + "Wilson coefficient C_ES^4 = 2 instead of 1")
    raise SystemExit(0 if ok and rejected and all(res.values()) else 1)
