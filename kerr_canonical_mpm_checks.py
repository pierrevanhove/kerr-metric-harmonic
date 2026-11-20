#!/usr/bin/env python3
"""
Checks for section 5.6, "Relation to canonical harmonic coordinates", and for the
exceptional sector of section 4.2, of

    A. Sadu and P. Vanhove,
    "The Kerr Metric in Canonical Harmonic Coordinates to All Orders".

Usage
-----
    python kerr_canonical_mpm_checks.py

The canonical harmonic metric of Kerr in the multipolar post-Minkowskian (MPM)
formalism is taken from Xing, He and Cao, arXiv:2608.05713.  Their eqs. (29), (32),
(33), (34), (61) and (69) are transcribed below, in their conventions:
h_can = sqrt(-g) g - eta = -hhat, G = c = 1, and their R is the flat oblate
radius varrho of the harmonic point.

 1. closed_forms_vs_canonical: the closed forms (h1), (h00PM)-(hijPM), (h004)-(hij4)
    against h_can(1..4), all ten components, as exact identities in the harmonic
    point and in the spin;
 2. direct_transformation_vs_canonical: Boyer-Lindquist metric -> chart (mainchart)
    -> Taylor coefficients in G M at a fixed harmonic point, against h_can(1..4);
 3. jiang_lin_difference: the difference between the Jiang-Lin and the canonical
    metrics at second order, their eq. (61), against (CSdiff), as an exact identity;
 4. azimuth_transformation: their transformation (69) reduces to varphi_2 = varphi
    through third order;
 5. exceptional_sector: dimension of the space of decaying harmonic tensors at order
    (3,p) compatible with the conditions (i)-(iii), and consistency of the recursion
    when the mode is added by hand;
 6. cook_scheel_relation: the relation (CStocanonical) between the Cook-Scheel
    (Jiang-Lin) coordinates and the chart (chart).

Names such as (h1) or (CSdiff) are the LaTeX labels of the equations in the source
of the paper.  Requires kerr_harmonic_checks.py in the same directory, and
python-flint, sympy, mpmath.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
from flint import fmpq
from mpmath import mp, mpf, taylor, tan
import kerr_harmonic_checks as khc

# Independent variables: a, M, R = varrho, c = cos(vartheta), the Cartesian coordinates X, Y of the
# harmonic point, and T = arctan(a/R).  They are related by X^2 + Y^2 = (R^2 + a^2)(1 - c^2), Z = R c.
a, M, R, m = sp.symbols('a M R m', positive=True)
c, X, Y, T = sp.symbols('c X Y T', real=True)
A = R**2 + a**2
Sig = R**2 + a**2*c**2
Z = R*c
D4 = a**2*Z**2 + R**4                                                   # = R^2 Sigma
at = T
s2 = 1 - c**2                                                           # sin(vartheta)^2
KEYS = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

def on_shell(expr):
    """numerator of expr reduced with X^2 + Y^2 = (R^2 + a^2)(1 - c^2); it vanishes identically if and
    only if expr vanishes at every harmonic point, for all values of the spin"""
    num = sp.fraction(sp.together(expr))[0]
    return sp.expand(sp.rem(sp.expand(num), Y**2 - (A*s2 - X**2), Y))

def at_point(rho, vth, av, digits):
    """numerical values of the variables at the harmonic point (rho, vth, phi = 0), with M = 1"""
    F = lambda v: sp.Float(str(v), digits)
    rho, vth, av = F(rho), F(vth), F(av)
    return {R: rho, c: sp.cos(vth), a: av, X: sp.sqrt(rho**2 + av**2)*sp.sin(vth), Y: 0,
            T: sp.atan(av/rho), M: 1}

def xing_metric():
    """h_can(n), n = 1..4: eqs. (29), (32), (33), (34) of arXiv:2608.05713"""
    W4 = a**3 + R*A*at - a*R**2
    h = {1: {}, 2: {}, 3: {}, 4: {}}
    h[1][(0, 0)] = -4*M*R**3/D4
    h[1][(0, 1)] = 2*a*M*R**3*Y/(A*D4)
    h[1][(0, 2)] = -2*a*M*R**3*X/(A*D4)
    h[2][(0, 0)] = -M**2*R**2*(3*a**2 + 7*R**2)/(A*D4)
    h[2][(0, 1)] = 2*a*M**2*R**2*Y/(A*D4)
    h[2][(0, 2)] = -2*a*M**2*R**2*X/(A*D4)
    h[2][(1, 1)] = -M**2*(R**2*X**2*(R**2 - a**2) + a**2*A*(R**2 - Z**2))/(A**2*D4)
    h[2][(2, 2)] = -M**2*(R**2*Y**2*(R**2 - a**2) + a**2*A*(R**2 - Z**2))/(A**2*D4)
    h[2][(3, 3)] = -M**2*Z**2/D4
    h[2][(1, 3)] = -M**2*R**2*X*Z/(A*D4)
    h[2][(2, 3)] = -M**2*R**2*Y*Z/(A*D4)
    h[2][(1, 2)] = -M**2*R**2*X*Y*(R**2 - a**2)/(A**2*D4)
    h[3][(0, 0)] = -8*M**3*R**3/(A*D4)
    h[3][(0, 1)] = 2*a*M**3*R**3*Y/(A**2*D4)
    h[3][(0, 2)] = -2*a*M**3*R**3*X/(A**2*D4)
    h[4][(0, 0)] = -M**4*(A**2*at*(D4*at - 2*a*R**3)
                          + a**2*R**2*(16*a**4 + a**2*(34*R**2 - Z**2) + R**4))/(4*a**4*A**2*D4)
    h[4][(0, 1)] = 2*a*M**4*R**2*Y/(A**2*D4)
    h[4][(0, 2)] = -2*a*M**4*R**2*X/(A**2*D4)
    h[4][(1, 1)] = M**4*(R**2*(X**2 - Y**2)*W4 - a**3*A*(R**2 - Z**2))/(2*a*A**3*D4)
    h[4][(2, 2)] = -M**4*(R**2*(X**2 - Y**2)*W4 + a**3*A*(R**2 - Z**2))/(2*a*A**3*D4)
    h[4][(3, 3)] = M**4*(A*at - a*R)*(a**3*R*(Z**2 - 2*R**2) + A*D4*at - a*R**5)/(4*a**4*A**2*D4)
    h[4][(1, 3)] = M**4*R*X*Z*(A*at - a*R)/(2*a*A**2*D4)
    h[4][(2, 3)] = M**4*R*Y*Z*(A*at - a*R)/(2*a*A**2*D4)
    h[4][(1, 2)] = M**4*R**2*X*Y*W4/(a*A**3*D4)
    return h

def manuscript_metric(n):
    """hhat at order n from (h1), (h00PM)-(hijPM), (h004)-(hij4), Cartesian components"""
    f = khc.closed_form_functions(R, s2, c, a, (at - a*R/A)/(2*a**2))
    get = lambda name: f[name].get(n, 0)
    o = {(0, 0): get('h00'), (0, 1): -get('W')*Y, (0, 2): get('W')*X, (0, 3): 0,
         (1, 1): get('alp') + get('beta')*X**2, (2, 2): get('alp') + get('beta')*Y**2,
         (1, 2): get('beta')*X*Y, (1, 3): get('kappa')*X, (2, 3): get('kappa')*Y, (3, 3): get('alz')}
    return {k: M**n*v for k, v in o.items()}

def closed_forms_vs_canonical(control=False):
    """h_can(n) + hhat^(n) = 0 for n = 1..4 and the ten components, as an identity between functions
    of the harmonic point and of the spin (symbolic).  Returns the list of (n, component) that fail.
    With control=True the numerator 4(2 varrho^2 + a^2) = 8 varrho^2 + 4 a^2 of the first term of
    hhat^{00} at fourth order is changed to 9 varrho^2 + 4 a^2, and the list must be [(4, (0, 0))]."""
    ref = xing_metric()
    bad = []
    for n in (1, 2, 3, 4):
        ours = manuscript_metric(n)
        if control and n == 4:
            ours[(0, 0)] += M**4*R**2/(A**2*Sig)
        for k in KEYS:
            if on_shell(ref[n].get(k, sp.Integer(0)) + ours[k]) != 0:  # h_can = -hhat
                bad.append((n, k))
    return bad

def direct_transformation_vs_canonical(dps=40):
    """Taylor coefficients in G M of the Boyer-Lindquist metric transformed to (mainchart), at fixed
    harmonic point, against h_can(1..4): largest deviation over two points and ten components"""
    ref = xing_metric()
    with mp.workdps(dps):
        worst = mpf(0)
        for pt in khc.HARMONIC_POINTS[:2]:
            rho, vth, av = map(mpf, pt)
            sub = at_point(*pt, dps + 5)
            for (i, j) in khc.IDX2:
                tc = taylor(lambda e: khc.hhat_at_x(e, rho, vth, av)[i, j], 0, 4)
                for n in (1, 2, 3, 4):
                    val = mpf(sp.N(ref[n][(i, j)].subs(sub), dps)) if (i, j) in ref[n] else mpf(0)
                    worst = max(worst, abs(tc[n] + val))
        return worst

def jiang_lin_difference():
    """their eq. (61), the Jiang-Lin metric minus the canonical one at second order, against (CSdiff)
    with h = -hhat (symbolic).  Returns the list of spatial components for which (61) + (CSdiff) != 0."""
    x61 = {(1, 1): -2*M**2*a*R**3*X*Y/(A**2*D4), (2, 2): 2*M**2*a*R**3*X*Y/(A**2*D4), (3, 3): 0,
           (1, 2): a*M**2*R**3*(X**2 - Y**2)/(A**2*D4), (2, 3): a*M**2*R*X*Z/(A*D4), (1, 3): -a*M**2*R*Y*Z/(A*D4)}
    # (CSdiff) = -M^2 a sin(vartheta)/(sqrt(A) Sigma) (u^i e_phi^j + e_phi^i u^j), u = dx/dvarrho, and
    # sin(vartheta) e_phi/sqrt(A) = (-Y, X, 0)/A,  u = (R X/A, R Y/A, c)
    u = sp.Matrix([R*X/A, R*Y/A, c])
    v = sp.Matrix([-Y, X, 0])/A
    CSdiff = -M**2*a/Sig*(u*v.T + v*u.T)
    return [k for k, val in x61.items() if on_shell(val + CSdiff[k[0] - 1, k[1] - 1]) != 0]

def azimuth_transformation():
    """their eq. (69): varphi_2 - varphi, expanded through order m^3 (m = G M); returns 0"""
    chi = sp.sqrt(a**2 - m**2)
    Theta = a/chi*sp.atan(chi/R)                       # int_r^oo a du/(u^2 - 2mu + a^2), R = r - m
    d = -Theta + sp.atan(a/R) - m**2/(2*a**2)*(a*R/A - sp.atan(a/R))
    return sp.simplify(sp.series(d, m, 0, 4).removeO())

def exceptional_mode(p, c_p):
    """the homogeneous solution (exceptionalgaugemetric), 2 c_p (a.grad)^p d_i d_j (1/|x|), with a
    along z and G M = 1, in the format of the recursion"""
    f = (khc.ONE, 1)
    for _ in range(p):
        f = khc.sreduce(khc.sdiff(f, 3))
    out = {}
    for i in (1, 2, 3):
        for j in range(i, 4):
            t = khc.sscale(khc.sreduce(khc.sdiff(khc.sdiff(f, i), j)), 2*c_p)
            if t is not None:
                out[(i, j)] = t
    return out

def exceptional_sector(spin_orders=(0, 2, 4), NMAX=6, PMAX=4):
    """Returns (dims, ok).

    dims[p] = (number of coefficients of the general rotationally covariant, parity-even ansatz for
    a symmetric tensor of order a^p decaying as |x|^-(p+3), dimension of its harmonic subspace);
    the dimension is 1: the mode (exceptionalgaugemetric).

    ok is True if the recursion with the modes c_0 and c_2 added by hand at order G^3 runs through
    order G^NMAX a^PMAX without resonance and with all its assertions satisfied."""
    x_, y_, z_ = sp.symbols('x y z', real=True)
    Xs = (x_, y_, z_); r = sp.sqrt(x_**2 + y_**2 + z_**2)
    lap = lambda f: sum(sp.diff(f, v, 2) for v in Xs)
    Q = sp.Rational
    pts = [(Q(1, 3), Q(2, 5), Q(3, 7)), (Q(2, 3), Q(-1, 4), Q(5, 6)), (Q(7, 5), Q(1, 2), Q(-2, 3)),
           (Q(-3, 4), Q(5, 3), Q(1, 7)), (Q(1, 9), Q(4, 3), Q(2, 1)), (Q(5, 2), Q(-7, 3), Q(1, 5)),
           (Q(3, 8), Q(1, 6), Q(-5, 4)), (Q(-2, 7), Q(-3, 5), Q(4, 9))]
    dims = {}
    for p in spin_orders:
        cc = z_/r; n = [v/r for v in Xs]; zh = [0, 0, 1]; coeffs = []
        def poly(maxdeg, parity, tag):
            e = 0
            for k in range(parity, maxdeg + 1, 2):
                sy = sp.Symbol('%s%d' % (tag, k)); coeffs.append(sy); e += sy*cc**k
            return e
        f = poly(p, 0, 'f'); g = poly(p, 0, 'g')
        h = poly(p - 2, 0, 'h') if p >= 2 else 0
        k_ = poly(p - 1, 1, 'k') if p >= 1 else 0
        T = sp.Matrix(3, 3, lambda i, j: ((1 if i == j else 0)*f + n[i]*n[j]*g + zh[i]*zh[j]*h
                                          + Q(1, 2)*(n[i]*zh[j] + n[j]*zh[i])*k_)/r**(p + 3))
        eqs = [sp.nsimplify(sp.simplify(lap(T[i, j]).subs(dict(zip(Xs, q)))))
               for i in range(3) for j in range(i, 3) for q in pts]
        Mx = sp.Matrix([[sp.diff(e, sy) for sy in coeffs] for e in eqs])
        dims[p] = (len(coeffs), len(Mx.nullspace()))
    ok = True
    try:
        khc.run(NMAX, PMAX, hom={(3, 0): exceptional_mode(0, fmpq(3, 2)), (3, 2): exceptional_mode(2, fmpq(5, 7))})
    except (khc.Resonance, AssertionError):
        ok = False
    return dims, ok

def cook_scheel_relation(dps=40):
    """(CStocanonical): (x + i y)_CS = (1 - i tan(delta)) (x + i y), z_CS = z.  Returns the largest
    deviation at the points khc.EXACT_POINTS."""
    with mp.workdps(dps):
        worst = mpf(0)
        for pt in khc.EXACT_POINTS:
            r, tht, mm, av = map(mpf, pt)
            phv = mpf('0.37')
            xc, yc, zc = khc.chart_map(r, tht, phv, mm, av)
            xs, ys, zs = khc.chart_map(r, tht, phv, mm, av, cs=True)
            delta = khc.Theta(r, mm, av) - mp.atan(av/(r - mm))
            w = (1 - 1j*tan(delta))*(xc + 1j*yc)
            worst = max(worst, abs(w - (xs + 1j*ys)), abs(zc - zs))
        return worst

if __name__ == "__main__":
    ok = True
    bad = closed_forms_vs_canonical()
    ok = ok and not bad
    print("1. closed forms vs h_can(1..4) of Xing et al., 10 components, symbolic: %s"
          % ("exact agreement" if not bad else "MISMATCH at %s" % bad))
    ctl = closed_forms_vs_canonical(control=True) == [(4, (0, 0))]
    ok = ok and ctl
    print("   " + khc.control_line("9 varrho^2 instead of 8 varrho^2 in the fourth-order hhat^00", ctl,
                                   "flagged at that component only"))
    dev = direct_transformation_vs_canonical()
    ok = ok and dev < 1e-30
    print("2. Boyer-Lindquist -> chart (mainchart) -> Taylor coefficients vs h_can(1..4): max deviation %.1e"
          % float(dev))
    bad = jiang_lin_difference()
    ok = ok and not bad
    print("3. Jiang-Lin minus canonical at 2PM, their eq. (61), vs (CSdiff), symbolic: %s"
          % ("exact agreement" if not bad else "MISMATCH at %s" % bad))
    az = azimuth_transformation()
    ok = ok and az == 0
    print("4. their eq. (69): varphi_2 - varphi through O(m^3) =", az)
    dims, consistent = exceptional_sector()
    ok = ok and consistent and all(d == 1 for _, d in dims.values())
    for p, (nc, d) in dims.items():
        print("5. order (3,%d): general covariant ansatz with %d coefficients, harmonic subspace of dimension %d"
              % (p, nc, d))
    print("   recursion with the modes c_0, c_2 added by hand at G^3: consistent through G^6, no resonance: %s"
          % consistent)
    dev = cook_scheel_relation()
    ok = ok and dev < 1e-30
    print("6. (CStocanonical), Cook-Scheel vs canonical coordinates: max deviation %.1e" % float(dev))
    sys.exit(0 if ok else 1)
