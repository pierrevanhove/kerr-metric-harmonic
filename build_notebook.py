#!/usr/bin/env python3
"""Builds and executes checks.ipynb (run from anywhere; the notebook is written in the repository)."""
import os
import sys
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient

import json
import re

REPO = os.path.dirname(os.path.abspath(__file__))

# Equation numbers of the paper, extracted from its .aux file by paper_labels.py.  In the text
# below, equations are written by LaTeX label, `(recY)`; they are replaced by their number, (19).
with open(os.path.join(REPO, "paper-labels.json")) as f:
    LABELS = json.load(f)
missing = set()

def number(label):
    "printed number of the equation with LaTeX label eq:<label>"
    if "eq:" + label not in LABELS:
        missing.add(label)
        return label
    return LABELS["eq:" + label]

def md_numbers(text):
    "`(recY)` -> (19) in markdown"
    return re.sub(r"`\(([A-Za-z][A-Za-z0-9]*)\)`", lambda m: "(%s)" % number(m.group(1)), text)

def code_numbers(text):
    "(recY) -> eq. (19) inside the string literals of a code cell, for labels of the paper only"
    def one(k):
        key = "eq:" + k.group(1)
        return "\x00(%s)" % LABELS[key] if key in LABELS else k.group(0)
    def literal(m):
        t = re.sub(r"\(([A-Za-z][A-Za-z0-9]*)\)", one, m.group(0))
        t = re.sub(r"\x00(\([\w.]+\))(-| and )\x00(\([\w.]+\))", r"eqs. \1\2\3", t)
        return t.replace("\x00", "eq. ")
    return re.sub(r'"[^"\n]*"', literal, text)

cells = []
md = lambda s: cells.append(new_markdown_cell(md_numbers(s.strip("\n"))))
code = lambda s: cells.append(new_code_cell(code_numbers(s.strip("\n"))))

# ------------------------------------------------------------------------------------------------
md(r"""
# The Kerr metric in harmonic coordinates: explicit checks

This notebook runs the checks reported in

> A. Sadu and P. Vanhove, *The Kerr Metric in Canonical Harmonic Coordinates to All Orders*.

Each section says what is checked and where it is used in the paper, calls the corresponding functions of the Python modules of this repository, and prints the result. Two kinds of result appear:

- **exact**: comparisons in exact rational arithmetic, or symbolic identities. They hold or they fail.
- **numerical**: multiprecision evaluations at generic points, for which the largest deviation is printed.

Several checks are followed by a *control*: the same check run on a deliberately wrong input, for instance one coefficient changed in a formula. A control succeeds when the check **rejects** the wrong input, so the mismatches it reports are expected: they show that the check is able to fail. A successful control prints `control OK, wrong input rejected: ...`; `CONTROL FAILED` would signal a problem. The controls are listed in a second table at the end.

Equation numbers are those of the paper. The Python modules refer to the equations by their LaTeX labels instead, for instance `recY` for the equation labelled `eq:recY` in the source; the file `paper-labels.json` gives the number of each label. The notebook runs in about one minute.

**Conventions.** Signature $(-,+,+,+)$; $D=4$ except in section 1; spin $\mathbf a$ along $z$. The gothic inverse metric is $\mathfrak g^{\mu\nu}=\sqrt{-g}\,g^{\mu\nu}$ and $\mathfrak h^{\mu\nu}=\eta^{\mu\nu}-\mathfrak g^{\mu\nu}$. The pair $(n,p)$ labels the term proportional to $G_N^n M^n a^p$.
""")

code(r"""
import time, platform
import flint, sympy, mpmath
import cubic_action_identities as cai
import worldline_source_checks as wsc
import kerr_harmonic_checks as khc
import kerr_relaxed_einstein_checks as kre
import kerr_canonical_mpm_checks as kmpm
import kerr_260519948_v1_4pm_checks as kd4

print("Python", platform.python_version(), "| python-flint", flint.__version__,
      "| sympy", sympy.__version__, "| mpmath", mpmath.__version__)

summary = []
def record(check, where, rng, result, passed):
    "store one line of the summary table and print it"
    summary.append((check, where, rng, result, bool(passed)))
    print(("PASS  " if passed else "FAIL  ") + check + " [" + rng + "]: " + result)

controls = []
def control(wrong_input, rejected, detail=""):
    "a check run on a deliberately wrong input, which it must reject: print and store the outcome"
    controls.append((wrong_input, bool(rejected)))
    print(khc.control_line(wrong_input, bool(rejected), detail))

t_start = time.time()
""")

# ------------------------------------------------------------------------------------------------
md(r"""
## 1. The cubic formulation of the action (section 2 and appendix D)

With the auxiliary field $A^\lambda{}_{\mu\nu}$ and the bilinear form

$$Q(A,B;\mathfrak g)=\Big(A^\lambda{}_{\mu\rho}B^\mu{}_{\lambda\sigma}-\frac{1}{D-1}A_\rho B_\sigma\Big)\mathfrak g^{\rho\sigma},\qquad A_\sigma=A^\lambda{}_{\lambda\sigma},$$

the Lagrangian `(action)` is cubic in $(A,\mathfrak g)$. The shift $A=A_{\rm lin}[\mathfrak h]+Y$ of `(Ydef)` brings it to the form `(Lbulk)`, with a diagonal quadratic part and three cubic vertices. The identities behind this rewriting and behind the field equations `(Yeq)` and `(heq)` are polynomial identities in the components of the tensors. They are tested in exact arithmetic on tensors with random integer components, for $D=3,\dots,6$ and three random draws. Derivatives of $\mathfrak h$ are replaced by random tensors with the same symmetries.
""")

code(r"""
dims = (3, 4, 5, 6)
table = cai.all_dimensions(dims)
width = max(map(len, table))
print("%-*s  %s" % (width, "identity", "  ".join("D=%d" % D for D in dims)))
for name, row in table.items():
    print("%-*s  %s" % (width, name, "  ".join("ok " if row[D] else "NO " for D in dims)))
print()
record("identities of the cubic action", "section 2, appendix D", "D = 3,...,6",
       "%d identities hold" % len(table), all(all(row.values()) for row in table.values()))
failing = [name for name, ok in cai.identities(4, control=True).items() if not ok]
control("1/(2(D-1)) instead of 1/(2(D-2)) in (E0inv), D = 4", failing,
        "%d identities fail, as expected" % len(failing))
""")

# ------------------------------------------------------------------------------------------------
md(r"""
## 2. The linear coupling of the worldline action (section 3)

For a body at rest and a static perturbation $\delta g_{\mu\nu}$ of flat space, the electric and magnetic parts of the curvature `(EB)` linearise to `(EBlin)`,

$$E_{ij}=-\tfrac12\,\partial_i\partial_j\,\delta g_{00},\qquad B_{ij}=\tfrac12\,\partial_j\big(\nabla\times\boldsymbol\gamma\big)_i,\qquad \gamma_i=\delta g_{0i}.$$

This is derived below from the Christoffel symbols, with ten arbitrary functions for $\delta g_{\mu\nu}$ and the conventions of the paper ($\epsilon_{0123}=+1$, $R^\rho{}_{\sigma\mu\nu}=\partial_\mu\Gamma^\rho_{\nu\sigma}-\dots$). With all Wilson coefficients equal to one, the couplings of the worldline action `(Swl)` then sum to the Kerr stress tensor `(Trest)`,

$$T^{00}(\mathbf k)=M\cosh(\mathbf a\cdot\mathbf k),\qquad T^{0i}(\mathbf k)=\frac i2\,(\mathbf k\times\mathbf S)^i\,\frac{\sinh(\mathbf a\cdot\mathbf k)}{\mathbf a\cdot\mathbf k}.$$

The comparison is made on a plane wave, order by order in the spin through $a^{13}$. The linearised spin kinetic term is used in the form `(spinconnectionlin)`.
""")

code(r"""
res = wsc.curvature_linearisation()
for name, ok in res.items():
    print("%-46s %s" % (name, "ok" if ok else "NO"))
print()
record("linearised curvature (EBlin) from the definitions (EB)", "section 3.2", "static perturbation",
       "all relations hold", all(res.values()))
ok, order = wsc.stress_tensor_resummation(13)
record("worldline action (Swl) vs stress tensor (Trest)", "section 3", "p <= %d" % order,
       "exact agreement" if ok else "mismatch", ok)
control("Wilson coefficient C_ES^4 = 2 instead of 1", not wsc.stress_tensor_resummation(13, wilson={4: 2})[0])
""")

# ------------------------------------------------------------------------------------------------
md(r"""
## 3. The recursion (sections 2.3, 2.4 and 7)

Outside the source and for a stationary field, the equations of the cubic action give the recursion `(recY)`–`(recJ)`,

$$Y^{(n)}=\mathcal E_0^{-1}\Big[\sum_{m=1}^{n-1}\mathcal E\big(A^{(m)};\mathfrak h^{(n-m)}\big)\Big],$$

$$\tfrac12\,\Delta\Big(\mathfrak h^{(n)}_{\mu\nu}-\tfrac12\,\eta_{\mu\nu}\mathfrak h^{(n)}\Big)=-\sum_{m=1}^{n-1}\Big(A^{(m)\rho}{}_{\sigma\mu}A^{(n-m)\sigma}{}_{\rho\nu}-\tfrac13A^{(m)}_\mu A^{(n-m)}_\nu\Big)-\partial_\lambda Y^{(n)\lambda}{}_{\mu\nu},$$

with $A^{(n)}=A_{\rm lin}[\mathfrak h^{(n)}]+Y^{(n)}$. It starts from the linear field `(h1)` of the Kerr multipoles. Every field is a sum of terms $N(\mathbf x)/|\mathbf x|^m$, with $N$ a polynomial with rational coefficients. The source of the Poisson equation is decomposed into harmonic polynomials $H_\ell$ and the equation is solved with `(invlap)`,

$$\Delta^{-1}\,\frac{H_\ell(\mathbf x)}{|\mathbf x|^{\mu}}=\frac{1}{(\mu-2)(\mu-2\ell-3)}\,\frac{H_\ell(\mathbf x)}{|\mathbf x|^{\mu-2}}.$$

No homogeneous solution is added. A source term with $\mu=2\ell+3$ would be resonant and would require a logarithm; the code raises an exception if one occurs.

`khc.run(NMAX, PMAX)` runs the recursion in units $G_N=M=1$ and returns the fields as `h[n][p][(mu, nu)]`. At each order it asserts the equation `(Yeq)` of the auxiliary field, which is `(Aeq)` written for $Y$, the Poisson equation `(recJ)` (the Laplacian of the solution gives back the source), and the harmonic gauge condition $\partial_\mu\mathfrak h^{\mu\nu}=0$.
""")

code(r"""
NMAX, PMAX = 5, 12
t0 = time.time()
h, A, Y, slots = khc.run(NMAX, PMAX)
print("recursion through G^%d a^%d in %.1f s" % (NMAX, PMAX, time.time() - t0))
print("orders (n, p):", sum(len(h[n]) for n in h),
      "| non-vanishing components of the metric:", sum(len(h[n][p]) for n in h for p in h[n]))
print("sources whose range of harmonic degrees contains the would-be resonant one:", len(slots))
print()
record("cubic recursion: equations of A and of the metric, harmonic gauge", "sections 2.3, 7",
       "n <= %d, p <= %d" % (NMAX, PMAX), "hold at every order", True)
record("resonances (logarithms)", "sections 4, 7", "n <= %d, p <= %d" % (NMAX, PMAX), "none", True)
""")

md(r"""
A few low orders, to show the format. For $a=0$ the recursion gives the Schwarzschild metric in harmonic coordinates, $\mathfrak h^{00}=4\rho+7\rho^2+8\rho^3+8\rho^4+8\rho^5+\dots$ and $\mathfrak h^{ij}=\rho^2\,n^in^j$, with $\rho=G_NM/|\mathbf x|$. The term linear in the spin is $\mathfrak h^{0i}=2\,a\,(\hat{\mathbf z}\times\mathbf x)^i/|\mathbf x|^{n+2}$ at every order $n$.
""")

code(r"""
for n in range(1, NMAX + 1):
    print("n = %d, p = 0:" % n, {k: khc.sstr(v) for k, v in h[n][0].items()})
print()
for n in (1, 2, 5):
    print("n = %d, p = 1:" % n, {k: khc.sstr(v) for k, v in h[n][1].items()})
print()
print("n = 2, p = 2, component (0, 0):", khc.sstr(h[2][2][(0, 0)]))
""")

# ------------------------------------------------------------------------------------------------
md(r"""
## 4. Closed forms of the post-Minkowskian expansion (section 6)

Section 6 gives the terms of order $G_N^n$, $n\le5$, in closed form in the flat oblate spheroidal coordinates $(\varrho,\vartheta)$ of the harmonic point: `(h00PM)`–`(hijPM)` through third order, `(h004)`–`(hij4)` at fourth order and `(h005)` at fifth order. From fourth order on they contain

$$\delta_2(\varrho)=\frac{1}{2a^2}\Big[\arctan\frac{a}{\varrho}-\frac{a\varrho}{\varrho^2+a^2}\Big].$$

These expressions are written once in the code, in `khc.closed_form_functions`, and that single transcription is used in sections 4, 5 and 9. Here they are expanded in the spin through $a^{12}$, and each term is compared with the output of the recursion as a rational function of $\mathbf x$: ten components at each of the $5\times13$ orders $(n,p)$.
""")

code(r"""
S = khc.structure_series(PMAX)                 # spin expansion of the closed forms
closed = khc.closed_fields(S, NMAX, PMAX)
bad = khc.compare(h, closed, NMAX, PMAX)
record("cubic recursion vs closed forms", "sections 6, 7", "n <= %d, p <= %d, 10 components" % (NMAX, PMAX),
       "exact agreement" if not bad else "mismatch at %s" % bad[:3], not bad)
for name, rejected in khc.negative_controls(h, closed, NMAX, PMAX).items():
    control(name, rejected)
""")

md(r"""
The three controls are deliberately wrong inputs, and the comparison rejects each of them: one coefficient of the closed forms changed; the leading time-odd term by which the Cook–Scheel chart differs from the recursion at order $G_N^2a$, $-2\,x^{(i}(\hat{\mathbf z}\times\mathbf x)^{j)}/|\mathbf x|^5$ from `(CSdiff)`, added to the closed forms; and $\cos$ instead of $\cosh$ in the momentum-space form factor of the source, which spoils the agreement at order $G_N^2a^2$.
""")

# ------------------------------------------------------------------------------------------------
md(r"""
## 5. The exact metric (section 5)

The chart `(chart)` is

$$T=t,\qquad x+iy=\sqrt{R^2+a^2}\,\cos\delta(r)\,\sin\theta\,e^{i\phi},\qquad z=R\cos\theta,\qquad R=r-G_NM,$$

in terms of the Boyer–Lindquist coordinates, with $\delta(r)=\Theta(r)-\arctan(a/R)$ and $\Theta(r)=\int_r^\infty a\,du/(u^2-2G_NMu+a^2)$. The checks of this section are numerical, with 40 to 50 digits. The points cover the sub-extremal, extremal and super-extremal cases and both signs of the spin.

- The closed forms `(Theta)` of $\Theta$ and the integral representation `(delta)` of $\delta$.
- The coordinates are harmonic, $\Box_g x^A=0$, for the chart `(chart)` and for the Cook–Scheel chart `(CS)`.
- The components `(gexact1)`–`(gexact3)` of the gothic metric, and the Cook–Scheel components `(CSframe)`, against the tensor-density transformation of the Boyer–Lindquist metric, with the Jacobian obtained by numerical differentiation of the coordinate map.
- The closed forms of section 6 are the expansion of `(gexact1)`–`(gexact3)` in $G_NM$ at a fixed harmonic point, through fifth order. This is checked twice: symbolically, as an identity in $\varrho$, $\vartheta$, the spin and $\delta_2$ (the only exact check of this section), and numerically, against the Taylor coefficients of the directly transformed metric. Together with section 4 this compares the recursion with the exact metric.
- The relation `(Rtheta4)` between the Boyer–Lindquist variables $(R,\theta)$ of a harmonic point and its flat oblate coordinates $(\varrho,\vartheta)$: they first differ at order $G_N^4$.
""")

code(r"""
TOL = 1e-30
d = float(khc.theta_check())
record("(Theta) and (delta) vs their integral representations", "sections 1, 5.2", "5 points",
       "max deviation %.0e" % d, d < TOL)
res = {k: float(v) for k, v in khc.harmonicity_check().items()}
record("harmonic coordinates, Box x = 0", "sections 5.1, 5.2", "chart and Cook-Scheel, 5 points",
       "relative residual %.0e and %.0e" % (res["chart"], res["Cook-Scheel"]), max(res.values()) < TOL)
res = {k: float(v) for k, v in khc.exact_formula_check().items()}
record("(gexact1)-(gexact3) vs direct transformation of Boyer-Lindquist", "section 5.3", "5 points",
       "max deviation %.0e" % res["chart"], res["chart"] < TOL)
record("Cook-Scheel components (CSframe) vs direct transformation", "section 5.3", "5 points",
       "max deviation %.0e" % res["Cook-Scheel"], res["Cook-Scheel"] < TOL)
bad = khc.closed_forms_vs_exact_formulas()
record("closed forms vs expansion of (gexact1)-(gexact3) in G_N M, symbolic", "sections 5.3, 6", "n <= 5, all orders in the spin",
       "exact agreement" if not bad else "mismatch at %s" % bad, not bad)
control("5 instead of 4 as coefficient of varrho delta_2^2/Sigma in the fifth-order hhat^00",
        khc.closed_forms_vs_exact_formulas(control=True) == [("00", 5)], "flagged at that component only")
d = float(khc.closed_vs_exact())
record("closed forms vs Taylor coefficients of the directly transformed metric", "sections 6, 7", "n <= 5, 4 points, 10 components",
       "max deviation %.0e" % d, d < TOL)
dev, low = map(float, khc.inversion_check())
record("(Rtheta4): (R, theta) vs flat oblate coordinates", "section 6.3", "orders G^1,...,G^5",
       "max deviation %.0e" % max(dev, low), max(dev, low) < TOL)
""")

# ------------------------------------------------------------------------------------------------
md(r"""
## 6. Orders beyond the fifth

The recursion runs to higher orders at little cost. As a test beyond the range quoted in the paper, its output through $G_N^8$, summed over the spin orders $p\le12$, is compared with the Taylor coefficients of the exact metric at the point $\mathbf x=(3/5,0,4/5)$ for the small spin $a/|\mathbf x|=10^{-3}$. The neglected spin orders are of relative size $a^{13}=10^{-39}$, which sets the expected deviation.
""")

code(r"""
h8, A8, Y8, slots8 = khc.run(8, PMAX)
worst, trunc = khc.recursion_vs_exact(h8, 8, PMAX)
print("  ".join("G^%d: %.0e" % (n, float(worst[n])) for n in range(1, 9)))
print()
record("cubic recursion vs exact metric, numerical", "beyond the paper", "n <= 8, a = 1/1000",
       "max deviation %.0e (neglected spin orders: %.0e)" % (float(max(worst.values())), float(trunc)),
       max(worst.values()) < 100*trunc)
""")

# ------------------------------------------------------------------------------------------------
md(r"""
## 7. Relaxed Einstein equations and the canonical iteration (appendix A and section 5.6)

In harmonic gauge the Einstein equations take the Landau–Lifshitz form `(LLwave)`, $\Box\,\mathfrak h^{\alpha\beta}=\Lambda^{\alpha\beta}$, with

$$\Lambda^{\alpha\beta}=\mathfrak h^{\mu\nu}\partial_\mu\partial_\nu\mathfrak h^{\alpha\beta}-\partial_\mu\mathfrak h^{\alpha\nu}\,\partial_\nu\mathfrak h^{\beta\mu}-16\pi G_N(-g)\,t_{LL}^{\alpha\beta}$$

and $t_{LL}$ the Landau–Lifshitz pseudo-tensor given in appendix A. The source $\Lambda$ is evaluated here on the output of the cubic recursion. Its term of order $n$ involves only the orders below $n$. For each $n\ge2$ the field $u_{(n)}=\Delta^{-1}\Lambda_{(n)}$, computed with `(invlap)`, is compared with $\mathfrak h^{(n)}$. By induction on $n$, the recursion based on `(LLwave)` and the cubic recursion have the same solution.

The same computation is the iteration `(MPMalgorithm)` of the multipolar post-Minkowskian formalism for the canonical metric, with $h=-\mathfrak h$. The check therefore also establishes, at the orders computed, two properties stated in section 5.6: no resonant term occurs, so that no finite part is needed, and $\partial_\mu u^{\mu\nu}_{(n)}=0$, so that the homogeneous term $v_{(n)}$ vanishes.
""")

code(r"""
def ll_report(h, nmax, pmax, where):
    "the Landau-Lifshitz source evaluated on the output h of the cubic recursion"
    res = kre.landau_lifshitz_iteration(h, nmax, pmax)
    ok = not (res["resonances"] or res["mismatches"] or res["divergence"])
    print("orders: %d | non-vanishing source components: %d | resonant terms: %d | mismatches: %d | "
          "orders with non-zero divergence: %d" % (res["orders"], res["source components"],
          len(res["resonances"]), len(res["mismatches"]), len(res["divergence"])))
    record("Landau-Lifshitz recursion vs cubic recursion", where, "n <= %d, p <= %d" % (nmax, pmax),
           "exact agreement, no resonance, zero divergence" if ok else "failure", ok)

ll_report(h, NMAX, PMAX, "appendix A, section 7")
ctl = kre.landau_lifshitz_iteration(h, 3, 2, control=True)
control("1/4 instead of 1/8 in the last term of t_LL", ctl["mismatches"],
        "%d mismatches for n <= 3, p <= 2, as expected" % len(ctl["mismatches"]))
""")

md(r"""
The iteration can also be run on its own, from the linear field `(h1)` and without reference to the cubic recursion: each order is built from the orders obtained before it. This is the check quoted in section 5.6, through $G_N^8$ and $a^{12}$. Its output is compared with that of the cubic recursion at the end.
""")

code(r"""
t0 = time.time()
hc, resc = kre.canonical_iteration(8, PMAX)
differences = kre.same_fields(hc, h8, 8, PMAX)
print("iteration through G^8 a^%d in %.1f s | resonant terms: %d | orders with non-zero divergence: %d | "
      "differences with the cubic recursion: %d" % (PMAX, time.time() - t0, len(resc["resonances"]),
      len(resc["divergence"]), len(differences)))
ok = not (resc["resonances"] or resc["divergence"] or differences)
record("canonical iteration (MPMalgorithm) from the linear field alone", "sections 5.6, 7", "n <= 8, p <= %d" % PMAX,
       "no resonance, zero divergence, same output as the cubic recursion" if ok else "failure", ok)
""")

md(r"""
On shell, the auxiliary field is the combination `(Aonshell)` of Christoffel symbols, $A^\lambda{}_{\mu\nu}=-\Gamma^\lambda_{\mu\nu}+\frac12\big(\delta^\lambda_\mu\partial_\nu+\delta^\lambda_\nu\partial_\mu\big)\log\sqrt{-g}$. In terms of $\mathfrak g^{\mu\nu}$ and of its inverse $K_{\mu\nu}$ this is, in four dimensions,

$$A^\lambda{}_{\mu\nu}=-\tfrac12\,\mathfrak g^{\lambda\sigma}\big(\partial_\mu K_{\sigma\nu}+\partial_\nu K_{\sigma\mu}-\partial_\sigma K_{\mu\nu}\big)-\tfrac14\,\mathfrak g^{\lambda\sigma}K_{\mu\nu}\,K_{\alpha\beta}\,\partial_\sigma\mathfrak h^{\alpha\beta},$$

which is expanded and compared with the auxiliary field $A^{(n)}=A_{\rm lin}[\mathfrak h^{(n)}]+Y^{(n)}$ of the recursion.
""")

code(r"""
bad = kre.onshell_auxiliary_field(h, A, NMAX, PMAX)
record("on-shell auxiliary field (Aonshell) vs recursion", "sections 2.1, 7", "n <= %d, p <= %d" % (NMAX, PMAX),
       "exact agreement" if not bad else "mismatch at %s" % bad[:3], not bad)
control("1/2 instead of 1/4 in the last term of (Aonshell)", kre.onshell_auxiliary_field(h, A, 2, 2, control=True))
""")

# ------------------------------------------------------------------------------------------------
md(r"""
## 8. The exceptional sector (section 4.2)

A decaying homogeneous solution compatible with rotational covariance, parity and time reversal can occur only in the spatial components at order $G_N^3a^p$, with angular degree $\ell=p+2$. Section 4.2 shows that this sector is spanned by the gauge mode `(exceptionalgaugevector)`, `(exceptionalgaugemetric)`,

$$\delta\mathfrak h^{ij}=\partial^i\xi^j+\partial^j\xi^i,\qquad \xi^i=c_p\,(G_NM)^3\,(\mathbf a\cdot\nabla)^p\,\partial^i\frac{1}{|\mathbf x|}.$$

- For $p=0,2,4$, the general symmetric tensor built from $\mathbf n$, $\hat{\mathbf z}$ and $\delta^{ij}$ with the required parity and decay is written with free coefficients, and the dimension of its harmonic subspace is computed. It is one.
- The recursion is run with the modes $c_0$ and $c_2$ added by hand at third order. It stays free of resonances and all its equations hold, here through $G_N^6a^4$: the mode is the linear term of a residual harmonic coordinate transformation.
""")

code(r"""
dims3, consistent = kmpm.exceptional_sector()
for p, (ncoef, dim) in dims3.items():
    print("order (3, %d): ansatz with %2d coefficients, harmonic subspace of dimension %d" % (p, ncoef, dim))
print()
record("exceptional sector at order (3, p)", "section 4.2", "p = 0, 2, 4",
       "one harmonic tensor for each p", all(dim == 1 for _, dim in dims3.values()))
record("recursion with the exceptional modes added by hand", "section 4.2", "n <= 6, p <= 4",
       "consistent, no resonance" if consistent else "failure", consistent)
""")

# ------------------------------------------------------------------------------------------------
md(r"""
## 9. Canonical harmonic coordinates (section 5.6)

The canonical harmonic metric of Kerr in the multipolar post-Minkowskian formalism was obtained through fourth order, to all orders in the spin, by Xing, He and Cao, [arXiv:2608.05713](https://arxiv.org/abs/2608.05713). Their eqs. (29), (32), (33), (34), (61) and (69) are transcribed in `kerr_canonical_mpm_checks.py`, with $h_{\rm can}=\sqrt{-g}\,g-\eta=-\mathfrak h$.

- $h_{{\rm can}(n)}+\mathfrak h^{(n)}=0$ for $n\le4$ and the ten components, with the closed forms of section 6. This is verified as an identity in the harmonic point and in the spin: $\arctan(a/\varrho)$ is treated as an independent variable and the relation $x^2+y^2=(\varrho^2+a^2)\sin^2\vartheta$ is imposed.
- The same comparison starting from the Boyer–Lindquist metric: transformation to the chart `(mainchart)`, then Taylor coefficients in $G_NM$ at a fixed harmonic point (numerical). At fourth order this tests the factor $\cos\delta$ of the chart.
- Their eq. (61), the Jiang–Lin metric minus the canonical one at second order, against `(CSdiff)` (identity).
- Their eq. (69) for the azimuth, which reduces to $\varphi_2=\varphi+\mathcal O(G_N^4)$.
- The relation `(CStocanonical)` between the Cook–Scheel (Jiang–Lin) coordinates and the chart, $(x+iy)_{\rm CS}=(1-i\tan\delta)\,(x+iy)$ (numerical).
""")

code(r"""
bad = kmpm.closed_forms_vs_canonical()
record("closed forms vs canonical MPM metric of Xing et al.", "section 5.6", "n <= 4, 10 components",
       "exact agreement" if not bad else "mismatch at %s" % bad, not bad)
control("9 varrho^2 instead of 8 varrho^2 in the fourth-order hhat^00",
        kmpm.closed_forms_vs_canonical(control=True) == [(4, (0, 0))], "flagged at that component only")
d = float(kmpm.direct_transformation_vs_canonical())
record("Boyer-Lindquist -> chart -> Taylor coefficients vs canonical metric", "section 5.6", "n <= 4, 2 points",
       "max deviation %.0e" % d, d < TOL)
bad = kmpm.jiang_lin_difference()
record("Jiang-Lin minus canonical, their eq. (61), vs (CSdiff)", "sections 5.5, 5.6", "order G^2",
       "exact agreement" if not bad else "mismatch at %s" % bad, not bad)
az = kmpm.azimuth_transformation()
record("their eq. (69): azimuth of the canonical chart", "section 5.6", "through G^3",
       "varphi_2 - varphi = %s" % az, az == 0)
d = float(kmpm.cook_scheel_relation())
record("(CStocanonical): Cook-Scheel vs canonical coordinates", "section 5.6", "5 points",
       "max deviation %.0e" % d, d < TOL)
""")

# ------------------------------------------------------------------------------------------------
md(r"""
## 10. The gauge-term-free recursion at fourth order (section 5.5)

Damgaard, Lee, Lee and Rahnuma, [arXiv:2605.19948](https://arxiv.org/abs/2605.19948), solve the harmonic-gauge recursion without the gauge term of order $G_N^2$ in the spatial components. They give the term of order $G_N^4a^p$ of the result, $\mathfrak h'^{\mu\nu}$, as finite sums of Jacobi polynomials $P_k^{(0,\beta)}(u)$ of $u=1-2z^2/r^2$: their eq. (5.19), repeated in (D.19), for $\mathfrak h'^{00}$; (D.19) for $\mathfrak h'^{0i}$; and (D.20) for the spatial components. Their field is defined by $\sqrt{-g}\,g^{\mu\nu}=\eta^{\mu\nu}-\mathfrak h^{\mu\nu}$, as here. The formulas are transcribed in `kerr_260519948_v1_4pm_checks.py`.

Eq. (D.20) contains misprints, and it is read as in the footnote of section 5.5 of the paper. $\mathfrak h'^{12}$ is the coefficient of $a^n$ with $n$ even. In $\mathfrak h'^{11}$, for the coefficient of $a^{2N}$, $m=N-1$ and the prefactor is $1/r^{2N+6}$ (printed: $1/r^{2n+8}$). The sum that appears in $\mathfrak h'^{12}$ is added to the coefficient of $x^2$, as required by axial symmetry, $\mathfrak h^{11}-\mathfrak h^{22}=(x^2-y^2)\,\mathfrak h^{12}/(xy)$.

Their expressions are compared, in exact arithmetic and for all ten components, with the spin expansion of the closed forms `(h004)`–`(hij4)` and with the output of the recursion at order $G_N^4$, through $a^{20}$. Damgaard et al. verified their general formulas through $a^{16}$.
""")

code(r"""
PD = 20
closed4, recursion4 = kd4.fourth_order_references(PD)
theirs4 = kd4.their_fourth_order(PD)
ncomp = sum(len(v) for v in theirs4.values())
bad = kd4.mismatches(theirs4, closed4, PD)
record("Damgaard et al. (5.19), (D.19), (D.20) vs closed forms (h004)-(hij4)", "sections 5.5, 7",
       "n = 4, p <= %d, 10 components" % PD,
       "exact agreement (%d non-vanishing components)" % ncomp if not bad else "mismatch at %s" % bad[:3], not bad)
bad = kd4.mismatches(theirs4, recursion4, PD)
record("Damgaard et al. (5.19), (D.19), (D.20) vs cubic recursion", "sections 5.5, 7", "n = 4, p <= %d" % PD,
       "exact agreement" if not bad else "mismatch at %s" % bad[:3], not bad)
control("59n+127 instead of 59n+126 in (5.19)",
        kd4.mismatches(kd4.their_fourth_order(PD, wrong_coefficient=True), closed4, PD))
lit = kd4.mismatches(kd4.their_fourth_order(PD, read_D20=False), closed4, PD)
print("reading of (D.20): without the sum of h'^12 in h'^11 and h'^22, the comparison fails at the orders a^p, p =",
      sorted({p for p, _ in lit}), "in the components", sorted({k for _, k in lit}))
""")

md(r"""
The last line shows that the reading of the footnote is needed. Without the sum of $\mathfrak h'^{12}$ in the diagonal components, $\mathfrak h'^{11}$ and $\mathfrak h'^{22}$ disagree from order $a^4$ on, where that sum first contributes. The other components agree as printed.
""")

md(r"""
## 11. The coefficients of appendix C

Appendix C writes the double expansion as

$$\mathfrak h^{00}=\sum_{n,p}\rho^n\Big(\frac{a}{|\mathbf x|}\Big)^p h_{n,p}(\xi),\qquad \mathfrak h^{0i}=(\hat{\mathbf z}\times\mathbf n)^i\sum_{n,p}\rho^n\Big(\frac{a}{|\mathbf x|}\Big)^p w_{n,p}(\xi),$$

and similarly for the functions $\alpha_\perp$, $\alpha_z$, $|\mathbf x|^2\beta$, $|\mathbf x|\,\kappa$ of the spatial components, with $\xi=\hat{\mathbf z}\cdot\mathbf n$ and $\rho=G_NM/|\mathbf x|$. The coefficients through $G_N^5a^{12}$ are in the file `kerr-harmonic-coefficients.m`. The cell below regenerates them from the expansion of section 4, which agrees with the recursion, compares them with the file, and prints the lowest spin orders. In the file the coefficients are called `h00`, `w`, `aPerp`, `aZ`, `b`, `k`.
""")

code(r"""
text = khc.coefficients_text(S, NMAX, PMAX)
with open("kerr-harmonic-coefficients.m") as f:
    same = f.read() == text
ncoef = sum(len(block) for block in khc.coefficient_lines(S, NMAX, PMAX))
record("coefficient file kerr-harmonic-coefficients.m", "appendix C", "n <= %d, p <= %d" % (NMAX, PMAX),
       "%d coefficients, identical to the file" % ncoef, same)
print()
for block in khc.coefficient_lines(S, NMAX, 2):
    print("\n".join(block), "\n")
""")

# ------------------------------------------------------------------------------------------------
md(r"""
## Summary
""")

code(r"""
from IPython.display import Markdown, display
rows = ["| check | in the paper | range | result | |", "|---|---|---|---|---|"]
for check, where, rng, result, passed in summary:
    rows.append("| %s | %s | %s | %s | %s |" % (check, where, rng.replace("<=", "≤"), result, "pass" if passed else "**FAIL**"))
display(Markdown("\n".join(rows)))
rows = ["| control: deliberately wrong input | rejected, as it must be |", "|---|---|"]
for wrong_input, rejected in controls:
    rows.append("| %s | %s |" % (wrong_input, "yes" if rejected else "**NO: CONTROL FAILED**"))
display(Markdown("\n".join(rows)))
print("%d checks, %d passed; %d controls, %d rejected their wrong input as expected; total time %.0f s"
      % (len(summary), sum(s[-1] for s in summary), len(controls), sum(c[1] for c in controls), time.time() - t_start))
assert all(s[-1] for s in summary) and all(c[1] for c in controls)
""")


if missing:
    sys.exit("labels not found in paper-labels.json: %s" % sorted(missing))
nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
out = os.path.join(REPO, "checks.ipynb")
if "--no-run" not in sys.argv:
    client = NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": REPO}})
    client.execute()
nbformat.write(nb, out)
print("written", out, "cells:", len(nb.cells))
