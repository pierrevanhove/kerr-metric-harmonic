# The Kerr metric in canonical harmonic coordinates: recursion and checks
[![DOI](https://zenodo.org/badge/1458929082.svg)](https://doi.org/10.5281/zenodo.24397113)
![arXiv: 26mm.nnnn](https://img.shields.io/badge/arXiv-26mm.nnnn-b31b1b.svg)

![Author: Amina Sadu](https://img.shields.io/badge/author-Amina_Sadu-blue)
[![Author: Pierre Vanhove](https://img.shields.io/badge/author-Pierre_Vanhove-blue)](https://pierrevanhove.github.io)


![Language: Python](https://img.shields.io/badge/Language-Python-yellow?logo=python)
![Python: 3.14.10](https://img.shields.io/badge/Python-3.14.10-blue?logo=python)
![python-flint: 0.9.0](https://img.shields.io/badge/python--flint-0.9.0-orange)
![SymPy: 1.14.0](https://img.shields.io/badge/SymPy-1.14.0-green)
![mpmath: 1.3.0](https://img.shields.io/badge/mpmath-1.3.0-purple)

Code and data for the paper

> Amina Sadu and Pierre Vanhove, *The Kerr Metric in Canonical Harmonic Coordinates to All Orders* (arXiv number to be added).

The paper reconstructs the Kerr metric in harmonic coordinates from a perturbative recursion in Newton's constant, started from the Kerr multipoles. It shows that the chart selected by the recursion is the canonical harmonic chart of the multipolar post-Minkowskian formalism, to all orders, and gives it in closed form. This repository contains

- an implementation of the recursion in position space, in exact rational arithmetic;
- the checks reported in the paper, as Python modules and as a notebook stored with its outputs, [`checks.ipynb`](checks.ipynb);
- the coefficients of the double expansion in $G_N$ and in the spin of appendix C, through order $G_N^5a^{12}$.

## Quick start

```
pip install -r requirements.txt
python run_all_checks.py            # all checks, about one minute
```

The notebook [`checks.ipynb`](checks.ipynb) runs the same checks with explanations. It is stored with its outputs, so it can be read on GitHub without running anything; [`checks.html`](checks.html) is the same run exported to HTML. To run the notebook, install JupyterLab (`pip install jupyterlab`) and open it, or execute it from the command line:

```
jupyter nbconvert --to notebook --execute --inplace checks.ipynb
jupyter nbconvert --to html checks.ipynb      # updates checks.html
```

Tested with Python 3.14.10 and 3.13, python-flint 0.9.0, SymPy 1.14.0 and mpmath 1.3.0. The combination python-flint 0.7.1, SymPy 1.12 also works.

## Files

| file | content |
|---|---|
| [`checks.ipynb`](checks.ipynb) | all the checks, with explanations and outputs |
| [`checks.html`](checks.html) | all the checks, with explanations and outputs in html format |
| [`kerr_harmonic_checks.py`](kerr_harmonic_checks.py) | arithmetic of rational functions $N(\mathbf x)/\lvert\mathbf x\rvert^m$, inverse Laplacian, cubic recursion; closed forms of the post-Minkowskian expansion; exact metric by transformation of Boyer–Lindquist; coefficients of appendix C |
| [`kerr_relaxed_einstein_checks.py`](kerr_relaxed_einstein_checks.py) | Landau–Lifshitz form of the relaxed Einstein equations, iterated on its own and compared with the cubic recursion; on-shell auxiliary field |
| [`kerr_canonical_mpm_checks.py`](kerr_canonical_mpm_checks.py) | comparison with the canonical harmonic metric of the multipolar post-Minkowskian formalism; exceptional sector |
| [`kerr_260519948_v1_4pm_checks.py`](kerr_260519948_v1_4pm_checks.py) | comparison with the fourth-order result of the gauge-term-free recursion of [arXiv:2605.19948v1](https://arxiv.org/abs/2605.19948) by Damgaard, Lee, Lee and Rahnuma |
| [`cubic_action_identities.py`](cubic_action_identities.py) | identities of the cubic formulation of the action in $D=3,\dots,6$ |
| [`worldline_source_checks.py`](worldline_source_checks.py) | linear coupling of the worldline action and Kerr stress tensor |
| [`run_all_checks.py`](run_all_checks.py) | runs the six modules from the command line |
| [`kerr-harmonic-coefficients.m`](kerr-harmonic-coefficients.m) | coefficients of appendix C through $G_N^5a^{12}$ |
| [`paper-labels.json`](paper-labels.json) | equation, section and appendix numbers of the paper, by LaTeX label |
| [`build_notebook.py`](build_notebook.py) | writes and executes `checks.ipynb` |

Each module can also be run on its own, for instance `python kerr_harmonic_checks.py 6 8` for the recursion through $G_N^6a^8$.

## The checks

*Exact* means exact rational arithmetic or a symbolic identity. *Numerical* means a multiprecision evaluation at generic points, with 40 to 50 digits. The pair $(n,p)$ labels the order $G_N^nM^na^p$.

| check | in the paper | range | kind | function |
|---|---|---|---|---|
| identities of the cubic action | section 2, appendix D | $D=3,\dots,6$ | exact | [`cubic_action_identities.all_dimensions`](cubic_action_identities.py) |
| linearised curvature and spin coupling | section 3.2 | static perturbation | exact | [`worldline_source_checks.curvature_linearisation`](worldline_source_checks.py) |
| worldline action vs Kerr stress tensor | section 3 | $p\le13$ | exact | [`worldline_source_checks.stress_tensor_resummation`](worldline_source_checks.py) |
| recursion: equations of the auxiliary field and of the metric, harmonic gauge, absence of resonances | sections 2.3, 4, 7 | $n\le5$, $p\le12$ | exact | [`kerr_harmonic_checks.run`](kerr_harmonic_checks.py) |
| recursion vs closed forms, ten components | sections 6, 7 | $n\le5$, $p\le12$ | exact | [`kerr_harmonic_checks.compare`](kerr_harmonic_checks.py) |
| closed forms vs expansion of the exact gothic metric in $G_N$ | sections 5.3, 6 | $n\le5$, all orders in the spin | exact | [`kerr_harmonic_checks.closed_forms_vs_exact_formulas`](kerr_harmonic_checks.py) |
| the same from the directly transformed metric | sections 6, 7 | $n\le5$, 4 points | numerical | [`kerr_harmonic_checks.closed_vs_exact`](kerr_harmonic_checks.py) |
| exact gothic metric, and Cook–Scheel components, vs transformation of Boyer–Lindquist | section 5.3 | 5 points | numerical | [`kerr_harmonic_checks.exact_formula_check`](kerr_harmonic_checks.py) |
| the coordinates are harmonic | sections 5.1, 5.2 | 5 points | numerical | [`kerr_harmonic_checks.harmonicity_check`](kerr_harmonic_checks.py) |
| $\Theta$ and $\delta$ vs their integral representations | sections 1, 5.2 | 5 points | numerical | [`kerr_harmonic_checks.theta_check`](kerr_harmonic_checks.py) |
| Boyer–Lindquist variables of a harmonic point at order $G_N^4$ | section 6.3 | 2 points | numerical | [`kerr_harmonic_checks.inversion_check`](kerr_harmonic_checks.py) |
| recursion vs exact metric beyond fifth order | not in the paper | $n\le8$, $a/\lvert\mathbf x\rvert=10^{-3}$ | numerical | [`kerr_harmonic_checks.recursion_vs_exact`](kerr_harmonic_checks.py) |
| Landau–Lifshitz recursion vs cubic recursion; no resonance; vanishing divergence | appendix A, section 7 | $n\le5$, $p\le12$ | exact | [`kerr_relaxed_einstein_checks.landau_lifshitz_iteration`](kerr_relaxed_einstein_checks.py) |
| canonical iteration from the linear field alone: no resonance, vanishing divergence, same output as the cubic recursion | sections 5.6, 7 | $n\le8$, $p\le12$ | exact | [`kerr_relaxed_einstein_checks.canonical_iteration`](kerr_relaxed_einstein_checks.py) |
| on-shell auxiliary field | sections 2.1, 7 | $n\le5$, $p\le12$ | exact | [`kerr_relaxed_einstein_checks.onshell_auxiliary_field`](kerr_relaxed_einstein_checks.py) |
| exceptional sector at third order | section 4.2 | $p=0,2,4$; $n\le6$ | exact | [`kerr_canonical_mpm_checks.exceptional_sector`](kerr_canonical_mpm_checks.py) |
| closed forms vs canonical metric of Xing, He and Cao, ten components | section 5.6 | $n\le4$, all orders in the spin | exact | [`kerr_canonical_mpm_checks.closed_forms_vs_canonical`](kerr_canonical_mpm_checks.py) |
| Boyer–Lindquist metric transformed to the chart vs canonical metric | section 5.6 | $n\le4$, 2 points | numerical | [`kerr_canonical_mpm_checks.direct_transformation_vs_canonical`](kerr_canonical_mpm_checks.py) |
| Jiang–Lin minus canonical metric at second order | sections 5.5, 5.6 | order $G_N^2$ | exact | [`kerr_canonical_mpm_checks.jiang_lin_difference`](kerr_canonical_mpm_checks.py) |
| azimuth of the canonical chart | section 5.6 | through $G_N^3$ | exact | [`kerr_canonical_mpm_checks.azimuth_transformation`](kerr_canonical_mpm_checks.py) |
| Cook–Scheel vs canonical coordinates | section 5.6 | 5 points | numerical | [`kerr_canonical_mpm_checks.cook_scheel_relation`](kerr_canonical_mpm_checks.py) |
| fourth-order metric of the gauge-term-free recursion of Damgaard et al., eqs. (5.19), (D.19), (D.20) of [arXiv:2605.19948](https://arxiv.org/abs/2605.19948), vs closed forms and recursion | sections 5.5, 7 | $n=4$, $p\le20$ | exact | [`kerr_260519948_v1_4pm_checks.their_fourth_order`](kerr_260519948_v1_4pm_checks.py) |
| coefficients of appendix C | appendix C | $n\le5$, $p\le12$ | exact | [`kerr_harmonic_checks.coefficients_text`](kerr_harmonic_checks.py) |

Most checks come with a control: the same check run on a deliberately wrong input, for instance one coefficient changed in a formula. A control succeeds when the check rejects that input, so the mismatches it reports are expected: they show that the check is able to fail. The output then reads `control OK, wrong input rejected: ...`; `CONTROL FAILED` would signal a problem. The controls count in the exit status of the scripts and in the summary of the notebook.

Eq. (D.20) of Damgaard et al. contains misprints, and [`kerr_260519948_v1_4pm_checks.py`](kerr_260519948_v1_4pm_checks.py) reads it as stated in the footnote of section 5.5 of the paper and in the docstring of the module. The module also shows that this reading is needed: without it, the diagonal spatial components disagree from order $a^4$ on.

## Using the recursion

```python
import kerr_harmonic_checks as khc

h, A, Y, slots = khc.run(6, 8)            # through order G^6 a^8
print(khc.sstr(h[2][2][(0, 0)]))          # term of order G^2 a^2 of hhat^{00}
# (3*x^2 + 3*y^2 - 11*z^2)/|x|^6
```

The recursion uses units $G_N=M=1$ and a spin along $z$. `h[n][p]` is the term of order $G_N^nM^na^p$ of $\mathfrak h^{\mu\nu}=\eta^{\mu\nu}-\sqrt{-g}\,g^{\mu\nu}$, stored as a dictionary `{(mu, nu): (N, m)}` with `mu <= nu`. The pair `(N, m)` stands for $N(x,y,z)/\lvert\mathbf x\rvert^m$, where `N` is a python-flint polynomial with rational coefficients. A component that vanishes is absent. `A` and `Y` contain the auxiliary field and its non-linear part in the same format, with keys `(lambda, mu, nu)` for $A^\lambda{}_{\mu\nu}$.

The run fails with an exception if a field equation or the harmonic gauge condition is violated, or if a source term is resonant, in which case the inverse Laplacian would need a logarithm. Homogeneous solutions can be added by hand with the argument `hom`, as in `kerr_canonical_mpm_checks.exceptional_sector`.

## The coefficient file

[`kerr-harmonic-coefficients.m`](kerr-harmonic-coefficients.m) lists the non-vanishing coefficients as lines `name[n, p] = polynomial in xi;`, which Mathematica reads directly. With $\mathbf n=\mathbf x/\lvert\mathbf x\rvert$, $\xi=\hat{\mathbf z}\cdot\mathbf n$ and $\rho=G_NM/\lvert\mathbf x\rvert$,

$$\mathfrak h^{00}=\sum_{n,p}\rho^n\Big(\frac{a}{\lvert\mathbf x\rvert}\Big)^p\,\texttt{h00[n,p]},\qquad
\mathfrak h^{0i}=(\hat{\mathbf z}\times\mathbf n)^i\sum_{n,p}\rho^n\Big(\frac{a}{\lvert\mathbf x\rvert}\Big)^p\,\texttt{w[n,p]},$$

$$\mathfrak h^{ij}=\alpha_\perp\,\delta_\perp^{ij}+\alpha_z\,\hat z^i\hat z^j+\beta\,x_\perp^ix_\perp^j+\kappa\,\big(x_\perp^i\hat z^j+\hat z^ix_\perp^j\big),$$

and `aPerp`, `aZ`, `b`, `k` are the coefficients of $\alpha_\perp$, $\alpha_z$, $\lvert\mathbf x\rvert^2\beta$ and $\lvert\mathbf x\rvert\,\kappa$ in the same expansion. The header of the file repeats these conventions.

## Conventions

Signature $(-,+,+,+)$. Four dimensions, except in `cubic_action_identities.py`.

The notebook quotes equations by their numbers in the paper. The Python modules refer to them by their LaTeX labels, without the prefix `eq:`, for instance `(recY)` or `(chart)`. The file [`paper-labels.json`](paper-labels.json) maps each label to its number, for instance `"eq:recY": "19"`. 
 
## Citing

If you use this code or these data, please cite the paper and this repository

```bibtex
@article{Sadu:2026kerr,
    author = "Sadu, Amina and Vanhove, Pierre",
    title  = "{The Kerr Metric in Canonical Harmonic Coordinates to All Orders}",
    year   = "2026"
}
```
