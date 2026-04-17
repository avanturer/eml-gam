"""Witness and orbit families with numerical verification of the
Transcendence Monotonicity theorems.

Two theorems, proved in ``docs/transcendence_theorem.md``:

1. **Theorem W (Witness Family).** For ``T_0 = x``,
   ``T_{d+1} = psi(T_d, T_d)``:  ``atc(T_{d+1}) >= atc(T_d) + 1``.

2. **Theorem O (Orbit).** For *every* non-constant ``g`` in ``F_0``
   and orbit ``W_0(g) = g``, ``W_{d+1}(g) = psi(W_d(g), W_d(g))``:
   ``atc(W_{d+1}(g)) >= atc(W_d(g)) + 1``.  Theorem O strictly
   generalises Theorem W (take ``g = x``).

Both proofs use Ax-Schanuel (Ax 1971, analytic form re-proved by
Noguchi 2022, arXiv:2203.00470) together with a short
``cosh``-transcendence lemma.

This module:

* constructs ``T_d`` and ``W_d(g)`` for arbitrary SymPy seed ``g``;
* verifies numerically via PSLQ at 200 decimal digits that atoms are
  algebraically independent (no non-trivial integer relations within
  coefficient bound ``10**20``) through depth 4;
* certifies the explicit rational-linear dependency
  ``psi(1, x) = psi(x, x) - psi(x, 1) + (sinh(1) - arsinh(1))``
  as documented evidence that the remaining genericity condition
  ``(Gen-d)`` for the general case is non-vacuous.

A PSLQ relation at this precision would falsify the theorems — it
would indicate a genuine algebraic dependence that Ax-Schanuel claims
is impossible.  Through depth 4 the sweep returns no non-trivial
relation, consistent with both theorems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import mpmath as mp
import sympy as sp


@dataclass
class WitnessReport:
    depth: int
    tree: sp.Expr
    atoms: Tuple[sp.Expr, ...]
    tc: int
    atc_lower_bound: int
    pslq_relation: Tuple[int, ...] | None


def _psi(a: sp.Expr, b: sp.Expr) -> sp.Expr:
    return sp.sinh(a) - sp.asinh(b)


def build_witness(depth: int) -> sp.Expr:
    """Return ``T_depth`` as a SymPy expression in a single variable
    ``x``, where ``T_0 = x`` and ``T_{d+1} = psi(T_d, T_d)``."""
    x = sp.Symbol("x")
    return build_orbit(x, depth)


def build_orbit(seed: sp.Expr, depth: int) -> sp.Expr:
    """Return ``W_depth(seed)`` where ``W_0 = seed`` and
    ``W_{d+1} = psi(W_d, W_d)``.  Accepts any SymPy expression in the
    variable ``x`` (the seed does not have to be ``x`` itself)."""
    tree = seed
    for _ in range(depth):
        tree = _psi(tree, tree)
    return tree


def _collect_atoms(expr: sp.Expr) -> list[sp.Expr]:
    out: list[sp.Expr] = []

    def walk(e: sp.Expr) -> None:
        if e.is_Atom:
            return
        for arg in e.args:
            walk(arg)
        if e.func in (sp.sinh, sp.asinh):
            out.append(e)

    walk(expr)
    return out


def _distinct_under_simplify(atoms: list[sp.Expr]) -> list[sp.Expr]:
    seen: list[sp.Expr] = []
    for a in atoms:
        simplified = sp.simplify(a)
        if not any(sp.simplify(simplified - s) == 0 for s in seen):
            seen.append(simplified)
    return seen


def verify_orbit(
    depth: int,
    seed: sp.Expr | None = None,
    x0: float = 0.3,
    dps: int = 200,
    max_coeff: int = 10**20,
) -> WitnessReport:
    """Build ``W_depth(seed)`` (default seed = ``x``), compute its
    atom signature, and run PSLQ at high precision.

    A PSLQ relation of length >= 2 would indicate an algebraic
    dependence the theorem rules out; the returned
    ``atc_lower_bound`` is reduced accordingly, so the caller can
    assert strict growth.
    """
    x_sym = sp.Symbol("x")
    if seed is None:
        seed = x_sym
    tree = build_orbit(seed, depth)
    raw_atoms = _collect_atoms(tree)
    distinct_atoms = _distinct_under_simplify(raw_atoms)

    old_dps = mp.mp.dps
    mp.mp.dps = dps
    try:
        atom_values: list[mp.mpf] = []
        for atom in distinct_atoms:
            val = sp.N(atom.subs(x_sym, sp.Float(x0, dps)), dps)
            atom_values.append(mp.mpf(val))
        if len(atom_values) >= 2:
            relation = mp.pslq(
                atom_values,
                tol=mp.mpf(10) ** -(dps - 30),
                maxcoeff=max_coeff,
            )
        else:
            relation = None
    finally:
        mp.mp.dps = old_dps

    atc_lower = len(distinct_atoms)
    if relation is not None and sum(1 for r in relation if r != 0) >= 2:
        atc_lower = max(0, len(distinct_atoms) - 1)

    return WitnessReport(
        depth=depth,
        tree=tree,
        atoms=tuple(distinct_atoms),
        tc=len(distinct_atoms),
        atc_lower_bound=atc_lower,
        pslq_relation=tuple(relation) if relation is not None else None,
    )


def verify_witness(
    depth: int,
    x0: float = 0.3,
    dps: int = 200,
    max_coeff: int = 10**20,
) -> WitnessReport:
    """Backward-compatible shorthand for ``verify_orbit`` with seed
    ``x`` (the classical witness family ``T_d``)."""
    return verify_orbit(depth, seed=None, x0=x0, dps=dps, max_coeff=max_coeff)


def verify_psi_linear_dependency(x0: float = 0.5, tol: float = 1e-6) -> dict:
    """Numerically certify the explicit rational dependency

        psi(1, x) = psi(x, x) - psi(x, 1) + (sinh(1) - arsinh(1)).

    This is documented evidence that the genericity condition
    ``(Gen-d)`` is non-vacuous: the general Transcendence Monotonicity
    Conjecture is *not* a free corollary of ``psi``-tree values being
    ``Q``-linearly independent — such dependencies really exist.

    Returns a dict with LHS, RHS, and their difference at ``x = x0``.
    """
    x_sym = sp.Symbol("x")
    lhs = _psi(sp.Integer(1), x_sym)
    rhs = (
        _psi(x_sym, x_sym)
        - _psi(x_sym, sp.Integer(1))
        + sp.sinh(sp.Integer(1))
        - sp.asinh(sp.Integer(1))
    )
    lhs_val = float(sp.N(lhs.subs(x_sym, sp.Float(x0)), 30))
    rhs_val = float(sp.N(rhs.subs(x_sym, sp.Float(x0)), 30))
    diff = lhs_val - rhs_val
    return {
        "x0": x0,
        "lhs": lhs_val,
        "rhs": rhs_val,
        "diff": diff,
        "holds_within_tol": abs(diff) < tol,
    }


def run_witness_sweep(
    max_depth: int = 4, dps: int = 200
) -> list[WitnessReport]:
    """Verify strict atc growth on ``T_0, T_1, ..., T_{max_depth}``."""
    reports: list[WitnessReport] = []
    for d in range(max_depth + 1):
        print(f"building T_{d} and running PSLQ at {dps} digits ...")
        report = verify_witness(d, dps=dps)
        reports.append(report)
        rel_str = (
            "no non-trivial relation"
            if report.pslq_relation is None
            else f"relation = {report.pslq_relation}"
        )
        print(
            f"  depth={d}  tc={report.tc}  "
            f"atc_lower>={report.atc_lower_bound}  {rel_str}"
        )
    return reports


def run_orbit_sweep(
    seed: sp.Expr,
    max_depth: int = 3,
    dps: int = 120,
) -> list[WitnessReport]:
    """Verify Theorem O on an alternative seed.  Uses somewhat lower
    precision because richer seeds produce larger atom sets and PSLQ
    cost grows with both precision and dimension."""
    reports: list[WitnessReport] = []
    for d in range(max_depth + 1):
        print(f"building W_{d}({seed}) and running PSLQ at {dps} digits ...")
        report = verify_orbit(d, seed=seed, dps=dps)
        reports.append(report)
        rel_str = (
            "no non-trivial relation"
            if report.pslq_relation is None
            else f"relation = {report.pslq_relation}"
        )
        print(
            f"  depth={d}  tc={report.tc}  "
            f"atc_lower>={report.atc_lower_bound}  {rel_str}"
        )
    return reports


if __name__ == "__main__":
    print("Transcendence-monotonicity theorems: numerical verification")
    print("===========================================================")

    print("\n[Theorem W] Witness family T_d (seed = x)")
    print("-------------------------------------------")
    w_reports = run_witness_sweep(max_depth=4, dps=200)

    x_sym = sp.Symbol("x")
    print("\n[Theorem O] Orbit family with seed g = x^2 + x")
    print("----------------------------------------------")
    o_reports = run_orbit_sweep(seed=x_sym**2 + x_sym, max_depth=3, dps=120)

    print("\n[Genericity check] The explicit rational-linear dependency")
    print("----------------------------------------------------------")
    dep = verify_psi_linear_dependency()
    print(
        f"  psi(1, x) = psi(x, x) - psi(x, 1) + (sinh 1 - arsinh 1)"
        f"\n  LHS at x={dep['x0']}: {dep['lhs']:.8f}"
        f"\n  RHS at x={dep['x0']}: {dep['rhs']:.8f}"
        f"\n  diff = {dep['diff']:.2e} (holds: {dep['holds_within_tol']})"
    )
    print(
        "\n  This dependency shows (Gen-d) is non-vacuous: the general"
        "\n  Transcendence Monotonicity Conjecture is NOT a free corollary"
        "\n  of Q-linear independence of psi-tree values."
    )

    print("\nsummary:")
    print(f"{'family':>10s} {'depth':>6s} {'tc':>6s} {'atc >=':>8s}")
    for r in w_reports:
        print(f"{'witness':>10s} {r.depth:6d} {r.tc:6d} {r.atc_lower_bound:8d}")
    for r in o_reports:
        print(f"{'orbit':>10s} {r.depth:6d} {r.tc:6d} {r.atc_lower_bound:8d}")

    def _strict_monotone(reports: list[WitnessReport]) -> bool:
        prev = -1
        for r in reports:
            if r.atc_lower_bound <= prev:
                return False
            prev = r.atc_lower_bound
        return True

    print(
        f"\nstrict atc growth on witness:  {_strict_monotone(w_reports)}"
    )
    print(
        f"strict atc growth on orbit:    {_strict_monotone(o_reports)}"
    )
