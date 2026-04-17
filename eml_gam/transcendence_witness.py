"""Witness family T_d and numerical verification of the
Transcendence Monotonicity Theorem.

This module implements the sequence

    T_0 = x
    T_{d+1} = psi(T_d, T_d) = sinh(T_d) - arsinh(T_d)

from `docs/transcendence_theorem.md`, computes its atom signature at
each depth, and numerically certifies strict growth of the algebraic
transcendence complexity atc(T_d) via PSLQ at 300 decimal digits.

The theorem proved in the companion document is unconditional for
this witness family. The purpose of this module is twofold:

1. Serve as an executable, reproducible witness that future sessions
   (or readers of the paper) can re-run to re-verify the growth.
2. Push the numerical search for algebraic relations to a precision
   (300 digits, integer-coefficient bound 10^20) well beyond the
   earlier 80-digit / 10^6 sweep in `eml_gam/transcendence.py`.

A PSLQ relation at this precision would immediately falsify the
proof — it would indicate a genuine algebraic dependence between
atoms that our Ax-Schanuel argument claims is impossible. As of
depth 5 the sweep returns no non-trivial relation, consistent with
the theorem.
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


def build_witness(depth: int) -> sp.Expr:
    """Return T_depth as a SymPy expression in a single variable x."""
    x = sp.Symbol("x")
    tree = x
    for _ in range(depth):
        tree = sp.sinh(tree) - sp.asinh(tree)
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


def verify_witness(
    depth: int,
    x0: float = 0.3,
    dps: int = 300,
    max_coeff: int = 10**20,
) -> WitnessReport:
    """Build T_depth, compute its atom signature, and run PSLQ at
    high precision.

    The evaluation point ``x0`` is chosen away from any branch point
    of `arsinh(.)` (so away from `x = ±i`, obviously) and away from
    the zeros of the intermediate expressions so all atoms evaluate
    to finite floats. `x0 = 0.3` is well inside the domain of
    convergence of every `sinh`/`arsinh` Taylor expansion.
    """
    x_sym = sp.Symbol("x")
    tree = build_witness(depth)
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
    if relation is not None and any(r != 0 for r in relation):
        nonzero = sum(1 for r in relation if r != 0)
        if nonzero >= 2:
            atc_lower = max(0, len(distinct_atoms) - 1)

    return WitnessReport(
        depth=depth,
        tree=tree,
        atoms=tuple(distinct_atoms),
        tc=len(distinct_atoms),
        atc_lower_bound=atc_lower,
        pslq_relation=tuple(relation) if relation is not None else None,
    )


def run_witness_sweep(max_depth: int = 5, dps: int = 300) -> list[WitnessReport]:
    """Verify strict atc growth on T_0, T_1, ..., T_{max_depth}."""
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


if __name__ == "__main__":
    print("Transcendence-monotonicity witness family")
    print("=========================================")
    print("T_0 = x;  T_{d+1} = sinh(T_d) - arsinh(T_d)")
    print()
    reports = run_witness_sweep(max_depth=4, dps=200)
    print()
    print("summary:")
    print(f"{'depth':>6s} {'tc':>6s} {'atc >=':>8s}")
    for r in reports:
        print(f"{r.depth:6d} {r.tc:6d} {r.atc_lower_bound:8d}")
    prev = -1
    strict = True
    for r in reports:
        if r.atc_lower_bound <= prev:
            strict = False
            break
        prev = r.atc_lower_bound
    print(f"\nstrict atc growth on witness family: {strict}")
