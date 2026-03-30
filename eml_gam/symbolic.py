"""Helpers for working with SymPy expressions returned by the model."""

from __future__ import annotations

from typing import Callable

import numpy as np
import sympy as sp


def format_formula(expr: sp.Expr, style: str = "pretty") -> str:
    """Render a SymPy expression as a string.

    ``style`` must be one of ``'pretty'``, ``'latex'``, ``'str'``.
    """
    if style == "pretty":
        return sp.pretty(expr, use_unicode=True)
    if style == "latex":
        return sp.latex(expr)
    if style == "str":
        return str(expr)
    raise ValueError(f"unknown style: {style}")


def complexity(expr: sp.Expr) -> int:
    """Rough complexity score: number of operations plus number of atoms."""
    return sp.count_ops(expr) + len(expr.atoms(sp.Symbol, sp.Number))


def to_numpy_fn(expr: sp.Expr, symbols: list[sp.Symbol]) -> Callable:
    """Compile a SymPy expression to a vectorised NumPy function."""
    return sp.lambdify(symbols, expr, modules="numpy")


def verify_formula(
    expr: sp.Expr,
    symbols: list[sp.Symbol],
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Evaluate a SymPy expression on data and return MSE plus predictions."""
    fn = to_numpy_fn(expr, symbols)
    args = [X[:, i] for i in range(X.shape[1])]
    y_pred = np.asarray(fn(*args), dtype=np.float64)
    mse = float(np.mean((y_pred - y) ** 2))
    return {"mse": mse, "y_pred": y_pred}
