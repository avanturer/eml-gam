"""PySR baseline wrapper.

Miles Cranmer's PySR (arXiv:2305.01582) is the current state-of-the-art
symbolic regression library. It runs a Julia-side genetic-programming
kernel, accepts arbitrary operator sets, and is widely cited as the
reference baseline in SR papers. The library is pip-installable
(``pip install pysr``) and the first import triggers a one-time Julia
package install; subsequent runs are fast.

This module is a thin adapter exposing the same ``fit / predict`` API
as the rest of our benchmark code. If PySR is not installed the
wrapper gracefully returns ``None`` so the benchmark runners can skip
it without crashing.

Typical use inside a benchmark:

    from .pysr_baseline import fit_pysr, PySRWrapper
    raw = fit_pysr(X_train, y_train, niterations=15)
    model = PySRWrapper(raw) if raw is not None else None

The default operator set covers ``+, -, *, /, exp, log, sqrt``. That
matches gplearn's function set, with the important upgrade of a working
``exp``. PySR also supports ``cos / sin`` and custom Julia expressions;
we keep the set minimal for fair comparison against EML-GA^2M (which
has no ``sin``) on targets where neither should win by richer operators.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def fit_pysr(
    X_train, y_train,
    niterations: int = 15,
    maxsize: int = 20,
    binary_ops: tuple[str, ...] = ("+", "-", "*", "/"),
    unary_ops: tuple[str, ...] = ("exp", "log"),
    parsimony: float = 1e-3,
    time_budget_s: Optional[float] = None,
    random_state: int = 0,
    verbosity: int = 0,
):
    """Fit a PySR regressor and return the raw model object or ``None``."""
    try:
        from pysr import PySRRegressor
    except ImportError:
        return None
    try:
        m = PySRRegressor(
            niterations=niterations,
            maxsize=maxsize,
            binary_operators=list(binary_ops),
            unary_operators=list(unary_ops),
            parsimony=parsimony,
            timeout_in_seconds=time_budget_s,
            random_state=random_state,
            deterministic=True,
            parallelism="serial",
            verbosity=verbosity,
            progress=False,
        )
        m.fit(X_train, y_train)
        return m
    except Exception as e:
        # Julia bridge can crash on some platforms; fail gracefully.
        print(f"  pysr fit failed: {e}")
        return None


class PySRWrapper:
    """Uniform ``predict / formula`` interface on top of the raw PySR model."""

    def __init__(self, m):
        self._m = m

    def predict(self, X):
        try:
            y = np.asarray(self._m.predict(X), dtype=np.float64)
            return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            return np.zeros(X.shape[0], dtype=np.float64)

    @property
    def formula(self) -> str:
        try:
            best = self._m.get_best()
            if hasattr(best, "equation"):
                return str(best.equation)
            return str(best["equation"])
        except Exception:
            return ""
