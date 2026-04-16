"""Nguyen symbolic regression benchmark.

The Nguyen family (Uy et al. 2011; Petersen et al. 2021) is the de-facto
standard micro-benchmark for symbolic regression systems. Each target is a
closed-form elementary function; a symbolic regressor is asked to recover
it from a small training sample and its output is evaluated by exact
recovery (syntactic match up to simplification) and by numerical fit on a
held-out test grid.

This module implements Nguyen-1..Nguyen-12 (the "classic" twelve targets)
and runs EML-GA^2M, gplearn, and plain linear regression against them. We
deliberately do not include the later SRBench extensions here; Nguyen-12
alone is enough to expose the qualitative difference between a tree-based
baseline and an EML-backed model on targets whose ground truth *is* an
elementary function.

Target specifications:

    N1  : x + x^2 + x^3                       x in U(-1, 1)
    N2  : x + x^2 + x^3 + x^4                 x in U(-1, 1)
    N3  : x + x^2 + x^3 + x^4 + x^5           x in U(-1, 1)
    N4  : x + x^2 + x^3 + x^4 + x^5 + x^6     x in U(-1, 1)
    N5  : sin(x^2) * cos(x) - 1               x in U(-1, 1)
    N6  : sin(x) + sin(x + x^2)               x in U(-1, 1)
    N7  : log(x + 1) + log(x^2 + 1)           x in U(0, 2)
    N8  : sqrt(x)                             x in U(0, 4)
    N9  : sin(x) + sin(y^2)                   (x, y) in U(0, 1)^2
    N10 : 2 * sin(x) * cos(y)                 (x, y) in U(0, 1)^2
    N11 : x ^ y                               (x, y) in U(0, 1)^2
    N12 : x^4 - x^3 + 0.5 * y^2 - y           (x, y) in U(0, 1)^2

We use the Petersen et al. training protocol: 20 training points, 1000
test points, single-seed per target; paired comparison across methods.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from ..gam import EMLGAM
from ..train import TrainConfig


@dataclass
class NguyenTarget:
    name: str
    n_vars: int
    domain: list[tuple[float, float]]
    formula_str: str
    fn: callable


def _mk1(name: str, fn, lo: float, hi: float, formula: str) -> NguyenTarget:
    return NguyenTarget(
        name=name, n_vars=1, domain=[(lo, hi)],
        formula_str=formula,
        fn=lambda X: fn(X[:, 0]),
    )


def _mk2(
    name: str, fn, lo: float, hi: float, formula: str
) -> NguyenTarget:
    return NguyenTarget(
        name=name, n_vars=2, domain=[(lo, hi), (lo, hi)],
        formula_str=formula,
        fn=lambda X: fn(X[:, 0], X[:, 1]),
    )


def _n11(x, y):
    return np.power(np.clip(x, 1e-6, None), y)


def all_nguyen_targets() -> list[NguyenTarget]:
    return [
        _mk1("N1", lambda x: x + x ** 2 + x ** 3,
             -1.0, 1.0, "x + x^2 + x^3"),
        _mk1("N2", lambda x: x + x ** 2 + x ** 3 + x ** 4,
             -1.0, 1.0, "x + x^2 + x^3 + x^4"),
        _mk1("N3", lambda x: x + x ** 2 + x ** 3 + x ** 4 + x ** 5,
             -1.0, 1.0, "x + x^2 + x^3 + x^4 + x^5"),
        _mk1("N4", lambda x: x + x ** 2 + x ** 3 + x ** 4 + x ** 5 + x ** 6,
             -1.0, 1.0, "x + x^2 + x^3 + x^4 + x^5 + x^6"),
        _mk1("N5", lambda x: np.sin(x ** 2) * np.cos(x) - 1.0,
             -1.0, 1.0, "sin(x^2) * cos(x) - 1"),
        _mk1("N6", lambda x: np.sin(x) + np.sin(x + x ** 2),
             -1.0, 1.0, "sin(x) + sin(x + x^2)"),
        _mk1("N7", lambda x: np.log(x + 1.0) + np.log(x ** 2 + 1.0),
             0.0, 2.0, "log(x + 1) + log(x^2 + 1)"),
        _mk1("N8", lambda x: np.sqrt(np.clip(x, 0.0, None)),
             0.0, 4.0, "sqrt(x)"),
        _mk2("N9", lambda x, y: np.sin(x) + np.sin(y ** 2),
             0.0, 1.0, "sin(x) + sin(y^2)"),
        _mk2("N10", lambda x, y: 2.0 * np.sin(x) * np.cos(y),
             0.0, 1.0, "2 sin(x) cos(y)"),
        _mk2("N11", _n11, 0.0, 1.0, "x^y"),
        _mk2("N12", lambda x, y: x ** 4 - x ** 3 + 0.5 * y ** 2 - y,
             0.0, 1.0, "x^4 - x^3 + 0.5 y^2 - y"),
    ]


@dataclass
class NguyenResult:
    target: str
    model: str
    r2_test: float
    mse_test: float
    time_s: float
    formula: str = ""


def _sample_dataset(
    target: NguyenTarget, n_train: int, n_test: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_train = np.empty((n_train, target.n_vars))
    X_test = np.empty((n_test, target.n_vars))
    for k, (lo, hi) in enumerate(target.domain):
        X_train[:, k] = rng.uniform(lo, hi, n_train)
        X_test[:, k] = rng.uniform(lo, hi, n_test)
    y_train = target.fn(X_train)
    y_test = target.fn(X_test)
    return X_train, y_train, X_test, y_test


def _fit_linear(X_tr, y_tr):
    m = LinearRegression()
    m.fit(X_tr, y_tr)
    return m


def _fit_gplearn(X_tr, y_tr, time_budget_s: float = 20.0):
    try:
        from gplearn.genetic import SymbolicRegressor
        from sklearn.utils.validation import validate_data
    except ImportError:
        return None

    if not hasattr(SymbolicRegressor, "_validate_data"):
        def _vd(self, *args, **kwargs):
            return validate_data(self, *args, **kwargs)
        SymbolicRegressor._validate_data = _vd  # type: ignore[attr-defined]

    m = SymbolicRegressor(
        population_size=1500,
        generations=30,
        function_set=(
            "add", "sub", "mul", "div", "log", "sqrt", "abs", "sin", "cos",
        ),
        parsimony_coefficient=0.001,
        random_state=0,
        n_jobs=1,
        verbose=0,
    )
    m.fit(X_tr, y_tr)
    return m


class _GPlearnWrapper:
    def __init__(self, m):
        self._m = m

    def predict(self, X):
        return np.asarray(self._m.predict(X), dtype=np.float64)

    @property
    def formula(self) -> str:
        try:
            return str(self._m._program)
        except Exception:
            return ""


def _fit_emlgam(
    X_tr, y_tr, n_vars: int, n_epochs: int = 1200,
) -> EMLGAM:
    pairs = list(combinations(range(n_vars), 2)) if n_vars >= 2 else None
    model = EMLGAM(
        n_features=n_vars,
        univariate_depth=2,
        bivariate_depth=2,
        interaction_pairs=pairs,
        standardize=True,
        scale_normalize=True,
    )
    cfg = TrainConfig(
        n_epochs=n_epochs, lr=5e-2,
        entropy_weight=1e-3,
        extrap_penalty_weight=0.01,
        extrap_max_std=50.0,
    )
    model.fit(
        X_tr, y_tr, cfg=cfg,
        warm_start=True, try_offsets=True, robust=True,
    )
    return model


def bench_nguyen(
    targets: list[NguyenTarget] | None = None,
    n_train: int = 20,
    n_test: int = 1000,
    seed: int = 0,
    models: tuple[str, ...] = ("linear", "gplearn", "emlgam"),
    verbose: bool = True,
) -> list[NguyenResult]:
    targets = targets or all_nguyen_targets()
    results: list[NguyenResult] = []
    for t in targets:
        X_tr, y_tr, X_te, y_te = _sample_dataset(t, n_train, n_test, seed)
        for name in models:
            t0 = time.perf_counter()
            if name == "linear":
                m = _fit_linear(X_tr, y_tr)
            elif name == "gplearn":
                raw = _fit_gplearn(X_tr, y_tr)
                m = _GPlearnWrapper(raw) if raw is not None else None
            elif name == "emlgam":
                m = _fit_emlgam(X_tr, y_tr, t.n_vars)
            else:
                raise ValueError(name)
            if m is None:
                continue
            elapsed = time.perf_counter() - t0
            y_pred = m.predict(X_te)
            if not np.all(np.isfinite(y_pred)):
                y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
            r2 = r2_score(y_te, y_pred)
            mse = float(np.mean((y_te - y_pred) ** 2))
            formula = ""
            if name == "emlgam":
                try:
                    formula = str(m.total_formula(simplify=True))[:160]
                except Exception:
                    pass
            elif name == "gplearn":
                formula = m.formula[:160] if isinstance(m, _GPlearnWrapper) else ""
            results.append(
                NguyenResult(
                    target=t.name, model=name,
                    r2_test=r2, mse_test=mse,
                    time_s=elapsed, formula=formula,
                )
            )
            if verbose:
                print(
                    f"  {t.name:4s} | {name:8s} | R2={r2:+.4f}  "
                    f"mse={mse:.3e}  t={elapsed:5.1f}s"
                )
    return results


def summary_table(results: list[NguyenResult]) -> None:
    by_target: dict[str, dict[str, NguyenResult]] = {}
    for r in results:
        by_target.setdefault(r.target, {})[r.model] = r
    models_seen: list[str] = []
    for r in results:
        if r.model not in models_seen:
            models_seen.append(r.model)
    header = f"  {'target':6s}  " + "  ".join(f"{m:>8s}" for m in models_seen)
    print()
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for t in sorted(by_target):
        row = by_target[t]
        cells = []
        for m in models_seen:
            if m in row:
                cells.append(f"{row[m].r2_test:+8.3f}")
            else:
                cells.append(" " * 8)
        print(f"  {t:6s}  " + "  ".join(cells))


if __name__ == "__main__":
    res = bench_nguyen()
    summary_table(res)
