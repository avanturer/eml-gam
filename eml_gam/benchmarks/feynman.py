"""Feynman-subset symbolic regression benchmark.

The Feynman-AI-100 dataset (Udrescu and Tegmark, 2020) catalogues 100
physical formulas from the Feynman Lectures. The full suite is widely
used as the gold-standard symbolic-regression benchmark. We include a
ten-equation subset biased toward targets that are expressible in the
EML grammar (exponential, logarithmic, power-law, rational). The
selection is designed to:

* include a mix of univariate and bivariate targets,
* cover the canonical exp/log/power pattern that motivates EML,
* omit trigonometry-heavy targets (III.8.54, I.26.2, etc.) for which
  EML has no competitive depth-2 primitive.

The model comparison is the same as in ``nguyen.py``: LinearRegression,
gplearn, and EML-GA^2M with robust multi-start and the extrapolation
penalty enabled. We also ship an in-sample and a light extrapolation
split so that the headline metric mirrors the extrapolation benchmark
that motivates the method.
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
class FeynmanTarget:
    name: str
    feynman_id: str
    n_vars: int
    train_domain: list[tuple[float, float]]
    extrap_domain: list[tuple[float, float]]
    formula_str: str
    fn: callable


def _t(name, fid, n_vars, train_dom, extrap_dom, formula, fn):
    return FeynmanTarget(
        name=name, feynman_id=fid, n_vars=n_vars,
        train_domain=train_dom, extrap_domain=extrap_dom,
        formula_str=formula, fn=fn,
    )


def all_feynman_targets() -> list[FeynmanTarget]:
    return [
        # exp(-theta^2 / 2), Gaussian, Feynman I.6.20a
        _t("gaussian", "I.6.20a", 1,
           [(-1.5, 1.5)], [(-3.0, 3.0)],
           "exp(-theta^2 / 2)",
           lambda X: np.exp(-X[:, 0] ** 2 / 2.0)),
        # I_0 * (exp(q V / kT) - 1), Shockley diode, Feynman III.14.14
        _t("diode", "III.14.14", 1,
           [(0.0, 1.5)], [(0.0, 3.0)],
           "exp(x) - 1",
           lambda X: np.exp(X[:, 0]) - 1.0),
        # N_0 * exp(-t / tau), radioactive decay
        _t("decay", "half_life", 1,
           [(0.0, 2.0)], [(0.0, 6.0)],
           "5 * exp(-0.8 * t)",
           lambda X: 5.0 * np.exp(-0.8 * X[:, 0])),
        # m * c^2 / sqrt(1 - v^2 / c^2), relativistic energy, Feynman I.16.6 style
        _t("relativistic", "I.16.6", 1,
           [(0.0, 0.85)], [(0.0, 0.95)],
           "1 / sqrt(1 - v^2)",
           lambda X: 1.0 / np.sqrt(np.clip(1.0 - X[:, 0] ** 2, 1e-6, None))),
        # kT / p, ideal gas-like (bivariate)
        _t("ideal_gas", "I.39.22-like", 2,
           [(1.0, 5.0), (1.0, 5.0)], [(1.0, 8.0), (0.5, 8.0)],
           "T / p",
           lambda X: X[:, 0] / np.clip(X[:, 1], 1e-3, None)),
        # q1 q2 / r^2, Coulomb, Feynman I.12.2-like
        _t("coulomb", "I.12.2", 2,
           [(0.5, 2.0), (0.5, 2.0)], [(0.5, 4.0), (0.5, 4.0)],
           "q / r^2",
           lambda X: X[:, 0] / np.clip(X[:, 1] ** 2, 1e-3, None)),
        # Planck radiation: 1 / (exp(x) - 1), Feynman I.41.16 simplified
        _t("planck", "I.41.16", 1,
           [(0.5, 3.0)], [(0.5, 6.0)],
           "1 / (exp(x) - 1)",
           lambda X: 1.0 / np.clip(np.exp(X[:, 0]) - 1.0, 1e-6, None)),
        # Centripetal force form: m v^2 / r
        _t("centripetal", "I.34.8-like", 2,
           [(0.5, 2.0), (0.5, 2.0)], [(0.5, 4.0), (0.5, 4.0)],
           "v^2 / r",
           lambda X: X[:, 0] ** 2 / np.clip(X[:, 1], 1e-3, None)),
        # Power dissipated: I^2 R
        _t("power_r", "power", 2,
           [(0.5, 2.0), (0.5, 2.0)], [(0.5, 4.0), (0.5, 4.0)],
           "I^2 * R",
           lambda X: X[:, 0] ** 2 * X[:, 1]),
        # Stefan-Boltzmann: sigma * T^4
        _t("stefan_boltzmann", "sigma_T4", 1,
           [(1.0, 3.0)], [(1.0, 6.0)],
           "T^4",
           lambda X: X[:, 0] ** 4),
    ]


@dataclass
class FeynmanResult:
    target: str
    model: str
    r2_interp: float
    r2_extrap: float
    mse_interp: float
    mse_extrap: float
    time_s: float
    formula: str = ""


def _sample_dataset(
    target: FeynmanTarget,
    n_train: int, n_test: int, seed: int,
):
    rng = np.random.default_rng(seed)
    d = target.n_vars
    X_train = np.empty((n_train, d))
    X_interp = np.empty((n_test, d))
    X_extrap = np.empty((n_test, d))
    for k in range(d):
        lo, hi = target.train_domain[k]
        lo_e, hi_e = target.extrap_domain[k]
        X_train[:, k] = rng.uniform(lo, hi, n_train)
        X_interp[:, k] = rng.uniform(lo, hi, n_test)
        X_extrap[:, k] = rng.uniform(lo_e, hi_e, n_test)
    y_train = target.fn(X_train)
    y_interp = target.fn(X_interp)
    y_extrap = target.fn(X_extrap)
    return X_train, y_train, X_interp, y_interp, X_extrap, y_extrap


def _fit_linear(X_tr, y_tr):
    m = LinearRegression()
    m.fit(X_tr, y_tr)
    return m


def _fit_gplearn(X_tr, y_tr):
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
        population_size=1500, generations=25,
        function_set=("add", "sub", "mul", "div", "log", "sqrt", "abs", "inv"),
        parsimony_coefficient=0.001, random_state=0, n_jobs=1, verbose=0,
    )
    m.fit(X_tr, y_tr)
    return m


class _GPW:
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


def _fit_emlgam(X_tr, y_tr, n_vars: int, n_epochs: int = 1200) -> EMLGAM:
    pairs = list(combinations(range(n_vars), 2)) if n_vars >= 2 else None
    model = EMLGAM(
        n_features=n_vars, univariate_depth=2, bivariate_depth=2,
        interaction_pairs=pairs,
        standardize=True, scale_normalize=True,
    )
    cfg = TrainConfig(
        n_epochs=n_epochs, lr=5e-2,
        entropy_weight=1e-3,
        extrap_penalty_weight=0.01, extrap_max_std=50.0,
    )
    model.fit(
        X_tr, y_tr, cfg=cfg,
        warm_start=True, try_offsets=True, robust=True,
    )
    return model


def bench_feynman(
    targets=None,
    n_train: int = 200, n_test: int = 500, seed: int = 0,
    models: tuple[str, ...] = ("linear", "gplearn", "emlgam"),
    verbose: bool = True,
) -> list[FeynmanResult]:
    targets = targets or all_feynman_targets()
    out: list[FeynmanResult] = []
    for t in targets:
        X_tr, y_tr, X_in, y_in, X_ex, y_ex = _sample_dataset(
            t, n_train, n_test, seed,
        )
        for name in models:
            t0 = time.perf_counter()
            if name == "linear":
                m = _fit_linear(X_tr, y_tr)
            elif name == "gplearn":
                raw = _fit_gplearn(X_tr, y_tr)
                m = _GPW(raw) if raw is not None else None
            elif name == "emlgam":
                m = _fit_emlgam(X_tr, y_tr, t.n_vars)
            else:
                raise ValueError(name)
            if m is None:
                continue
            elapsed = time.perf_counter() - t0
            y_in_pred = m.predict(X_in)
            y_ex_pred = m.predict(X_ex)
            if not np.all(np.isfinite(y_in_pred)):
                y_in_pred = np.nan_to_num(y_in_pred, nan=0.0, posinf=0.0, neginf=0.0)
            if not np.all(np.isfinite(y_ex_pred)):
                y_ex_pred = np.nan_to_num(y_ex_pred, nan=0.0, posinf=0.0, neginf=0.0)
            r2_in = r2_score(y_in, y_in_pred)
            r2_ex = r2_score(y_ex, y_ex_pred)
            mse_in = float(np.mean((y_in - y_in_pred) ** 2))
            mse_ex = float(np.mean((y_ex - y_ex_pred) ** 2))
            formula = ""
            if name == "emlgam":
                try:
                    formula = str(m.total_formula(simplify=True))[:160]
                except Exception:
                    pass
            elif name == "gplearn":
                formula = m.formula[:160] if isinstance(m, _GPW) else ""
            out.append(FeynmanResult(
                target=t.name, model=name,
                r2_interp=r2_in, r2_extrap=r2_ex,
                mse_interp=mse_in, mse_extrap=mse_ex,
                time_s=elapsed, formula=formula,
            ))
            if verbose:
                print(
                    f"  {t.name:18s} | {name:8s} | "
                    f"R2_in={r2_in:+7.3f} | R2_ex={r2_ex:+7.3f} | "
                    f"t={elapsed:5.1f}s"
                )
    return out


if __name__ == "__main__":
    bench_feynman()
