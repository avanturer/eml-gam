"""Extrapolation benchmark: EML-GA²M against linear regression, EBM,
XGBoost, and gplearn on eight synthetic scientific targets.

The headline metric is the test R² evaluated on inputs outside the training
range. Tree-based regressors flatten at the edge of their training support
and therefore receive large negative R² on such a test set. A
symbolic-regression model that recovers the correct functional form
continues to predict correctly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from ..gam import EMLGAM
from ..train import TrainConfig
from .scientific import ScientificDataset, all_scientific


@dataclass
class ExtrapResult:
    dataset: str
    model: str
    r2_interp: float
    r2_extrap: float
    mse_interp: float
    mse_extrap: float
    train_time_s: float
    formula: str = ""


def _fit_linear(X_tr, y_tr):
    m = LinearRegression()
    m.fit(X_tr, y_tr)
    return m


def _fit_ebm(X_tr, y_tr, feature_names):
    try:
        from interpret.glassbox import ExplainableBoostingRegressor
    except ImportError:
        return None
    m = ExplainableBoostingRegressor(
        interactions=5, random_state=0, feature_names=feature_names
    )
    m.fit(X_tr, y_tr)
    return m


def _fit_xgboost(X_tr, y_tr):
    try:
        import xgboost as xgb
    except ImportError:
        return None
    m = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        random_state=0, n_jobs=-1,
    )
    m.fit(X_tr, y_tr)
    return m


def _fit_gplearn(X_tr, y_tr):
    """Genetic-programming symbolic regression baseline."""
    try:
        from gplearn.genetic import SymbolicRegressor
        from sklearn.utils.validation import validate_data
    except ImportError:
        return None

    # gplearn 0.4.2 expects an instance method that was removed in modern
    # scikit-learn releases.
    if not hasattr(SymbolicRegressor, "_validate_data"):
        def _vd(self, *args, **kwargs):
            return validate_data(self, *args, **kwargs)
        SymbolicRegressor._validate_data = _vd  # type: ignore[attr-defined]

    m = SymbolicRegressor(
        population_size=1500,
        generations=30,
        function_set=(
            "add", "sub", "mul", "div", "log", "sqrt", "abs", "neg", "inv",
        ),
        parsimony_coefficient=0.001,
        random_state=0,
        n_jobs=1,
        verbose=0,
    )
    m.fit(X_tr, y_tr)
    return m


class _GPlearnWrapper:
    """Uniform predict interface for gplearn."""

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
    X_tr,
    y_tr,
    feature_names: list[str],
    univariate_depth: int = 2,
    bivariate_depth: int = 2,
    interaction_pairs=None,
    auto_pairs: bool = True,
    n_epochs: int = 1200,
    standardize: bool = True,
    scale_normalize: bool = True,
) -> EMLGAM:
    if interaction_pairs is None and auto_pairs and X_tr.shape[1] >= 2:
        from itertools import combinations
        interaction_pairs = list(combinations(range(X_tr.shape[1]), 2))

    model = EMLGAM(
        n_features=X_tr.shape[1],
        univariate_depth=univariate_depth,
        bivariate_depth=bivariate_depth,
        feature_names=feature_names,
        interaction_pairs=interaction_pairs,
        standardize=standardize,
        scale_normalize=scale_normalize,
    )
    cfg = TrainConfig(n_epochs=n_epochs, lr=5e-2, entropy_weight=1e-3)
    model.fit(X_tr, y_tr, cfg=cfg, warm_start=True, verbose=False)
    return model


def _pred(model, X):
    return model.predict(X)


def bench_dataset(
    ds: ScientificDataset,
    models: tuple[str, ...] = ("linear", "ebm", "xgboost", "gplearn", "emlgam"),
    **emlgam_kwargs,
) -> list[ExtrapResult]:
    results: list[ExtrapResult] = []
    for name in models:
        t0 = time.perf_counter()
        model: Optional[Any] = None
        if name == "linear":
            model = _fit_linear(ds.X_train, ds.y_train)
        elif name == "ebm":
            model = _fit_ebm(ds.X_train, ds.y_train, ds.feature_names)
        elif name == "xgboost":
            model = _fit_xgboost(ds.X_train, ds.y_train)
        elif name == "gplearn":
            raw = _fit_gplearn(ds.X_train, ds.y_train)
            model = _GPlearnWrapper(raw) if raw is not None else None
        elif name == "emlgam":
            model = _fit_emlgam(
                ds.X_train, ds.y_train, ds.feature_names, **emlgam_kwargs
            )
        else:
            raise ValueError(name)
        if model is None:
            continue
        tr_time = time.perf_counter() - t0

        y_in_pred = _pred(model, ds.X_test_interp)
        y_ex_pred = _pred(model, ds.X_test_extrap)
        r2_in = r2_score(ds.y_test_interp, y_in_pred)
        r2_ex = r2_score(ds.y_test_extrap, y_ex_pred)
        mse_in = mean_squared_error(ds.y_test_interp, y_in_pred)
        mse_ex = mean_squared_error(ds.y_test_extrap, y_ex_pred)

        formula = ""
        if name == "emlgam":
            try:
                formula = str(model.total_formula(simplify=True))[:200]
            except Exception:
                pass
        elif name == "gplearn":
            formula = model.formula[:200] if isinstance(model, _GPlearnWrapper) else ""

        results.append(
            ExtrapResult(
                dataset=ds.name,
                model=name,
                r2_interp=r2_in,
                r2_extrap=r2_ex,
                mse_interp=mse_in,
                mse_extrap=mse_ex,
                train_time_s=tr_time,
                formula=formula,
            )
        )
    return results


def run_all(verbose: bool = True, **emlgam_kwargs) -> list[ExtrapResult]:
    all_results: list[ExtrapResult] = []
    for ds in all_scientific():
        if verbose:
            print(f"\n=== {ds.name} ({ds.domain}) ===")
            print(f"  target: {ds.description}")
        res = bench_dataset(ds, **emlgam_kwargs)
        all_results.extend(res)
        if verbose:
            header = (
                f"  {'model':8s} | {'R2 interp':>10s} | {'R2 extrap':>10s} | "
                f"{'MSE in':>10s} | {'MSE ex':>10s} | time"
            )
            print(header)
            print("  " + "-" * (len(header) - 2))
            for r in res:
                print(
                    f"  {r.model:8s} | {r.r2_interp:+10.4f} | "
                    f"{r.r2_extrap:+10.4f} | {r.mse_interp:10.3e} | "
                    f"{r.mse_extrap:10.3e} | {r.train_time_s:5.1f}s"
                )
            for r in res:
                if r.model == "emlgam" and r.formula:
                    print(f"  emlgam formula: {r.formula}")
    return all_results


if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    run_all()
