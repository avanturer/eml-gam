"""Real-world benchmarks on UCI datasets.

Two datasets are included:

  - Yacht Hydrodynamics (``yacht.csv``): residuary resistance of a yacht
    hull as a function of six geometric parameters. Training is restricted
    to slow regimes (Froude number ``Fr < 0.3``); the test set contains the
    planing regime ``Fr >= 0.3``. The textbook result is a steep power-law
    dependence on ``Fr``.

  - Auto-MPG (``auto_mpg.csv``): fuel economy of vehicles as a function of
    seven engine and body parameters. This is the classical commercial
    tabular benchmark; tree-based methods are expected to dominate.

An ``isolated_physics`` flag restricts the input to the single feature that
carries the dominant physical effect (Froude number for Yacht, vehicle
weight for Auto-MPG). This gives a cleaner view of extrapolation in the
regime where the symbolic model has a structural advantage.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .extrapolation import (
    _GPlearnWrapper,
    _fit_ebm,
    _fit_emlgam,
    _fit_gplearn,
    _fit_linear,
    _fit_xgboost,
)


DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
)
AUTO_MPG_PATH = os.path.join(DATA_DIR, "auto_mpg.csv")
YACHT_PATH = os.path.join(DATA_DIR, "yacht.csv")


@dataclass
class RWResult:
    dataset: str
    model: str
    split: str
    r2: float
    mse: float
    time_s: float
    formula: str = ""


def _load_auto_mpg() -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(AUTO_MPG_PATH)
    features = [
        "cylinders", "displacement", "horsepower", "weight",
        "acceleration", "model_year", "origin",
    ]
    X = df[features].to_numpy(dtype=np.float64)
    y = df["mpg"].to_numpy(dtype=np.float64)
    return X, y, features


def _load_yacht() -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(YACHT_PATH)
    features = [
        "longitudinal_pos", "prismatic_coef", "length_disp_ratio",
        "beam_draught_ratio", "length_beam_ratio", "froude_number",
    ]
    X = df[features].to_numpy(dtype=np.float64)
    y = df["residuary_resistance"].to_numpy(dtype=np.float64)
    return X, y, features


def _fit_and_score(
    name: str, fit_fn, X_tr, y_tr, X_te, y_te, **fit_kwargs
) -> RWResult | None:
    t0 = time.perf_counter()
    model = fit_fn(X_tr, y_tr, **fit_kwargs) if fit_kwargs else fit_fn(X_tr, y_tr)
    if model is None:
        return None
    dt = time.perf_counter() - t0
    if name == "gplearn":
        pred = _GPlearnWrapper(model).predict(X_te)
    else:
        pred = model.predict(X_te)
    r2 = r2_score(y_te, pred)
    mse = mean_squared_error(y_te, pred)
    formula = ""
    if name == "emlgam":
        try:
            formula = str(model.total_formula(simplify=True))[:300]
        except Exception:
            pass
    elif name == "gplearn":
        try:
            formula = str(model._program)[:300]
        except Exception:
            pass
    return RWResult(
        dataset="",
        model=name,
        split="",
        r2=r2,
        mse=mse,
        time_s=dt,
        formula=formula,
    )


def _run_dataset(
    X: np.ndarray,
    y: np.ndarray,
    features: list[str],
    dataset_name: str,
    split_feature: str,
    split_threshold: float,
    verbose: bool,
) -> list[RWResult]:
    all_results: list[RWResult] = []

    # Interpolation
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    if verbose:
        print(
            f"\n=== {dataset_name} interpolation "
            f"(n_train={len(X_tr)}, n_test={len(X_te)}) ==="
        )
    for name, fit in [
        ("linear", _fit_linear),
        ("ebm", lambda Xt, yt: _fit_ebm(Xt, yt, features)),
        ("xgboost", _fit_xgboost),
        ("gplearn", _fit_gplearn),
    ]:
        r = _fit_and_score(name, fit, X_tr, y_tr, X_te, y_te)
        if r is None:
            continue
        r.split = "interp"
        r.dataset = dataset_name
        all_results.append(r)
        if verbose:
            print(
                f"  {r.model:10s}  R2={r.r2:+.4f}  MSE={r.mse:.2f}  "
                f"time={r.time_s:5.1f}s"
            )

    r = _fit_and_score(
        "emlgam", _fit_emlgam, X_tr, y_tr, X_te, y_te,
        feature_names=features,
        n_epochs=1500,
        univariate_depth=2,
        bivariate_depth=2,
        auto_pairs=len(features) >= 2,
    )
    if r is not None:
        r.split = "interp"
        r.dataset = dataset_name
        all_results.append(r)
        if verbose:
            print(
                f"  {r.model:10s}  R2={r.r2:+.4f}  MSE={r.mse:.2f}  "
                f"time={r.time_s:5.1f}s"
            )
            if r.formula:
                print(f"    formula: {r.formula[:200]}")

    # Extrapolation split by one physical feature
    if split_feature in features:
        split_idx = features.index(split_feature)
        values = X[:, split_idx]
        tr_mask = values < split_threshold
        te_mask = values >= split_threshold
        X_tr, y_tr = X[tr_mask], y[tr_mask]
        X_te, y_te = X[te_mask], y[te_mask]
        if verbose:
            print(
                f"\n=== {dataset_name} extrapolation by {split_feature} "
                f"(threshold {split_threshold:g}, "
                f"n_train={len(X_tr)}, n_test={len(X_te)}) ==="
            )
        for name, fit in [
            ("linear", _fit_linear),
            ("ebm", lambda Xt, yt: _fit_ebm(Xt, yt, features)),
            ("xgboost", _fit_xgboost),
            ("gplearn", _fit_gplearn),
        ]:
            r = _fit_and_score(name, fit, X_tr, y_tr, X_te, y_te)
            if r is None:
                continue
            r.split = "extrap"
            r.dataset = dataset_name
            all_results.append(r)
            if verbose:
                print(
                    f"  {r.model:10s}  R2={r.r2:+.4f}  MSE={r.mse:.2f}  "
                    f"time={r.time_s:5.1f}s"
                )

        r = _fit_and_score(
            "emlgam", _fit_emlgam, X_tr, y_tr, X_te, y_te,
            feature_names=features,
            n_epochs=1500,
            univariate_depth=2,
            bivariate_depth=2,
            auto_pairs=len(features) >= 2,
        )
        if r is not None:
            r.split = "extrap"
            r.dataset = dataset_name
            all_results.append(r)
            if verbose:
                print(
                    f"  {r.model:10s}  R2={r.r2:+.4f}  MSE={r.mse:.2f}  "
                    f"time={r.time_s:5.1f}s"
                )
                if r.formula:
                    print(f"    formula: {r.formula[:200]}")
    return all_results


def run_yacht(
    verbose: bool = True, isolated_physics: bool = True
) -> list[RWResult]:
    """UCI Yacht Hydrodynamics benchmark."""
    X, y, features = _load_yacht()
    if isolated_physics:
        idx = features.index("froude_number")
        X = X[:, [idx]]
        features = ["froude_number"]
    return _run_dataset(
        X, y, features, dataset_name="yacht",
        split_feature="froude_number", split_threshold=0.3,
        verbose=verbose,
    )


def run_auto_mpg(
    verbose: bool = True, isolated_physics: bool = False
) -> list[RWResult]:
    """UCI Auto-MPG benchmark."""
    X, y, features = _load_auto_mpg()
    if isolated_physics:
        idx = features.index("weight")
        X = X[:, [idx]]
        features = ["weight"]
    return _run_dataset(
        X, y, features, dataset_name="auto_mpg",
        split_feature="weight",
        split_threshold=float(np.percentile(X[:, features.index("weight")], 60))
        if "weight" in features else 0.0,
        verbose=verbose,
    )


if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    run_yacht(isolated_physics=True)
    run_auto_mpg(isolated_physics=False)
