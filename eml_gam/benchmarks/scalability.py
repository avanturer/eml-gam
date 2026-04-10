"""High-dimensional benchmark: 20 features, 5 of which carry signal.

The target is

    y = 3 * log(x1) + 2 * exp(x2) - 4 * x3 + 5 * (x4 / x5) + noise,

so features 1..5 are informative and features 6..20 are standard-normal
noise. Correlation-based pair selection is applied inside EMLGAM; the
ground-truth interaction is the ratio ``x4 / x5``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from ..gam import EMLGAM
from ..interaction_select import select_pairs
from ..train import TrainConfig
from .extrapolation import (
    _GPlearnWrapper,
    _fit_ebm,
    _fit_gplearn,
    _fit_linear,
    _fit_xgboost,
)


@dataclass
class ScaleResult:
    model: str
    r2: float
    time_s: float
    n_params: int = 0
    selected_pairs: str = ""


def make_high_dim_dataset(
    n_samples: int = 2000,
    n_features: int = 20,
    noise: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.5, 2.0, n_samples)
    x2 = rng.uniform(-1.0, 1.0, n_samples)
    x3 = rng.uniform(-1.0, 1.0, n_samples)
    x4 = rng.uniform(1.0, 3.0, n_samples)
    x5 = rng.uniform(1.0, 3.0, n_samples)
    noise_feats = rng.normal(0, 1, (n_samples, n_features - 5))

    X = np.concatenate(
        [np.stack([x1, x2, x3, x4, x5], axis=1), noise_feats], axis=1
    )
    y = (
        3 * np.log(x1)
        + 2 * np.exp(x2)
        - 4 * x3
        + 5 * (x4 / x5)
        + rng.normal(0, noise, n_samples)
    )
    names = [f"x{i + 1}" for i in range(n_features)]
    return X, y, names


def run_scalability(verbose: bool = True) -> list[ScaleResult]:
    X, y, names = make_high_dim_dataset(n_samples=2000, n_features=20, seed=0)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    if verbose:
        print(
            f"=== scalability: n={X_tr.shape[0]} features={X_tr.shape[1]} ==="
        )
        print("target: 3 log(x1) + 2 exp(x2) - 4 x3 + 5 (x4/x5) + noise")
        print("signal features: x1..x5; noise features: x6..x20\n")

    results: list[ScaleResult] = []

    for name, fit in [
        ("linear", _fit_linear),
        ("ebm", lambda Xt, yt: _fit_ebm(Xt, yt, names)),
        ("xgboost", _fit_xgboost),
        ("gplearn", _fit_gplearn),
    ]:
        t0 = time.perf_counter()
        m = fit(X_tr, y_tr)
        if m is None:
            continue
        dt = time.perf_counter() - t0
        if name == "gplearn":
            m = _GPlearnWrapper(m)
        pred = m.predict(X_te)
        r2 = r2_score(y_te, pred)
        results.append(ScaleResult(model=name, r2=r2, time_s=dt))
        if verbose:
            print(f"  {name:10s}  R2={r2:+.4f}  time={dt:5.1f}s")

    t0 = time.perf_counter()
    from sklearn.linear_model import LinearRegression

    residual = y_tr - LinearRegression().fit(X_tr, y_tr).predict(X_tr)
    pairs = select_pairs(
        X_tr, y_tr, method="correlation", top_k=5, residual=residual
    )

    model = EMLGAM(
        n_features=X.shape[1],
        univariate_depth=2,
        bivariate_depth=2,
        feature_names=names,
        interaction_pairs=pairs,
        standardize=True,
        scale_normalize=True,
    )
    cfg = TrainConfig(n_epochs=1000, lr=5e-2, entropy_weight=1e-3)
    model.fit(X_tr, y_tr, cfg=cfg, warm_start=True, verbose=False)
    dt = time.perf_counter() - t0
    pred = model.predict(X_te)
    r2 = r2_score(y_te, pred)
    n_params = sum(p.numel() for p in model.parameters())
    results.append(
        ScaleResult(
            model="emlgam",
            r2=r2,
            time_s=dt,
            n_params=n_params,
            selected_pairs=str(pairs),
        )
    )
    if verbose:
        print(
            f"  emlgam     R2={r2:+.4f}  time={dt:5.1f}s  params={n_params}"
        )
        print(
            f"    pair selection: {pairs}  "
            f"(truth: (3, 4) = indices of x4 / x5)"
        )
        try:
            forms = model.get_formulas()
            print("    top univariate formulas:")
            for k in names[:5]:
                expr = forms.get(k)
                if expr is not None:
                    print(f"      {k}: {str(expr)[:80]}")
        except Exception:
            pass

    return results


if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    run_scalability()
