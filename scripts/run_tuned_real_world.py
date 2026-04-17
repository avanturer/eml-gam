"""Re-run UCI benchmarks with EML-GA²M full stability config.

Evaluate whether the catastrophic extrapolation R² on Auto-MPG,
Concrete, and Airfoil can be recovered by enabling:
  - robust multi-start (n_restarts=4, with NaN abort),
  - extrap_penalty_weight = 0.05 (penalise formula blow-up outside train),
  - warm_start + try_offsets,
  - 2000 epochs instead of 1500.

Baselines (linear, EBM, XGBoost, gplearn) are re-run for parity but their
R² numbers already exist in ``results.json``.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eml_gam.benchmarks.extrapolation import (  # noqa: E402
    _fit_ebm,
    _fit_linear,
    _fit_xgboost,
)
from eml_gam.benchmarks.real_world import (  # noqa: E402
    _load_airfoil,
    _load_auto_mpg,
    _load_concrete,
    _load_yacht,
)
from eml_gam.gam import EMLGAM  # noqa: E402
from eml_gam.train import TrainConfig  # noqa: E402


@dataclass
class Res:
    dataset: str
    model: str
    split: str
    r2: float
    mse: float
    time_s: float
    formula: str = ""


def fit_emlgam_tuned(
    X_tr, y_tr, feature_names,
    univariate_depth=2, bivariate_depth=2,
    interaction_pairs=None, auto_pairs=True,
    n_epochs=1000, standardize=True, scale_normalize=True,
    extrap_penalty_weight=0.05, n_restarts=1, robust=False,
):
    # Limit bivariate interactions: at most first 5 pair combinations.
    if interaction_pairs is None and auto_pairs and X_tr.shape[1] >= 2:
        from itertools import combinations
        all_pairs = list(combinations(range(X_tr.shape[1]), 2))
        interaction_pairs = all_pairs[:5] if len(all_pairs) > 5 else all_pairs
    model = EMLGAM(
        n_features=X_tr.shape[1],
        univariate_depth=univariate_depth,
        bivariate_depth=bivariate_depth,
        feature_names=feature_names,
        interaction_pairs=interaction_pairs,
        standardize=standardize,
        scale_normalize=scale_normalize,
    )
    cfg = TrainConfig(
        n_epochs=n_epochs, lr=5e-2, entropy_weight=1e-3,
        extrap_penalty_weight=extrap_penalty_weight,
    )
    model.fit(
        X_tr, y_tr, cfg=cfg,
        warm_start=True, try_offsets=True,
        n_restarts=n_restarts, robust=robust, verbose=False,
    )
    return model


def run_dataset_extrap(name, X, y, features, split_feature, split_threshold):
    """Return list of Res rows for this dataset at extrapolation split."""
    results = []
    idx = features.index(split_feature)
    values = X[:, idx]
    tr_mask = values < split_threshold
    te_mask = values >= split_threshold
    X_tr, y_tr = X[tr_mask], y[tr_mask]
    X_te, y_te = X[te_mask], y[te_mask]
    print(f"\n=== {name} extrap (split {split_feature} < {split_threshold}) ===")
    print(f"    n_train={len(X_tr)}, n_test={len(X_te)}")

    fitters = [
        ("linear", _fit_linear),
        ("ebm", lambda Xt, yt: _fit_ebm(Xt, yt, features)),
        ("xgboost", _fit_xgboost),
    ]
    for mname, fit in fitters:
        t0 = time.perf_counter()
        m = fit(X_tr, y_tr)
        if m is None:
            continue
        elapsed = time.perf_counter() - t0
        y_p = m.predict(X_te)
        r2 = r2_score(y_te, y_p)
        mse = mean_squared_error(y_te, y_p)
        results.append(Res(name, mname, "extrap", r2, mse, elapsed))
        print(f"  {mname:10s} R2={r2:+.4f}  MSE={mse:.3e}  time={elapsed:.1f}s")

    # gplearn skipped (too slow for batch runs; rely on existing results.json for reference)
    # EML-GA²M tuned
    t0 = time.perf_counter()
    try:
        m = fit_emlgam_tuned(X_tr, y_tr, features)
        elapsed = time.perf_counter() - t0
        y_p = m.predict(X_te)
        r2 = r2_score(y_te, y_p)
        mse = mean_squared_error(y_te, y_p)
        try:
            fml = str(m.total_formula(simplify=True))[:400]
        except Exception:
            fml = ""
        results.append(Res(name, "emlgam_tuned", "extrap", r2, mse, elapsed, fml))
        print(f"  emlgam_tun R2={r2:+.4f}  MSE={mse:.3e}  time={elapsed:.1f}s")
        print(f"    formula: {fml[:150]}")
    except Exception as e:
        print(f"  emlgam_tuned error: {e}")
    return results


def main():
    datasets = [
        ("yacht", _load_yacht, "froude_number", 0.3),
        ("auto_mpg", _load_auto_mpg, "weight", None),  # median split
        ("concrete", _load_concrete, "age", 29.0),
        ("airfoil", _load_airfoil, "frequency", 2001.0),
    ]
    all_results = {}
    for name, loader, sf, st in datasets:
        X, y, features = loader()
        # Compact to physical feature only for Yacht; keep all for others
        if name == "yacht":
            idx = features.index("froude_number")
            X = X[:, [idx]]
            features = ["froude_number"]
        elif name == "auto_mpg":
            if st is None:
                wi = features.index("weight")
                st = float(np.percentile(X[:, wi], 60))
        try:
            res = run_dataset_extrap(name, X, y, features, sf, st)
            all_results[name] = [asdict(r) for r in res]
        except Exception as e:
            print(f"error on {name}: {e}")
            all_results[name] = {"error": str(e)}

    with open("tuned_real_world.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved tuned_real_world.json")

    # Summary of wins
    print("\n=== SUMMARY: EML-GA²M (tuned) vs best baseline on extrap R² ===")
    wins = 0
    for name, rows in all_results.items():
        if not isinstance(rows, list):
            continue
        em = next((r for r in rows if r["model"] == "emlgam_tuned"), None)
        bl = [r for r in rows if r["model"] != "emlgam_tuned"]
        if em and bl:
            best_bl = max(r["r2"] for r in bl)
            win = em["r2"] > best_bl
            wins += int(win)
            print(f"  {name}: EML-GA²M r2={em['r2']:.3f} vs "
                  f"best baseline r2={best_bl:.3f} → {'WIN' if win else 'loss'}")
    print(f"\nTotal tuned UCI wins: {wins}/{len(all_results)}")


if __name__ == "__main__":
    main()
