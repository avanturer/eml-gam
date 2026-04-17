"""Run EML-GA²M tuned + baselines on two more UCI datasets:

- Abalone: predict rings (age proxy) from shell measurements.
  Classical allometric scaling relation: rings ~ log(whole_weight).
  Extrap: train on small shells (whole_weight < median), test on
  large shells.

- Forest Fires: predict log1p(area) from meteorological features.
  The target is already log-transformed; relationship to features is
  roughly exponential (fire spread rate). Extrap: train on low
  temperatures, test on high.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eml_gam.benchmarks.extrapolation import (  # noqa: E402
    _fit_ebm, _fit_gplearn, _fit_linear, _fit_xgboost, _GPlearnWrapper,
)
from eml_gam.gam import EMLGAM  # noqa: E402
from eml_gam.train import TrainConfig  # noqa: E402

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)


@dataclass
class Res:
    dataset: str
    model: str
    r2_extrap: float
    mse_extrap: float
    time_s: float
    formula: str = ""


def fit_emlgam_tuned(X_tr, y_tr, feature_names):
    from itertools import combinations
    all_pairs = list(combinations(range(X_tr.shape[1]), 2)) if X_tr.shape[1] >= 2 else None
    pairs = all_pairs[:3] if all_pairs and len(all_pairs) > 3 else all_pairs
    model = EMLGAM(
        n_features=X_tr.shape[1],
        univariate_depth=2,
        bivariate_depth=2,
        feature_names=feature_names,
        interaction_pairs=pairs,
        standardize=True,
        scale_normalize=True,
    )
    cfg = TrainConfig(
        n_epochs=800, lr=5e-2, entropy_weight=1e-3,
        extrap_penalty_weight=0.05,
    )
    model.fit(
        X_tr, y_tr, cfg=cfg,
        warm_start=True, try_offsets=True,
        n_restarts=1, robust=False, verbose=False,
    )
    return model


def run_bench(name, X, y, feature_names, split_feature, split_value):
    idx = feature_names.index(split_feature)
    values = X[:, idx]
    tr_mask = values < split_value
    te_mask = values >= split_value
    X_tr, y_tr = X[tr_mask], y[tr_mask]
    X_te, y_te = X[te_mask], y[te_mask]
    print(f"\n=== {name} extrap: {split_feature} < {split_value:g} ===")
    print(f"    n_train={len(X_tr)}, n_test={len(X_te)}")
    if len(X_tr) < 50 or len(X_te) < 20:
        print("  Skipped (insufficient data)")
        return []

    res = []
    fitters = [
        ("linear", _fit_linear),
        ("ebm", lambda Xt, yt: _fit_ebm(Xt, yt, feature_names)),
        ("xgboost", _fit_xgboost),
    ]
    for mname, fit in fitters:
        t0 = time.perf_counter()
        m = fit(X_tr, y_tr)
        if m is None:
            continue
        el = time.perf_counter() - t0
        y_p = m.predict(X_te)
        r2 = r2_score(y_te, y_p)
        mse = mean_squared_error(y_te, y_p)
        res.append(Res(name, mname, r2, mse, el))
        print(f"  {mname:10s} R2={r2:+.4f}  MSE={mse:.3e}  time={el:.1f}s")

    # gplearn skipped for speed.
    t0 = time.perf_counter()
    try:
        m = fit_emlgam_tuned(X_tr, y_tr, feature_names)
        el = time.perf_counter() - t0
        y_p = m.predict(X_te)
        r2 = r2_score(y_te, y_p)
        mse = mean_squared_error(y_te, y_p)
        try:
            fml = str(m.total_formula(simplify=True))[:400]
        except Exception:
            fml = ""
        res.append(Res(name, "emlgam_tuned", r2, mse, el, fml))
        print(f"  emlgam_tun R2={r2:+.4f}  MSE={mse:.3e}  time={el:.1f}s")
        print(f"    formula: {fml[:200]}")
    except Exception as e:
        print(f"  emlgam_tuned error: {e}")
    return res


def load_abalone():
    df = pd.read_csv(os.path.join(DATA_DIR, "abalone.csv"))
    feature_cols = [
        "length", "diameter", "height",
        "whole_weight", "shucked_weight", "viscera_weight", "shell_weight",
    ]
    X = df[feature_cols].values.astype(np.float32)
    y = df["rings"].values.astype(np.float32)
    return X, y, feature_cols


def load_forest_fires():
    df = pd.read_csv(os.path.join(DATA_DIR, "forest_fires.csv"))
    feature_cols = [
        "X", "Y", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain",
    ]
    X = df[feature_cols].values.astype(np.float32)
    # Use log1p(area) as target (already stored as area_log1p)
    y = df["area_log1p"].values.astype(np.float32)
    return X, y, feature_cols


def main():
    all_results = {}

    print(">>> Abalone")
    X, y, features = load_abalone()
    med_ww = float(np.percentile(X[:, features.index("whole_weight")], 60))
    res = run_bench("abalone", X, y, features, "whole_weight", med_ww)
    all_results["abalone"] = [asdict(r) for r in res]

    print("\n>>> Forest Fires")
    X, y, features = load_forest_fires()
    # Split by temperature
    med_t = float(np.percentile(X[:, features.index("temp")], 75))
    res = run_bench("forest_fires", X, y, features, "temp", med_t)
    all_results["forest_fires"] = [asdict(r) for r in res]

    with open("abalone_forest_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved abalone_forest_results.json")

    # Summary
    print("\n=== SUMMARY ===")
    wins = 0
    for name, rows in all_results.items():
        if not isinstance(rows, list):
            continue
        em = next((r for r in rows if r["model"] == "emlgam_tuned"), None)
        bl = [r for r in rows if r["model"] != "emlgam_tuned"]
        if em and bl:
            best_bl = max(r["r2_extrap"] for r in bl)
            win = em["r2_extrap"] > best_bl
            wins += int(win)
            print(f"  {name}: EML-GA²M r2={em['r2_extrap']:.3f} vs "
                  f"best baseline r2={best_bl:.3f} → {'WIN' if win else 'loss'}")
    print(f"\nAbalone+ForestFires wins: {wins}/{len(all_results)}")


if __name__ == "__main__":
    main()
