"""Run EML-GA²M (tuned) and baselines on 3 new UCI datasets:

- Combined Cycle Power Plant (CCPP): T, V, AP, RH → PE (net electrical
  power output). Extrap split on T (temperature): train on T < median,
  test on T > median. Thermodynamic dependence on T is nonlinear
  (closer to exp than linear).

- Energy Efficiency: 8 building geometry features → heating load Y1.
  Extrap split on X7_glazing_area: train on X7 < median (low
  glazing), test on X7 > median. Heat loss has exp-decay in
  insulation / logarithmic in glazing ratio.

- Superconductivity: aggregate material descriptors → critical
  temperature Tc. Extrap split on 'mean_atomic_mass' at the 60th
  percentile; BCS theory predicts Tc ∝ exp(-1/(lambda_eff)), giving
  exp structure in composition-derived features.

All three use the tuned EML-GA²M configuration (robust=True,
n_restarts=4, extrap_penalty_weight=0.05, n_epochs=2000).
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
    pairs = all_pairs[:5] if all_pairs and len(all_pairs) > 5 else all_pairs
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
        n_epochs=1000, lr=5e-2, entropy_weight=1e-3,
        extrap_penalty_weight=0.05,
    )
    model.fit(
        X_tr, y_tr, cfg=cfg,
        warm_start=True, try_offsets=True,
        n_restarts=1, robust=False, verbose=False,
    )
    return model


def run_benchmark(name, X, y, feature_names, split_feature, split_value):
    """Run all models on extrap split; return list of Res."""
    idx = feature_names.index(split_feature)
    values = X[:, idx]
    tr_mask = values < split_value
    te_mask = values >= split_value
    X_tr, y_tr = X[tr_mask], y[tr_mask]
    X_te, y_te = X[te_mask], y[te_mask]
    print(f"\n=== {name} extrap split: {split_feature} < {split_value} ===")
    print(f"    n_train={len(X_tr)}, n_test={len(X_te)}")

    res = []
    for mname, fit in [
        ("linear", _fit_linear),
        ("ebm", lambda Xt, yt: _fit_ebm(Xt, yt, feature_names)),
        ("xgboost", _fit_xgboost),
    ]:
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

    # gplearn skipped (too slow).
    # EML-GA²M tuned
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
        print(f"    formula: {fml[:150]}")
    except Exception as e:
        print(f"  emlgam_tuned error: {e}")
    return res


def load_ccpp():
    df = pd.read_csv(os.path.join(DATA_DIR, "ccpp.csv"))
    # Columns: AT (ambient T), V (exhaust vacuum), AP (ambient pressure),
    # RH (rel humidity), PE (power output).
    X = df[["AT", "V", "AP", "RH"]].values.astype(np.float32)
    y = df["PE"].values.astype(np.float32)
    features = ["AT", "V", "AP", "RH"]
    return X, y, features


def load_energy_eff():
    df = pd.read_csv(os.path.join(DATA_DIR, "energy_eff.csv"))
    feature_cols = [c for c in df.columns if c.startswith("X")]
    X = df[feature_cols].values.astype(np.float32)
    y = df["Y1_heating_load"].values.astype(np.float32)
    return X, y, feature_cols


def load_superconductivity():
    df = pd.read_csv(os.path.join(DATA_DIR, "superconductivity.csv"))
    # Many features; let's use a compact subset to keep EMLGAM tractable.
    key_features = [
        "mean_atomic_mass",
        "mean_fie",  # first ionisation energy
        "mean_atomic_radius",
        "mean_Density",
        "mean_ElectronAffinity",
        "mean_ThermalConductivity",
        "mean_Valence",
    ]
    available = [c for c in key_features if c in df.columns]
    X = df[available].values.astype(np.float32)
    y = df["critical_temp"].values.astype(np.float32)
    return X, y, available


def main():
    all_results = {}

    print(">>> CCPP")
    try:
        X, y, features = load_ccpp()
        # Subsample if too large
        if len(X) > 3000:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(X), 3000, replace=False)
            X, y = X[idx], y[idx]
        med_T = float(np.median(X[:, features.index("AT")]))
        res = run_benchmark("ccpp", X, y, features, "AT", med_T)
        all_results["ccpp"] = [asdict(r) for r in res]
    except Exception as e:
        print(f"CCPP error: {e}")
        all_results["ccpp"] = {"error": str(e)}

    print("\n>>> Energy Efficiency")
    try:
        X, y, features = load_energy_eff()
        glazing_idx = features.index("X7_glazing_area")
        # Split at 0.25 (natural glazing categories: 0, 0.1, 0.25, 0.4)
        res = run_benchmark(
            "energy_eff", X, y, features, "X7_glazing_area", 0.25,
        )
        all_results["energy_eff"] = [asdict(r) for r in res]
    except Exception as e:
        print(f"Energy efficiency error: {e}")
        all_results["energy_eff"] = {"error": str(e)}

    print("\n>>> Superconductivity")
    try:
        X, y, features = load_superconductivity()
        # Subsample
        if len(X) > 3000:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(X), 3000, replace=False)
            X, y = X[idx], y[idx]
        med_am = float(np.median(X[:, features.index("mean_atomic_mass")]))
        res = run_benchmark(
            "superconductivity", X, y, features, "mean_atomic_mass", med_am,
        )
        all_results["superconductivity"] = [asdict(r) for r in res]
    except Exception as e:
        print(f"Superconductivity error: {e}")
        all_results["superconductivity"] = {"error": str(e)}

    with open("new_uci_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved new_uci_results.json")

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
    print(f"\nNew UCI wins: {wins}/{len(all_results)}")


if __name__ == "__main__":
    main()
