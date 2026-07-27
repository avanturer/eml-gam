"""9-dataset UCI extrapolation benchmark: SafePoolGAM vs baselines.

Re-runs the full physical-extrapolation-split sweep from the earlier
README table with the new SafePoolGAM, alongside fresh runs of the
linear / EBM / XGBoost baselines and the earlier (unguarded) EML-GA²M
on exactly the same data. All models see identical train/test splits.

Data files come from public GitHub mirrors of the UCI datasets
(archive.ics.uci.edu is not reachable from every environment); the
mirror provenance is recorded in ``scripts/download_datasets.py`` and
the loaders below. Column conventions may differ cosmetically from the
original UCI files; since every model consumes the same arrays, the
comparison is internally consistent.

Splits (train = below threshold, test = above):
    yacht           froude          >= 0.3 test
    concrete        age             > 28 test
    superconduct    mean_atomic_mass q50 (median) test above
    auto_mpg        weight          q60 test above
    energy_eff      X7 glazing area > 0.25 test
    abalone         whole_weight    q60 test above
    ccpp            AT (ambient T)  median test above
    airfoil         frequency       > 2000 test
    forest_fires    temp            > 22.8 test

Writes ``safe_uci_results.json``.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

from eml_gam.safe import SafePoolGAM  # noqa: E402

DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)

SUPERCONDUCT_SUBSAMPLE = 5000  # train rows, applied identically to all models
SEED = 0


# ---------------------------------------------------------------------------
# Loaders (mirror files) — return (X, y, feature_names, split_idx, split_val)
# ---------------------------------------------------------------------------


def load_yacht():
    df = pd.read_csv(
        os.path.join(DATA, "yacht_hydrodynamics.data"),
        sep=r"\s+", header=None,
        names=["long_pos", "prism_coef", "len_disp", "beam_draught",
               "len_beam", "froude", "resistance"],
    )
    X = df.iloc[:, :6].to_numpy(float)
    y = df["resistance"].to_numpy(float)
    return X, y, list(df.columns[:6]), 5, 0.3


def load_concrete():
    df = pd.read_csv(os.path.join(DATA, "concrete.csv"))
    y = df["strength"].to_numpy(float)
    X = df.drop(columns=["strength"]).to_numpy(float)
    names = [c for c in df.columns if c != "strength"]
    return X, y, names, names.index("age"), 28.0 + 1e-9


def load_superconduct():
    df = pd.read_csv(os.path.join(DATA, "superconduct_train.csv"))
    y = df["critical_temp"].to_numpy(float)
    X = df.drop(columns=["critical_temp"]).to_numpy(float)
    names = [c for c in df.columns if c != "critical_temp"]
    j = names.index("mean_atomic_mass")
    return X, y, names, j, float(np.median(X[:, j]))


def load_auto_mpg():
    df = pd.read_csv(os.path.join(DATA, "auto_mpg_seaborn.csv")).dropna()
    y = df["mpg"].to_numpy(float)
    cols = ["cylinders", "displacement", "horsepower", "weight",
            "acceleration", "model_year"]
    X = df[cols].to_numpy(float)
    j = cols.index("weight")
    return X, y, cols, j, float(np.quantile(X[:, j], 0.6))


def load_energy_eff():
    df = pd.read_csv(os.path.join(DATA, "ENB2012_data.csv"))
    y = df["Y1"].to_numpy(float)
    cols = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
    X = df[cols].to_numpy(float)
    return X, y, cols, cols.index("X7"), 0.25 + 1e-9


def load_abalone():
    df = pd.read_csv(
        os.path.join(DATA, "abalone_jb.csv"), header=None,
        names=["sex", "length", "diameter", "height", "whole_weight",
               "shucked_weight", "viscera_weight", "shell_weight", "rings"],
    )
    y = df["rings"].to_numpy(float)
    cols = ["length", "diameter", "height", "whole_weight",
            "shucked_weight", "viscera_weight", "shell_weight"]
    X = df[cols].to_numpy(float)
    j = cols.index("whole_weight")
    return X, y, cols, j, float(np.percentile(X[:, j], 60))


def load_ccpp():
    df = pd.read_csv(os.path.join(DATA, "ccpp.csv"))
    y = df["PE"].to_numpy(float)
    cols = ["AT", "V", "AP", "RH"]
    X = df[cols].to_numpy(float)
    return X, y, cols, cols.index("AT"), float(np.median(X[:, 0]))


def load_airfoil():
    df = pd.read_csv(
        os.path.join(DATA, "airfoil_self_noise.dat"), sep=r"\s+", header=None,
        names=["frequency", "aoa", "chord", "velocity", "thickness", "spl"],
    )
    y = df["spl"].to_numpy(float)
    cols = ["frequency", "aoa", "chord", "velocity", "thickness"]
    X = df[cols].to_numpy(float)
    return X, y, cols, cols.index("frequency"), 2000.0 + 1e-9


def load_forest_fires():
    df = pd.read_csv(os.path.join(DATA, "forestfires.csv"))
    y = np.log1p(df["area"].to_numpy(float))
    cols = ["X", "Y", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]
    X = df[cols].to_numpy(float)
    return X, y, cols, cols.index("temp"), 22.8


LOADERS = {
    "yacht": load_yacht,
    "concrete": load_concrete,
    "superconductivity": load_superconduct,
    "auto_mpg": load_auto_mpg,
    "energy_eff": load_energy_eff,
    "abalone": load_abalone,
    "ccpp": load_ccpp,
    "airfoil": load_airfoil,
    "forest_fires": load_forest_fires,
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def fit_linear(X_tr, y_tr, X_te, names):
    t0 = time.time()
    m = LinearRegression().fit(X_tr, y_tr)
    return m.predict(X_te), time.time() - t0, ""


def fit_ebm(X_tr, y_tr, X_te, names):
    from interpret.glassbox import ExplainableBoostingRegressor

    t0 = time.time()
    m = ExplainableBoostingRegressor(random_state=SEED)
    m.fit(X_tr, y_tr)
    return m.predict(X_te), time.time() - t0, ""


def fit_xgb(X_tr, y_tr, X_te, names):
    from xgboost import XGBRegressor

    t0 = time.time()
    m = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        random_state=SEED, verbosity=0,
    )
    m.fit(X_tr, y_tr)
    return m.predict(X_te), time.time() - t0, ""


def fit_emlgam_unguarded(X_tr, y_tr, X_te, names):
    """The earlier tuned EML-GA²M configuration (no safety gating)."""
    from itertools import combinations

    from eml_gam.gam import EMLGAM
    from eml_gam.train import TrainConfig

    t0 = time.time()
    p = X_tr.shape[1]
    pairs = list(combinations(range(p), 2))[:5] if p >= 2 else None
    model = EMLGAM(
        n_features=p, univariate_depth=2, bivariate_depth=2,
        feature_names=list(names), interaction_pairs=pairs,
        standardize=True, scale_normalize=True,
    )
    cfg = TrainConfig(
        n_epochs=1000, lr=5e-2, entropy_weight=1e-3,
        extrap_penalty_weight=0.05,
    )
    model.fit(
        X_tr, y_tr, cfg=cfg, warm_start=True, try_offsets=True,
        n_restarts=1, robust=False, verbose=False,
    )
    pred = model.predict(X_te)
    return pred, time.time() - t0, ""


def fit_safe(X_tr, y_tr, X_te, names):
    t0 = time.time()
    m = SafePoolGAM(depth=4, random_state=SEED)
    m.fit(X_tr, y_tr)
    fs = m.formulas(list(names))
    desc = "; ".join(
        f"{k}[{v['gate']}]: {str(v['expr'])[:60]}" for k, v in fs.items()
    )
    return m.predict(X_te), time.time() - t0, desc


MODELS = {
    "linear": fit_linear,
    "ebm": fit_ebm,
    "xgboost": fit_xgb,
    "emlgam_unguarded": fit_emlgam_unguarded,
    "safepoolgam": fit_safe,
}


def main(out_path="safe_uci_results.json", skip_unguarded=False):
    results = {}
    for name, loader in LOADERS.items():
        X, y, names, j, split_val = loader()
        tr = X[:, j] < split_val
        te = ~tr
        X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]
        if name == "superconductivity" and len(y_tr) > SUPERCONDUCT_SUBSAMPLE:
            rng = np.random.default_rng(SEED)
            keep = rng.choice(len(y_tr), SUPERCONDUCT_SUBSAMPLE, replace=False)
            X_tr, y_tr = X_tr[keep], y_tr[keep]
        print(
            f"\n=== {name}: split {names[j]} @ {split_val:g} | "
            f"train {len(y_tr)}, test {len(y_te)} ==="
        )
        row = {
            "split_feature": names[j],
            "split_value": split_val,
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "models": {},
        }
        for mname, fitter in MODELS.items():
            if skip_unguarded and mname == "emlgam_unguarded":
                continue
            try:
                pred, secs, desc = fitter(X_tr, y_tr, X_te, names)
                r2 = float(r2_score(y_te, pred))
            except Exception as exc:  # noqa: BLE001
                print(f"  {mname}: FAILED ({exc})")
                row["models"][mname] = {"r2": None, "error": str(exc)}
                continue
            row["models"][mname] = {
                "r2": r2, "seconds": secs, "formulas": desc,
            }
            print(f"  {mname:18s} R2 = {r2:+9.3f}  ({secs:.0f}s)")
        results[name] = row
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
