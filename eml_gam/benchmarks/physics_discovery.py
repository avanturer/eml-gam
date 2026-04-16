"""Physics discovery case studies.

Three end-to-end demonstrations that EML-GA^2M recovers known physical
laws from noisy observational data. The cases are chosen so that the
ground-truth law sits inside EML's depth-2 expressibility class:

    arrhenius_rate
        Reaction rate as a function of absolute temperature.
        ``k = A * exp(-Ea / (R T))``. Train on a narrow temperature
        window, extrapolate both hotter and colder. A classic case where
        tree baselines flat-line and the formula wins by structure.

    shockley_diode
        Current through a p-n junction.
        ``I = I_s * (exp(V / V_T) - 1)``. Noisy Gaussian sampling on the
        training range; extrapolation doubles the voltage.

    radioactive_decay
        Radon-222 decay series (N_0 exp(-t/tau) with Poisson-like noise
        and known half-life 3.8235 d). Train on 30 days, extrapolate to
        180 days. This is the cleanest "rediscover the law" narrative.

Kepler's third law ``T = a^{1.5}`` and Planck's spectral radiance
``u = x^3 / (exp(x) - 1)`` were attempted as additional cases but do
not fit inside the depth-2 atlas (pure power-law / rational-exp
structure). They are discussed in ``docs/theory.md`` as honest
limitations rather than included as failed cases here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from ..gam import EMLGAM
from ..train import TrainConfig


@dataclass
class DiscoveryResult:
    case: str
    r2_train: float
    r2_extrap: float
    time_s: float
    recovered_formula: str
    expected_formula: str
    baseline_r2_extrap: float


def arrhenius_data(
    n_train: int = 128, n_test: int = 512, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Arrhenius rate ``k = A * exp(-Ea / (R T))``.

    ``A = 1e4``, ``Ea / R = 4500 K``. Training temperatures in
    [350, 450] K; extrapolation in [250, 700] K. The feature passed to
    the regressor is ``1/T`` so the target becomes a pure ``exp`` in
    that variable.
    """
    rng = np.random.default_rng(seed)
    A, Ea_R = 1.0e4, 4500.0
    T_tr = rng.uniform(350.0, 450.0, n_train)
    y_tr_clean = A * np.exp(-Ea_R / T_tr)
    y_tr = y_tr_clean * (1.0 + 0.02 * rng.standard_normal(n_train))
    T_te = rng.uniform(250.0, 700.0, n_test)
    y_te = A * np.exp(-Ea_R / T_te)
    return (1.0 / T_tr).reshape(-1, 1), y_tr, (1.0 / T_te).reshape(-1, 1), y_te


def shockley_data(
    n_train: int = 128, n_test: int = 512, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shockley diode equation ``I = I_s * (exp(V/V_T) - 1)``.

    Using ``I_s = 1e-12 A``, ``V_T = 0.02585 V`` (room temperature). We
    train on ``V in [0.4, 0.55] V`` and extrapolate to ``V in [0.3, 0.8]
    V`` (exp response is 10^6 larger at the extrapolation extreme).
    """
    rng = np.random.default_rng(seed)
    Is, VT = 1.0e-12, 0.02585
    V_tr = rng.uniform(0.40, 0.55, n_train)
    I_tr_clean = Is * (np.exp(V_tr / VT) - 1.0)
    # Heteroscedastic noise: 2 per cent of mean.
    I_tr = I_tr_clean * (1.0 + 0.02 * rng.standard_normal(n_train))
    V_te = rng.uniform(0.30, 0.80, n_test)
    I_te = Is * (np.exp(V_te / VT) - 1.0)
    return V_tr.reshape(-1, 1), I_tr, V_te.reshape(-1, 1), I_te


def radon_decay_data(
    n_train: int = 64, n_test: int = 256,
    half_life_days: float = 3.8235, n0: float = 100.0, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Radon-222 decay ``N(t) = N_0 * exp(-t / tau)``.

    Training: 30 days (~8 half lives). Extrapolation: 180 days. Noise is
    a Gaussian approximation to Poisson shot noise on the count rate.
    """
    rng = np.random.default_rng(seed)
    tau = half_life_days / np.log(2.0)
    t_tr = rng.uniform(0.0, 30.0, n_train)
    mean_tr = n0 * np.exp(-t_tr / tau)
    noise_tr = np.sqrt(np.maximum(mean_tr, 1.0)) * rng.standard_normal(n_train)
    y_tr = np.maximum(mean_tr + noise_tr, 0.0)
    t_te = np.linspace(0.0, 180.0, n_test)
    y_te = n0 * np.exp(-t_te / tau)
    return t_tr.reshape(-1, 1), y_tr, t_te.reshape(-1, 1), y_te


def _fit_and_report(
    case: str, expected: str,
    X_tr, y_tr, X_te, y_te,
    train_desc: str, extrap_desc: str,
    verbose: bool,
) -> DiscoveryResult:
    lin = LinearRegression().fit(X_tr, y_tr)
    r2_lin = r2_score(y_te, lin.predict(X_te))

    t0 = time.perf_counter()
    model = EMLGAM(
        n_features=X_tr.shape[1], univariate_depth=2,
        standardize=True, scale_normalize=True,
    )
    cfg = TrainConfig(
        n_epochs=2000, lr=5e-2,
        extrap_penalty_weight=0.01, extrap_max_std=50.0,
    )
    model.fit(
        X_tr, y_tr, cfg=cfg,
        warm_start=True, try_offsets=True, robust=True,
    )
    elapsed = time.perf_counter() - t0
    r2_tr = r2_score(y_tr, model.predict(X_tr))
    r2_te = r2_score(y_te, model.predict(X_te))
    formula = str(model.total_formula(simplify=True))[:200]

    if verbose:
        print(f"\n=== {case} ===")
        print(f"  Target:              {expected}")
        print(f"  Train:               {train_desc}")
        print(f"  Extrap:              {extrap_desc}")
        print(f"  Linear R2_extrap:    {r2_lin:+.4f}")
        print(f"  EML-GA^2M R2_train:  {r2_tr:+.4f}")
        print(f"  EML-GA^2M R2_extrap: {r2_te:+.4f}")
        print(f"  Formula:             {formula}")
    return DiscoveryResult(
        case=case, r2_train=r2_tr, r2_extrap=r2_te,
        time_s=elapsed, recovered_formula=formula,
        expected_formula=expected, baseline_r2_extrap=r2_lin,
    )


def run_arrhenius(verbose: bool = True) -> DiscoveryResult:
    X_tr, y_tr, X_te, y_te = arrhenius_data()
    return _fit_and_report(
        "arrhenius_rate", "k = A * exp(-Ea / (R T))",
        X_tr, y_tr, X_te, y_te,
        "T in [350, 450] K  (1/T in [0.0022, 0.0029])",
        "T in [250, 700] K  (1/T in [0.0014, 0.0040])",
        verbose,
    )


def run_shockley(verbose: bool = True) -> DiscoveryResult:
    X_tr, y_tr, X_te, y_te = shockley_data()
    return _fit_and_report(
        "shockley_diode", "I = I_s * (exp(V / V_T) - 1)",
        X_tr, y_tr, X_te, y_te,
        "V in [0.40, 0.55] V",
        "V in [0.30, 0.80] V  (5-order-of-magnitude extrapolation in I)",
        verbose,
    )


def run_radon(verbose: bool = True) -> DiscoveryResult:
    X_tr, y_tr, X_te, y_te = radon_decay_data()
    return _fit_and_report(
        "radon_decay", "N(t) = N_0 * exp(-t / tau)",
        X_tr, y_tr, X_te, y_te,
        "t in [0, 30] days  (~8 half-lives)",
        "t in [0, 180] days  (~47 half-lives)",
        verbose,
    )


def run_all(verbose: bool = True) -> list[DiscoveryResult]:
    results = [
        run_arrhenius(verbose=verbose),
        run_shockley(verbose=verbose),
        run_radon(verbose=verbose),
    ]
    if verbose:
        print()
        print("=" * 60)
        print("  Summary (physics discovery)")
        print("=" * 60)
        print(f"  {'case':18s} {'R2 extrap (linear)':>20s}  {'R2 extrap (EML)':>18s}")
        for r in results:
            print(
                f"  {r.case:18s} {r.baseline_r2_extrap:+20.4f}  "
                f"{r.r2_extrap:+18.4f}"
            )
    return results


if __name__ == "__main__":
    run_all()
