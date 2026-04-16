"""Synthetic scientific regression targets with known ground-truth formulas.

Each dataset is generated from a textbook physical, chemical or economic
law. Training and test samples are drawn from different slices of the input
space so that a positive test R² requires extrapolation rather than
interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import sympy as sp


@dataclass
class ScientificDataset:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test_interp: np.ndarray
    y_test_interp: np.ndarray
    X_test_extrap: np.ndarray
    y_test_extrap: np.ndarray
    feature_names: list[str]
    true_formula: sp.Expr
    description: str
    domain: str


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def exponential_decay(
    n_train: int = 512, noise: float = 0.02, seed: int = 2
) -> ScientificDataset:
    """Radioactive / first-order decay: y = y0 * exp(-lambda * t).

    y0 = 5, lambda = 0.4. Train t in [0, 3], extrapolate to t in [3, 8].
    """
    rng = _rng(seed)
    y0, lam = 5.0, 0.4
    fn = lambda t: y0 * np.exp(-lam * t)

    t_tr = rng.uniform(0, 3, n_train)
    y_tr = fn(t_tr) + rng.normal(0, noise * y0, n_train)
    t_in = rng.uniform(0, 3, 256); y_in = fn(t_in)
    t_ex = rng.uniform(3, 8, 256); y_ex = fn(t_ex)

    tsym = sp.Symbol("t")
    return ScientificDataset(
        name="exp_decay",
        X_train=t_tr.reshape(-1, 1), y_train=y_tr,
        X_test_interp=t_in.reshape(-1, 1), y_test_interp=y_in,
        X_test_extrap=t_ex.reshape(-1, 1), y_test_extrap=y_ex,
        feature_names=["t"],
        true_formula=y0 * sp.exp(-lam * tsym),
        description="exponential decay y = y0 * exp(-lambda * t)",
        domain="physics",
    )


def arrhenius(
    n_train: int = 512, noise: float = 0.05, seed: int = 1
) -> ScientificDataset:
    """Arrhenius rate law: k = A * exp(-Ea / (R * T)).

    Reparameterised with x = 1/T so the dependence becomes a pure
    ``exp(-Ea * x)``. A = 1e6, Ea / R = 3000. Train T in [350, 450] K,
    extrapolate to T in [300, 600] K.
    """
    rng = _rng(seed)
    A, Ea_over_R = 1e6, 3000.0
    fn = lambda inv_T: A * np.exp(-Ea_over_R * inv_T)

    def _gen(tmin, tmax, n):
        T = rng.uniform(tmin, tmax, n)
        return 1.0 / T, fn(1.0 / T)

    x_tr, y_tr_clean = _gen(350, 450, n_train)
    y_tr = y_tr_clean + rng.normal(0, noise * y_tr_clean.std(), n_train)
    x_in, y_in = _gen(350, 450, 256)
    x_ex, y_ex = _gen(300, 600, 256)

    xsym = sp.Symbol("inv_T")
    return ScientificDataset(
        name="arrhenius",
        X_train=x_tr.reshape(-1, 1), y_train=y_tr,
        X_test_interp=x_in.reshape(-1, 1), y_test_interp=y_in,
        X_test_extrap=x_ex.reshape(-1, 1), y_test_extrap=y_ex,
        feature_names=["inv_T"],
        true_formula=A * sp.exp(-Ea_over_R * xsym),
        description="Arrhenius rate k = A * exp(-Ea / (R T)), x = 1/T",
        domain="physics",
    )


def michaelis_menten(
    n_train: int = 512, noise: float = 0.02, seed: int = 0
) -> ScientificDataset:
    """Michaelis-Menten kinetics: v = Vmax * [S] / (Km + [S]).

    Vmax = 10, Km = 2. Train [S] in [0.5, 3], extrapolate to [S] in [3, 10].
    """
    rng = _rng(seed)
    Vmax, Km = 10.0, 2.0
    fn = lambda S: Vmax * S / (Km + S)

    S_tr = rng.uniform(0.5, 3.0, n_train)
    y_tr = fn(S_tr) + rng.normal(0, noise * Vmax, n_train)
    S_in = rng.uniform(0.5, 3.0, 256)
    y_in = fn(S_in)
    S_ex = rng.uniform(3.0, 10.0, 256)
    y_ex = fn(S_ex)

    s = sp.Symbol("S")
    return ScientificDataset(
        name="michaelis_menten",
        X_train=S_tr.reshape(-1, 1), y_train=y_tr,
        X_test_interp=S_in.reshape(-1, 1), y_test_interp=y_in,
        X_test_extrap=S_ex.reshape(-1, 1), y_test_extrap=y_ex,
        feature_names=["S"],
        true_formula=Vmax * s / (Km + s),
        description="Michaelis-Menten v = Vmax * S / (Km + S)",
        domain="pharmacology",
    )


def cobb_douglas(
    n_train: int = 512, noise: float = 0.02, seed: int = 3
) -> ScientificDataset:
    """Log-linear Cobb-Douglas production: log Y = log A + a log L + b log K.

    A = 2, a = 0.7, b = 0.3. Train L, K in [1, 5], extrapolate to [5, 15].
    """
    rng = _rng(seed)
    A, alpha, beta = 2.0, 0.7, 0.3
    fn = lambda L, K: np.log(A) + alpha * np.log(L) + beta * np.log(K)

    def _gen(lo, hi, n):
        L = rng.uniform(lo, hi, n); K = rng.uniform(lo, hi, n)
        return np.stack([L, K], axis=1), fn(L, K)

    X_tr, y_tr_clean = _gen(1.0, 5.0, n_train)
    y_tr = y_tr_clean + rng.normal(
        0, noise * np.abs(y_tr_clean).mean(), n_train
    )
    X_in, y_in = _gen(1.0, 5.0, 256)
    X_ex, y_ex = _gen(5.0, 15.0, 256)

    L, K = sp.symbols("L K")
    return ScientificDataset(
        name="cobb_douglas",
        X_train=X_tr, y_train=y_tr,
        X_test_interp=X_in, y_test_interp=y_in,
        X_test_extrap=X_ex, y_test_extrap=y_ex,
        feature_names=["L", "K"],
        true_formula=sp.log(A) + alpha * sp.log(L) + beta * sp.log(K),
        description="log Cobb-Douglas: log Y = log A + a log L + b log K",
        domain="economics",
    )


def logistic_growth(
    n_train: int = 512, noise: float = 0.01, seed: int = 4
) -> ScientificDataset:
    """Logistic growth: y = K / (1 + exp(-r * (t - t0))).

    K = 1, r = 1.5, t0 = 0. Train t in [-2, 2], extrapolate to [-5, 5].
    """
    rng = _rng(seed)
    K, r, t0 = 1.0, 1.5, 0.0
    fn = lambda t: K / (1.0 + np.exp(-r * (t - t0)))

    t_tr = rng.uniform(-2, 2, n_train)
    y_tr = fn(t_tr) + rng.normal(0, noise, n_train)
    t_in = rng.uniform(-2, 2, 256); y_in = fn(t_in)
    t_ex = rng.uniform(-5, 5, 256); y_ex = fn(t_ex)

    tsym = sp.Symbol("t")
    return ScientificDataset(
        name="logistic_growth",
        X_train=t_tr.reshape(-1, 1), y_train=y_tr,
        X_test_interp=t_in.reshape(-1, 1), y_test_interp=y_in,
        X_test_extrap=t_ex.reshape(-1, 1), y_test_extrap=y_ex,
        feature_names=["t"],
        true_formula=K / (1 + sp.exp(-r * (tsym - t0))),
        description="logistic growth y = K / (1 + exp(-r (t - t0)))",
        domain="biology",
    )


def power_law(
    n_train: int = 512, noise: float = 0.02, seed: int = 5
) -> ScientificDataset:
    """Power law in log domain: log y = log a + b log x.

    a = 3, b = 1.5. Train x in [1, 5], extrapolate to [5, 20].
    """
    rng = _rng(seed)
    a, b = 3.0, 1.5
    fn = lambda x: np.log(a) + b * np.log(x)

    x_tr = rng.uniform(1, 5, n_train); y_tr_clean = fn(x_tr)
    y_tr = y_tr_clean + rng.normal(
        0, noise * np.abs(y_tr_clean).mean(), n_train
    )
    x_in = rng.uniform(1, 5, 256); y_in = fn(x_in)
    x_ex = rng.uniform(5, 20, 256); y_ex = fn(x_ex)

    xsym = sp.Symbol("x")
    return ScientificDataset(
        name="power_law_log",
        X_train=x_tr.reshape(-1, 1), y_train=y_tr,
        X_test_interp=x_in.reshape(-1, 1), y_test_interp=y_in,
        X_test_extrap=x_ex.reshape(-1, 1), y_test_extrap=y_ex,
        feature_names=["x"],
        true_formula=sp.log(a) + b * sp.log(xsym),
        description="log power law: log y = log a + b log x",
        domain="physics",
    )


def competitive_inhibition(
    n_train: int = 512, noise: float = 0.02, seed: int = 6
) -> ScientificDataset:
    """Competitive inhibition enzyme kinetics: truly bivariate target.

    v = Vmax * S / (Km * (1 + I / Ki) + S), with Vmax = 10, Km = 2, Ki = 1.5.
    This expression is not additively separable in (S, I) and therefore
    exercises the bivariate component of the model.
    Train S, I in [0.5, 3], extrapolate to S in [3, 10] and I in [0.5, 5].
    """
    rng = _rng(seed)
    Vmax, Km, Ki = 10.0, 2.0, 1.5
    fn = lambda S, I: Vmax * S / (Km * (1.0 + I / Ki) + S)

    def _gen(s_range, i_range, n):
        S = rng.uniform(*s_range, n)
        I = rng.uniform(*i_range, n)
        return np.stack([S, I], axis=1), fn(S, I)

    X_tr, y_tr_clean = _gen((0.5, 3.0), (0.5, 3.0), n_train)
    y_tr = y_tr_clean + rng.normal(0, noise * Vmax, n_train)
    X_in, y_in = _gen((0.5, 3.0), (0.5, 3.0), 256)
    X_ex, y_ex = _gen((3.0, 10.0), (0.5, 5.0), 256)

    s, i = sp.symbols("S I")
    return ScientificDataset(
        name="competitive_inhibition",
        X_train=X_tr, y_train=y_tr,
        X_test_interp=X_in, y_test_interp=y_in,
        X_test_extrap=X_ex, y_test_extrap=y_ex,
        feature_names=["S", "I"],
        true_formula=Vmax * s / (Km * (1 + i / Ki) + s),
        description="competitive inhibition v = Vmax * S / (Km (1 + I/Ki) + S)",
        domain="pharmacology",
    )


def combined_gas_law(
    n_train: int = 512, noise: float = 0.02, seed: int = 7
) -> ScientificDataset:
    """Combined gas law: P = n R T / V (bivariate, multiplicative).

    n R = 0.5. Train T in [250, 350] K and V in [1, 5] L,
    extrapolate to T in [350, 500] K and V in [0.5, 8] L.
    """
    rng = _rng(seed)
    nR = 0.5
    fn = lambda T, V: nR * T / V

    def _gen(t_range, v_range, n):
        T = rng.uniform(*t_range, n)
        V = rng.uniform(*v_range, n)
        return np.stack([T, V], axis=1), fn(T, V)

    X_tr, y_tr_clean = _gen((250, 350), (1.0, 5.0), n_train)
    y_tr = y_tr_clean + rng.normal(
        0, noise * np.abs(y_tr_clean).mean(), n_train
    )
    X_in, y_in = _gen((250, 350), (1.0, 5.0), 256)
    X_ex, y_ex = _gen((350, 500), (0.5, 8.0), 256)

    T, V = sp.symbols("T V")
    return ScientificDataset(
        name="combined_gas_law",
        X_train=X_tr, y_train=y_tr,
        X_test_interp=X_in, y_test_interp=y_in,
        X_test_extrap=X_ex, y_test_extrap=y_ex,
        feature_names=["T", "V"],
        true_formula=nR * T / V,
        description="combined gas law P = n R T / V",
        domain="physics",
    )


ALL_SCIENTIFIC: list[Callable[..., ScientificDataset]] = [
    exponential_decay,
    arrhenius,
    michaelis_menten,
    cobb_douglas,
    logistic_growth,
    power_law,
    competitive_inhibition,
    combined_gas_law,
]


def all_scientific(n_train: int = 512) -> list[ScientificDataset]:
    return [fn(n_train=n_train) for fn in ALL_SCIENTIFIC]
