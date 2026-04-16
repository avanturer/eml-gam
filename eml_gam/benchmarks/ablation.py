"""Ablation study for the warm-start, affine, scale-normalisation and
hold-out validation components of EMLGAM.

Each variant adds one capability on top of the previous one:

  V0 baseline     : random init, no affine, no normalization.
  V1 +warm_start  : atlas-based warm-start (in-sample OLS, no holdout).
  V2 +affine      : learnable per-tree input scale and offset.
  V3 +scale_norm  : conditional feature-scale normalisation.
  V4 +holdout     : two-sided holdout validation with adaptive simplicity
                    tolerance (the full EMLGAM defaults).

Extrapolation R² is measured on six representative datasets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
from sklearn.metrics import r2_score

from ..gam import EMLGAM
from ..train import TrainConfig
from .scientific import (
    arrhenius,
    cobb_douglas,
    competitive_inhibition,
    exponential_decay,
    michaelis_menten,
    power_law,
)


@dataclass
class AblationResult:
    dataset: str
    variant: str
    r2_interp: float
    r2_extrap: float


def _run(
    ds,
    *,
    warm_start: bool,
    use_input_affine: bool,
    use_holdout: bool,
    scale_normalize: bool,
    standardize: bool,
    univariate_depth: int = 2,
    bivariate_depth: int = 2,
    interaction_pairs=None,
    n_epochs: int = 1200,
) -> tuple[float, float]:
    torch.manual_seed(0)
    np.random.seed(0)
    if interaction_pairs is None and ds.X_train.shape[1] >= 2:
        interaction_pairs = list(combinations(range(ds.X_train.shape[1]), 2))
    model = EMLGAM(
        n_features=ds.X_train.shape[1],
        univariate_depth=univariate_depth,
        bivariate_depth=bivariate_depth,
        feature_names=ds.feature_names,
        interaction_pairs=interaction_pairs,
        use_input_affine=use_input_affine,
        standardize=standardize,
        scale_normalize=scale_normalize,
    )
    cfg = TrainConfig(n_epochs=n_epochs, lr=5e-2, entropy_weight=1e-3)
    model.fit(
        ds.X_train, ds.y_train, cfg=cfg,
        warm_start=warm_start, use_holdout=use_holdout,
        verbose=False,
    )
    y_in = model.predict(ds.X_test_interp)
    y_ex = model.predict(ds.X_test_extrap)
    return (
        r2_score(ds.y_test_interp, y_in),
        r2_score(ds.y_test_extrap, y_ex),
    )


def run_ablation(verbose: bool = True) -> list[AblationResult]:
    datasets = [
        exponential_decay(n_train=512),
        michaelis_menten(n_train=512),
        cobb_douglas(n_train=512),
        power_law(n_train=512),
        competitive_inhibition(n_train=512),
        arrhenius(n_train=512),
    ]
    variants = [
        ("V0_baseline", dict(
            warm_start=False, use_input_affine=False,
            use_holdout=False, scale_normalize=False, standardize=False,
        )),
        ("V1_+warm", dict(
            warm_start=True, use_input_affine=False,
            use_holdout=False, scale_normalize=False, standardize=False,
        )),
        ("V2_+affine", dict(
            warm_start=True, use_input_affine=True,
            use_holdout=False, scale_normalize=False, standardize=False,
        )),
        ("V3_+scale_norm", dict(
            warm_start=True, use_input_affine=True,
            use_holdout=False, scale_normalize=True, standardize=True,
        )),
        ("V4_+holdout", dict(
            warm_start=True, use_input_affine=True,
            use_holdout=True, scale_normalize=True, standardize=True,
        )),
    ]

    results: list[AblationResult] = []
    for ds in datasets:
        if verbose:
            print(f"\n=== {ds.name} ({ds.domain}) ===")
        for vname, vcfg in variants:
            t0 = time.perf_counter()
            try:
                r2_in, r2_ex = _run(ds, **vcfg)
            except Exception as e:
                if verbose:
                    print(f"  {vname:16s} FAILED: {e}")
                continue
            results.append(
                AblationResult(
                    dataset=ds.name,
                    variant=vname,
                    r2_interp=r2_in,
                    r2_extrap=r2_ex,
                )
            )
            if verbose:
                print(
                    f"  {vname:16s}  R2_in={r2_in:+7.3f}  R2_ex={r2_ex:+7.3f}"
                    f"  time={time.perf_counter() - t0:5.1f}s"
                )
    return results


if __name__ == "__main__":
    results = run_ablation()
    print("\n" + "=" * 72)
    print(
        f"{'dataset':25s} {'variant':17s} "
        f"{'R2 interp':>10s} {'R2 extrap':>10s}"
    )
    print("-" * 72)
    for r in results:
        print(
            f"{r.dataset:25s} {r.variant:17s} "
            f"{r.r2_interp:+10.3f} {r.r2_extrap:+10.3f}"
        )
