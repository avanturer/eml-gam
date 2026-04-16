"""AEES vs gradient-descent recovery on the landscape benchmark.

This module directly compares three warm-start strategies on the
landscape targets of ``benchmarks/landscape.py``:

1. Random-init gradient descent (the baseline that fails at depth >= 3).
2. Hand-coded atlas warm-start (the production default in ``EMLGAM``).
3. **AEES** — exhaustive enumeration of the entire snap space at the
   requested depth, ranking candidates by OLS R^2.

The headline table is the recovery rate per depth for each strategy.
AEES is exhaustive at depth <= 3 (feasible — 186k configurations for
univariate depth 3), so its recovery is analytic, not stochastic: if
the target snap lies in the enumerated space, AEES finds it.

Run with:
    python -m eml_gam.benchmarks.aees_vs_gradient
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from ..atlas_expansion import aees_search, aees_search_unbranched
from ..eml_tree import EMLTree
from ..primitives import default_atlas, warm_start_tree
from ..train import TrainConfig, train_tree
from ..utils import DTYPE
from .landscape import target_snap_elog_iterated_exp


@dataclass
class RecoveryRow:
    depth: int
    strategy: str
    n_trials: int
    n_recovered: int
    success_rate: float
    median_r2: float
    time_s_per_trial: float


def _target_data(depth: int, n: int = 256, seed: int = 0):
    rng = np.random.default_rng(seed)
    if depth == 1:
        x_np = rng.uniform(0.2, 3.0, size=n)
    elif depth == 2:
        x_np = rng.uniform(-1.5, 1.5, size=n)
    elif depth == 3:
        x_np = rng.uniform(-2.0, 0.5, size=n)
    else:
        x_np = rng.uniform(-3.0, -0.3, size=n)
    x = torch.tensor(x_np, dtype=DTYPE).unsqueeze(1)
    target_snap = target_snap_elog_iterated_exp(depth)
    tree = EMLTree(depth=depth, n_inputs=1, use_input_affine=False)
    tree.set_snap_config(target_snap)
    with torch.no_grad():
        y = tree(x)
    return x, y


def _random_init_gd(depth: int, n_trials: int = 20,
                    n_epochs: int = 1500) -> RecoveryRow:
    x, y = _target_data(depth)
    n_ok = 0
    r2s: list[float] = []
    t0 = time.perf_counter()
    for trial in range(n_trials):
        torch.manual_seed(trial * 31 + 7)
        tree = EMLTree(depth=depth, n_inputs=1, use_input_affine=False)
        cfg = TrainConfig(
            n_epochs=n_epochs, lr=5e-2,
            entropy_weight=1e-3, warmup_frac=0.7, hardening_frac=0.3,
        )
        try:
            info = train_tree(tree, x, y, cfg=cfg)
            final_mse = info["final_mse"]
        except Exception:
            final_mse = float("inf")
        # Convert to R^2 for uniform reporting.
        var_y = float(torch.var(y).item())
        r2 = 1.0 - final_mse / max(var_y, 1e-20) if np.isfinite(final_mse) else -np.inf
        r2s.append(r2)
        if np.isfinite(r2) and r2 >= 0.999:
            n_ok += 1
    elapsed = (time.perf_counter() - t0) / max(n_trials, 1)
    return RecoveryRow(
        depth=depth, strategy="random_init_gd",
        n_trials=n_trials, n_recovered=n_ok,
        success_rate=n_ok / n_trials,
        median_r2=float(np.median(r2s)) if r2s else float("-inf"),
        time_s_per_trial=elapsed,
    )


def _atlas_warm_start(depth: int, n_trials: int = 20) -> RecoveryRow:
    if depth not in (1, 2):
        # Hand-coded atlas only exists for depth 1 and 2 univariate.
        return RecoveryRow(
            depth=depth, strategy="atlas_warm",
            n_trials=n_trials, n_recovered=0, success_rate=0.0,
            median_r2=float("-inf"), time_s_per_trial=0.0,
        )
    x, y = _target_data(depth)
    atlas = default_atlas(depth=depth, n_inputs=1)
    n_ok = 0
    r2s: list[float] = []
    t0 = time.perf_counter()
    for trial in range(n_trials):
        torch.manual_seed(trial * 31 + 7)
        tree = EMLTree(depth=depth, n_inputs=1, use_input_affine=False)
        try:
            warm_start_tree(tree, atlas, x, y, try_offsets=False)
            with torch.no_grad():
                pred = tree(x).detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            m, v = pred.mean(), pred.var()
            cov = np.mean((pred - m) * (y_np - y_np.mean()))
            beta = cov / max(v, 1e-30)
            alpha = y_np.mean() - beta * m
            resid = y_np - (alpha + beta * pred)
            ss_tot = float(np.sum((y_np - y_np.mean()) ** 2))
            r2 = 1.0 - float(np.sum(resid ** 2)) / max(ss_tot, 1e-20)
        except Exception:
            r2 = float("-inf")
        r2s.append(r2)
        if np.isfinite(r2) and r2 >= 0.999:
            n_ok += 1
    elapsed = (time.perf_counter() - t0) / max(n_trials, 1)
    return RecoveryRow(
        depth=depth, strategy="atlas_warm",
        n_trials=n_trials, n_recovered=n_ok,
        success_rate=n_ok / n_trials,
        median_r2=float(np.median(r2s)) if r2s else float("-inf"),
        time_s_per_trial=elapsed,
    )


def _aees(depth: int) -> RecoveryRow:
    x, y = _target_data(depth)
    t0 = time.perf_counter()
    top = aees_search(x, y, depth=depth, n_inputs=1, top_k=1, verbose=False)
    elapsed = time.perf_counter() - t0
    best_r2 = top[0].r2 if top else -np.inf
    recovered = np.isfinite(best_r2) and best_r2 >= 0.999
    return RecoveryRow(
        depth=depth, strategy="aees",
        n_trials=1, n_recovered=int(recovered),
        success_rate=1.0 if recovered else 0.0,
        median_r2=best_r2,
        time_s_per_trial=elapsed,
    )


def _aees_unbranched(depth: int) -> RecoveryRow:
    """Unbranched AEES — scales to depth 8+ on unbranched targets."""
    x, y = _target_data(depth)
    t0 = time.perf_counter()
    top = aees_search_unbranched(x, y, depth=depth, top_k=1, verbose=False)
    elapsed = time.perf_counter() - t0
    best_r2 = top[0].r2 if top else -np.inf
    recovered = np.isfinite(best_r2) and best_r2 >= 0.999
    return RecoveryRow(
        depth=depth, strategy="aees_unbranched",
        n_trials=1, n_recovered=int(recovered),
        success_rate=1.0 if recovered else 0.0,
        median_r2=best_r2,
        time_s_per_trial=elapsed,
    )


def run_comparison(
    depths: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    n_gd_trials: int = 20,
    verbose: bool = True,
    include_full_aees_up_to: int = 3,
) -> list[RecoveryRow]:
    """Compare all four recovery strategies on the landscape target family.

    ``include_full_aees_up_to`` caps the depth at which the exhaustive
    (branched) AEES is run — beyond depth 3 the state space blows up
    combinatorially (~10^9 at depth 4) and only the unbranched variant
    remains tractable.
    """
    rows: list[RecoveryRow] = []
    if verbose:
        print(f"{'=' * 76}")
        print("  AEES vs gradient-descent vs hand-coded atlas")
        print(f"{'=' * 76}")
        print(
            f"  {'depth':>5}  {'strategy':<18}  "
            f"{'success':>8}  {'median R2':>12}  {'time/trial':>12}"
        )
    for depth in depths:
        new_rows: list[RecoveryRow] = []
        new_rows.append(_random_init_gd(depth, n_trials=n_gd_trials))
        new_rows.append(_atlas_warm_start(depth, n_trials=min(n_gd_trials, 5)))
        if depth <= include_full_aees_up_to:
            new_rows.append(_aees(depth))
        new_rows.append(_aees_unbranched(depth))
        rows.extend(new_rows)
        if verbose:
            for r in new_rows:
                print(
                    f"  {r.depth:>5}  {r.strategy:<18}  "
                    f"{r.n_recovered}/{r.n_trials:<5}  "
                    f"{r.median_r2:+12.4f}  {r.time_s_per_trial:10.2f}s"
                )
    return rows


if __name__ == "__main__":
    import json
    rows = run_comparison()
    with open("aees_vs_gradient.json", "w") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)
    print("  saved -> aees_vs_gradient.json")
