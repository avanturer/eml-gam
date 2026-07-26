"""The depth-wall benchmark: canonical-pool search vs gradient descent
vs uniform syntactic sampling on random branched EML targets.

Protocol
--------
* Data grid: 320 sorted uniform-random points on [0.3, 2.5] (seeded).
  All methods see exactly the same ``(x, y)``.
* Targets: random *live* trees of exact depth ``d`` (root-to-leaf spine
  forced), rejection-sampled to be **depth-critical**: a target is kept
  only if no depth-``(d-1)`` candidate from the complete pool passes
  dense in-range verification. This excludes the (large) fraction of
  random deep trees whose function collapses to a shallower one —
  recovering those is not evidence of breaking the wall.
* Recovery criterion (all methods): the returned expression matches the
  target on a dense 1111-point in-range grid to relative tolerance
  1e-6 (max-abs error, scaled by max |y|). Grid-R^2 alone is NOT
  accepted — R^2 = 1.0 ties on the training grid are common at depth
  >= 4 and are exactly the failure mode a symbolic-recovery claim must
  exclude.
* Methods:
    - ``pool``    : lookup_root_pairs (exact inversion) with streamed
                    verification, falling back to the complete
                    OLS sweep (exhaustive_root) at depth <= 4.
    - ``gd``      : multi-start Adam with temperature annealing (the
                    reference approach of Odrzywolek 2026 and of this
                    repository's earlier landscape study).
    - ``uniform`` : uniform syntactic snap sampling with OLS readout,
                    budget-matched in candidate count.

Outputs ``depth_wall_results.json``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

from ..canonical_pool import (
    FunctionPool,
    sample_live_nested,
    syntactic_live_count,
    syntactic_snap_count,
)

GRID_SEED = 12345
N_GRID = 320
X_LO, X_HI = 0.3, 2.5
N_DENSE = 1111
N_PROBE = 64
REL_TOL = 1e-6


def make_grid() -> np.ndarray:
    g = np.random.default_rng(GRID_SEED)
    return np.sort(g.uniform(X_LO, X_HI, N_GRID))


@dataclass
class Verifier:
    """Dense in-range verification against a ground-truth function."""

    pool: FunctionPool
    nested_true: tuple

    def __post_init__(self):
        self.x_probe = np.linspace(X_LO + 0.05, X_HI - 0.05, N_PROBE)
        self.x_dense = np.linspace(X_LO, X_HI, N_DENSE)
        self.y_probe = self.pool.evaluate_nested(self.nested_true, self.x_probe)
        self.y_dense = self.pool.evaluate_nested(self.nested_true, self.x_dense)
        self.thr_probe = REL_TOL * (1.0 + np.max(np.abs(self.y_probe)))
        self.thr_dense = REL_TOL * (1.0 + np.max(np.abs(self.y_dense)))
        self._cache_probe: dict = {}
        self._cache_dense: dict = {}

    def check_pair(self, ia: int, ib: int) -> bool:
        hp = self.pool.evaluate_pair(ia, ib, self.x_probe, self._cache_probe)
        if not np.all(np.isfinite(hp)):
            return False
        if np.max(np.abs(hp - self.y_probe)) >= self.thr_probe:
            return False
        h = self.pool.evaluate_pair(ia, ib, self.x_dense, self._cache_dense)
        return bool(
            np.all(np.isfinite(h))
            and np.max(np.abs(h - self.y_dense)) < self.thr_dense
        )

    def check_values(self, h_probe, h_dense, alpha=0.0, beta=1.0) -> bool:
        hp = alpha + beta * h_probe
        if not (
            np.all(np.isfinite(hp))
            and np.max(np.abs(hp - self.y_probe)) < self.thr_probe
        ):
            return False
        h = alpha + beta * h_dense
        return bool(
            np.all(np.isfinite(h))
            and np.max(np.abs(h - self.y_dense)) < self.thr_dense
        )

    def check_nested(self, nested, alpha=0.0, beta=1.0) -> bool:
        return self.check_values(
            self.pool.evaluate_nested(nested, self.x_probe),
            self.pool.evaluate_nested(nested, self.x_dense),
            alpha,
            beta,
        )


def is_depth_critical(pool: FunctionPool, nested: tuple, depth: int) -> bool:
    """True if no depth-(d-1) candidate passes dense verification."""
    y = pool.evaluate_nested(nested, pool.x[:, 0])
    ver = Verifier(pool, nested)
    d_prev = depth - 1
    if d_prev <= pool.max_depth:
        # Complete OLS sweep at the previous depth (top candidates).
        for cand in pool.exhaustive_root(y, d_prev, top_k=64):
            if ver.check_nested(cand["nested"], cand["alpha"], cand["beta"]):
                return False
        return True
    # Previous depth beyond the exhaustive range: exact lookup, streamed.
    for ia, ib in pool.lookup_root_pairs(y, d_prev, cap=50_000):
        if ver.check_pair(ia, ib):
            return False
    return True


def sample_critical_targets(
    pool: FunctionPool,
    depth: int,
    n_targets: int,
    seed: int,
    max_attempts_factor: int = 200,
) -> tuple[list[tuple], dict]:
    """Sample depth-critical targets; also report collapse statistics."""
    rng = np.random.default_rng(seed)
    kept: list[tuple] = []
    n_sampled = n_degenerate = n_shallow = 0
    max_attempts = max_attempts_factor * n_targets
    while len(kept) < n_targets and n_sampled < max_attempts:
        n_sampled += 1
        nested = sample_live_nested(depth, pool.n_inputs, rng)
        y = pool.evaluate_nested(nested, pool.x[:, 0])
        if not (np.all(np.isfinite(y)) and np.std(y) >= 1e-8):
            n_degenerate += 1
            continue
        if not is_depth_critical(pool, nested, depth):
            n_shallow += 1
            continue
        kept.append(nested)
    stats = {
        "n_sampled": n_sampled,
        "n_degenerate": n_degenerate,
        "n_collapsed_to_shallower": n_shallow,
        "n_critical": len(kept),
        "critical_fraction": len(kept) / max(n_sampled, 1),
    }
    return kept, stats


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


def run_pool_method(
    pool: FunctionPool, nested: tuple, depth: int, time_budget_s: float = 300.0
) -> dict:
    y = pool.evaluate_nested(nested, pool.x[:, 0])
    ver = Verifier(pool, nested)
    t0 = time.time()
    n_checked = 0
    # Exact inversion lookup, streamed verification.
    if depth <= pool.max_depth + 1:
        pairs = pool.lookup_root_pairs(y, depth, cap=300_000, per_run_cap=64)
        for ia, ib in pairs:
            n_checked += 1
            if ver.check_pair(ia, ib):
                return {
                    "recovered": True,
                    "seconds": time.time() - t0,
                    "n_verified_candidates": n_checked,
                    "via": "lookup",
                }
            if time.time() - t0 > time_budget_s:
                break
    # Complete OLS sweep (also covers alpha/beta-scaled targets).
    if depth <= pool.max_depth + 1 and pool.level_end[
        min(depth - 1, pool.max_depth)
    ] <= 30_000:
        for cand in pool.exhaustive_root(y, depth, top_k=512):
            n_checked += 1
            if ver.check_nested(cand["nested"], cand["alpha"], cand["beta"]):
                return {
                    "recovered": True,
                    "seconds": time.time() - t0,
                    "n_verified_candidates": n_checked,
                    "via": "exhaustive",
                }
    return {
        "recovered": False,
        "seconds": time.time() - t0,
        "n_verified_candidates": n_checked,
        "via": None,
    }


def run_functional_coverage(
    pool: FunctionPool, nested: tuple, solve_depth: int, time_budget_s: float = 300.0
) -> dict:
    """For targets *deeper* than the pool supports: how often does the
    depth-``solve_depth`` lookup still recover the function (because the
    deep tree's children collapse functionally into the pool)?"""
    return run_pool_method(pool, nested, solve_depth, time_budget_s)


def run_gd_method(
    x: np.ndarray,
    nested: tuple,
    pool: FunctionPool,
    depth: int,
    n_restarts: int = 10,
    n_epochs: int = 1200,
) -> dict:
    import torch

    from ..eml_tree import EMLTree
    from ..train import TrainConfig, train_tree

    y = pool.evaluate_nested(nested, x)
    ver = Verifier(pool, nested)
    xt = torch.tensor(x[:, None], dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    y_var = float(np.var(y))
    cfg = TrainConfig(n_epochs=n_epochs, lr=5e-2, verbose=False)
    t0 = time.time()
    mse_success = False
    for _ in range(n_restarts):
        tree = EMLTree(depth=depth, n_inputs=1, use_input_affine=False)
        try:
            train_tree(tree, xt, yt, cfg)
        except Exception:
            continue
        tree.snap()
        with torch.no_grad():
            pred = tree(xt).cpu().numpy()
        if not np.all(np.isfinite(pred)):
            continue
        mse = float(np.mean((pred - y) ** 2))
        if mse < 1e-3 * max(y_var, 1e-12):
            mse_success = True
            # dense verification of the snapped formula
            with torch.no_grad():
                h_probe = tree(
                    torch.tensor(ver.x_probe[:, None], dtype=torch.float64)
                ).cpu().numpy()
                h_dense = tree(
                    torch.tensor(ver.x_dense[:, None], dtype=torch.float64)
                ).cpu().numpy()
            if ver.check_values(h_probe, h_dense):
                return {
                    "recovered": True,
                    "mse_success": True,
                    "seconds": time.time() - t0,
                }
    return {
        "recovered": False,
        "mse_success": mse_success,
        "seconds": time.time() - t0,
    }


def run_uniform_method(
    pool: FunctionPool, nested: tuple, depth: int, budget: int = 200_000
) -> dict:
    """Uniform syntactic snap sampling with OLS scoring + verification."""
    import random as pyrandom

    import torch

    from ..neural_beam import SnapSchema, sample_random_snap, tree_from_flat

    x = pool.x[:, 0]
    y = pool.evaluate_nested(nested, x)
    ver = Verifier(pool, nested)
    schema = SnapSchema(depth=depth, n_inputs=1)
    rng = pyrandom.Random(0)
    xt = torch.tensor(x[:, None], dtype=torch.float64)
    y_mean = y.mean()
    ss_tot = float(np.sum((y - y_mean) ** 2)) or 1.0
    t0 = time.time()
    for i in range(budget):
        flat = sample_random_snap(schema, rng)
        tree = tree_from_flat(schema, flat)
        with torch.no_grad():
            pred = tree(xt).cpu().numpy()
        if not np.all(np.isfinite(pred)) or np.std(pred) < 1e-12:
            continue
        cov = np.mean((pred - pred.mean()) * (y - y_mean))
        beta = cov / max(pred.var(), 1e-30)
        alpha = y_mean - beta * pred.mean()
        r2 = 1.0 - np.sum((y - alpha - beta * pred) ** 2) / ss_tot
        if r2 > 1.0 - 1e-9:
            with torch.no_grad():
                h_probe = tree(
                    torch.tensor(ver.x_probe[:, None], dtype=torch.float64)
                ).cpu().numpy()
                h_dense = tree(
                    torch.tensor(ver.x_dense[:, None], dtype=torch.float64)
                ).cpu().numpy()
            if ver.check_values(h_probe, h_dense, alpha, beta):
                return {
                    "recovered": True,
                    "seconds": time.time() - t0,
                    "n_sampled": i + 1,
                }
    return {"recovered": False, "seconds": time.time() - t0, "n_sampled": budget}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(
    depths=(3, 4, 5),
    coverage_depths=(6, 7, 8),
    n_targets: int = 40,
    n_targets_gd: int = 12,
    n_targets_uniform: int = 12,
    uniform_budget: int = 100_000,
    out_path: str = "depth_wall_results.json",
) -> dict:
    x = make_grid()
    print(f"building canonical pool on {N_GRID}-point grid ...")
    pool = FunctionPool(x, operator="eml")
    t0 = time.time()
    pool.build(4, verbose=True)
    build_s = time.time() - t0

    results: dict = {
        "protocol": {
            "grid_seed": GRID_SEED,
            "n_grid": N_GRID,
            "x_range": [X_LO, X_HI],
            "rel_tol": REL_TOL,
            "n_targets": n_targets,
            "pool_build_seconds": build_s,
            "pool_sizes": pool.level_end,
            "pool_build_stats": pool.build_stats,
        },
        "depths": {},
        "coverage": {},
    }

    for depth in depths:
        print(f"\n=== depth {depth} ===")
        targets, tstats = sample_critical_targets(
            pool, depth, n_targets, seed=1000 + depth
        )
        print(
            f"  targets: {tstats['n_critical']} critical / "
            f"{tstats['n_sampled']} sampled "
            f"({tstats['n_collapsed_to_shallower']} collapsed, "
            f"{tstats['n_degenerate']} degenerate)"
        )
        row: dict = {
            "target_stats": tstats,
            "syntactic_snaps": syntactic_snap_count(depth, 1),
            "syntactic_live": syntactic_live_count(depth, 1),
        }

        pool_res = [run_pool_method(pool, t, depth) for t in targets]
        row["pool"] = {
            "n": len(pool_res),
            "recovered": sum(r["recovered"] for r in pool_res),
            "mean_seconds": float(np.mean([r["seconds"] for r in pool_res])),
            "max_seconds": float(np.max([r["seconds"] for r in pool_res])),
        }
        print(f"  pool: {row['pool']['recovered']}/{row['pool']['n']} "
              f"({row['pool']['mean_seconds']:.2f}s avg)")

        gd_res = [
            run_gd_method(x, t, pool, depth) for t in targets[:n_targets_gd]
        ]
        row["gd"] = {
            "n": len(gd_res),
            "recovered": sum(r["recovered"] for r in gd_res),
            "mse_success": sum(r["mse_success"] for r in gd_res),
            "mean_seconds": float(np.mean([r["seconds"] for r in gd_res])),
        }
        print(f"  gd: {row['gd']['recovered']}/{row['gd']['n']} verified, "
              f"{row['gd']['mse_success']} mse-hits "
              f"({row['gd']['mean_seconds']:.0f}s avg)")

        uni_res = [
            run_uniform_method(pool, t, depth, budget=uniform_budget)
            for t in targets[:n_targets_uniform]
        ]
        row["uniform"] = {
            "n": len(uni_res),
            "recovered": sum(r["recovered"] for r in uni_res),
            "budget": uniform_budget,
            "mean_seconds": float(np.mean([r["seconds"] for r in uni_res])),
        }
        print(f"  uniform: {row['uniform']['recovered']}/{row['uniform']['n']}")

        results["depths"][str(depth)] = row
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    # Functional-coverage sweep beyond the enumerated depth: how many
    # random depth-d targets remain solvable because their children
    # collapse into the depth-4 pool?
    for depth in coverage_depths:
        print(f"\n=== functional coverage at depth {depth} ===")
        rng = np.random.default_rng(2000 + depth)
        n_cov = 30
        n_deg = n_solved = n_shallow_like = 0
        secs = []
        sampled = 0
        while sampled - n_deg < n_cov and sampled < 100 * n_cov:
            sampled += 1
            nested = sample_live_nested(depth, 1, rng)
            y = pool.evaluate_nested(nested, pool.x[:, 0])
            if not (np.all(np.isfinite(y)) and np.std(y) >= 1e-8):
                n_deg += 1
                continue
            res = run_functional_coverage(pool, nested, pool.max_depth + 1)
            if res["recovered"]:
                n_solved += 1
                if res["via"] == "exhaustive":
                    n_shallow_like += 1
            secs.append(res["seconds"])
        n_valid = sampled - n_deg
        results["coverage"][str(depth)] = {
            "n_sampled": sampled,
            "n_degenerate": n_deg,
            "n_valid": n_valid,
            "n_recovered_via_depth5_solver": n_solved,
            "coverage_fraction": n_solved / max(n_valid, 1),
            "mean_seconds": float(np.mean(secs)) if secs else None,
        }
        print(
            f"  {n_solved}/{n_valid} random depth-{depth} targets recovered "
            f"by the depth-5 solver"
        )
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nwrote {out_path}")
    return results


if __name__ == "__main__":
    main()
