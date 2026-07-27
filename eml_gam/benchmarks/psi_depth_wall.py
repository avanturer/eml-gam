"""Operator-generality check: the canonical-pool method on ψ trees.

The pool machinery never looks inside the operator beyond three
callables (node evaluation and the two side-canonicalisations), so the
entire depth-wall pipeline — enumeration, dedup, exhaustive sweeps,
root inversion (via ``a = arsinh(y + arsinh(b))`` and
``b = sinh(sinh(a) − y)``), verification — applies verbatim to
``ψ(x, y) = sinh(x) − arsinh(y)``. This benchmark repeats the
depth-critical recovery protocol for ψ targets at depths 3–5.

Because ψ has no clamps, its function space collapses far less
(zero degeneracy at level 3), so this is also the harder enumeration
case: the level-4 pool is 1.43M distinct functions versus EML's 0.61M.

Writes ``psi_depth_wall_results.json``.
"""

from __future__ import annotations

import json
import time

import numpy as np

from ..canonical_pool import FunctionPool, sample_live_nested

GRID_SEED = 12345
N_GRID = 256
X_LO, X_HI = 0.3, 2.5
N_DENSE = 1111
REL_TOL = 1e-6


def main(n_targets: int = 20, out_path: str = "psi_depth_wall_results.json"):
    g = np.random.default_rng(GRID_SEED)
    x = np.sort(g.uniform(X_LO, X_HI, N_GRID))
    pool = FunctionPool(x, operator="psi")
    t0 = time.time()
    pool.build(4, verbose=True)
    pool.compact()
    build_s = time.time() - t0

    x_probe = np.linspace(X_LO + 0.05, X_HI - 0.05, 64)
    x_dense = np.linspace(X_LO, X_HI, N_DENSE)

    def verified(nested_true, pairs) -> bool:
        y_probe = pool.evaluate_nested(nested_true, x_probe)
        y_dense = pool.evaluate_nested(nested_true, x_dense)
        thr_p = REL_TOL * (1 + np.max(np.abs(y_probe)))
        thr_d = REL_TOL * (1 + np.max(np.abs(y_dense)))
        cp: dict = {}
        cd: dict = {}
        for ia, ib in pairs:
            hp = pool.evaluate_pair(ia, ib, x_probe, cp)
            if not (np.all(np.isfinite(hp))
                    and np.max(np.abs(hp - y_probe)) < thr_p):
                continue
            h = pool.evaluate_pair(ia, ib, x_dense, cd)
            if np.all(np.isfinite(h)) and np.max(np.abs(h - y_dense)) < thr_d:
                return True
        return False

    def is_critical(nested, depth) -> bool:
        y = pool.evaluate_nested(nested, x)
        if depth - 1 >= 4:
            pairs = pool.lookup_root_pairs(y, depth - 1, cap=50_000)
            return not verified(nested, pairs)
        for cand in pool.exhaustive_root(y, depth - 1, top_k=64):
            hv_p = cand["alpha"] + cand["beta"] * pool.evaluate_nested(
                cand["nested"], x_probe
            )
            y_probe = pool.evaluate_nested(nested, x_probe)
            thr_p = REL_TOL * (1 + np.max(np.abs(y_probe)))
            if not (np.all(np.isfinite(hv_p))
                    and np.max(np.abs(hv_p - y_probe)) < thr_p):
                continue
            hv = cand["alpha"] + cand["beta"] * pool.evaluate_nested(
                cand["nested"], x_dense
            )
            y_dense = pool.evaluate_nested(nested, x_dense)
            thr_d = REL_TOL * (1 + np.max(np.abs(y_dense)))
            if np.all(np.isfinite(hv)) and np.max(np.abs(hv - y_dense)) < thr_d:
                return False
        return True

    results = {"protocol": {
        "grid_seed": GRID_SEED, "n_grid": N_GRID, "rel_tol": REL_TOL,
        "n_targets": n_targets, "pool_sizes": pool.level_end,
        "build_seconds": build_s,
    }, "depths": {}}

    for depth in (3, 4, 5):
        rng = np.random.default_rng(3000 + depth)
        kept = []
        n_sampled = n_deg = n_shallow = 0
        while len(kept) < n_targets and n_sampled < 200 * n_targets:
            n_sampled += 1
            nested = sample_live_nested(depth, 1, rng)
            y = pool.evaluate_nested(nested, x)
            if not (np.all(np.isfinite(y)) and np.std(y) >= 1e-8):
                n_deg += 1
                continue
            if not is_critical(nested, depth):
                n_shallow += 1
                continue
            kept.append(nested)
        n_rec = 0
        secs = []
        for nested in kept:
            y = pool.evaluate_nested(nested, x)
            t0 = time.time()
            pairs = pool.lookup_root_pairs(y, depth, cap=300_000,
                                           per_run_cap=64)
            ok = verified(nested, pairs)
            secs.append(time.time() - t0)
            n_rec += ok
        results["depths"][str(depth)] = {
            "n_critical": len(kept), "n_sampled": n_sampled,
            "n_collapsed": n_shallow, "n_degenerate": n_deg,
            "recovered": n_rec,
            "mean_seconds": float(np.mean(secs)) if secs else None,
        }
        print(f"psi depth {depth}: {n_rec}/{len(kept)} recovered "
              f"({np.mean(secs):.1f}s avg)", flush=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    print(f"wrote {out_path}")
    return results


if __name__ == "__main__":
    main()
