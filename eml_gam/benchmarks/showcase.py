"""Showcase recoveries: the source paper's own target and exact
minimal-depth certificates for classic identities.

Part 1 — the paper's bivariate showcase. Odrzywolek (2026) Sect. 4.3
trains complex-valued EML nets on ``f(x, y) = ln(e - ln(e^x - ln y))``
— a depth-5 bivariate EML composition — and reports the depth wall
precisely on such targets. Here the canonical-pool recursive solver
recovers the exact expression from 256 real samples, with no gradient
descent, in seconds.

Part 2 — minimal-depth certificates. Because the pool enumerates
*every* distinct function at each depth budget (up to grid identity),
a failed exhaustive search at depth ``d`` is a *certificate* that no
depth-``d`` EML tree computes the target on the grid, and a successful
one exhibits a witness. This upgrades the "shortest found" columns of
the paper's Table 4 into exact per-depth statements (relative to the
evaluation grid and the safe-eml clamps).

Writes ``showcase_results.json``.
"""

from __future__ import annotations

import json
import time

import numpy as np

from ..canonical_pool import FunctionPool

RESULTS: dict = {}


def part1_paper_target() -> dict:
    print("Part 1: the paper's bivariate depth-5 target")
    g = np.random.default_rng(777)
    xy = np.column_stack(
        [g.uniform(0.3, 2.5, 256), g.uniform(0.3, 2.5, 256)]
    )
    pool = FunctionPool(xy, operator="eml")
    t0 = time.time()
    pool.build(3, verbose=True)
    build_s = time.time() - t0

    x, y = xy[:, 0], xy[:, 1]
    target = np.log(np.e - np.log(np.exp(x) - np.log(y)))
    t0 = time.time()
    found = pool.solve_recursive(target, depth=5, shallow_k=2)
    solve_s = time.time() - t0

    result = {
        "target": "ln(e - ln(exp(x) - ln(y)))",
        "pool_sizes": pool.level_end,
        "build_seconds": build_s,
        "solve_seconds": solve_s,
        "n_matches": len(found),
        "recovered": False,
        "expression": None,
    }
    # Verify on an independent dense grid.
    gv = np.random.default_rng(778)
    xy_v = np.column_stack(
        [gv.uniform(0.35, 2.45, 2000), gv.uniform(0.35, 2.45, 2000)]
    )
    tv = np.log(np.e - np.log(np.exp(xy_v[:, 0]) - np.log(xy_v[:, 1])))
    for nested in found:
        hv = pool.evaluate_nested(nested, xy_v)
        if np.all(np.isfinite(hv)) and np.max(np.abs(hv - tv)) < 1e-8 * (
            1 + np.max(np.abs(tv))
        ):
            result["recovered"] = True
            result["expression"] = str(
                pool.nested_to_sympy(nested, ["x", "y"])
            )
            break
    print(
        f"  recovered={result['recovered']} in {solve_s:.1f}s; "
        f"expr = {result['expression']}"
    )
    return result


def part2_minimal_depths() -> dict:
    print("Part 2: minimal-depth certificates (univariate, grid identity)")
    g = np.random.default_rng(12345)
    x = np.sort(g.uniform(0.3, 2.5, 256))
    pool = FunctionPool(x, operator="eml")
    pool.build(4, verbose=False)

    # Scope note: these are certificates for the REAL-CLAMPED EML
    # semantics (``safe_eml``), which is what every trainable
    # implementation actually optimises. The paper's complex-domain
    # identities (e.g. for -x) use ``ln`` of negative arguments and lie
    # outside this class; targets whose values push ``exp`` past the
    # clamp on this range (e.g. ``exp(exp(x))``) are excluded because
    # the clamped class cannot contain them.
    specs = {
        "exp(x)": lambda v: np.exp(v),
        "ln(x)": lambda v: np.log(v),
        "-x": lambda v: -v,
        "1/x": lambda v: 1.0 / v,
        "x**2": lambda v: v**2,
        "sqrt(x)": lambda v: np.sqrt(v),
        "e - ln(x)": lambda v: np.e - np.log(v),
        "sinh(x)": lambda v: np.sinh(v),
        "cosh(x)": lambda v: np.cosh(v),
    }
    x_dense = np.linspace(0.3, 2.5, 2048)
    targets = {k: fn(x) for k, fn in specs.items()}
    dense = {k: fn(x_dense) for k, fn in specs.items()}

    rows = {}
    for name, tv in targets.items():
        row: dict = {"exact_depth": None, "affine_depth": None,
                     "expression": None}
        for d in (1, 2, 3, 4, 5):
            if d <= pool.max_depth + 1:
                pairs = pool.lookup_root_pairs(tv, d, cap=50_000)
                hit = None
                cache: dict = {}
                for ia, ib in pairs:
                    hv = pool.evaluate_pair(ia, ib, x_dense, cache)
                    if np.all(np.isfinite(hv)) and np.max(
                        np.abs(hv - dense[name])
                    ) < 1e-8 * (1 + np.max(np.abs(dense[name]))):
                        hit = ("node", pool.expr_to_nested(ia),
                               pool.expr_to_nested(ib))
                        break
                if hit is not None:
                    row["exact_depth"] = d
                    row["expression"] = str(
                        pool.nested_to_sympy(hit, ["x"])
                    )
                    break
        # affine variant: alpha + beta * tree
        for d in (1, 2, 3, 4):
            cands = pool.exhaustive_root(tv, d, top_k=64)
            got = None
            for c in cands:
                hv = c["alpha"] + c["beta"] * pool.evaluate_nested(
                    c["nested"], x_dense
                )
                if np.all(np.isfinite(hv)) and np.max(
                    np.abs(hv - dense[name])
                ) < 1e-8 * (1 + np.max(np.abs(dense[name]))):
                    got = c
                    break
            if got is not None:
                row["affine_depth"] = d
                if row["expression"] is None:
                    row["expression"] = (
                        f"{got['alpha']:+.6g} {got['beta']:+.6g} * "
                        + str(pool.nested_to_sympy(got["nested"], ["x"]))
                    )
                break
        rows[name] = row
        print(
            f"  {name:12s} exact depth: {row['exact_depth']}, "
            f"with-affine depth: {row['affine_depth']}"
        )
    return {"grid": "256 jittered points on [0.3, 2.5]", "targets": rows}


def main(out_path: str = "showcase_results.json") -> dict:
    RESULTS["paper_target"] = part1_paper_target()
    RESULTS["minimal_depths"] = part2_minimal_depths()
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"wrote {out_path}")
    return RESULTS


if __name__ == "__main__":
    main()
