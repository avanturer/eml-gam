"""Numerical verification that no ψ-expression over ``{1, x}`` is
identically zero on R, at depth ≤ 3.

Strategy
--------
A real-analytic function f on R satisfies f ≡ 0 iff f vanishes on any
open interval, iff all Taylor coefficients at any single point are zero.
It therefore suffices to check f(x₀) ≠ 0 at a single generic point
``x₀``: if f(x₀) ≠ 0 then f is not identically zero; contrapositively,
f ≡ 0 forces f(x₀) = 0 for any x₀.

We pick two algebraically independent test points (x₁, x₂) =
(1/2, 7/5) and evaluate every ψ-expression over ``{1, x}`` of depth
at most 3 at each test point with ``mpmath`` at 150 decimal digits.
If every tree has |f(x_i)| > 10^{-100} at at least one test point,
no tree in the enumerated subset can be identically zero as a real
function.

This is an empirical companion to the theoretical reduction in
``docs/sheffer_analysis.md §4.2``. It strengthens the evidence for
Conjecture 7 by extending the numerical enumeration from the
constant-only case (ψ-closure of ``{1}``) to the function case
(ψ-closure of ``{1, x}``).

Complexity
----------
Number of trees of depth ≤ d over ``{1, x}``:
    N(0) = 2, N(d) = N(d-1) + N(d-1)² → 2, 6, 42, 1806, 3 263 442
At depth 3 the count is 1806, which with two ψ evaluations per tree at
150 digits completes in under a minute.
"""

from __future__ import annotations

import json
import time

from mpmath import asinh, mp, mpf, sinh


def psi(a, b):
    return sinh(a) - asinh(b)


def enumerate_over_1x(
    max_depth: int, test_points: tuple[mpf, ...],
) -> list[dict]:
    """Return a list of per-tree records containing depth, repr, and
    the vector of evaluations at each test point."""
    # Initial leaf values: (repr, (value_at_x₁, value_at_x₂, ...)).
    T = [
        (0, "1", tuple(mpf(1) for _ in test_points)),
        (0, "x", tuple(x for x in test_points)),
    ]
    for d in range(1, max_depth + 1):
        new_trees = []
        for _, ra, va in T:
            for _, rb, vb in T:
                new_vals = tuple(psi(a, b) for a, b in zip(va, vb))
                new_trees.append((d, f"psi({ra},{rb})", new_vals))
        T.extend(new_trees)
    return [
        {"depth": d, "repr": r, "values": vs}
        for d, r, vs in T
    ]


def stream_depth_plus_one(
    trees_at_depth_leq: list[dict],
) -> tuple[mpf, str]:
    """Stream-enumerate depth-(d+1) trees without storing them, tracking
    the minimum max|f| across test points.

    Returns (infimum of max|f| observed, repr of argmin tree)."""
    min_max_abs = mpf("+inf")
    argmin_repr = ""
    n_pairs = len(trees_at_depth_leq) ** 2
    check_every = max(n_pairs // 20, 1)
    count = 0
    for left in trees_at_depth_leq:
        for right in trees_at_depth_leq:
            new_vals = tuple(
                psi(a, b)
                for a, b in zip(left["values"], right["values"])
            )
            mx = max(abs(v) for v in new_vals)
            if mx < min_max_abs:
                min_max_abs = mx
                argmin_repr = f"psi({left['repr']},{right['repr']})"
            count += 1
            if count % check_every == 0:
                print(f"    {count:,}/{n_pairs:,}")
    return min_max_abs, argmin_repr


def main(
    max_depth: int = 3, precision_digits: int = 150,
    extend_to_depth_4: bool = False,
) -> dict:
    mp.dps = precision_digits
    test_points = (mpf(1) / mpf(2), mpf(7) / mpf(5))
    print("Subproblem (A) over {1, x} — numerical check")
    print(f"  max_depth = {max_depth}")
    print(f"  precision = {precision_digits} digits")
    print(f"  test points: x1 = {float(test_points[0])}, "
          f"x2 = {float(test_points[1])}")

    t0 = time.perf_counter()
    trees = enumerate_over_1x(max_depth, test_points)
    elapsed = time.perf_counter() - t0
    print(f"  enumerated {len(trees)} trees in {elapsed:.1f}s")

    # For each tree, take min |f(x_i)| across test points. If this
    # minimum is zero across all test points, the tree COULD be
    # identically zero; if it is > threshold, tree is certainly not
    # identically zero.
    zero_threshold = mpf(10) ** -100

    suspicious: list[dict] = []
    min_per_tree_overall = mpf("inf")
    count_numerically_zero = 0

    for tr in trees:
        vals = tr["values"]
        # If ALL test-point evaluations are below threshold, flag.
        max_abs = max(abs(v) for v in vals)
        min_per_tree_overall = min(
            min_per_tree_overall, max_abs,
        )
        if max_abs < zero_threshold:
            count_numerically_zero += 1
            suspicious.append({
                "depth": tr["depth"],
                "repr": tr["repr"][:200],
                "max_abs": mp.nstr(max_abs, 20),
            })

    # Also record the overall minimum across trees of max|f| across
    # test points (this is the "worst-case" distance-from-zero).
    best_candidate = min(
        trees, key=lambda tr: max(abs(v) for v in tr["values"]),
    )
    best_max_abs = max(abs(v) for v in best_candidate["values"])

    result = {
        "max_depth": max_depth,
        "precision_digits": precision_digits,
        "n_trees": len(trees),
        "test_points": [float(x) for x in test_points],
        "zero_threshold": "1e-100",
        "count_numerically_zero": count_numerically_zero,
        "infimum_max_abs_f_across_test_points": mp.nstr(
            best_max_abs, 20,
        ),
        "attained_by": best_candidate["repr"][:200],
        "attained_by_depth": best_candidate["depth"],
        "elapsed_s": elapsed,
        "suspicious_trees": suspicious,
    }

    print()
    print("Result")
    print(f"  trees whose max|f(x_i)| < 10^{-100}: "
          f"{count_numerically_zero}")
    print(f"  smallest max|f| observed: "
          f"{result['infimum_max_abs_f_across_test_points']}")
    print(f"  attained by: {result['attained_by'][:120]}")
    print(f"  depth: {result['attained_by_depth']}")

    if extend_to_depth_4 and max_depth == 3:
        print()
        print(f"Streaming depth-4 enumeration "
              f"({len(trees) ** 2:,} pairs)...")
        t_stream = time.perf_counter()
        min_abs_d4, argmin_d4 = stream_depth_plus_one(trees)
        stream_elapsed = time.perf_counter() - t_stream
        print(f"  depth-4 streaming completed in {stream_elapsed:.0f}s")
        print(f"  smallest max|f| at depth 4: {mp.nstr(min_abs_d4, 20)}")
        print(f"  attained by: {argmin_d4[:120]}")
        result["depth_4_min_max_abs"] = mp.nstr(min_abs_d4, 20)
        result["depth_4_argmin"] = argmin_d4[:250]
        result["depth_4_stream_elapsed_s"] = stream_elapsed
    return result


if __name__ == "__main__":
    import sys
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    extend = "--extend" in sys.argv
    res = main(max_depth=d, extend_to_depth_4=extend)
    with open("subproblem_a_over_1x.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nsaved subproblem_a_over_1x.json")
