"""Numerical check of Subproblem (A) from docs/sheffer_analysis.md.

Subproblem (A) asks whether there exists a finite ψ-expression over the
terminal set ``{1}`` that equals exactly zero. A positive answer would
give a constructive proof that ``psi`` is a Sheffer operator. A
strong-enough negative numerical result — every value bounded away
from zero by a margin incompatible with float round-off — provides
empirical support for Conjecture 7.

We enumerate every ψ-tree of depth at most 4 (1 806 values, counting
duplicates), evaluate each with ``mpmath`` at 150 decimal digits, and
report:

* the minimum ``|value|`` over all trees,
* which tree attains the minimum, and
* how the minimum compares with the prediction from the ``T_self``
  iteration ``t_{k+1} = psi(t_k, t_k)`` starting from ``t_0 = 1``, which
  contracts super-cubically and gives the candidate smallest value at
  each depth.

150 decimal digits is far more precision than needed to distinguish
non-zero from zero in the depth-4 regime (T_self iteration produces
values around ``3^{-(3^4 - 1)/2} ~ 2e-20`` at depth 4). If the minimum
matches the ``T_self`` prediction and no other composition beats it, we
conclude that the ψ-closure of ``{1}`` has no element equal to zero up
to depth 4, and that the closest-to-zero trajectory is exactly the
super-contraction tower described in Proposition 6.

The script writes ``subproblem_a_numerical.json`` with the full
result table.
"""

from __future__ import annotations

import json
import time

from mpmath import asinh, mp, mpf, sinh


def psi(a, b):
    return sinh(a) - asinh(b)


def enumerate_psi_closure(max_depth: int, precision_digits: int = 150):
    """Return a list of tuples ``(depth, expr_description, value)``.

    Every ψ-tree of depth ``<= max_depth`` is included (with duplicates
    because distinct trees may evaluate to the same real number).
    """
    mp.dps = precision_digits
    # Representation: each tree is a (depth, repr, mpf value) triple.
    T = [(0, "1", mpf(1))]
    for d in range(1, max_depth + 1):
        new_trees = []
        for da, ra, va in T:
            for db, rb, vb in T:
                vv = psi(va, vb)
                new_trees.append((d, f"psi({ra},{rb})", vv))
        T.extend(new_trees)
    return T


def main(max_depth: int = 4, precision_digits: int = 150) -> dict:
    print("Running numerical Subproblem (A) check")
    print(f"  max_depth = {max_depth}")
    print(f"  precision = {precision_digits} decimal digits")

    t0 = time.perf_counter()
    trees = enumerate_psi_closure(max_depth, precision_digits)
    elapsed = time.perf_counter() - t0
    print(f"  enumerated {len(trees)} trees in {elapsed:.1f}s")

    # T_self iteration baseline: t_0 = 1, t_{k+1} = psi(t_k, t_k).
    t_self = [mpf(1)]
    for _ in range(max_depth):
        t_self.append(psi(t_self[-1], t_self[-1]))

    # Find minimum |value|.
    best = min(trees, key=lambda tr: abs(tr[2]))
    min_abs = abs(best[2])

    # Compare to T_self predictions.
    t_self_depth_k = t_self[max_depth]

    # Numerical-zero threshold. With 150-digit precision,
    # treat abs < 10^{-100} as "numerically zero" (though for this
    # experiment we expect much larger values).
    numerical_zero_threshold = mpf(10) ** -100
    is_numerically_zero = min_abs < numerical_zero_threshold

    result = {
        "max_depth": max_depth,
        "precision_digits": precision_digits,
        "n_trees": len(trees),
        "min_abs_value": mp.nstr(min_abs, 20),
        "best_tree_repr": best[1],
        "best_tree_depth": best[0],
        "t_self_at_max_depth": mp.nstr(abs(t_self_depth_k), 20),
        "is_numerically_zero_at_10_to_minus_100": bool(is_numerically_zero),
        "elapsed_s": elapsed,
        "t_self_orbit": [mp.nstr(abs(v), 15) for v in t_self],
    }

    print()
    print("Result")
    print(f"  min |value| over {len(trees)} trees: "
          f"{result['min_abs_value']}")
    print(f"  achieved by tree: {result['best_tree_repr'][:100]}...")
    print(f"  tree depth: {result['best_tree_depth']}")
    print(f"  T_self prediction at depth {max_depth}: "
          f"{result['t_self_at_max_depth']}")
    print(f"  ratio min / T_self: "
          f"{float(min_abs / abs(t_self_depth_k)):.4e}")
    print(f"  numerically zero? {is_numerically_zero}")
    print()
    print("T_self orbit (absolute values):")
    for k, v in enumerate(t_self):
        print(f"  depth {k}: |t_{k}| = {mp.nstr(abs(v), 15)}")

    return result


if __name__ == "__main__":
    import sys
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    res = main(max_depth=depth)
    with open("subproblem_a_numerical.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nsaved subproblem_a_numerical.json")
