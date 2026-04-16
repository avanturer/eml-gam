"""Depth-5 extension of the Subproblem (A) numerical check.

At depth 5 the tree count is ``1806^2 ~ 3.3M`` which is too much to
store in memory at 150-digit precision. We instead stream the
enumeration: hold only the T_4 values in memory and scan through the
3.3M T_5 products, tracking only the running minimum.

Time budget is about 30 minutes on a single CPU core at 150-digit
precision. The script also tracks the top-5 smallest values so we
can verify the T_self prediction is the strict optimum.
"""

from __future__ import annotations

import heapq
import json
import time

from mpmath import asinh, mp, mpf, sinh


def psi(a, b):
    return sinh(a) - asinh(b)


def enumerate_trees_depth_le_4(precision_digits: int = 150):
    mp.dps = precision_digits
    T = [(0, "1", mpf(1))]
    for d in range(1, 5):
        new_trees = []
        for da, ra, va in T:
            for db, rb, vb in T:
                new_trees.append((d, f"psi({ra},{rb})", psi(va, vb)))
        T.extend(new_trees)
    return T


def main(precision_digits: int = 150, top_k: int = 5) -> dict:
    print("Running Subproblem (A) numerical check at depth 5")
    print(f"  precision = {precision_digits} decimal digits")

    t0 = time.perf_counter()
    T4 = enumerate_trees_depth_le_4(precision_digits)
    print(f"  enumerated {len(T4)} trees up to depth 4 in "
          f"{time.perf_counter() - t0:.1f}s")

    # Track top-k smallest |values| at depth 5 via a max-heap of size k.
    # (Python heapq is a min-heap; we store (-abs_value, ...) to invert.)
    top: list[tuple[float, str]] = []

    t0 = time.perf_counter()
    n_pairs = len(T4) ** 2
    check_every = n_pairs // 20
    count = 0
    for da, ra, va in T4:
        for db, rb, vb in T4:
            vv = psi(va, vb)
            abs_v = abs(vv)
            entry = (-float(abs_v), f"psi({ra},{rb})")
            if len(top) < top_k:
                heapq.heappush(top, entry)
            elif entry > top[0]:
                heapq.heapreplace(top, entry)
            count += 1
            if count % check_every == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"    {count:,}/{n_pairs:,} "
                    f"({100 * count / n_pairs:.1f}%)  "
                    f"elapsed {elapsed:.0f}s"
                )

    # Extract top-k sorted by ascending |value|.
    top_sorted = sorted(top, key=lambda e: -e[0])

    result = {
        "max_depth": 5,
        "precision_digits": precision_digits,
        "n_pairs_at_depth_5": n_pairs,
        "top_k": top_k,
        "top": [
            {"abs_value_float": -e[0], "tree_repr": e[1][:160]}
            for e in top_sorted
        ],
        "elapsed_s": time.perf_counter() - t0,
    }

    print()
    print(f"Top-{top_k} smallest |value| at depth 5:")
    for entry in result["top"]:
        print(f"  |v| ~ {entry['abs_value_float']:.4e}  "
              f"tree: {entry['tree_repr'][:120]}")

    with open("subproblem_a_depth5.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nsaved subproblem_a_depth5.json")

    return result


if __name__ == "__main__":
    main()
