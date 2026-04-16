"""Constant-variant check of the Pure-sinh Non-representability Lemma.

Complements ``symbolic_lemma_check.py`` which handles the function-level
case over ``F_0 = psi``-closure of ``{x}``. Here we check the
**constant-level** claim that

    sinh(sinh(c)) is not in F_c  for any non-zero c in F_c,

where ``F_c = psi``-closure of ``{1}``.  This together with the
Lemma-on-F_0 closes the full Pure-sinh Non-representability Lemma
because the Reduction splits into two cases depending on whether
``g(0) = 0`` or not.

Strategy — mirror ``subproblem_a_numerical.py``: enumerate ``F_c`` up
to a given depth at 150 decimal digits, compute the orbit
``{sinh(sinh(c)) : c in F_c non-zero}``, and check that none of these
values coincides (numerically, with generous slack) with any element
of ``F_c``.

A positive numerical result (every pair separated by much more than
round-off) is empirical evidence for the constant variant of the
Lemma analogous to Conjecture 7.
"""
from __future__ import annotations

import json
import time

from mpmath import asinh, mp, mpf, sinh


def psi(a, b):
    return sinh(a) - asinh(b)


def enumerate_Fc(max_depth: int) -> list[tuple[int, str, "mpf"]]:
    T = [(0, "1", mpf(1))]
    for d in range(1, max_depth + 1):
        new = []
        for _, ra, va in T:
            for _, rb, vb in T:
                new.append((d, f"psi({ra},{rb})", psi(va, vb)))
        T.extend(new)
    return T


def main(max_depth: int = 4, precision_digits: int = 150) -> dict:
    mp.dps = precision_digits
    print(f"[constant_variant_check] max_depth={max_depth}, prec={precision_digits}d")

    t0 = time.perf_counter()
    Fc = enumerate_Fc(max_depth)
    print(f"F_c up to depth {max_depth}: {len(Fc)} trees in {time.perf_counter()-t0:.1f}s")

    # Deduplicate values with tolerance
    tolerance = mpf(10) ** -(precision_digits - 20)
    distinct_vals: list[tuple[str, "mpf"]] = []
    for _, r, v in Fc:
        is_new = True
        for _, v_ref in distinct_vals:
            if abs(v - v_ref) < tolerance:
                is_new = False
                break
        if is_new:
            distinct_vals.append((r, v))
    print(f"distinct values (tolerance {tolerance}): {len(distinct_vals)}")

    # Compute targets sinh(sinh(c)) for each non-zero c
    print("Computing sinh(sinh(c)) for each c...")
    targets = []
    for r, v in distinct_vals:
        if abs(v) < tolerance:  # skip zero
            continue
        sv = sinh(v)
        ssv = sinh(sv)
        targets.append((r, v, ssv))

    # For each target, check minimum distance to any c in F_c
    print("Checking target -> F_c distance...")
    min_dists = []
    worst = None
    worst_pair = None
    for r, v, ssv in targets:
        min_d = None
        closest = None
        for r_ref, v_ref in distinct_vals:
            d = abs(ssv - v_ref)
            if min_d is None or d < min_d:
                min_d = d
                closest = (r_ref, v_ref)
        min_dists.append(min_d)
        if worst is None or min_d < worst:
            worst = min_d
            worst_pair = (r, v, ssv, closest)

    print(
        f"\nworst-case (minimum) distance from sinh(sinh(c)) to F_c: "
        f"{mp.nstr(worst, 20)}"
    )
    if worst_pair:
        r_t, v_t, ssv_t, (r_c, v_c) = worst_pair
        print(f"  target c (repr): {r_t[:80]}")
        print(f"  target sinh(sinh(c)) = {mp.nstr(ssv_t, 15)}")
        print(f"  closest c' (repr): {r_c[:80]}")
        print(f"  c' value = {mp.nstr(v_c, 15)}")

    # Distance distribution
    min_dists_sorted = sorted(min_dists)
    print(f"\ndistance quantiles over {len(min_dists)} targets:")
    for q, lab in [(0, "min"), (0.05, "5%"), (0.5, "median"), (0.95, "95%"), (-1, "max")]:
        idx = min(int(q * (len(min_dists) - 1)), len(min_dists) - 1) if q >= 0 else -1
        print(f"  {lab}: {mp.nstr(min_dists_sorted[idx], 15)}")

    result = {
        "max_depth": max_depth,
        "precision_digits": precision_digits,
        "n_trees_Fc": len(Fc),
        "n_distinct_values": len(distinct_vals),
        "n_targets": len(targets),
        "worst_case_min_distance": mp.nstr(worst, 30),
        "target_min_distances_percentiles": {
            "min": mp.nstr(min_dists_sorted[0], 15),
            "median": mp.nstr(min_dists_sorted[len(min_dists)//2], 15),
            "max": mp.nstr(min_dists_sorted[-1], 15),
        },
    }
    return result


if __name__ == "__main__":
    import sys
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    r = main(max_depth=d)
    with open("constant_variant_check.json", "w") as f:
        json.dump(r, f, indent=2)
    print("\nSaved constant_variant_check.json")
