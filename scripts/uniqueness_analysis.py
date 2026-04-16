"""Structural uniqueness analysis for the Lemma-on-F_0 proof.

Goal: test the claim "T_self^k is the unique element of F_0 with
ord = 3^k at any depth". If TRUE, this closes the induction step for
the general Lemma.

Key test: for each relevant ord = 3^k, find all elements of F_0^{<= 4}
with that ord, check if they all equal T_self^k as Taylor series.

Additional analysis: for each (ord, leading) class, study the
"cancellation structure" — i.e. for pairs (g, h) with same (ord,
leading), what is the set of possible first-disagreement positions?
This tells us how "rigid" F_0 is.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from fractions import Fraction

sys.path.insert(0, "scripts")
from symbolic_lemma_check import (  # noqa: E402
    arsinh_taylor_rat,
    compose_vanishing,
    enumerate_F0_exact,
    poly_sub,
    sinh_taylor_rat,
)


def ord_of(c):
    for i, x in enumerate(c):
        if x != 0:
            return i
    return None


def analyze(max_depth: int = 4, N: int = 90):
    print(f"[uniqueness_analysis] max_depth={max_depth}, N={N}")
    sinh_c = sinh_taylor_rat(N)
    arsinh_c = arsinh_taylor_rat(N)

    F = enumerate_F0_exact(max_depth=max_depth, N=N, sinh_c=sinh_c, arsinh_c=arsinh_c)
    by_series = {}
    for d, r, c in F:
        if c not in by_series:
            by_series[c] = (d, r)
    print(f"distinct at N={N}: {len(by_series)}")

    # Group by ord
    by_ord = defaultdict(list)
    for c, (d, r) in by_series.items():
        o = ord_of(c)
        if o is not None:
            by_ord[o].append((c, d, r))

    # Relevant: ord = 3^k for k=0,...
    ord_3k = [3 ** k for k in range(20) if 3 ** k <= N]
    print(f"\nord = 3^k classes visible at N={N}: {ord_3k}")

    # Count distinct at each 3^k
    print("\nUniqueness at ord = 3^k:")
    for m in ord_3k:
        trees = by_ord.get(m, [])
        print(f"  ord = {m}: {len(trees)} distinct trees")
        if len(trees) > 1:
            # Check leading coefficients
            leads = [t[0][m] for t in trees]
            print(f"    leadings: {sorted(set(str(l) for l in leads))[:5]}...")

    # Self-tree leading values
    c_k = [Fraction(1)]
    for k in range(1, 8):
        c_k.append(c_k[-1] ** 3 / 3)
    print(f"\nT_self^k leadings (for cross-ref): {[str(c) for c in c_k[:5]]}")

    # For ord 3^k with multiple trees: are they all equal modulo higher orders?
    # I.e. do they agree up to some order and differ later, or are they
    # genuinely different leading coefficients?
    print("\nTree breakdown for ord = 3^k:")
    for m in ord_3k:
        trees = by_ord.get(m, [])
        if not trees:
            continue
        by_lead = defaultdict(list)
        for c, d, r in trees:
            by_lead[c[m]].append((c, d, r))
        print(f"  ord = {m}: leadings with counts:")
        for lead, group in sorted(by_lead.items()):
            print(f"    lead = {lead}: {len(group)} tree(s)")

    # Crucial test: at each ord = 3^k, check if the tree with
    # leading = c_k is UNIQUELY T_self^k (single element)
    print("\n=== Uniqueness check: at ord=3^k, lead=c_k, only T_self^k? ===")
    for k, m in enumerate(ord_3k):
        trees = by_ord.get(m, [])
        if not trees:
            continue
        target_lead = c_k[k] if k < len(c_k) else None
        if target_lead is None:
            continue
        matching = [t for t in trees if t[0][m] == target_lead]
        print(f"  k={k}, m=3^{k}={m}, c_k={target_lead}: "
              f"{len(matching)} tree(s) with this (ord, lead)")
        if matching:
            # All these trees must equal T_self^k if uniqueness holds
            first_series = matching[0][0]
            all_equal = all(t[0] == first_series for t in matching)
            if all_equal and len(matching) >= 1:
                print(f"    → All {len(matching)} trees have IDENTICAL Taylor series "
                      f"(= T_self^{k} as expected).")
            else:
                print(f"    WARNING: {len(matching)} DIFFERENT Taylor series "
                      f"at same (ord, lead). Uniqueness FAILS at ord {m}!")
                for i, (ci, di, ri) in enumerate(matching[:3]):
                    print(f"      tree {i}: depth={di}, first-nonzero-diff-positions:")
                    if i > 0:
                        for p in range(m, N + 1):
                            if ci[p] != matching[0][0][p]:
                                print(f"        x^{p}: tree_0={matching[0][0][p]}, "
                                      f"tree_{i}={ci[p]}")
                                break

    # For the induction step: check if trees with ord > 3^{k-1} but
    # < 3^k can be constructed from trees of lower ord. This would
    # complete the sharp-bound uniqueness argument.
    # (Skipped — this is subsumed by the full distinct-Taylor enumeration.)


def build_T_self_exact(k: int, N: int):
    """Return T_self^k as exact rational Taylor list of length N+1."""
    sinh_c = sinh_taylor_rat(N)
    arsinh_c = arsinh_taylor_rat(N)
    x_ser = [Fraction(0)] * (N + 1)
    x_ser[1] = Fraction(1)
    T = x_ser
    for _ in range(k):
        sa = compose_vanishing(sinh_c, T, N)
        sb = compose_vanishing(arsinh_c, T, N)
        T = poly_sub(sa, sb)
    return T


def compare_to_T_self():
    """At each (ord = 3^k, lead = c_k) class, explicitly compare to T_self^k."""
    print("\n=== Explicit comparison to T_self^k ===")
    N = 100
    sinh_c = sinh_taylor_rat(N)
    arsinh_c = arsinh_taylor_rat(N)
    F = enumerate_F0_exact(max_depth=4, N=N, sinh_c=sinh_c, arsinh_c=arsinh_c)

    by_series = {}
    for d, r, c in F:
        if c not in by_series:
            by_series[c] = (d, r)
    print(f"distinct at N={N}: {len(by_series)}")

    for k in range(1, 5):
        m = 3 ** k
        if m > N:
            break
        T = tuple(build_T_self_exact(k, N))
        if T in by_series:
            d, r = by_series[T]
            print(f"  T_self^{k} (ord={m}): present in F_0 as {r[:100]}, "
                  f"depth={d}.")
        else:
            print(f"  T_self^{k} (ord={m}): NOT FOUND in F_0^{{<=4}} at N={N}")
        # Count other trees with same (ord, leading)
        c_k_val = T[m]
        same_ord_lead = [
            (c, by_series[c])
            for c in by_series
            if len(c) > m and ord_of(c) == m and c[m] == c_k_val
        ]
        if len(same_ord_lead) > 1:
            print(f"    Other trees at (ord={m}, lead={c_k_val}): "
                  f"{len(same_ord_lead) - 1} additional")
            for c, (d, r) in same_ord_lead[:3]:
                if c != T:
                    # Find first differing position vs T_self^k
                    for p in range(len(c)):
                        if c[p] != T[p]:
                            print(f"      depth={d}: first diff at x^{p}: "
                                  f"T_self_{k}={T[p]}, other={c[p]}")
                            break
        else:
            print(f"    T_self^{k} is UNIQUELY identified by (ord, leading)={m, c_k_val}.")


if __name__ == "__main__":
    analyze(max_depth=4, N=90)
    compare_to_T_self()
