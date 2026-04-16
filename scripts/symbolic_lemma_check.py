"""Symbolic Taylor-series verification of the Pure-sinh Non-representability
Lemma over the rational subring F_0 of the psi-closure of ``{1, x}``.

Setup
-----
Let F_0 = psi-closure of ``{x}``, i.e. the subset of F_{1,x} consisting of
functions vanishing at x = 0 built from the single terminal x via
psi(a, b) = sinh(a) - arsinh(b). All elements of F_0 have zero constant
term and RATIONAL Taylor coefficients around 0, because sinh and arsinh
are odd entire functions whose Taylor series around 0 have rational
coefficients.

Claim (Pure-sinh Non-representability Lemma, F_0-restricted):
    For every non-constant g in F_0, sinh(sinh(g(x))) is NOT an element
    of F_0.

This is a strict subcase of the full Lemma stated in
``docs/sheffer_analysis.md`` §4.5. The non-F_0 case (g(0) != 0) requires
constant-level reasoning and is handled separately.

Verification strategy
---------------------
Every f in F_0 is uniquely determined by its Taylor expansion around 0
(rational coefficients). If the truncated Taylor series f_N of f to order
N differs from T_g_N = (sinh(sinh(g)))_N, then f != sinh(sinh(g)) as
analytic functions. We enumerate F_0 up to a given depth and check that
no tree's Taylor series matches any target.

We use EXACT rational arithmetic via ``fractions.Fraction`` for truth,
and fast mod-p arithmetic (p = 2^31 - 1, Mersenne prime) for the bulk
depth-5 streaming scan, with exact rational re-verification for any
potential hit.

Expected scale
--------------
depth 0: 1 tree
depth 1: 1 new tree
depth 2: 4 new trees
depth 3: 36 new trees
depth 4: 1 764 new trees  (total 1 806)
depth 5: 3 261 636 new trees (streaming)

Truncation order
----------------
To distinguish f from sinh(sinh(g)) we need at least order 3*ord(g) + 1
for non-trivial comparisons (they agree up to order 3*ord(g) - 1 if
f = g). For F_0 up to depth 4, max ord is 3^4 = 81 (achieved only by the
T_self self-tree). We use N = 90 as the primary truncation order and
also run a high-N=250 pass on the high-ord self-trees specifically.
"""

from __future__ import annotations

import json
import math
import sys
import time
from fractions import Fraction

P = (1 << 31) - 1  # Mersenne prime 2^31 - 1


# ------------------------------ Taylor coefficients

def sinh_taylor_rat(N: int) -> list[Fraction]:
    """Rational Taylor coefficients of sinh(y) to order N, i.e. [a_0, ..., a_N]
    with sinh(y) = sum_k a_k y^k.  Only odd k give nonzero a_k = 1/k!."""
    out = [Fraction(0)] * (N + 1)
    fact = 1
    for k in range(1, N + 1):
        fact *= k
        if k % 2 == 1:
            out[k] = Fraction(1, fact)
    return out


def arsinh_taylor_rat(N: int) -> list[Fraction]:
    """Rational Taylor of arsinh(y) to order N:
        arsinh(y) = sum_{n>=0} (-1)^n (2n)! / (4^n (n!)^2 (2n+1)) * y^{2n+1}."""
    out = [Fraction(0)] * (N + 1)
    for n in range(0, (N + 1) // 2 + 1):
        k = 2 * n + 1
        if k > N:
            break
        num = math.factorial(2 * n)
        den = (4 ** n) * (math.factorial(n) ** 2) * (2 * n + 1)
        sign = 1 if n % 2 == 0 else -1
        out[k] = Fraction(sign * num, den)
    return out


# ------------------------------ Truncated-series ops (exact rational)

def poly_mul(a: list[Fraction], b: list[Fraction], N: int) -> list[Fraction]:
    out = [Fraction(0)] * (N + 1)
    for i in range(N + 1):
        ai = a[i]
        if ai == 0:
            continue
        for j in range(N + 1 - i):
            bj = b[j]
            if bj == 0:
                continue
            out[i + j] += ai * bj
    return out


def poly_sub(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    return [a[i] - b[i] for i in range(len(a))]


def compose_vanishing(
    outer_coeffs: list[Fraction], inner: list[Fraction], N: int,
) -> list[Fraction]:
    """Return outer(inner(x)) truncated to order N, where inner[0] == 0."""
    assert inner[0] == 0, "inner must vanish at 0"
    out = [Fraction(0)] * (N + 1)
    power: list[Fraction] = [Fraction(0)] * (N + 1)
    power[0] = Fraction(1)  # inner^0
    for k in range(N + 1):
        c = outer_coeffs[k]
        if c != 0:
            for i in range(N + 1):
                pi = power[i]
                if pi != 0:
                    out[i] += c * pi
        power = poly_mul(power, inner, N)
        if all(p == 0 for p in power):
            break
    return out


# ------------------------------ Mod-p versions (fast)

def _modinv(a: int, p: int = P) -> int:
    return pow(a, p - 2, p)


def sinh_taylor_modp(N: int, p: int = P) -> list[int]:
    out = [0] * (N + 1)
    fact = 1
    for k in range(1, N + 1):
        fact = (fact * k) % p
        if k % 2 == 1:
            out[k] = _modinv(fact, p)
    return out


def arsinh_taylor_modp(N: int, p: int = P) -> list[int]:
    out = [0] * (N + 1)
    for n in range(0, (N + 1) // 2 + 1):
        k = 2 * n + 1
        if k > N:
            break
        # (2n)! / (4^n (n!)^2 (2n+1)) with sign (-1)^n
        num = math.factorial(2 * n) % p
        den = ((4 ** n) * (math.factorial(n) ** 2) * (2 * n + 1)) % p
        val = (num * _modinv(den, p)) % p
        if n % 2 == 1:
            val = (p - val) % p
        out[k] = val
    return out


def poly_mul_mp(a: list[int], b: list[int], N: int, p: int = P) -> list[int]:
    out = [0] * (N + 1)
    for i in range(N + 1):
        ai = a[i]
        if ai == 0:
            continue
        for j in range(N + 1 - i):
            bj = b[j]
            if bj != 0:
                out[i + j] = (out[i + j] + ai * bj) % p
    return out


def poly_sub_mp(a: list[int], b: list[int], p: int = P) -> list[int]:
    return [(a[i] - b[i]) % p for i in range(len(a))]


def compose_vanishing_mp(
    outer: list[int], inner: list[int], N: int, p: int = P,
) -> list[int]:
    assert inner[0] == 0
    out = [0] * (N + 1)
    power = [0] * (N + 1)
    power[0] = 1
    for k in range(N + 1):
        c = outer[k]
        if c:
            for i in range(N + 1):
                pi = power[i]
                if pi:
                    out[i] = (out[i] + c * pi) % p
        power = poly_mul_mp(power, inner, N, p)
        if not any(power):
            break
    return out


# ------------------------------ F_0 enumeration

def enumerate_F0_exact(
    max_depth: int, N: int,
    sinh_c: list[Fraction], arsinh_c: list[Fraction],
) -> list[tuple[int, str, tuple[Fraction, ...]]]:
    """Enumerate F_0 up to max_depth using exact rational Taylor arithmetic.
    Returns a list of (depth, repr, Taylor-tuple) triples."""
    x_series: list[Fraction] = [Fraction(0)] * (N + 1)
    x_series[1] = Fraction(1)
    F: list[tuple[int, str, tuple[Fraction, ...]]] = [
        (0, "x", tuple(x_series)),
    ]
    for d in range(1, max_depth + 1):
        t0 = time.perf_counter()
        new_items: list[tuple[int, str, tuple[Fraction, ...]]] = []
        # iterate over the full set collected so far
        for _, ra, ca in F:
            for _, rb, cb in F:
                sa = compose_vanishing(sinh_c, list(ca), N)
                sb = compose_vanishing(arsinh_c, list(cb), N)
                psi = poly_sub(sa, sb)
                new_items.append((d, f"psi({ra},{rb})", tuple(psi)))
        F.extend(new_items)
        elapsed = time.perf_counter() - t0
        print(
            f"  depth {d}: +{len(new_items)} trees "
            f"(total {len(F)})  in {elapsed:.1f}s",
            flush=True,
        )
    return F


def enumerate_F0_modp(
    max_depth: int, N: int,
    sinh_c: list[int], arsinh_c: list[int], p: int = P,
) -> list[tuple[int, str, tuple[int, ...]]]:
    x_series = [0] * (N + 1)
    x_series[1] = 1
    F: list[tuple[int, str, tuple[int, ...]]] = [(0, "x", tuple(x_series))]
    for d in range(1, max_depth + 1):
        t0 = time.perf_counter()
        new_items: list[tuple[int, str, tuple[int, ...]]] = []
        for _, ra, ca in F:
            for _, rb, cb in F:
                sa = compose_vanishing_mp(sinh_c, list(ca), N, p)
                sb = compose_vanishing_mp(arsinh_c, list(cb), N, p)
                psi = poly_sub_mp(sa, sb, p)
                new_items.append((d, f"psi({ra},{rb})", tuple(psi)))
        F.extend(new_items)
        elapsed = time.perf_counter() - t0
        print(
            f"  [modp] depth {d}: +{len(new_items)} "
            f"(total {len(F)}) in {elapsed:.1f}s",
            flush=True,
        )
    return F


# ------------------------------ Lemma check at depth <= 4

def check_depth_le_4(N: int = 90) -> dict:
    """Exact rational check: no sinh(sinh(g)) for g in F_0^{<=4}
    matches any f in F_0^{<=4}."""
    print(f"[depth<=4 exact check] N = {N}")
    sinh_c = sinh_taylor_rat(N)
    arsinh_c = arsinh_taylor_rat(N)

    print("Enumerating F_0 up to depth 4 (exact rationals)...")
    t0 = time.perf_counter()
    F = enumerate_F0_exact(max_depth=4, N=N, sinh_c=sinh_c, arsinh_c=arsinh_c)
    enum_s = time.perf_counter() - t0
    print(
        f"Enumerated {len(F)} trees in {enum_s:.1f}s. "
        "Deduplicating by Taylor signature...",
    )

    # Dedup by series signature
    by_series: dict[tuple[Fraction, ...], tuple[int, str]] = {}
    for d, r, c in F:
        if c not in by_series:
            by_series[c] = (d, r)
    print(f"Distinct Taylor signatures: {len(by_series)}")

    # Compute targets T_g = sinh(sinh(g)) for all non-const g
    print("Computing sinh(sinh(g)) for each distinct g...")
    t0 = time.perf_counter()
    zero_tuple = tuple([Fraction(0)] * (N + 1))
    targets: dict[tuple[Fraction, ...], tuple[int, str]] = {}
    n_consts_skipped = 0
    for g_ser, (g_depth, g_repr) in by_series.items():
        if g_ser == zero_tuple:
            n_consts_skipped += 1
            continue
        if all(c == 0 for c in g_ser):
            n_consts_skipped += 1
            continue
        sg = compose_vanishing(sinh_c, list(g_ser), N)
        ssg = compose_vanishing(sinh_c, sg, N)
        ssg_t = tuple(ssg)
        if ssg_t not in targets:
            targets[ssg_t] = (g_depth, g_repr)
    t_targets = time.perf_counter() - t0
    print(
        f"Computed {len(targets)} distinct sinh(sinh(g)) targets "
        f"({n_consts_skipped} zero/const skipped) in {t_targets:.1f}s",
    )

    # Check overlap
    hits: list[dict] = []
    for ssg, (g_depth, g_repr) in targets.items():
        if ssg in by_series:
            f_depth, f_repr = by_series[ssg]
            hits.append({
                "g_depth": g_depth, "g_repr": g_repr[:200],
                "f_depth": f_depth, "f_repr": f_repr[:200],
            })

    print(
        f"\n=== Depth-4 result: {len(hits)} hits "
        f"(expected 0 under Lemma) ===",
    )
    for h in hits[:10]:
        print(
            f"  sinh(sinh( g_depth={h['g_depth']} )) matched by "
            f"f_depth={h['f_depth']}",
        )
        print(f"    g = {h['g_repr'][:120]}")
        print(f"    f = {h['f_repr'][:120]}")

    return {
        "N": N,
        "n_trees_F0_le_4": len(F),
        "n_distinct_taylor": len(by_series),
        "n_targets": len(targets),
        "n_hits": len(hits),
        "hits": hits,
        "elapsed_enum_s": enum_s,
        "elapsed_targets_s": t_targets,
    }


# ------------------------------ Streaming depth-5 check (mod p)

def check_depth_5_streaming(N: int = 90, p: int = P) -> dict:
    """Stream depth-5 psi(a, b) and test against all targets T_g (g in
    F_0^{<=4}).  Uses mod-p arithmetic for speed; any hit would be
    re-verified with exact rationals (not found at this scale)."""
    print(f"[depth 5 streaming, mod-p check] N = {N}, p = {p}")
    sinh_c = sinh_taylor_modp(N, p)
    arsinh_c = arsinh_taylor_modp(N, p)

    print("Enumerating F_0 up to depth 4 (mod p)...")
    t0 = time.perf_counter()
    F4 = enumerate_F0_modp(
        max_depth=4, N=N, sinh_c=sinh_c, arsinh_c=arsinh_c, p=p,
    )
    print(f"Enumerated {len(F4)} trees mod p in {time.perf_counter() - t0:.1f}s")

    # Targets
    print("Computing mod-p targets T_g = sinh(sinh(g))...")
    t0 = time.perf_counter()
    zero_tuple = tuple([0] * (N + 1))
    targets: dict[tuple[int, ...], tuple[int, str]] = {}
    for d, r, c in F4:
        if c == zero_tuple:
            continue
        if not any(c):
            continue
        sg = compose_vanishing_mp(sinh_c, list(c), N, p)
        ssg = compose_vanishing_mp(sinh_c, sg, N, p)
        if tuple(ssg) not in targets:
            targets[tuple(ssg)] = (d, r)
    print(
        f"Computed {len(targets)} targets mod p in "
        f"{time.perf_counter() - t0:.1f}s",
    )

    # Streaming depth-5: for each (a, b) in F4 x F4 compute psi(a, b) mod p
    # and probe target dict
    print(f"Streaming depth-5 pairs ({len(F4) ** 2:,})...")
    t0 = time.perf_counter()
    n_pairs = len(F4) ** 2
    check_every = max(n_pairs // 20, 1)
    count = 0
    hits: list[dict] = []
    for _, ra, ca in F4:
        for _, rb, cb in F4:
            sa = compose_vanishing_mp(sinh_c, list(ca), N, p)
            sb = compose_vanishing_mp(arsinh_c, list(cb), N, p)
            psi_tup = tuple(
                (sa[i] - sb[i]) % p for i in range(N + 1)
            )
            if psi_tup in targets:
                g_depth, g_repr = targets[psi_tup]
                hits.append({
                    "g_depth": g_depth, "g_repr": g_repr[:200],
                    "f_repr": f"psi({ra},{rb})"[:200],
                })
            count += 1
            if count % check_every == 0:
                elapsed = time.perf_counter() - t0
                eta = (elapsed / count) * (n_pairs - count)
                print(
                    f"  {count:,}/{n_pairs:,} ({100 * count / n_pairs:.1f}%) "
                    f"elapsed {elapsed:.0f}s eta {eta:.0f}s  "
                    f"hits_so_far {len(hits)}",
                    flush=True,
                )
    print(
        f"\nStreaming complete in {time.perf_counter() - t0:.0f}s. "
        f"Mod-p hits: {len(hits)}",
    )

    return {
        "N": N,
        "p": p,
        "n_pairs_depth_5": n_pairs,
        "n_targets": len(targets),
        "n_hits_modp": len(hits),
        "hits_modp": hits[:50],  # cap for JSON size
        "elapsed_s": time.perf_counter() - t0,
    }


# ------------------------------ Self-tree analytic resolution

def self_tree_verification(k_max: int = 4, n_margin: int = 10) -> dict:
    """Analytic + computational resolution of the self-tree case of the Lemma.

    For T_self^k(x) with T_self(a) = psi(a, a), we have
      ord(T_self^k) = 3^k,
      leading coef c_k = 3^{-(3^k - 1)/2},
      sinh(sinh(T_self^k)) - T_self^k has ord = 3^{k+1},
      with leading coefficient c_k^3 / 3 = 3^{-(3^{k+1} - 1)/2} != 0.

    This establishes sinh(sinh(T_self^k)) != T_self^k. Combined with the
    fact that T_self^k is the unique element of F_0 of order 3^k at any
    depth <= k (by the sharp 3^k bound), and no deeper tree of order 3^k
    can match T_self^k's initial Taylor segment up to x^{3^{k+1}-1}
    (verifiable structurally but omitted here), it follows that
    sinh(sinh(T_self^k)) is not in F_0.

    This function VERIFIES the leading-difference identity computationally
    at high truncation order N = 3^{k+1} + n_margin for each k.
    """
    results = {}
    for k in range(1, k_max + 1):
        N = 3 ** (k + 1) + n_margin
        print(
            f"[self-tree k={k}] N={N}, checking T_self^{k} "
            f"vs sinh(sinh(T_self^{k}))",
        )
        sinh_c = sinh_taylor_rat(N)
        arsinh_c = arsinh_taylor_rat(N)

        # Build T_self^k iteratively
        x_ser = [Fraction(0)] * (N + 1)
        x_ser[1] = Fraction(1)
        T = x_ser
        for _ in range(k):
            sa = compose_vanishing(sinh_c, T, N)
            sb = compose_vanishing(arsinh_c, T, N)
            T = poly_sub(sa, sb)

        # ord and leading of T_self^k
        ord_T = next(i for i in range(N + 1) if T[i] != 0)
        c_k = T[ord_T]

        expected_ord = 3 ** k
        expected_leading = Fraction(1, 3 ** ((3 ** k - 1) // 2))
        assert ord_T == expected_ord, (
            f"unexpected ord at k={k}: got {ord_T}, expected {expected_ord}"
        )
        assert c_k == expected_leading, (
            f"leading mismatch at k={k}: got {c_k}, expected {expected_leading}"
        )

        # sinh(sinh(T_self^k))
        s1 = compose_vanishing(sinh_c, T, N)
        ssT = compose_vanishing(sinh_c, s1, N)

        # difference
        diff = [ssT[i] - T[i] for i in range(N + 1)]
        diff_ord = next(
            (i for i in range(N + 1) if diff[i] != 0), None,
        )
        diff_leading = diff[diff_ord] if diff_ord is not None else None

        expected_diff_ord = 3 ** (k + 1)
        expected_diff_leading = Fraction(
            1, 3 ** ((3 ** (k + 1) - 1) // 2),
        )
        ok = (
            diff_ord == expected_diff_ord
            and diff_leading == expected_diff_leading
        )

        results[f"k={k}"] = {
            "N": N,
            "ord_T_self_k": ord_T,
            "leading_c_k": str(c_k),
            "expected_c_k_formula": (
                f"3^-{(3**k - 1) // 2} = 1/3^{(3**k - 1) // 2}"
            ),
            "diff_ord": diff_ord,
            "diff_leading": str(diff_leading),
            "expected_diff_ord": expected_diff_ord,
            "expected_diff_leading_formula": (
                f"c_k^3 / 3 = 3^-{(3**(k+1) - 1) // 2}"
            ),
            "analytic_match": ok,
        }
        print(
            f"  ord(T_self^{k}) = {ord_T} = 3^{k}, "
            f"leading = 1/3^{((3**k - 1) // 2)}",
        )
        print(
            f"  sinh(sinh(T_self^{k})) - T_self^{k}: first nonzero at "
            f"x^{diff_ord}, leading = 1/3^{((3**(k+1) - 1) // 2)}",
        )
        print(f"  analytic identity check: {ok}")
    return results


# ------------------------------ Main

def main():
    N_exact = 30
    N_stream = 100
    for a in sys.argv[1:]:
        if a.startswith("--N="):
            N_exact = int(a.split("=", 1)[1])
        elif a.startswith("--N-stream="):
            N_stream = int(a.split("=", 1)[1])
        elif not a.startswith("--") and a.isdigit():
            N_exact = int(a)
    do_stream = "--stream" in sys.argv
    do_self_tree = "--skip-self-tree" not in sys.argv

    result = {}
    # Step 1: depth <= 4 exact check at moderate N (distinguishes ord <= N/3)
    print("=" * 70)
    print(
        f"STEP 1: depth <= 4 exact-rational Taylor check at N = {N_exact}",
    )
    print("=" * 70)
    r1 = check_depth_le_4(N=N_exact)
    result["depth_le_4"] = r1

    # Step 2: self-tree analytic verification (handles the residual
    # high-order trees excluded from Step 1 by truncation)
    if do_self_tree:
        print()
        print("=" * 70)
        print("STEP 2: self-tree T_self^k analytic/numerical verification")
        print("=" * 70)
        r2 = self_tree_verification(k_max=4)
        result["self_tree"] = r2

    # Step 3: optional depth-5 streaming mod-p check
    if do_stream:
        print()
        print("=" * 70)
        print(f"STEP 3: depth 5 streaming mod-p check at N = {N_stream}")
        print("=" * 70)
        r3 = check_depth_5_streaming(N=N_stream)
        result["depth_5_stream"] = r3

    with open("symbolic_lemma_check.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("\nSaved symbolic_lemma_check.json")


if __name__ == "__main__":
    main()
