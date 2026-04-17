# Transcendence-complexity invariant for ψ-trees

## Motivation

The `(ord, leading)` uniqueness induction proposed as Path 1 to close the
Pure-sinh Non-representability Lemma was empirically refuted at depth 5
(commit `7c34f491`): two distinct trees share the same `(ord, leading)`
pair, and applying ψ once more preserves the match while the Taylor
series diverge at higher order. A refined invariant is therefore
required. This note documents the first such candidate, implemented in
`eml_gam/transcendence.py`.

## Hypothesis

For a ψ-tree `f ∈ F_0` over the terminal set `{1, x}` define the
*syntactic transcendence complexity*

```
tc(f) = |{distinct innermost sinh / arsinh subexpressions of f}|,
```

where "distinct" is decided by `sp.simplify`. This is an upper bound on
the algebraic transcendence degree of the atom set of `f` over `ℚ(x)`:
two atoms that simplify to the same SymPy expression cannot contribute
independent transcendental dimensions, and the reverse relation is a
conjecture we do not try to prove here.

**Transcendence monotonicity conjecture.** For every non-constant
`g ∈ F_0` at ψ-depth `d`, any ψ-composition that introduces a new
atom layer satisfies `tc(ψ(ψ(g, g), ψ(g, g))) > tc(f)` for every
`f ∈ F_0` with ψ-depth ≤ d.

If the conjecture holds, the Pure-sinh Non-representability Lemma
follows: composition increases `tc`, so no bounded-depth tree can
reproduce the `tc` of an arbitrarily-deep target.

## Empirical test

The module enumerates every ψ-tree of exact depth `d` over `{1, x}` for
`d = 0, 1, 2, 3` (up to a cap of 16 trees per depth for tractability),
computes `tc` for each, and records whether `max tc` is strictly
increasing in `d`. It also evaluates a PSLQ sanity check at 80 decimal
digits on the numerical atom values — a non-trivial integer relation
at that precision would immediately falsify the conjecture.

### Results (depth ≤ 3, 16 trees per depth)

| depth | n_trees | min tc | max tc | distinct tc | max tc increases |
|------:|--------:|------:|------:|------------:|------------------:|
| 0     | 2       | 0     | 0     | 1           | yes (baseline)    |
| 1     | 8       | 1     | 2     | 2           | yes (0 → 2)       |
| 2     | 16      | 2     | 4     | 3           | yes (2 → 4)       |
| 3     | 16      | 3     | 5     | 3           | yes (4 → 5)       |

Every ψ-composition step adds at least one new transcendence layer.
The growth is strictly monotone through depth 3.

### PSLQ check

At depth 3 the module computes 44 atom values for the first eight
trees and runs PSLQ at 80 digits with coefficient bound 10⁶. The only
relation reported is the trivial

```
[1, 0, ..., 0, -1]
```

indicating that `atoms[0] == atoms[43]`, which is expected: different
trees share the constant sub-atom `sinh(1)`. No non-trivial relation
of degree ≤ 1 and integer coefficients ≤ 10⁶ was detected.

## What this proves, and what it does not

The monotonicity test is empirical and bounded. It is **not** a proof
that `tc` keeps growing for all depths, nor that `tc` is preserved
under all algebraic operations a ψ-target could be hiding. It rules out
the cheapest way the conjecture could fail — a collapse of the number
of distinct sinh/arsinh subexpressions at a specific depth — and it
provides a numerical fingerprint (the atom signature of
`eml_gam.transcendence.AtomSignature`) that is strong enough to reject
the counterexample of commit `7c34f491` (those two trees differ in
their atom multiset although their `(ord, leading)` coincides).

A proof at all depths would require a transcendence-degree argument
over `ℚ(x)`: specifically, that every `ψ(·, ·)` application increases
the algebraic transcendence degree of the atom set by at least one,
and that `sinh` and `arsinh` are never roots of polynomials over
`ℚ(x)` whose coefficients are themselves atom-expressible. The
Schanuel conjecture is the natural tool; no depth-uniform unconditional
statement appears to be available with current transcendence theory.

## How to reproduce

```bash
python -m eml_gam.transcendence
# writes a summary table to stdout; no artefact file.
```

Parameters: `max_depth`, `max_trees_per_depth`, `x0` (the numerical
evaluation point), and the PSLQ precision can be edited in
`run_transcendence_experiment`.
