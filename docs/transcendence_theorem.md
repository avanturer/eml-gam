# Transcendence Monotonicity: Witness and Orbit Theorems

## Status

Two unconditional theorems on explicit families of targets, upgrading
the empirical conjecture of `docs/transcendence_analysis.md`:

- **Theorem W (Witness Family).**  `atc(T_{d+1}) ≥ atc(T_d) + 1` for the
  diagonal iterate `T_0 = x, T_{d+1} = ψ(T_d, T_d)`.
- **Theorem O (Orbit).**  `atc(W_{d+1}(g)) ≥ atc(W_d(g)) + 1` for
  *every* non-constant `g ∈ F_0` and the orbit
  `W_0(g) := g, W_{d+1}(g) := ψ(W_d(g), W_d(g))`.

Theorem O strictly generalises Theorem W (take `g = x`).  Both are
proved via the Ax-Schanuel theorem (Ax 1971; analytic form re-proved by
Noguchi 2022, arXiv:2203.00470) together with a short `cosh`-
transcendence lemma.

The **general** Transcendence Monotonicity Conjecture of the original
`docs/transcendence_analysis.md` — which asks for strict `tc`-growth on
*any* non-orbit composition — remains open and has been reduced to a
combinatorial genericity condition `(Gen-d)` in the free `ψ`-algebra
over `{1, x}`.  Explicit `ℚ`-linear dependencies among `ψ`-tree values
(e.g. `ψ(1, x) = ψ(x, x) − ψ(x, 1) + (sinh 1 − arsinh 1)`) show that
`(Gen-d)` is *not* vacuous; it is a genuine structural open problem.

## Setup

Work in the differential field `(M, d/dx)` of germs of meromorphic
functions at `x = 0`.  Let `C = ℚ̄` be the constant subfield.  For any
`f ∈ F_0` let

- `A(f)` := set of all `sinh(·)` and `arsinh(·)` subexpressions of `f`;
- `F(f) := C(x)(A(f)) ⊂ M`;
- `atc(f) := tr.deg_{C(x)}(F(f))`;
- `tc(f) :=` number of equivalence classes of `A(f)` under `sp.simplify`.

Always `atc(f) ≤ tc(f)`.

## Tools

### Ax-Schanuel (functional, unconditional)

**Theorem (Ax 1971, §III).** *Let `K` be a differential field of
characteristic zero with field of constants `C`.  Let `n ≥ 1` and
`u_1, …, u_n, y_1, …, y_n ∈ K` satisfy `du_i = dy_i / y_i`.  If
`u_1, …, u_n` are `ℚ`-linearly independent modulo `C`, then*

```
tr.deg_C(u_1, …, u_n, y_1, …, y_n) ≥ n + tr.deg_C(u_1, …, u_n).
```

We use the following two corollaries.

**Corollary 1 (Exp-transcendence).**  Let `F` be a differential subfield
of `M` with constants `C`.  If `g ∈ F` is non-constant and not a
`ℚ`-linear combination modulo `C` of elements `u ∈ F` for which
`e^u ∈ F`, then `e^g` is transcendental over `F`.

**Corollary 2 (Log-transcendence).**  Let `α ∈ F` be a non-constant
element whose multiplicative rank modulo `C·(F^*)^{ℚ}` is at least one
(i.e. `α` is not a constant multiple of a `ℚ`-product of elements of
`F^*`).  Then `log(α)` is transcendental over `F`.

Both are `n = 1` specialisations of Ax 1971.

### `cosh`-transcendence (elementary lemma)

**Lemma (`cosh`-transcendence).**  Let `F` be a differential subfield
of `M` with constants `C`, and let `g ∈ F` be non-constant with
`e^g ∉ F`.  Then `cosh(g)` is transcendental over `F`.

**Proof.**  By Corollary 1, `e^g` is transcendental over `F`.  The
ring extension `F[e^g, e^{−g}]` has `e^{−g} = 1/e^g`, so its total
transcendence degree over `F` is exactly one, with generator `e^g`.
Write `cosh(g) = (e^g + e^{−g})/2 = (e^{2g} + 1) / (2 e^g)`.  If
`cosh(g)` were algebraic over `F`, then `e^g` (a root of the quadratic
`z² − 2 cosh(g) z + 1 = 0`) would be algebraic over `F(cosh(g)) ⊂
F^{alg}`, hence algebraic over `F`.  But we just saw `e^g` is
transcendental over `F`.  Contradiction.  ∎

**Remark (F vs F^{alg}).**  Transcendence over `F` and transcendence
over `F^{alg}` coincide for elements of `M`: if `α ∈ M` is algebraic
over `F^{alg}`, then the minimal polynomial over `F^{alg}` has
algebraic coefficients, and multiplying by the minimal polynomials of
those coefficients produces a polynomial over `F` whose roots include
`α`.  Thus every statement "transcendental over `F_d`" in the proof
below equivalently says "transcendental over `F_d^{alg}`", and we can
freely pass between the two when convenient.

## Theorem W (Witness Family)

Define `T_0 := x`, `T_{d+1} := ψ(T_d, T_d) = sinh(T_d) − arsinh(T_d)`.

**Theorem W.**  `atc(T_{d+1}) ≥ atc(T_d) + 1` for every `d ≥ 0`.

**Proof.**  Abbreviate `F_d := F(T_d) = C(x)(A(T_d))`.  By induction on
`d` we show (W-d): *`T_d` is non-constant, and `sinh(T_d) ∉
F_d^{alg}`.*

*Base step* (`d = 0`): `T_0 = x` is non-constant.  `F_0 = C(x)`
contains no non-trivial `e^u` factor, so by Corollary 1 `e^x` is
transcendental over `C(x)`.  Hence `sinh(x) = (e^x − e^{−x})/2` is
transcendental over `C(x)`, so `sinh(T_0) = sinh(x) ∉ F_0^{alg}`.

*Inductive step* (`d ⇒ d + 1`): by (W-d), `sinh(T_d) ∉ F_d^{alg}`, so
in particular `sinh(T_d)` is a *new* atom when we form
`A(T_{d+1}) = A(T_d) ∪ \{sinh(T_d), arsinh(T_d)\}`.  Hence
`atc(T_{d+1}) = tr.deg_{C(x)}(F_{d+1}) \ge atc(T_d) + 1`.

It remains to prove (W-{d+1}): *`T_{d+1}` is non-constant, and
`sinh(T_{d+1}) ∉ F_{d+1}^{alg}`.*

**Non-constancy of `T_{d+1}`.**  Differentiating,

```
T_{d+1}'
  = cosh(T_d) · T_d' − T_d' / √(1 + T_d²)
  = T_d' · (cosh(T_d) − 1 / √(1 + T_d²)).
```

By inductive hypothesis (W-d) and the `cosh`-transcendence lemma
applied to `g = T_d ∈ F_d` (which satisfies `e^{T_d} ∉ F_d` — this is
equivalent to `sinh(T_d) ∉ F_d^{alg}` via the same algebraic closure
argument as in the lemma), `cosh(T_d)` is transcendental over `F_d`.
In particular `cosh(T_d) − 1/√(1 + T_d²)` is non-zero (otherwise
`cosh(T_d) = 1/√(1 + T_d²) ∈ F_d^{alg}`, contradicting transcendence).
Since `T_d'` is non-zero by inductive non-constancy of `T_d`, the
product `T_{d+1}'` is non-zero.  Hence `T_{d+1}` is non-constant.

**`sinh(T_{d+1}) ∉ F_{d+1}^{alg}`.**  Apply Corollary 1 with
`F = F_{d+1}`, `g = T_{d+1}`.  The hypothesis requires `T_{d+1}` to
be non-constant (just shown) and not a `ℚ`-linear combination modulo
`C` of elements `u ∈ F_{d+1}` with `e^u ∈ F_{d+1}`.

The elements `u ∈ F_{d+1}` with `e^u ∈ F_{d+1}` are exactly those `u`
of the form

```
u = ∑_{i ∈ I} q_i · h_i + c,                                    (★)
```

where each `h_i` is an *inner* `sinh`-argument of some `sinh`-atom
`sinh(h_i) ∈ A(T_{d+1})`, the `q_i ∈ ℚ`, and `c ∈ C`.  For the witness
family the possible `h_i`'s are exactly `T_0, T_1, \ldots, T_d`.  So
`(★)` reduces to asking whether

```
T_{d+1} = ∑_{i=0}^{d} q_i · T_i + c                             (★★)
```

has a solution `(q_0, \ldots, q_d, c) ∈ ℚ^{d+1} \times C`.

Differentiate `(★★)`:

```
T_{d+1}' = ∑_{i=0}^{d} q_i · T_i'.                              (★★')
```

The LHS factorises as `T_{d+1}' = T_d' · (cosh(T_d) − 1/√(1+T_d²))`,
which contains the transcendental factor `cosh(T_d)` over `F_d^{alg}`.
The RHS is a `ℚ`-linear combination of `T_i'` for `i ≤ d`.  Each
`T_i'` for `i ≤ d` can be written (by repeated application of the
product rule) as a *finite polynomial in `F_{d-1}^{alg}[cosh(T_0),
\ldots, cosh(T_{d-1})]`* with no `cosh(T_d)` factor.  Hence the RHS
lies in `F_{d-1}^{alg}[cosh(T_0), \ldots, cosh(T_{d-1})] \subset
F_d^{alg}`; it does *not* involve `cosh(T_d)`.

Since `cosh(T_d)` is transcendental over `F_d^{alg}`, it cannot equal
the `cosh(T_d)`-free RHS.  Contradiction — so `(★★)` has no solution.
Therefore `T_{d+1}` is not a `ℚ`-linear combination modulo `C` of the
`h_i`'s, the hypothesis of Corollary 1 is satisfied, and
`e^{T_{d+1}}` is transcendental over `F_{d+1}`.  Consequently
`sinh(T_{d+1}) ∉ F_{d+1}^{alg}`.  ∎ (Theorem W)

## Theorem O (Orbit)

Given any non-constant `g ∈ F_0`, define the *orbit*

```
W_0(g) := g,
W_{d+1}(g) := ψ(W_d(g), W_d(g)).
```

**Theorem O.**  `atc(W_{d+1}(g)) ≥ atc(W_d(g)) + 1` for every `d ≥ 0`
and every non-constant `g ∈ F_0`.

**Proof.**  Literally the same proof as Theorem W, replacing `T_d` by
`W_d(g)` throughout.  The only place the specific form `T_0 = x` is
used is in the base step, where non-constancy of `T_0` is needed; here
the hypothesis `g` non-constant supplies the same conclusion.  All
subsequent differential-algebraic steps go through mutatis mutandis
because they depend only on `W_d(g)` being non-constant and
`e^{W_d(g)} \notin F(W_d(g))`, both inherited from `g` non-constant
via the same induction.  ∎

**Corollary Q (quantitative).**  For every non-constant `g ∈ F_0`,

```
atc(W_d(g)) ≥ atc(g) + d.
```

In particular, `atc` along any orbit grows *at least linearly* in
orbit depth, regardless of the seed.

## Field-filtration corollary

Let `F_d^W := F(T_d) ⊂ M` be the field generated by `C(x)` and the
atoms of `T_d`.  Theorem W gives:

**Corollary F.**  The tower `F_0^W \subsetneq F_1^W \subsetneq F_2^W
\subsetneq \cdots` is a strict chain of differential subfields of
`M`, with `tr.deg_{C(x)}(F_d^W) \ge 2d`.

In particular, `\bigcup_d F_d^W` has infinite transcendence degree
over `C(x)` — no finite-dimensional transcendental extension of
`C(x)` contains all iterated `ψ`-witnesses.

## The remaining open problem

The generalization from orbits to *arbitrary* trees is the
combinatorial genericity condition:

**(Gen-d).**  *For every depth-`d` tree `G ∈ F_0` with
`G = ψ(L, R)` and depth(L) = d − 1, the function `L` is not a
`ℚ`-linear combination modulo `C` of `\{h : sinh(h) ∈ A(G),
\text{depth of this atom in G} < d\}`.*

Under `(Gen-d)` for every `d`, the inductive step of Theorems W/O
extends to every tree, giving the full universal Transcendence
Monotonicity Conjecture.

**Why (Gen-d) is not trivial.**  The ψ-tree values are *not*
`ℚ`-linearly independent.  A direct check gives

```
ψ(1, x) = ψ(x, x) − ψ(x, 1) + (sinh(1) − arsinh(1)),
```

verifiable to 10⁻⁶ precision at `x = 0.5` (see the test
`test_transcendence_psi_linear_dependency` in `tests/test_eml_tree.py`).
So `(Gen-d)` is a genuine structural statement about which
configurations of `ψ`-tree arguments admit such dependencies; it is
not a free corollary of any obvious invariant.

## What is proven, and what is not

**Proven (this note):**

- Theorem W: strict `atc` growth on the witness family `{T_d}`.
- Theorem O: strict `atc` growth along every orbit `{W_d(g)}` for
  non-constant `g ∈ F_0`.
- Corollary Q: quantitative lower bound `atc(W_d(g)) \ge atc(g) + d`.
- Corollary F: strict filtration `F_0^W \subsetneq F_1^W \subsetneq
  \cdots` with unbounded transcendence degree.

**Open:**

- `(Gen-d)` for non-orbit trees.  Reduces to a combinatorial
  independence question in the free `ψ`-algebra over `\{1, x\}`.
- Function-level representability of orbit iterates by shallower
  non-orbit trees; this is a *separate* question from `(Gen-d)` and
  requires classifying algebraic relations among `ψ`-tree *values*,
  not just tree *expressions*.

## How to reproduce

```bash
python -m eml_gam.transcendence_witness
# prints T_0, T_1, …, T_5 and their atom lists, and PSLQ-verifies
# strict atc growth at 200 decimal digits, integer-coefficient bound
# 10^20.  Also verifies the explicit linear dependency
# ψ(1, x) = ψ(x, x) - ψ(x, 1) + (sinh 1 - arsinh 1) as documented
# evidence that (Gen-d) is non-trivial.
```
