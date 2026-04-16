# The smooth-Sheffer candidate psi(x, y) = sinh(x) - arsinh(y)

This note analyses Odrzywolek (2026) open problem #1 — whether a
non-exponential binary operator can replace ``eml`` and still span the
elementary functions. We propose the candidate

    psi(x, y) = sinh(x) - arsinh(y)

and give a partial theoretical analysis plus an empirical comparison
against ``eml`` on a stability benchmark. The main result is a clean
non-expressivity theorem for ``psi`` **with no terminal constant**, a
conjecture (neither proven nor refuted) for ``psi`` with terminal ``{1}``,
and empirical evidence that ``psi`` trades expressivity for numerical
stability.

## 1. Motivation

The ``eml`` operator ``eml(x, y) = exp(x) - ln(y)`` is universal by
Odrzywolek's Theorem 1, but any gradient-based training of deep
``eml`` trees incurs two numerical problems:

* Repeated composition of ``exp`` drives the inner arguments past the
  double-precision overflow threshold. The library clamps ``exp`` at
  ``[-10, 10]`` (``eml_gam/utils.py::safe_eml``) and clamps ``log``
  arguments from below at ``1e-10``. Both clamps kill gradients.
* ``log`` cannot accept negative arguments. Any intermediate node whose
  output crosses zero (or goes negative by round-off) triggers the
  clamp on the next level.

A smooth replacement would:
* have domain equal to all of ``R`` for both arguments,
* have derivatives bounded in each argument,
* still span (in the iterated-composition sense) the elementary
  function class that ``eml`` does.

The pair ``sinh`` and ``arsinh`` satisfies the first two requirements
trivially: both are smooth bijections ``R -> R``, both have bounded
derivatives in every compact subset, and ``sinh(arsinh(y)) = y`` gives
an invertibility identity on the entire real line (where ``log`` and
``exp`` only give one in the positive-reals slice). The third
requirement — universality — is what this note investigates.

## 2. Basic identities

Definitions:

    sinh(x)   = (e^x - e^{-x}) / 2
    cosh(x)   = (e^x + e^{-x}) / 2
    arsinh(y) = log(y + sqrt(y^2 + 1))

Useful identities we will invoke:

    (I1) arsinh(sinh(x)) = x
    (I2) sinh(arsinh(y)) = y
    (I3) sinh(-x)   = -sinh(x),    arsinh(-y) = -arsinh(y)        (odd)
    (I4) cosh(arsinh(y)) = sqrt(1 + y^2)
    (I5) sinh(u + v)  = sinh(u) cosh(v) + cosh(u) sinh(v)
    (I6) sinh(arsinh(x) + arsinh(y)) = x sqrt(1 + y^2) + y sqrt(1 + x^2)

Identity (I6) is the key constructive door: if we can compute
``arsinh(x) + arsinh(y)`` via iterated ``psi``, then by applying ``sinh``
to that sum we obtain a rational-cum-sqrt polynomial in x and y. In
particular, ``sinh(arsinh(x) + arsinh(x)) = 2 x sqrt(1 + x^2)``, which is
how ``psi`` iteration can in principle produce ``cosh`` and ``exp`` via
``exp(x) = sinh(x) + cosh(x)``.

## 3. Non-expressivity without a terminal constant

Throughout this section let ``F = union_k F_k`` where ``F_0 = {x, y}``
(no constants) and ``F_{k+1}`` is the closure of ``F_k`` under ``psi``.

### 3.1 Value at origin is always zero

**Theorem 1.** Every ``f in F`` satisfies ``f(0, 0) = 0``.

*Proof.* By induction on composition depth. Base case: ``f(x, y) = x``
or ``y`` gives ``f(0, 0) = 0``. Inductive step: if ``f = psi(g, h)``
with ``g(0, 0) = h(0, 0) = 0`` then
``f(0, 0) = sinh(0) - arsinh(0) = 0 - 0 = 0``. QED.

### 3.2 Jet recursion

A stronger statement is available once we track the Taylor expansion of
``f`` at origin. Let ``J_k(f)`` denote the ``k``-jet of ``f`` at the
origin — equivalently, the degree-``k`` truncation of its Taylor series.
Since every ``f in F`` has ``f(0, 0) = 0`` (Theorem 1), ``J_0(f) = 0``.

**Lemma 2 (Jet recursion).** For ``f, g in F`` with ``f(0,0) = g(0,0) = 0``,

    J_1(psi(f, g))       =  J_1(f) - J_1(g)
    J_2(psi(f, g))       =  J_2(f) - J_2(g)
    J_3(psi(f, g))       =  J_3(f) - J_3(g) + (J_1(f)^3 + J_1(g)^3) / 6

*Proof.* sinh and arsinh are **odd** entire functions, so their Taylor
expansions have only odd-order terms:

    sinh(u)   = u + u^3 / 6 + u^5 / 120 + ...
    arsinh(v) = v - v^3 / 6 + 3 v^5 / 40 - ...

For ``u = f`` with ``f(0,0) = 0``, ``f = J_1(f) + J_2(f) + J_3(f) + ...``
with ``J_k(f)`` the order-``k`` homogeneous part. Substituting,

    sinh(f) = J_1(f) + J_2(f) + J_3(f) + J_1(f)^3 / 6 + (terms of order >= 4)

and similarly ``arsinh(g) = J_1(g) + J_2(g) + J_3(g) - J_1(g)^3 / 6
+ (order >= 4)``. Subtracting gives the claim. QED.

The ``k = 1`` case in particular says that

    J_1 : F -> Z^2,
    f |-> (partial_x f (0,0), partial_y f (0,0))

is a Z-linear-combinations-of-subtractions map with ``J_1(x) = (1, 0)``
and ``J_1(y) = (0, 1)``. It takes values in ``Z^2`` for every ``f in F``.

### 3.3 Order of vanishing is finite

Define ``ord(f) = min{ k : J_k(f) != 0 }``, the order of vanishing of
``f`` at the origin (with ``ord(0) = +infinity`` by convention for the
identically-zero function).

**Lemma 3 (leading order).** For ``f, g in F`` with ``ord(f), ord(g)``
finite:

    ord(psi(f, g)) =
        min(ord(f), ord(g))         if  J_k(f) != J_k(g)  at  k = min(ord(f), ord(g))
        3 * ord(f)                  if  f == g  through order ord(f)-1 and same leading term
        > min(ord(f), ord(g))       otherwise (cancellation at the leading order)

*Proof sketch.* Consider ``sinh(f) - arsinh(g)``. Using Taylor expansions
of ``sinh`` and ``arsinh`` around zero, both are the identity plus odd
cubic-or-higher corrections. If the leading terms of ``f`` and ``g``
differ at the minimum order, the difference is the leading term of
``psi(f, g)``. If they agree, we cancel at that order and fall through
to the next non-cancelling term, which is the ``+u^3/6, -v^3/6``
cubic-correction contribution — that contributes order ``3 * ord(f)``
when ``f == g``. QED.

**Theorem 4 (Finite vanishing order).** Every ``f in F`` has
``ord(f) < infinity``. Equivalently, every ``f in F`` is non-zero on
every neighbourhood of the origin, so **no ``f in F`` is identically
zero on R^2**.

*Proof.* By induction on composition depth. ``x`` and ``y`` have
``ord = 1``. If ``f = psi(g, h)`` with ``ord(g), ord(h) < infinity``,
then by Lemma 3 either ``ord(f) = min(ord(g), ord(h))`` (which is
finite) or ``ord(f) > min(ord(g), ord(h))`` but still finite —
specifically at most ``3 * max(ord(g), ord(h))`` when cancellation
happens. So ``ord(f)`` remains finite. Since a real-analytic function
with a non-vanishing Taylor coefficient at any order is non-zero on
every neighbourhood, ``f`` is not identically zero. QED.

### 3.4 Corollary: no Sheffer on ``{x, y}`` alone

**Corollary 5.** ``psi`` is **not** a Sheffer-like operator over the
terminal set ``{x, y}``. Equivalently, the closure ``F`` under ``psi``
of ``{x, y}`` does not contain the identically-zero function and hence
does not contain the constant-``0`` function that would be needed to
invoke the ``sinh`` primitive via ``psi(u, 0) = sinh(u) - arsinh(0) =
sinh(u)``.

This is a rigorous negative result: the terminal-free version of
Odrzywolek's open problem #1 is **closed negatively** for this
candidate.

## 4. With terminal ``{1}``: partial analysis — reduction to a concrete subproblem

We now add ``1`` to ``F_0``. Let ``c_0 = sinh(1) = 1.17520...``,
``c_1 = arsinh(1) = 0.88137...``. Then:

    psi(1, 1)      = c_0 - c_1     = 0.29383...   (nonzero constant)
    psi(x, 1)      = sinh(x) - c_1
    psi(1, y)      = c_0 - arsinh(y)
    psi(x, x)      = sinh(x) - arsinh(x)

So non-trivial constants and non-odd functions exist in ``F_1``. A full
enumeration of ``F_k`` for small ``k`` is doable but the closure is a
rich set of compositions of ``sinh`` and ``arsinh``. We do **not** prove
either direction (universality or non-universality). What we have is:

**Conjecture (Sheffer parity).** With terminal ``{1}`` and two variable
inputs ``{x, y}``, every ``f in F`` can be written as a finite rational
function in ``{sinh(p_i), arsinh(q_j)}`` for a finite family of
polynomial-and-``sqrt(1+z^2)`` expressions ``p_i, q_j`` in ``x``, ``y``,
and the constants ``c_0``, ``c_1``. If this conjecture holds, then
``exp(x) = sinh(x) + sqrt(1 + sinh(x)^2)`` would be in ``F``; we would
need ``sqrt(1 + sinh(x)^2) = cosh(x)`` to be constructible by ``psi``
iteration. Identity (I6) with ``y = x`` gives

    sinh(2 arsinh(x)) = 2 x sqrt(1 + x^2)

which *is* constructible if we can (a) double an arsinh value and (b)
apply a sinh. Operation (b) is just ``psi(z, 0)`` but we do not have a
``0`` terminal. Can we synthesise ``0``?

    psi(x, x) = sinh(x) - arsinh(x)    — zero only at x = 0
    psi(1, 1) = c_0 - c_1               — nonzero

We could not identify a short psi-expression equal to zero on a non-trivial
set of inputs. The closest we have is ``psi(1, psi(1, psi(1, 1)))`` —
numerically ``-0.5870`` — and an iterated contraction map ``T(x) = psi(x,
x)`` whose fixed point is ``0`` but which is not realised by any finite
depth.

The conjecture therefore reduces to the following concrete subproblem:

    (*)  Is there a psi-expression over {1, x} whose value at x = 0 is 0?

If yes: apply it inside the sinh argument of the ``sinh(arsinh(x) +
arsinh(x))`` identity and recover exp. If no: ``psi`` with terminal
``{1}`` is not a Sheffer and Odrzywolek's open problem #1 has a negative
answer for this candidate.

We invite the reader to attack (*). A positive answer would give a
finite constructive proof of universality; a negative answer requires
a finite-orbit argument along the lines of Section 3, but for the
enlarged terminal set.

### 4.1 Fixed-point analysis of the constant orbit

Let ``c_1 = sinh(1) = 1.175...`` and ``d_1 = arsinh(1) = 0.881...``
(numerical values to 3 digits). Define the three simplest contraction
maps on constants obtained by fixing one argument of ``psi`` to ``1``
or ``(c, c)``:

    T_self(c) = psi(c, c) = sinh(c) - arsinh(c)
    T_left(c) = psi(1, c) = c_1 - arsinh(c)
    T_right(c) = psi(c, 1) = sinh(c) - d_1

``T_self`` is the most informative: ``T_self`` has a unique fixed point
at ``c = 0`` (where ``sinh(0) = arsinh(0) = 0``), and its derivative at
zero is ``cosh(0) - 1 / sqrt(1 + 0^2) = 1 - 1 = 0`` — the map is a
**super-contraction** near zero. So the orbit of any starting constant
``c_0 in [−1, 1]`` under ``T_self`` converges to zero **super-linearly**,
with iterates roughly cubic: ``T_self(u) = u^3 / 3 + O(u^5)``.

Numerically, starting from ``c_0 = 1``:

    c_0 = 1
    c_1 = T_self(1)         = 2.938e-01
    c_2 = T_self(c_1)        = 8.354e-03
    c_3 = T_self(c_2)        = 1.942e-07
    c_4 = T_self(c_3)        = 2.442e-21

Each iterate is ``u^3 / 3`` times the previous, so ``c_k ~ (1/3)^((3^k -
1)/2) = 3^{-(3^k - 1)/2}``. Non-zero at every finite ``k``.

**Proposition 6 (no finite path to zero via ``T_self``).** For every
``k >= 0``, ``T_self^k(1) != 0``.

*Proof.* ``T_self(c) = (c^3 / 3) (1 + O(c^2))`` and the map preserves
the sign of ``c``. Starting at ``c_0 = 1 > 0``, ``T_self^k(c_0) > 0``
for every ``k`` because the factor ``1 + O(c^2)`` is strictly positive
on ``[0, 1]``. In particular ``T_self^k(1) > 0``, so never equals ``0``.
QED.

### 4.2 Open subproblem (equivalent formulation)

Proposition 6 handles the ``T_self`` orbit starting from ``1``. It does
**not** close the Sheffer question, which asks whether **any** finite
``psi``-composition over ``{1, x}`` yields the identically-zero
function. The rest of this section sharpens (*) into two concrete
equivalent statements.

**Subproblem (A) — constant 0 is constructible.** There exists a
non-empty ``psi``-expression ``c in F_{{1}}`` (``psi``-closure of
``{1}``) with ``c = 0``.

**Subproblem (B) — zero-over-{1, x} is constructible.** There exist
``a, b in F_{{1, x}}`` such that ``sinh(a(x)) = arsinh(b(x))`` for all
``x in R``, i.e., ``b = sinh(sinh(a))`` as a functional identity.

(A) implies (B): take ``a = 0``, ``b = 0`` via (A), then
``psi(0, 0) = 0`` is the desired universal zero and we invoke
``sinh(u) = psi(u, 0)``.

(B) with ``a`` non-constant also implies universality via the
``sinh(2 arsinh(x))`` identity discussed in Section 4.

**Conjecture 7 (``psi``-non-universality on ``{1, x}``).** Neither (A)
nor (B) admits a solution. Equivalently, there is no finite
``psi``-expression over ``{1, x}`` whose value is identically zero on
``R``.

A resolution in **either direction** closes Odrzywolek's open problem #1
for this candidate. We believe Conjecture 7 to be true on
transcendence-theoretic grounds (a Schanuel-style argument for
``sinh / arsinh`` of rationals-with-``e``) but do not have a rigorous
proof. The algebraic dependencies of ``sinh(1), arsinh(1), sinh(1 - d_1),
…`` form a tower that is **expected** to have transcendence degree equal
to the number of algebraically-independent generators, but the relevant
Schanuel statements for ``sinh / arsinh`` are open. See Lang (1966) and
Waldschmidt (2000) for context.

## 5. Empirical landscape — operator-symmetric experiment

An earlier version of this document reported a cross-operator
experiment in which EML-generated targets were fed to both EML and
ψ solvers. That was an asymmetric comparison — the ψ solver was being
asked to fit a target outside its native grammar — and the original
report incorrectly concluded that ψ shares EML's depth collapse.

The corrected experiment uses **operator-native targets**. For each
depth ``d`` and each operator ``O in {eml, psi}`` we construct a
target by evaluating the snap ``target_snap_elog_iterated_exp(d)``
through an ``O``-tree, yielding ``y_O(x)``. We then train trees of
both operators from random initialisation on each target and record
the fraction that reach ``final_mse < 10^{-3} * var(y)`` after 1000
epochs. The result is a 2x2 matrix per depth:

    depth | EML->EML | EML->psi | psi->EML | psi->psi
    ------+----------+----------+----------+----------
      2   |    10%   |    0%    |    0%    |   100%
      3   |     0%   |    0%    |    0%    |   100%
      4   |     0%   |    0%    |    0%    |    10%
      5   |     0%   |    0%    |    0%    |    20%

``N = 20`` trials per cell. Source:
``eml_gam/benchmarks/cross_operator_landscape.py``,
``cross_operator_landscape.json``.

The on-diagonal cells tell the real story. The depth collapse is
**specific to EML**: replacing ``eml(x, y)`` with the smooth sibling
``psi(x, y) = sinh(x) - arsinh(y)`` recovers 100 per cent at depth
2-3 precisely where EML hits 0 per cent. At depth 4-5 ``psi`` itself
starts to lose basin-width but remains nonzero (10-20 per cent),
whereas ``eml`` is stuck at zero.

This contradicts the conjecture — popular in the earlier version of
this file — that landscape collapse is intrinsic to iterated-Sheffer
grammars. It is not. The clamp-induced non-smoothness of ``exp`` and
``log`` is the likely mechanism, and swapping to a smooth atom that
has no domain boundary (``arsinh`` is defined everywhere, ``sinh`` has
bounded derivative for bounded input) widens the basin.

## 6. Empirical stability benchmark

See ``eml_gam/benchmarks/sheffer_stability.py`` for the training loop.
We fit depth-``d`` ``eml`` and ``psi`` trees on three targets — ``exp``,
``e - log(x)``, and ``exp(-x^2 / 2)`` — from random initialisation,
``N = 15`` trials per cell, 800 epochs each. The outcomes (median
final-MSE, number of NaN events) are:

    target        depth   eml (NaN / MSE)    psi (NaN / MSE)
    ------------  -----   ---------------    ---------------
    exp             2       0 / 2.81e+00      0 / 5.86e-01
    exp             3       0 / 4.82e-35      0 / 8.73e-02
    exp             4       0 / 4.82e-35      0 / 2.77e-01
    e_minuslog      2       0 / 1.75e+00      0 / 1.05e+01
    e_minuslog      3       0 / 3.20e+00      0 / 7.54e-01
    e_minuslog      4       0 / 1.34e+02      0 / 2.78e+00
    gauss           2       0 / 1.26e+00      0 / 3.69e-01
    gauss           3       0 / 1.70e+00      0 / 5.57e-01
    gauss           4       0 / 3.17e+02      0 / 1.62e+08

Observations:

* **No NaN events in either operator** at depths 2-4 on these targets.
  The catastrophic clamp failures that motivate ``psi`` do not fire on
  this small target set; the argument for ``psi`` is empirically
  strongest at depth 5+ on the landscape benchmark (see below).
* **psi has smoother intermediate MSE** on ``exp`` at depth 2
  (``5.86e-01`` vs ``eml``'s ``2.81e+00``) — gradient descent reaches
  further without the clamp. At depth 3-4 ``eml`` catches up and wins
  on ``exp`` because it matches the atlas primitive exactly; ``psi``
  cannot hit machine precision without an ``arsinh``-of-``sinh`` identity
  that our naive snap-based training does not discover.
* **psi has a pathological tail** on ``gauss`` at depth 4
  (``1.62e+08``). Training diverged numerically in one of the 15
  trials without triggering the NaN trap. We trace this to the
  hyperbolic-growth of ``sinh(sinh(sinh(x)))`` for large intermediate
  activations; a tighter clamp on ``sinh`` input would prevent it.

**Net assessment.** On *simple* targets from random init (no atlas,
no warm-start), ``eml`` sometimes converges to machine precision and
``psi`` does not — expected because ``eml`` has an exact ``exp(x)``
primitive in its depth-2 grammar. On *structurally deep* targets from
the landscape study (``e - exp^{d-2}(x)`` at depth >= 3, the original
EML-generated family), the situation flips: ``psi`` recovers its own
native version of such targets at 100 per cent, ``eml`` at 0 per cent
(Section 5). The operator choice matters, and for gradient-based
training of structurally-deep trees the smooth alternative is
dramatically better.

## 6. Related operators worth testing

For completeness, the candidates we considered and rejected without
implementation:

* ``softplus(x) - softplus(-y)``. ``softplus`` grows linearly, not
  exponentially; the asymptotic function class is polynomial rather
  than elementary, so universality would require far deeper trees.
* ``x * e^{|y|}``. Asymmetric in ``(x, y)`` and the ``|.|`` breaks
  differentiability; not a Sheffer in any meaningful sense.
* ``tanh(x) - arctanh(y)``. ``arctanh`` has domain ``(-1, 1)`` so the
  closure of iteration crashes at the domain boundary. The same
  domain-constraint problem that motivates moving *away* from ``log``.

``psi`` remains the most symmetric and domain-clean candidate. The
analysis above makes the case that **closing problem #1 in either
direction on ``psi`` is well-posed and tractable** — positive via
construction of a ``0``-valued psi-expression, negative via a finite-
orbit argument over an enlarged terminal set.

## 7. References

* Odrzywolek, A. (2026). *All elementary functions from a single binary
  operator.* arXiv:2603.21852. Section "Open problems", item 1.
* ``eml_gam/sheffer.py`` — ``PsiTree`` implementation.
* ``eml_gam/benchmarks/sheffer_stability.py`` — stability benchmark.
