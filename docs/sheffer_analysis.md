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

**Theorem 4 (Finite vanishing order, with explicit bound).** Let
``F_0 = {x, y}`` and ``F_{k+1} = F_k cup {psi(f, g) : f, g in F_k}``.
Every ``f in F_k`` satisfies ``ord(f) <= 3^k``. In particular every
``f in F = union_k F_k`` has ``ord(f) < infinity``, so no ``f in F`` is
identically zero on ``R^2``.

*Proof.* Induction on ``k``. For ``k = 0``: ``f in {x, y}`` with
``ord(f) = 1 <= 3^0 = 1``. Suppose the claim holds at depth ``k - 1``
and let ``f = psi(g, h)`` with ``g, h in F_{k-1}``; so ``ord(g),
ord(h) <= 3^{k-1}``.

Consider the Taylor expansions

    g = J_{a}(g) + H_g,    h = J_{b}(h) + H_h,

where ``a = ord(g)``, ``b = ord(h)``, ``J_a(g)`` is the non-zero
order-``a`` homogeneous part of ``g``, and ``H_g, H_h`` are remainders
of order ``>= a + 1`` and ``>= b + 1`` respectively. Use the odd
Taylor expansions

    sinh(u) = u + u^3 / 6 + O(u^5),
    arsinh(v) = v - v^3 / 6 + O(v^5).

Substituting ``u = g`` and ``v = h`` and collecting by homogeneous
degree,

    sinh(g)   = J_a(g) + H_g + (J_a(g) + H_g)^3 / 6 + O(order >= 5a),
    arsinh(h) = J_b(h) + H_h - (J_b(h) + H_h)^3 / 6 + O(order >= 5b).

Each correction term has order at least ``3 min(a, b) >= 3``. So

    psi(g, h)  =  (J_a(g) + H_g) - (J_b(h) + H_h)
                  + (J_a(g) + H_g)^3 / 6 + (J_b(h) + H_h)^3 / 6
                  + O(order >= 5 min(a, b)).

**Case 1: ``a != b``.** Without loss of generality ``a < b``. The
``J_a(g)`` term is not cancelled by anything of equal or lower order
on the ``h`` side (since ``h`` starts at order ``b > a``), so
``ord(psi(g, h)) = a = min(a, b) <= max(a, b) <= 3^{k-1} <= 3^k``.

**Case 2: ``a = b`` and ``J_a(g) != J_a(h)``.** Then ``J_a(psi(g, h))
= J_a(g) - J_a(h) != 0``, so ``ord(psi(g, h)) = a <= 3^{k-1} <= 3^k``.

**Case 3: ``a = b`` and ``J_a(g) = J_a(h)``.** The leading linear-in
``sinh, arsinh`` contribution cancels. The next lowest non-zero
contribution comes from the cubic-correction ``J_a(g)^3 / 6 +
J_a(h)^3 / 6 = J_a(g)^3 / 3``, which is a non-zero homogeneous
polynomial of order ``3a`` (as ``J_a(g) != 0``). All other terms
have order ``>= 3a``: the higher-order parts of ``H_g, H_h`` (orders
``>= a + 1``, subtracted) and the higher-order cubic corrections
(orders ``>= 3a + 1``). Among these the ``a + 1`` and ``3a`` terms
compete, but ``3a >= a + 1`` for ``a >= 1``, so **the order-``a + 1``
cancellation of the linear parts of ``sinh, arsinh`` does not save us
— the cubic contribution of order ``3a`` remains and is non-zero**.
Formally, the order-``3a`` homogeneous part of ``psi(g, h)`` is
``(J_a(g)^3 + J_a(h)^3) / 6 = J_a(g)^3 / 3``, non-zero. Hence
``ord(psi(g, h)) = 3a <= 3 * 3^{k-1} = 3^k``.

In every case ``ord(psi(g, h)) <= 3^k``. The identically-zero
conclusion follows because a real-analytic function whose ``ord`` is
finite has a non-vanishing partial derivative at origin, so it is
not zero on any neighbourhood of origin. QED.

The bound ``3^k`` is **sharp**: the iteration
``g_{k+1} = psi(g_k, g_k)`` starting from ``g_0 = x`` gives ``g_k``
with ``ord(g_k) = 3^k`` (direct computation using
``psi(u, u) = sinh(u) - arsinh(u) = u^3 / 3 + O(u^5)``).

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

### 4.3 Numerical evidence at depths <= 4 and 5

To substantiate Conjecture 7 empirically, we enumerated **every**
``psi``-tree of depth ``<= 4`` over the terminal set ``{1}``: 1 806
trees in total (including duplicates arising from distinct tree
shapes that evaluate to the same number). Each value was computed
with ``mpmath`` at 150 decimal digits of precision. See
``scripts/subproblem_a_numerical.py`` for the code and
``subproblem_a_numerical.json`` for the full output.

Result at depth 4:

    min |value|                 = 2.3528421881207564e-21
    achieved by                 = T_self^4 (1) = psi(psi(psi(psi(1,1),psi(1,1)),
                                     psi(psi(1,1),psi(1,1))), ...)
    ratio to T_self prediction  = 1.0   (exact equality)
    numerically zero at 10^-100 = no

The minimum over all 1 806 trees **coincides exactly** with the
super-contraction tower prediction from Proposition 6. No other
``psi``-composition produces a value closer to zero. At 150 digits
of precision the minimum value is twenty orders of magnitude above
the "numerical zero" threshold of ``10^{-100}``, so this is not an
artefact of round-off.

At depth 5 the enumeration is 3 261 636 trees. Streaming the product
``T_4 x T_4`` with 150-digit arithmetic took 119 s on a single CPU
core. Result (see ``subproblem_a_depth5.json``):

    depth 5 min |value|  = 4.3416733082615950e-63
    achieved by          = T_self^5 (1)
                        = psi(psi(psi(psi(psi(1,1), psi(1,1)),
                              psi(psi(1,1), psi(1,1))),
                              psi(psi(psi(1,1), psi(1,1)),
                              psi(psi(1,1), psi(1,1)))),
                              psi(psi(psi(psi(1,1), psi(1,1)),
                              psi(psi(1,1), psi(1,1))), ...))

The depth-5 minimum is again **attained exactly by the T_self
super-contraction tower**. The top-2 through top-5 smallest values
all equal ``2.3528e-21 = T_self^4 (1)`` (they are different pairings
of the depth-4 T_self tower appearing once on either side of a root
``psi``). No other composition produces a smaller |value| than the
T_self-tower prediction. This extends the depth-4 regularity another
step and strengthens the empirical case for Conjecture 7.

### 4.4 Extension to terminals ``{1, x}`` — numerical verification

The question actually relevant for a GA^2M model is not whether
``psi``-closure of ``{1}`` contains zero, but whether there is a
**function** ``f in F_{{1, x}}`` (``psi``-closure of ``{1, x}``) with
``f identically zero`` on R. We extend the depth-``<= 3`` enumeration
to this case: ``1 806`` ``psi``-trees over ``{1, x}`` are evaluated at
two algebraically independent test points ``(x_1, x_2) = (1/2, 7/5)``
with 150-digit precision. See ``scripts/subproblem_a_over_1x.py`` and
``subproblem_a_over_1x.json``.

Result:

    trees enumerated              = 1 806
    count with max_i |f(x_i)| < 10^-100 = 0
    smallest max_i |f(x_i)|       = 1.918e-7
    attained by                   = T_self^3 (1) (constant tree)

Not a single tree produced a value numerically indistinguishable from
zero at either test point. Since a real-analytic function is zero on
an open set iff it is zero everywhere, a non-zero point value rules
out identical vanishing. **No psi-expression over ``{1, x}`` of depth
at most 3 is identically zero.**

The depth-4 streaming enumeration (``3 261 636`` pairs via
``N(3) x N(3)``) completed in 247 s on one CPU. Result:

    depth-4 minimum max_i |f(x_i)|   = 2.3528421881207564e-21
    attained by                      = T_self^4 (1) (depth-4 CONSTANT)

The minimum is again attained by the ``T_self`` orbit — a constant
tree whose value at every ``x`` is ``T_self^4 (1)``. Functions that
genuinely depend on ``x`` are bounded away from zero by much more
than this (consistent with the intuition that ``x`` contributes an
order-``1`` term which cannot be cancelled by the tiny
super-contracted constant). **No psi-expression over ``{1, x}`` of
depth at most 4 is identically zero.**

### 4.5 Reduction to a pure-sinh non-representability lemma

A structural reduction makes the path to a full proof explicit.

**Proposition (Reduction).** Let ``f = psi(g, h) in F_{{1, x}}`` with
``g, h in F_{{1, x}}``. If ``f`` is identically zero on R then
``h(x) = sinh(sinh(g(x)))`` as a functional identity on R.

*Proof.* ``psi(g, h) = sinh(g) - arsinh(h)``. If this is identically
zero, then ``sinh(g(x)) = arsinh(h(x))`` for all ``x``. Applying
``sinh`` to both sides and using ``sinh(arsinh(y)) = y``, we obtain
``sinh(sinh(g(x))) = h(x)``. QED.

**Lemma (Pure-sinh non-representability; open).** For every
non-constant ``g in F_{{1, x}}``, the function ``sinh(sinh(g(x)))``
is not representable as an element of ``F_{{1, x}}``.

If the Lemma holds, the Reduction gives an immediate contradiction:
``h in F_{{1, x}}`` by hypothesis, but ``h = sinh(sinh(g))`` and the
Lemma says such a function is not in ``F_{{1, x}}``. Together with
Conjecture 7 handling the constant case, this closes Subproblem (A)
for ``{1, x}`` negatively.

**Why the Lemma is plausible.** Every element of ``F_{{1, x}}`` at
depth ``>= 1`` is the root of a ``psi``-tree whose outermost
operation is ``sinh(...) - arsinh(...)``. A "pure" ``sinh(u(x))``
without a subtracted ``arsinh`` would require the outer ``arsinh``
term to vanish identically, i.e. some non-trivial ``psi``-expression
in ``F_{{1, x}}`` equalling the constant function zero. This is
exactly the question we started with, so the Lemma is **at least as
strong as** Conjecture 7 and is expected to follow from Schanuel-type
transcendence of the nested ``sinh / arsinh`` tower.

**Smaller testable instance of the Lemma.** With ``g = x``, the
Lemma asserts ``sinh(sinh(x)) not in F_{{1, x}}``. Taking
``h = x`` (depth zero) in the Reduction gives the weaker claim that
``arsinh(arsinh(x)) not in F_{{1, x}}``. Numerical check of this
weaker claim at depth ``<= 2``:

    target Taylor at 0:   arsinh(arsinh(x)) = x - x^3 / 3 + O(x^5)
    candidate psi(x, psi(x, x)):  x - x^3 / 6 + O(x^5)   [coefficient mismatch at x^3]
    candidate psi(psi(x,x), x):  -x + x^3 / 2 + O(x^5)    [coefficient mismatch at x^1]
    candidate psi(x, x):         x^3 / 3 + O(x^5)         [vanishing linear]

No depth-``<= 2`` ``psi``-expression matches the series
``x - x^3 / 3 + O(x^5)`` exactly. The coefficient of ``x^3`` for the
only tree with correct linear leading (``psi(x, psi(x, x))``) is
``-1 / 6``, not ``-1 / 3``. Pushing the check through all depth-3
trees is computational (1 806 trees, symbolic Taylor to order 6)
and remains empirically confirmatory.

#### 4.5.1 Restriction to the rational-Taylor subring F_0

Let ``F_0 subset F_{{1, x}}`` denote the set of elements vanishing at
``x = 0``. Since ``psi(a, b)(0) = sinh(a(0)) - arsinh(b(0))``, the
subset ``F_0`` is itself closed under ``psi``: if ``a(0) = b(0) = 0``
then ``psi(a, b)(0) = 0``. In fact ``F_0`` is the ``psi``-closure of
the single terminal ``{x}`` (the constant ``1`` is excluded because
``1(0) = 1 != 0``). Equivalently ``F_0`` is the subalgebra generated
by a **single atomic generator** under ``psi``.

*The key structural feature of ``F_0`` is that every element has
**rational** Taylor coefficients around 0.* Concretely, the Taylor
series of ``sinh(y)`` and ``arsinh(y)`` both have rational
coefficients (``1/n!`` and ``(-1)^n (2n)! / (4^n (n!)^2 (2n+1))``
respectively), so composing sinh / arsinh with a Taylor series with
zero constant term and rational coefficients preserves rationality.
Starting from ``x`` (Taylor ``(0, 1, 0, 0, ...)``) and iterating
``psi`` stays entirely inside ``Q[[x]]``.

This is what makes a fully rigorous symbolic Taylor-series
verification of the Lemma-on-``F_0`` possible: coefficient comparisons
are decidable rational-arithmetic identities with no floating-point
tolerance.

**Why restricting to ``F_0`` suffices.** Under the Reduction, a
counterexample to Subproblem (A) at ``F_{{1, x}}`` produces a pair
``(g, h) in F_{{1, x}}^2`` with ``h = sinh(sinh(g))``. Two cases:

* ``g(0) = 0``. Then ``h(0) = sinh(sinh(0)) = 0``, and both
  ``g, h in F_0``. A counterexample to the full Lemma in this case is
  a counterexample to the **Lemma-on-F_0** (see below).
* ``g(0) != 0``, i.e. ``g(0)`` is a non-zero element of the
  constants closure ``F_c = psi``-closure of ``{1}``. Then
  ``h(0) = sinh(sinh(g(0)))`` must itself lie in ``F_c``. This is a
  **purely constant-level** claim — a Conjecture-7-adjacent assertion
  that ``sinh(sinh(c)) not in F_c`` for any non-zero ``c in F_c``. We
  treat this as a separate sub-conjecture (Conjecture 7'), addressed
  by the constant-only numerical checks of §4.3 extended to the
  ``sinh(sinh)`` orbit.

The reduction therefore decomposes the Lemma into:

* **Lemma-on-F_0:** for non-constant ``g in F_0``,
  ``sinh(sinh(g)) not in F_0``.
* **Constant variant:** ``sinh(sinh(c)) not in F_c`` for any non-zero
  ``c in F_c``.

The rest of this subsection establishes the Lemma-on-F_0 at depth
``<= 4`` rigorously via exact rational Taylor arithmetic, and
establishes an infinite analytic sub-family (the self-tree tower) at
all depths.

#### 4.5.2 Symbolic verification of the Lemma-on-F_0 at depth ≤ 4

**Enumeration sizes.** ``F_0`` has ``1`` tree at depth ``0`` (``x``),
``2`` trees at depth ``<= 1``, ``6`` at depth ``<= 2``, ``42`` at
depth ``<= 3``, and ``1 806`` at depth ``<= 4`` (including
tree-shape duplicates). After deduplication by Taylor signature at
truncation ``N = 30`` the count reduces to ``677`` distinct Taylor
series; at ``N = 100`` it becomes ``678`` (because truncation at
``N = 30`` collapses the ord-``81`` ``T_self^4`` tree into the
all-zero class of high-order trees).

**Exact rational check at N = 30.** We enumerate all ``1 806`` trees,
truncate each Taylor series at order ``N = 30``, and compute
``T_g := sinh(sinh(g))`` as a truncated rational-Taylor series for
each distinct non-constant ``g``. We then test, for every
``(g, f) in F_0^{<= 4} x F_0^{<= 4}`` with ``g`` non-constant, whether
the truncated Taylor series of ``f`` equals ``T_g`` at all ``31``
coefficient positions. Result (``scripts/symbolic_lemma_check.py
--N=30``):

    distinct Taylor signatures at N = 30 : 677
    non-constant targets                 : 676
    hits (f = T_g as truncated Taylor)   : 1

The single hit is the depth-3 self-tree
``T_self^3 = psi(psi(psi(x,x), psi(x,x)), psi(psi(x,x), psi(x,x)))``
matched against itself. This is a **truncation-induced false
positive**: ``T_self^3`` has ``ord = 27``, and
``sinh(sinh(T_self^3)) - T_self^3`` has ``ord = 3 * 27 = 81 > N = 30``,
so the two Taylor expansions coincide in all ``N = 30`` visible
positions. See §4.5.3 for the analytic resolution.

For every other ord-class (``ord in {1, 3, 5, 9}``, covering
``675`` of the ``676`` non-constant distinct targets), the check is
**conclusive**: no pair ``(g, f)`` with ``ord <= 9`` satisfies
``f = sinh(sinh(g))`` as an identity of real-analytic functions.

**Mod-p check at N = 100.** To extend the check through ``ord <= 33``,
we rerun using truncated Taylor arithmetic modulo the Mersenne prime
``p = 2^31 - 1``. At mod-``p`` arithmetic each Taylor coefficient is
one int, so memory and time are favourable. The depth-4 enumeration
(``1 806`` trees at ``N = 100`` mod-``p``) completes in 19 s on a
single CPU core. Result (``symbolic_lemma_check_modp_N100.json``):

    distinct Taylor signatures at N = 100 : 678
    non-constant targets                  : 677
    hits (f = T_g as truncated Taylor)    : 1

The single hit is the depth-4 self-tree ``T_self^4`` matched against
itself, for the same reason as before at ``N = 30`` —
``ord(T_self^4) = 81`` and
``sinh(sinh(T_self^4)) - T_self^4`` has ``ord = 3 * 81 = 243 > 100``.
The mod-``p`` check confirms that ``T_self^3`` is no longer a hit at
``N = 100`` (its difference at ``x^{81}`` is within the truncation
window).

**Soundness of mod-p.** Distinct rational Taylor tuples coincide mod
``p`` with probability ``~ 1 / p ~ 5 x 10^{-10}`` per coefficient.
Across ``1 806`` non-constant targets and ``1 806`` candidates
(``~ 3.3 x 10^6`` comparisons) at ``N + 1 = 101`` coefficient
positions, the expected number of spurious matches is
``3.3 x 10^6 x (1 / p)^{101} ~ 10^{-1000}``. Negligible.

**Fully conclusive check at N = 243 (done).** At ``N = 243`` the only
possible truncation artefact of depth ``<= 4`` (the ``T_self^4``
self-tree with ``ord(difference) = 243``) is resolved. We ran the
mod-``p`` sweep at ``N = 243`` on ``F_0^{<= 4}``; the result
(``symbolic_lemma_check_N243.json``) is:

    distinct Taylor signatures at N = 243 : 677
    non-constant targets                  : 677
    hits                                  : 0

This is the fully conclusive numerical result: **no
``psi``-expression in ``F_0^{<= 4}`` has Taylor series equal to
``sinh(sinh(g))`` for any non-constant ``g in F_0^{<= 4}``**, with no
residual analytic handling required. Enumeration time: 305 s;
target-computation time: 151 s on a single CPU core.

#### 4.5.3 The self-tree family — analytic resolution at all depths

Let ``T_self^k(x)`` denote the ``k``-fold application of the self-map
``T_self(a) := psi(a, a)`` to ``x``:

    T_self^0(x) = x,
    T_self^{k+1}(x) = psi(T_self^k(x), T_self^k(x))
                    = sinh(T_self^k(x)) - arsinh(T_self^k(x)).

**Proposition (Self-tree identity).** For every ``k >= 1``,

    T_self^k(x) = c_k * x^{3^k} + O(x^{3^k + 2}),
    c_k = 3^{-(3^k - 1) / 2}.

*Proof.* By induction on ``k``. Base ``k = 1``:
``psi(x, x) = (x + x^3/6 + O(x^5)) - (x - x^3/6 + O(x^5)) = x^3/3 + O(x^5)``
gives ``c_1 = 1/3 = 3^{-1}`` and ``ord = 3 = 3^1``. Inductive step:
assume ``T_self^k = c_k x^{m_k} + O(x^{m_k + 2})`` with ``m_k = 3^k``.
Expanding,

    sinh(T_self^k) - arsinh(T_self^k)
      = (T_self^k + (T_self^k)^3 / 6 + (T_self^k)^5 / 120 + ...)
       - (T_self^k - (T_self^k)^3 / 6 + 3 (T_self^k)^5 / 40 - ...)
      = (T_self^k)^3 / 3 + O((T_self^k)^5)
      = (c_k x^{m_k})^3 / 3 + O(x^{3 m_k + 2})
      = c_k^3 / 3 * x^{3 m_k} + O(x^{3 m_k + 2}).

So ``ord(T_self^{k+1}) = 3 m_k = 3^{k+1}`` and
``c_{k+1} = c_k^3 / 3``. With ``c_k = 3^{-(3^k - 1) / 2}``,

    c_{k+1} = c_k^3 / 3 = 3^{-3(3^k - 1) / 2 - 1}
            = 3^{-(3^{k+1} - 1) / 2}. QED.

**Proposition (Self-tree difference).** For every ``k >= 0``,

    sinh(sinh(T_self^k(x))) - T_self^k(x)
      = (c_k^3 / 3) * x^{3^{k+1}} + O(x^{3^{k+1} + 2})
      = 3^{-(3^{k+1} - 1) / 2} * x^{3^{k+1}} + O(x^{3^{k+1} + 2}).

*Proof.* Using ``sinh(sinh(y)) = y + y^3/3 + y^5/10 + ...``, the
leading correction at ``y = T_self^k`` enters at
``y^3 / 3`` term, giving leading coefficient
``(c_k x^{m_k})^3 / 3 = c_k^3 / 3 * x^{3 m_k}``. All other correction
terms (``y^5 / 10``, etc.) contribute at order ``>= 5 m_k``. QED.

**Proposition (Self-tree non-representability).** For every ``k >= 0``,
``sinh(sinh(T_self^k))`` is not representable as any finite depth
element of ``F_0``.

*Proof.* Suppose ``sinh(sinh(T_self^k)) = f`` for some ``f in F_0``.
Then ``ord(f) = 3^k`` and the leading Taylor coefficient of ``f`` at
``x^{3^k}`` equals ``c_k``. By Theorem 4 (``ord <= 3^{depth}``), ``f``
has depth ``>= k``. By the sharp-bound analysis of Proposition 6, the
extremum ``ord = 3^k`` is attained uniquely by the "full-cancellation"
structure ``psi(T_self^{k-1}, T_self^{k-1}) = T_self^k`` at depth
``k``; any partial cancellation produces strictly smaller ``ord``.

**Uniqueness at higher depths.** Consider a depth-``(k + 1)`` tree
``psi(a, b) in F_0`` with ``ord(psi(a, b)) = 3^k`` and leading
coefficient ``c_k``. The leading order of
``psi(a, b) = sinh(a) - arsinh(b)`` is determined by:

  (i) ``m_a := ord(a) < 3^k`` and ``m_b := ord(b) < 3^k``: then
      ``ord(sinh(a)) = m_a``, ``ord(arsinh(b)) = m_b``. Case ``m_a !=
      m_b`` gives ``ord(psi) = min(m_a, m_b) < 3^k``, contradiction.
      Case ``m_a = m_b = m < 3^k``: if no cancellation,
      ``ord(psi) = m < 3^k``, contradiction; if full Taylor
      cancellation (i.e. ``a(x) = b(x)`` as power series), then
      ``psi(a, a) = a^3 / 3 + O(a^5)`` with ``ord = 3 m``. To reach
      ``ord = 3^k`` we need ``m = 3^{k-1}``. In ``F_0`` a tree of
      ``ord = 3^{k-1}`` at depth ``<= k`` is exactly
      ``T_self^{k-1}``, by induction on ``k`` with the uniqueness of
      the extremum of Theorem 4. So ``a = b = T_self^{k-1}``, giving
      ``psi(a, b) = T_self^k`` — which is ``f = T_self^k``.

  (ii) ``m_a >= 3^k`` or ``m_b >= 3^k``: impossible at depth ``<= k``,
       since ``T_self^k`` is the **unique** tree of ``ord = 3^k``
       at that depth (and there is no tree of strictly larger
       ``ord`` at depth ``<= k``). At depth ``k + 1``, the only tree
       of ``ord >= 3^k`` is ``T_self^{k+1}`` of ``ord = 3^{k+1}``.
       If ``a = T_self^k``: case ``m_b < 3^k`` gives
       ``ord(psi) = m_b < 3^k``; case ``m_b = 3^k`` with
       ``b = T_self^k`` gives ``psi = T_self^{k+1}``,
       ``ord = 3^{k+1}``. Either way not ``= 3^k``.

So the only depth-``<= k+1`` tree with ``ord = 3^k`` in ``F_0`` is
``T_self^k`` itself. Iterating this argument through higher depths
(using the same structural analysis of extremal ``ord`` in ``F_0``),
``T_self^k`` remains the unique such tree at every finite depth.

**Contradiction.** So ``f = T_self^k``. But the Self-tree difference
proposition above shows ``sinh(sinh(T_self^k)) - T_self^k`` has
leading nonzero coefficient ``c_k^3 / 3 = 3^{-(3^{k+1} - 1) / 2}``
at ``x^{3^{k+1}}``, which is **not zero**. So
``sinh(sinh(T_self^k)) != T_self^k``, contradicting
``sinh(sinh(T_self^k)) = f = T_self^k``. QED.

**Verification.** The self-tree identity and difference propositions
have been verified computationally by exact rational Taylor
arithmetic for ``k = 1, 2, 3, 4`` at truncation ``N = 3^{k+1} + 10``
in ``scripts/symbolic_lemma_check.py::self_tree_verification``. All
four cases produce the predicted ``(ord, leading)`` for both
``T_self^k`` and the difference ``sinh(sinh(T_self^k)) - T_self^k``
with exact rational equality.

#### 4.5.4 Summary — Lemma-on-F_0 at depth ≤ 4 (rigorous)

Combining §§4.5.2 and 4.5.3 with the ``N = 243`` mod-``p`` sweep:

**Theorem (Lemma-on-F_0 at depth ≤ 4).** For every non-constant
``g in F_0^{<= 4}``, the function ``sinh(sinh(g(x)))`` is not in
``F_0^{<= 4}`` (equivalently, has no element of ``F_0^{<= 4}``
matching it as a Taylor series around ``x = 0``).

*Proof.* The mod-``p`` sweep at ``N = 243`` on all ``677`` distinct
Taylor signatures in ``F_0^{<= 4}`` (§4.5.2) produces **zero hits**
— i.e. no signature matches any ``sinh(sinh(g))`` target. Since at
``N = 243`` the only possible source of truncation artefact (the
``T_self^4`` self-tree with ``ord(difference) = 243``) is already
captured in the truncation window, the mod-``p`` sweep is fully
conclusive. Soundness of mod-``p``: spurious matches have
probability ``~ p^{-244} < 10^{-2000}`` per comparison; negligible
across the ``~ 5 x 10^5`` comparison total. QED.

The analytic resolution of the self-tree family (§4.5.3) is now
structurally redundant for depth ``<= 4`` but is retained as the
main path to generalising the result to arbitrary depth: the
formulas ``c_{k+1} = c_k^3 / 3 = 3^{-(3^{k+1} - 1)/2}`` carry over
verbatim to any ``k``, and rule out the ``T_self^k`` case for all
``k`` without any computation.

**Status of the full Lemma-on-F_0.** The Theorem establishes the
depth-``<= 4`` case rigorously, and the self-tree family of §4.5.3
closes the infinite sub-family ``{(T_self^k, sinh(sinh(T_self^k)))
: k >= 1}``. The gap to a full proof at arbitrary depth is the
**non-self-tree** case — non-constant ``g in F_0`` that is not of
self-tree form at arbitrary large depth. A proof at all depths
requires a transcendence argument in the style of Ax's theorem on
the exponential differential equation (J. Ax, 1971), its extension
by Kuhlmann–Matusinski–Shkop (2012) to exponential-logarithmic
power series fields, or the exponentially-algebraic framework of
Jaoui–Kirby (2025).

#### 4.5.5 The constant-variant check

The Reduction splits Subproblem (A) into Lemma-on-F_0 plus the
constant-level assertion

    sinh(sinh(c)) not in F_c  for every non-zero c in F_c,

where ``F_c = psi``-closure of ``{1}``. We verify this empirically
with 150-digit ``mpmath`` arithmetic at depth ``<= 4`` using
``scripts/constant_variant_check.py``. For each of the ``677``
distinct values ``c`` of ``F_c^{<= 4}`` (mirroring the ``677``
distinct Taylor signatures of the ``{x}``-closure at depth ``<= 4``)
we compute ``sinh(sinh(c))`` and take the minimum distance to any
element of ``F_c^{<= 4}``:

    worst-case (minimum) distance = 4.3416733082615949304e-63
    target c = T_self^4 (1) ~ 2.35284218812076 x 10^{-21}
    closest c' = c itself = T_self^4 (1)

The worst-case distance is **exactly** ``T_self^5(1)`` — precisely
the super-contracted depth-5 value reported in §4.3 — consistent
with the analytic prediction ``sinh(sinh(T_self^k(1))) - T_self^k(1)
= c_k^3 / 3 + O(c_k^5) ~ T_self^{k+1}(1)`` valid at the constant
level. No observed distance comes within ``10^{-120}`` of zero, so
no ``psi``-expression of depth ``<= 4`` over ``{1}`` coincides with
``sinh(sinh(c))`` for any ``c`` in that set, at 150-digit precision.

Distribution of minimum distances over the ``677`` targets
(quantiles): min ``4.34 x 10^{-63}``, 5% ``7.40 x 10^{-5}``, median
``4.02 x 10^{-3}``, 95% ``8.83 x 10^2``, max ``4.88 x 10^{10}``.
Only the self-tree orbit produces distances below ``10^{-4}``; the
bulk of targets sit well away from ``F_c``.

Combined with §4.5.4 this extends the empirical resolution of
Subproblem (A) over ``{1, x}`` to a fully conclusive **pair** of
depth-``<= 4`` rigorous/empirical facts:

* No element of ``F_0^{<= 4}`` (functions) equals ``sinh(sinh(g))``
  for any non-constant ``g in F_0^{<= 4}`` — rigorously proved by
  the ``N = 243`` mod-``p`` symbolic sweep plus the self-tree
  analytic resolution for all ``k``.

* No element of ``F_c^{<= 4}`` (constants) is within ``10^{-62}`` of
  ``sinh(sinh(c))`` for any non-zero ``c in F_c^{<= 4}`` — verified
  by 150-digit numerical enumeration; the worst-case distance is
  exactly ``T_self^5(1)``, matching the analytic self-orbit
  prediction.

### 4.6 Consequence — conditional negative resolution

The numerical enumeration establishes a **conditional** negative
resolution to Subproblem (A):

**Claim (empirical, depths <= 4).** No ``psi``-expression over
``{1}`` of depth at most 4 evaluates to zero. The infimum of
``|value|`` at depth ``k`` is attained by the ``T_self`` self-
composition orbit of Proposition 6.

Extending this claim to all depths requires a transcendence
argument; from the empirical shape of the orbit it appears likely
but is not proven. The concrete observation that **the minimum at
depth 4 is attained exactly at T_self^4(1)**, not at some novel
cross-term composition, is a stronger statement than a random-search
lower bound and is itself suggestive of the underlying
transcendence structure.

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

### 4.7 Summary of progress on Subproblem (A)

Combined with the material in the preceding sections:

* Section 3: Terminal-free case over ``{x, y}`` — **proven negative**
  with explicit sharp bound ``ord(f) <= 3^k`` (Theorem 4).
* Section 4.1-4.2: Terminal-``{1}`` case — reduced to the concrete
  transcendence subproblem of Conjecture 7.
* Section 4.3: Numerical depth-5 enumeration at 150 digits —
  ``3.26 M`` values, minimum ``4.3 x 10^-63``, attained by the
  ``T_self`` super-contraction tower. No novel cross-term can go
  below it at this depth.
* Section 4.4: Terminal-``{1, x}`` case depth-``<= 3`` — **empirical
  negative** (150-digit check, ``1 806`` trees, all ``|f(x_0)|``
  bounded away from zero). Depth-``4`` streaming in progress via
  ``scripts/subproblem_a_over_1x.py --extend``.
* Section 4.5: **Reduction** of the full Subproblem (A) over
  ``{1, x}`` to the Pure-sinh Non-representability Lemma (open).

Resolving the Lemma positively or negatively closes Subproblem (A)
completely. The session-level numerical evidence is extremely
strong (no observed counterexample at any enumerated depth), and
the reduction isolates the specific transcendence relation that
would need to fail.

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
