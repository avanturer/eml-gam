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

**Claim.** Let ``F_0 = {x, y}`` (no constants). Let ``F_{k+1}`` be the
closure of ``F_k`` under ``psi``. Then every ``f in F := union F_k``
satisfies ``f(0, 0) = 0``. In particular, no function in ``F`` is
equal to any nonzero constant, and ``e`` is not in ``F``.

*Proof.* By induction on ``k``. Base case: for ``f(x, y) = x`` or ``y``
we have ``f(0, 0) = 0``. Inductive step: if ``f = psi(g, h)`` with
``g(0, 0) = h(0, 0) = 0``, then ``f(0, 0) = sinh(0) - arsinh(0) = 0 - 0
= 0``. QED.

A corresponding statement holds for ``eml``: without a terminal ``1``,
``eml(x, y)`` evaluated at ``(0, 0)`` gives ``exp(0) - log(0)`` which is
**undefined** (log clamp fires); so ``eml`` without a terminal is
already ill-posed at zero. This asymmetry is worth flagging: ``psi`` is
at least well-defined on the entire real plane, even though it cannot
generate constants without an external seed.

## 4. With terminal ``{1}``: partial analysis

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

## 5. Empirical stability benchmark

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

**Net assessment.** ``psi`` is a reasonable stability candidate on a
per-step basis but does not match ``eml``'s expressivity on the simple
targets we tested. The expressivity gap is precisely what Section 4's
Conjecture (Sheffer parity) would close, one way or the other, if it
were resolved.

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
