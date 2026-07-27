# Why the depth wall exists — and why it is a mirage

This note gives the mechanism behind the two central empirical facts of
this repository:

1. **Gradient-based recovery of EML trees collapses at depth ≥ 3–5**
   (Odrzywołek 2026 reports 100% blind recovery at depth 2, ~25% at
   depths 3–4, <1% at depth 5, 0/448 at depth 6; our harder
   depth-critical targets give 0% for multi-start Adam at depth ≥ 3).
2. **The function space of EML trees is vastly smaller than its
   syntactic description** — small enough that *complete enumeration*
   replaces optimisation entirely (see `eml_gam/canonical_pool.py` and
   `depth_wall_results.json`).

Both facts have the same root cause: the composed-exponential dynamic
range of `eml(a, b) = exp(a) − log(b)` under the numerically necessary
clamps.

## 1. Setup

Every trainable implementation of EML trees — the original paper's
complex-valued torch nets and this repository's real-valued trees —
must clamp: `safe_eml` computes

    eml_c(a, b) = exp(clip(a, −C, C)) − log(max(b, ε)),   C = 10, ε = 1e−10.

Training selects each slot input as a softmax mixture over
`{1, x, f_child}`; at standard random initialisation (logits ~
N(0, 0.1²), temperature 1) the mixture weights are within a few per
cent of uniform.

## 2. Proposition 1 (exact gradient blocking)

*If, at the current parameters, the pre-clamp exp-argument of some
node exceeds `C` on every training point (or the log-argument is below
`ε` on every point), then every parameter whose influence on the
output flows only through that argument has **identically zero**
gradient.*

Proof: `d/dz clip(z)` is exactly 0 outside `[−C, C]`, and
`d/dz max(z, ε)` is exactly 0 below `ε`; apply the chain rule. ∎

This is qualitatively different from the familiar "vanishing
gradients" of deep sigmoids: no learning-rate schedule, no optimiser,
and no amount of patience recovers information through an exactly-zero
derivative. The slot choices below a saturated argument are invisible
to the loss.

## 3. Proposition 2 (mean-field saturation depth)

Under uniform mixing, every slot at level `k` (counting from the
leaves) carries approximately the mean of its options,

    m_k = (1 + x̄ + V_k) / 3,        m_0 = (1 + x̄) / 2,

where `x̄` is the mean input and `V_k` the typical node value at level
`k`, and node values follow

    V_{k+1} = exp(min(m_k, C)) − ln(max(m_k, ε)).

For `x̄ = 1.4` (inputs on `[0.3, 2.5]`) the trajectory is

    V₁ ≈ 3.1 → V₂ ≈ 5.7 → V₃ ≈ 14.2 → V₄ ≈ 242 → m₄ ≈ 82 ≫ C.

The recursion is doubly exponential and crosses the clamp at the
**fifth** composition level; every additional level multiplies the
surviving log-side sensitivity by `1/m_k` (the derivative of
`−log`), i.e. suppresses it doubly exponentially. Depths 1–4 are
(mean-field) clamp-free — matching the regime where blind recovery has
non-trivial success — and depth ≥ 5 is saturated, matching the <1%
observed. (Exact trajectory values: `saturation_results.json`,
`mean_field` block.)

## 4. Proposition 3 (the smooth sibling contracts)

The same recursion for `ψ(a, b) = sinh(a) − arsinh(b)`,

    W_{k+1} = sinh(m_k) − arsinh(m_k),   m_k = (1 + x̄ + W_k)/3,

has a **stable fixed point** `W* ≈ 0.19` (`m* ≈ 0.86` at `x̄ = 1.4`)
with contraction factor

    ρ = (cosh(m*) − (1 + m*²)^{−1/2}) / 3 ≈ 0.21 < 1.

ψ-activations neither explode nor saturate at init, and the per-level
gradient attenuation is a benign geometric factor — the measured
median bottom-level gradient of ψ trees decays smoothly by ≈ 0.4 per
level (`saturation_results.json`), the classic *trainable* vanishing
regime, not a wall. This is precisely the empirical cross-operator
landscape: ψ trains from random init at depths where EML is at 0%.

EML gradients, by contrast, are simultaneously **exploding and
dying**, in two regimes that follow the mean-field trajectory
(measured over 300 random inits per depth,
`saturation_results.json`):

* **Cliff regime (depths 3–7).** While `m_k` climbs toward the clamp,
  the exp side multiplies sensitivities: the median *maximum*
  bottom-logit gradient explodes from 6.8 at depth 2 to 2×10⁵ at
  depth 3 and 4×10⁷ at depth 4 — six orders of magnitude of curvature
  for a first-order optimiser to survive.
* **Plateau regime (depth ≥ 6).** Once saturation spreads, Proposition
  1 bites: the fraction of bottom slots receiving *exactly zero*
  gradient grows from 1% (depth 6) to 22% (depth 7) to 54% (depth 8),
  and the median surviving gradient collapses to 3×10⁻⁵.

A landscape made of cliffs and plateaus is the worst case for any
first-order method. ψ trees show neither pathology: max gradients stay
below 0.06 at every depth through 8, and no slot ever receives an
exactly-zero gradient.

## 5. The same mechanism collapses the function space

Saturation does not only block gradients — it *identifies functions*.
Any two subtrees whose exp-side inputs both exceed `C` everywhere
contribute the same `exp(C)` term; any subtree whose value drops below
`ε` on the log side is indistinguishable from the constant `ε`. Add
dead slots (a slot snapped to `1` or `x` disconnects its entire
subtree — only "live" pruned trees matter) and the syntactic snap
space collapses:

| depth | raw snaps | live trees | distinct functions (measured) |
|------:|----------:|-----------:|------------------------------:|
| 1     | 4         | 4          | 4                             |
| 2     | 144       | 36         | 32                            |
| 3     | 186 624   | 1 444      | 1 142                         |
| 4     | 3.13×10¹¹ | 2 090 916  | 613 593                       |

(Distinct-function counts are new entries per level on the reference
320-point grid; see `depth_wall_results.json → protocol →
pool_build_stats`. The raw→distinct ratio at depth 4 is ≈ 5×10⁵.)

The clamp's role is isolated by the ψ-operator control
(`psi_pool_stats.json`, same 256-point grid): ψ trees — no clamps —
show **zero** functional degeneracy at depth 3 (1,446 distinct
functions = all 1,444 live trees + 2 leaves) and retain 68% of live
trees as distinct functions at depth 4 (1,431,542 of 2,090,916),
versus EML's 29% (613,593). Saturation is not incidental to the
collapse; it *is* the collapse.

The practical consequence is the reversal at the heart of this
repository: **the correct response to the depth wall is not a better
optimiser but the observation that there is nothing left to optimise**
— at depth 4 the entire reachable function space fits in memory, and
at depth 5 the root node can be *inverted analytically* against it
(`lookup_root`), reducing exact recovery to a hash join.

## 6. Scope and honesty

* The mean-field argument is a heuristic (it tracks means, not
  distributions); its predictions (clamp-free through depth 4,
  saturation at 5) are validated empirically in
  `saturation_results.json`, not proven distributionally.
* Proposition 1 is exact but conditional on full saturation across the
  batch; partially saturated nodes retain (erratic) gradients.
* Function-space collapse counts are grid-relative (320 points,
  8-decimal rounding); all recovery claims built on them are
  re-verified on independent dense grids before being reported.
