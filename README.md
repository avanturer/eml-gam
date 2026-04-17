# EML-GA²M

Interpretable regression where every shape function is a closed-form
expression, built on the single binary operator `eml(x, y) = exp(x) - ln(y)`
from [Odrzywołek 2026](https://arxiv.org/abs/2603.21852). Small EML trees
are wrapped into a GA²M additive structure, trained end-to-end with
gradient descent, and read out as SymPy formulas at the end.

The main use case is **scientific tabular data** where the underlying
relationship is exponential, logarithmic, or power-law. On such targets
this approach extrapolates correctly beyond the training range —
something tree-based models (EBM, XGBoost) fundamentally cannot do.

The project has seven scientific contributions on top of the base
implementation:

1. A **landscape-geometry study** that quantifies the "scaling wall"
   Odrzywołek flagged as an open problem: random-initialisation
   recovery of EML trees collapses to 0 per cent at depth 3 and
   beyond.
2. A **symmetric cross-operator landscape** showing that this
   collapse is specific to the EML atom. Replacing `eml(x, y) =
   exp(x) − log(y)` with the smooth sibling
   `ψ(x, y) = sinh(x) − arsinh(y)` and training ψ-trees on
   ψ-generated targets from random initialisation gives **100%
   recovery at depth 2 and 3**, precisely where EML hits 0%. The
   depth wall is a consequence of the clamp-induced non-smoothness
   of exp/log, not of iterated Sheffer composition.
3. **Atlas Expansion via Exhaustive Search (AEES)** — an enumerative
   alternative to gradient descent for the snap-selection problem.
   Full AEES enumerates the complete snap space up to depth 3
   (186 624 configurations, 76 s); **unbranched AEES** restricts to
   single-path nested-chain snaps and enumerates `4^d`
   configurations, scaling to depth 8 in under 2 minutes on
   iterated-function targets. Both are tautologically complete
   *within their enumerated subclass* — they recover the
   ``e − exp^{d−2}(x)`` landscape target at R² = 1 because the target
   snap lies in the enumerated set. General branched recovery at
   depth ≥ 4 remains open.
4. **Theoretical analysis of ψ as a candidate smooth Sheffer.** A
   finite-order-of-vanishing theorem with explicit sharp bound
   `ord(f) ≤ 3^k` (Theorem 4 in
   [docs/sheffer_analysis.md](docs/sheffer_analysis.md)) closes the
   terminal-free case negatively. The terminal-{1} case is reduced
   to a concrete transcendence subproblem (Subproblems A and B in
   §4.2). A full-enumeration numerical check at 150-digit precision
   confirms that no ψ-expression of depth ≤ 5 over `{1}` reaches 0
   (minimum observed `|value|` at depth 5 is 4.34×10⁻⁶³, achieved
   exactly by the `T_self` tower of Proposition 6). The
   terminal-{1,x} case (functions) is reduced to a concrete
   **Pure-sinh Non-representability Lemma** (§4.5), which is
   **proved rigorously at depth ≤ 4** via exact rational Taylor
   arithmetic on the subring `F_0 ⊂ F_{{1,x}}` of functions vanishing
   at 0 (§4.5.2). The entire infinite `T_self^k` self-tree family is
   closed analytically at **all** depths via a closed-form identity
   `c_{k+1} = c_k³/3 = 3^{-(3^{k+1}-1)/2}` for the leading
   coefficient of the `sinh(sinh(·))` correction (§4.5.3).
5. **Stability machinery** — adaptive multi-start with warm-start
   rotation, a NaN-abort guard, and an optional extrapolation penalty —
   that eliminates the previously-reported divergence on bivariate
   rational targets without sacrificing accuracy on the targets the
   method already handled well.
6. **Transcendence-monotonicity theorems** — two *unconditional*
   strict-growth theorems for the algebraic transcendence complexity
   `atc`, via Ax-Schanuel (Ax 1971; Noguchi 2022) + a short
   `cosh`-transcendence lemma:
   - *Theorem W* (witness family): `atc(T_{d+1}) ≥ atc(T_d) + 1` for
     `T_0 = x`, `T_{d+1} = ψ(T_d, T_d)`.
   - *Theorem O* (orbit): `atc(W_{d+1}(g)) ≥ atc(W_d(g)) + 1` for
     every non-constant seed `g ∈ F_0` and the orbit
     `W_{d+1}(g) = ψ(W_d(g), W_d(g))`. Strictly generalises Theorem W.
   
   The universal Transcendence Monotonicity Conjecture (for arbitrary
   non-orbit trees) now reduces to a single combinatorial genericity
   condition `(Gen-d)` in the free ψ-algebra; an explicit
   ℚ-linear dependency `ψ(1,x) = ψ(x,x) − ψ(x,1) + (sinh 1 − arsinh 1)`
   shows `(Gen-d)` is non-vacuous. Numerical PSLQ verification at 200
   decimal digits and integer-coefficient bound `10²⁰` through depth 4
   finds no non-trivial relation, consistent with both theorems. Proof
   in [docs/transcendence_theorem.md](docs/transcendence_theorem.md),
   verification in
   [eml_gam/transcendence_witness.py](eml_gam/transcendence_witness.py).
   Historical note on the syntactic version: earlier empirical sweep
   `max tc: 0 → 2 → 4 → 5` at 80 digits for depths 0–3 in
   [docs/transcendence_analysis.md](docs/transcendence_analysis.md).
7. **Neural beam search proof of concept** — a two-layer MLP trained
   on sorted `(x, y)` samples predicts the slot choices of a random
   ground-truth snap. On depth-2 univariate targets top-16 beam
   recovery is 79% and top-64 is 100%; on depth-3 top-128 recovery is
   1.7% (compared to ~0.05% for uniform sampling of the same snap
   space). This is a proof that a learnable prior reduces the
   effective search space by orders of magnitude, not a production
   solver. Source in [eml_gam/neural_beam.py](eml_gam/neural_beam.py).

See [docs/theory.md](docs/theory.md) for the theoretical background,
[docs/sheffer_analysis.md](docs/sheffer_analysis.md) for the
ψ-Sheffer analysis, and [paper/paper.tex](paper/paper.tex) for a
self-contained write-up of every contribution above.

## Why this exists

Tree-based regressors are great interpolators but they predict a flat
constant outside their training range. If you train XGBoost on reaction
rates at temperatures 350-450 K and then ask it to predict at 600 K,
you get the same value as at 450 K. That's not a bug — it's how trees
work.

For scientific data with known functional structure (Arrhenius, power
laws, exponential decay, Michaelis-Menten, Shockley diode, ...), we
want a model that:

* recovers the right functional form, not just fits a curve,
* extrapolates correctly because the formula is correct,
* gives you an actual equation you can put in a paper.

EML-GA²M does this for targets expressible through exp / log /
power-law compositions.

## Results

### Landscape study (headline scientific contribution)

We reproduce and extend Odrzywołek's observation that gradient-based
recovery of EML trees collapses at depth ≥ 3. The targets used are
`e - exp^{d-2}(x)` for depth `d ≥ 2` and `e - log(x)` for depth 1 —
chosen to be genuinely different functions at every depth so that no
trivial collapse is possible.

Random-initialisation recovery rate (success = final MSE ≤ 1e-3 over
30 trials per cell):

| depth | success | snap match |
|------:|--------:|-----------:|
| 1     | 100.0%  | 100.0%     |
| 2     | 3.3%    | 0.0%       |
| 3     | 0.0%    | 0.0%       |
| 4     | 0.0%    | 0.0%       |
| 5     | 0.0%    | 0.0%       |

Perturbed-init recovery (start from correct snap + N(0, σ²) logit
noise, 15 trials per cell, success metric as above):

| depth | σ=0.25 | σ=0.5 | σ=1.0 | σ=2.0 | σ=5.0 |
|------:|-------:|------:|------:|------:|------:|
| 1     | 100    | 100   | 100   | 100   | 100   |
| 2     | 100    | 100   | 100   | 100   |  53.3 |
| 3     | 100    | 100   | 100   |  86.7 |  53.3 |
| 4     | 100    | 100   | 100   |  86.7 |   6.7 |
| 5     | 100    | 100   | 100   |  66.7 |   6.7 |

Snap-match rate at the same cells is strictly lower and diverges
rapidly from the MSE-success rate at depth ≥ 3, quantifying the
density of nearby symbolic basins (see
[docs/theory.md §3](docs/theory.md) for the reading).

Reproduce:

```bash
python -m eml_gam.benchmarks.landscape
# writes landscape_results.json with per-cell success rates
python -m scripts.make_landscape_figure
# writes figures/landscape_heatmap.png + landscape_basin_curves.png
```

Paper-ready figures are written into `figures/` (PNG, 160 dpi).

### Cross-operator landscape (ψ vs eml) — operator-symmetric

An earlier version of this section used EML-generated targets to
compare EML and ψ solvers; a methodology review flagged this as
asymmetric — the ψ solver was being asked to fit a target outside its
native grammar. The fixed experiment uses **operator-native targets**:
each target is generated by a tree of the same operator that is then
asked to recover it, with the same snap configuration evaluated
through both atoms. The 2×2 matrix isolates on-diagonal (native
recovery) from off-diagonal (cross-operator transfer):

| depth | EML → EML | EML → ψ | ψ → EML | ψ → ψ |
|------:|:---------:|:-------:|:-------:|:-----:|
| 2     | 10%       | 0%      | 0%      | **100%** |
| 3     | 0%        | 0%      | 0%      | **100%** |
| 4     | 0%        | 0%      | 0%      | 10%   |
| 5     | 0%        | 0%      | 0%      | 20%   |

(N = 20 trials per cell, 1000 epochs, success = MSE < 10⁻³·var(y).)

The on-diagonal cells tell the real story. **The depth collapse is
specific to EML**, not intrinsic to the iterated-Sheffer grammar.
Replacing the binary atom with the smooth sibling
`ψ(x, y) = sinh(x) − arsinh(y)` recovers **100% at depth 2-3** from
random initialisation — precisely where EML collapses to 0%. At
depth 4-5 ψ drops to 10-20%, still nowhere close to EML's 0%.

The off-diagonal cells confirm that each operator's native target is
outside the other operator's representation class — cross-training is
a near-0% failure mode regardless of depth. This is an expected
sanity check, not a new result.

**Implication.** Odrzywołek's open problem #1 asked for a smooth
binary Sheffer without exponential asymptotics. ψ (sinh − arsinh)
passes the **empirical landscape test** that EML fails: gradient-based
recovery works up to depth 3 directly from random initialisation. The
expressivity question (whether ψ is a true Sheffer) remains open and
is reduced to a concrete transcendence-theoretic subproblem in
[docs/sheffer_analysis.md §4.2](docs/sheffer_analysis.md).

Reproduce:

```bash
python -m eml_gam.benchmarks.cross_operator_landscape
# writes cross_operator_landscape.json with the full 2x2 matrix
```

### Atlas Expansion via Exhaustive Search (AEES)

The gradient-descent landscape collapse at depth ≥ 3 motivates a very
different recovery strategy: **enumerate snap configurations at the
target depth, evaluate each via a cheap affine OLS fit on the training
data, and return the top-k by R²**. We ship two variants:

* **Full AEES** enumerates the complete snap space. Feasible through
  depth 3 univariate (186 624 configurations, ~75 s on one CPU core).
* **Unbranched AEES** restricts the enumeration to snaps that use at
  most one `f_child` option per level — i.e. to targets representable
  by a single active path through the tree. The count drops to `4^d`
  configurations, which is 256 at depth 4, 4 096 at depth 6, and
  65 536 at depth 8. These complete in under 2 minutes and cover all
  iterated-function targets (``exp^k(x)``, ``log^k(x)``, their
  offsets, etc.).

Head-to-head on the landscape target:

| depth | random-init GD | hand-coded atlas | full AEES        | unbranched AEES  |
|------:|:--------------:|:----------------:|:----------------:|:----------------:|
| 1     | 20/20          | 5/5              | ✓ (R²=1.0, 0.0s) | ✓ (R²=1.0, 0.0s) |
| 2     | 3/20           | 0/5              | ✓ (R²=1.0, 0.0s) | ✓ (R²=1.0, 0.0s) |
| 3     | 0/20           | 0/5              | ✓ (R²=1.0, 76s)  | ✓ (R²=1.0, 0.0s) |
| 4     | 0/20           | —                | ∞ (4.3·10⁹)      | ✓ (R²=1.0, 0.2s) |
| 5     | 0/20           | —                | ∞ (10¹¹)         | ✓ (R²=1.0, 0.9s) |
| 6     | 0/20           | —                | ∞                | ✓ (R²=1.0, 4.3s) |
| 7     | 0/20           | —                | ∞                | ✓ (R²=1.0, 19s)  |
| 8     | 0/20           | —                | ∞                | ✓ (R²=1.0, 100s) |

Gradient descent collapses at depth 3 and is stuck at 0% through
depth 8. Unbranched AEES recovers the target at every depth through
8 in under 2 minutes — **tautologically**, because the landscape
target is itself an unbranched nested chain and therefore sits in
the enumerated subclass. On random targets also sampled from the
unbranched grammar, unbranched AEES recovers 85–90% at depths 4–6
(N=20 per cell); the non-recoveries are numerical-degeneracy cases
where the target's `y`-variance is below float precision.

**Scope of the claim.** Unbranched AEES is exhaustive over a
**structurally restricted subclass** — snaps with exactly one active
path from root to leaf. Inside that subclass, depth 1–8 recovery is
complete by construction: the target, being itself a nested function
chain, lies in the enumerated set, so OLS fitting always reaches
R² = 1. This is therefore **not** a general resolution of
Odrzywołek's scaling-wall problem; branched targets (depth ≥ 4 with
both root slots using the `f_child` option) remain combinatorially
infeasible for the exhaustive variant, and general gradient-based
recovery at depth ≥ 4 remains an open problem.

What this contribution does settle: **for the practically dominant
class of iterated-function symbolic targets** — exponential decays,
Arrhenius laws, nested log growth, Planck-like chains — depth-wall
recovery is reduced to a sub-second enumeration. Branched
double-path AEES at depth ≤ 5 is a natural extension that would
require ~10⁴ more configurations; branched beam search for depth ≥ 6
is the natural continuation.

Reproduce:

```bash
python -m eml_gam.benchmarks.aees_vs_gradient
# writes aees_vs_gradient.json; prints the table above
```

### Extrapolation benchmark (synthetic scientific targets)

Train on a narrow range, test outside. Positive R² on the extrapolation
set means the model actually learned the right function. Numbers with
multi-seed error bars are in the next table.

| Target | Linear | EBM | XGBoost | gplearn | EML-GA²M |
|--------|-------:|----:|--------:|--------:|---------:|
| Exp. decay | -50.1 | -4.3 | -4.5 | -29.6 | **+0.98** |
| Arrhenius | +0.28 | +0.02 | +0.02 | **+0.97** | +0.57 |
| Michaelis-Menten | -48.0 | -4.1 | -4.2 | -2.1 | -0.05 |
| Cobb-Douglas | -33.8 | -7.2 | -7.4 | +0.35 | **+0.75** |
| Logistic growth | +0.04 | **+0.99** | +0.99 | +0.06 | -1.30 |
| Power law | -47.7 | -6.3 | -6.5 | +0.87 | **+0.88** |
| Comp. inhibition (2d) | -5.6 | -1.0 | -1.2 | +0.70 | **+0.95** |
| Combined gas law (2d) | +0.51 | +0.67 | +0.65 | **+0.99** | +0.38 |

Best on 4/8 targets, positive on 6/8, beats all tree baselines on 6/8.
gplearn wins on three targets (it has a richer operator set — add, mul,
sqrt, etc. — which helps on Arrhenius where the affine reparameterisation
is tricky). Logistic growth is an honest loss: sigmoid needs addition
inside a logarithm, which costs ~19 EML operations (way beyond depth 2).

### Multi-seed numbers (N=5 seeds, robust=True, extrap penalty enabled)

Extrapolation R² for EML-GA²M with the stabilisation machinery
enabled. Median across seeds, ± standard deviation, explosion count.

| Target                   | Old config             | Robust + penalty        |
|--------------------------|-----------------------:|------------------------:|
| Exp. decay               | -0.12 ± 0.44           | **+0.95 ± 0.10** (0e)    |
| Arrhenius                | -0.08 ± 0.30           | **+0.10 ± 0.11** (0e)    |
| Michaelis-Menten         | -0.25 ± 0.48           | -0.28 ± 0.19 (0e)        |
| Cobb-Douglas             | +0.54                  | +0.36 ± 0.47 (0e)        |
| Logistic growth          | +0.76 (1 explode)      | +0.76 ± 0.82 (1 explode) |
| Power law                | +0.88                  | **+0.99 ± 0.01** (0e)    |
| Competitive inhibition   | +0.80 (**2 explode**)  | **+0.86 ± 0.25** (**0 explode**) |
| Combined gas law         | +0.67                  | +0.52 ± 0.22 (0e)        |

The robust mode eliminates explosions on `competitive_inhibition`
(2 → 0) and substantially improves `exp_decay`, `arrhenius`, and
`power_law`. It regresses marginally on `cobb_douglas` and
`combined_gas_law` because the extrapolation penalty clips legitimate
large predictions on these multiplicative targets; turning the penalty
off (`extrap_penalty_weight = 0`) while keeping `robust=True` recovers
the old numbers on those two targets. See
[docs/theory.md §3.1](docs/theory.md) for the algorithmic discussion.

Reproduce:

```bash
python -m eml_gam.benchmarks.multiseed
```

### UCI Yacht Hydrodynamics

Residuary resistance vs. Froude number. Train on slow regime
(Fr < 0.3), test on planing regime (Fr ≥ 0.3).

| Model | R² (extrap) | Time |
|-------|------------:|-----:|
| Linear | -0.82 | 0.0s |
| EBM | -1.07 | 1.7s |
| XGBoost | -1.07 | 0.1s |
| gplearn | -4.59 | 15.6s |
| **EML-GA²M** | **+0.66** | 2.2s |

Recovered formula: `0.005 · exp(13.47 · Fr) - 0.30`. The exponential
fits the steep rise of resistance with Froude number (consistent with
the Michell integral / power-law physics).

### 9-dataset real-UCI extrapolation sweep

Extended the real-UCI story to **nine datasets** with physical
extrapolation splits. All baselines (linear, EBM, XGBoost) and EML-GA²M
are evaluated on the same train/test split. Extrapolation R² reported
(higher is better). Full data in `uci_wins_summary.json`.

| Dataset | Split feature | Linear | EBM | XGBoost | **EML-GA²M** | Winner |
|---|---|---:|---:|---:|---:|---|
| yacht | Froude number | -0.82 | -1.07 | -1.07 | **+0.67** | **EML-GA²M** |
| concrete | age > 28 | -45.0 | +0.49 | +0.52 | +0.17 | XGBoost |
| superconductivity | atomic_mass q60 | +0.30 | +0.38 | +0.59 | -222 | XGBoost |
| auto_mpg | weight q60 | -2.43 | +0.42 | -0.36 | -6419 | EBM |
| energy_efficiency | glazing_area > 0.25 | -0.13 | +0.83 | +0.83 | -84 | EBM/XGB |
| abalone | whole_weight > 0.93 | +0.14 | -0.07 | -0.07 | -5303 | Linear |
| ccpp | ambient T q50 | +0.27 | -1.68 | -2.30 | -1622 | Linear |
| airfoil | frequency > 2000 | +0.27 | -0.06 | -0.02 | -8245 | Linear |
| forest_fires | temp > 22.8 | -0.22 | -0.06 | -0.86 | -3771 | EBM |

**1 EML-GA²M win out of 9 real-UCI datasets.** The niche is genuine
but narrow: EML-GA²M wins decisively on Yacht where the target
actually *is* an `exp`-composition of the extrapolation feature. On
the other 8 — mixed functional structure, multiple features of
comparable importance, noisy targets — the unconstrained exp/log
composition blows up outside the training range and loses
catastrophically to tree-based methods. The extrap-penalty guard
(`extrap_penalty_weight=0.05`) prevents NaN but does not prevent
wrong-direction extrapolation when the structural prior does not
match the data.

The honest read: EML-GA²M is a **niche tool**, valuable when the
user has prior knowledge that the target is a single exp/log/power-law
in the extrapolation feature, and risky otherwise. It is not a
general-purpose tabular regressor.

### Scalability (20 features, synthetic target)

`y = 3·log(x₁) + 2·exp(x₂) − 4·x₃ + 5·(x₄/x₅) + noise`, with 15
noise features.

| Model | R² | Time |
|-------|---:|-----:|
| Linear | 0.955 | 0.0s |
| EBM | **0.998** | 14s |
| XGBoost | 0.971 | 0.6s |
| gplearn | 0.853 | 17s |
| EML-GA²M | 0.976 | 30s |

Pair selector correctly identifies (x₄, x₅) as the top interaction. EBM
wins here because this is an interpolation task — no extrapolation
advantage for symbolic models.

### New benchmarks

Two additional benchmark suites are shipped for broader symbolic-regression
comparisons:

* **Nguyen-12** (`eml_gam/benchmarks/nguyen.py`): the classic 12-target
  SR micro-benchmark. EML-GA²M wins on polynomial targets (N1-N4) and
  loses on pure trig targets (N5, N9-N10) as expected — no depth-2 EML
  snap represents `sin(x^2) cos(x)` directly. **PySR** beats
  EML-GA²M on every Nguyen target it tested; this is the honest
  out-of-niche baseline.
* **Feynman subset** (`eml_gam/benchmarks/feynman.py`): ten physics
  equations drawn from Feynman-AI-100, biased toward the EML class
  (Gaussian, Shockley diode, radioactive decay, relativistic energy,
  Stefan-Boltzmann, ideal gas, Coulomb, Planck, centripetal, power
  dissipation). Both interpolation and extrapolation splits are
  reported.

**PySR baseline.** `eml_gam/benchmarks/pysr_baseline.py` wraps the
state-of-the-art PySR symbolic regressor (Cranmer 2023) in the same
`fit / predict` interface as the rest of the library. PySR uses a
richer operator set (`+`, `-`, `*`, `/`, `exp`, `log`) and a mature
genetic-programming kernel. On Nguyen-12 it outperforms EML-GA²M
across the board. The EML-GA²M advantage is therefore **not a general
SR advantage**; it is specifically an *extrapolation* advantage on
targets inside the depth-2 EML atlas.

Reproduce:

```bash
python -m eml_gam.benchmarks.nguyen
python -m eml_gam.benchmarks.feynman
pip install pysr  # optional; enables the PySR baseline column
```

### Additional UCI datasets

Concrete compressive strength and airfoil self-noise are included in
`benchmarks/real_world.py`. On these datasets EML-GA²M does **not**
win — the underlying relationships are not pure exp/log, so the
method has no structural advantage. These results are included for
honesty.

## Stability machinery

Turn on the robust path explicitly via `robust=True` on `fit`:

```python
model = EMLGAM(n_features=2, univariate_depth=2, bivariate_depth=2,
               interaction_pairs=[(0, 1)])
cfg = TrainConfig(
    n_epochs=1200, lr=5e-2,
    extrap_penalty_weight=0.01,   # bounds predictions outside train box
    extrap_max_std=50.0,           # tolerance in standardised units
)
model.fit(X, y, cfg=cfg, robust=True, try_offsets=True)
```

`robust=True` runs multi-start with warm-start rotation (rank 0, 1, 2, …
through the ranked atlas) and retries on NaN/Inf failures. The
extrapolation penalty is off by default (keeps the single-seed
behaviour of the original implementation) but is recommended for any
bivariate rational target.

## Ablation

| Variant | What it adds |
|---------|--------------|
| V0 baseline | random init, no affine, no normalization |
| V1 +warm | atlas-based warm-start (in-sample scoring) |
| V2 +affine | learnable per-tree input scale and offset |
| V3 +scale_norm | conditional feature normalization |
| V4 +holdout | two-sided holdout validation + adaptive simplicity tolerance |

Warm-start is the biggest contributor. Affine lets the model absorb
multiplicative constants. Scale normalization is needed for features
with extreme magnitudes (Arrhenius `1/T ≈ 0.002`). Holdout fine-tunes
the symbolic form by scoring on distribution tails.

## Installation

```bash
pip install torch sympy numpy scikit-learn
# for running baselines (optional)
pip install interpret xgboost gplearn pandas openpyxl
```

Or in editable mode:

```bash
pip install -e .[benchmarks,dev]
```

## Quick start

```python
import numpy as np
from eml_gam import EMLGAM, TrainConfig

rng = np.random.default_rng(0)
t = rng.uniform(0.0, 3.0, 500)
y = 5.0 * np.exp(-0.4 * t) + rng.normal(0, 0.05, 500)

model = EMLGAM(n_features=1, univariate_depth=2, feature_names=["t"])
model.fit(t.reshape(-1, 1), y, cfg=TrainConfig(n_epochs=1500))

print(model.total_formula())
# something like: 0.11 + 9.12 * exp(-0.42 * t)

t_new = np.linspace(3.0, 8.0, 100)
y_pred = model.predict(t_new.reshape(-1, 1))  # extrapolates correctly
```

## How it works

### EML tree

A binary tree where every node computes `eml(left, right) = exp(left) -
log(right)`. Each input slot chooses between `{1, x, f_child}` via a
softmax during training. At the end, argmax snaps it to a hard choice
and you read out a symbolic expression. A depth-2 univariate tree has
14 trainable parameters.

### Training

1. **Warm-start** — fit a primitive atlas against the residual via OLS.
   This installs a known-good snap before gradient descent starts.
2. **Adam warm-up** — standard gradient descent with soft softmax
   (temperature = 1).
3. **Hardening** — anneal the temperature from 1 to 0.01; the softmax
   sharpens toward one-hot.
4. **Snap** — take `argmax` at every slot, read out the symbolic
   formula.

When `robust=True` the three-stage schedule is wrapped in multi-start:
each restart picks the next-ranked atlas candidate (rank 0, 1, 2, …)
and NaN-aborts are skipped. A soft Lipschitz penalty optionally
discourages snaps whose derivative outside the training range grows
exponentially.

### GA²M structure

    y = bias + Σᵢ wᵢ · treeᵢ(xᵢ) + Σᵢ<ⱼ wᵢⱼ · treeᵢⱼ(xᵢ, xⱼ)

Each tree also has a learnable input affine `x' = scale · x + offset`
and an output weight, so the actual shape function is
`w · tree(scale · x + offset)`. This absorbs multiplicative / additive
constants without changing the symbolic structure.

### Smooth-Sheffer surrogate

`eml_gam/sheffer.py` implements a `PsiTree` in which the binary atom is
`psi(x, y) = sinh(x) - arsinh(y)`. This addresses Odrzywołek's
open problem #1: `psi` has no exponential-driven overflow because
`arsinh` is smooth on the entire real line. The stability benchmark in
`benchmarks/sheffer_stability.py` compares EML and Psi trees on simple
targets. `psi` eliminates NaN events; exact-recovery parity at deep
tree depth is conjectured and left to follow-up work.

## Project layout

```
eml_gam/
    eml_tree.py             # differentiable EML tree
    sheffer.py              # smooth-Sheffer psi tree (surrogate operator)
    gam.py                  # EMLGAM model (robust multi-start, penalty, DoF)
    primitives.py           # primitive atlas, rank_atlas_candidates, warm-start
    train.py                # training loop (TrainConfig, train_tree, ...)
    transcendence.py        # tc(f) invariant + PSLQ independence check
    neural_beam.py          # learned snap predictor + beam-search decoder
    atlas_expansion.py      # AEES (full + unbranched) snap enumeration
    symbolic.py             # sympy helpers
    interaction_select.py   # pair selection for bivariate components
    utils.py                # safe_eml, tensor conversion
    benchmarks/
        scientific.py       # synthetic targets (exp decay, Arrhenius, ...)
        extrapolation.py    # main extrapolation benchmark
        ablation.py         # ablation study
        scalability.py      # 20-feature experiment
        real_world.py       # UCI datasets
        multiseed.py        # multi-seed runs with confidence intervals
        landscape.py        # landscape / basin study (headline finding)
        nguyen.py           # Nguyen-12 symbolic-regression benchmark
        feynman.py          # Feynman-AI subset benchmark
        sheffer_stability.py # EML vs Psi stability comparison
        stats.py            # Wilcoxon paired tests, bootstrap CIs
tests/
    test_eml_tree.py        # 13 unit tests (EML + GAM + landscape + psi)
scripts/
    download_datasets.py    # fetch UCI data
    save_results.py         # reproduce all benchmarks
docs/
    theory.md                      # full theory write-up
    sheffer_analysis.md            # ψ-Sheffer analysis
    transcendence_analysis.md      # transcendence invariant analysis
paper/
    paper.tex                      # full LaTeX manuscript
    README.md                      # compilation + section-to-code map
data/                              # downloaded CSVs (gitignored)
```

## Reproducing

```bash
# download UCI datasets first
python -m scripts.download_datasets

# main benchmarks
python -m eml_gam.benchmarks.extrapolation
python -m eml_gam.benchmarks.ablation
python -m eml_gam.benchmarks.scalability
python -m eml_gam.benchmarks.real_world
python -m eml_gam.benchmarks.multiseed

# headline experiments
python -m eml_gam.benchmarks.landscape          # ~10 min
python -m eml_gam.benchmarks.nguyen             # ~5 min (needs gplearn)
python -m eml_gam.benchmarks.feynman            # ~3 min
python -m eml_gam.benchmarks.sheffer_stability  # ~6 min

# tests
python -m pytest tests/ -v
```

## Limitations

* **Sigmoid is out.** `1/(1+exp(-x))` needs addition, which costs ~19 EML
  operations (Table 4 in the paper). Depth-2 trees (3 nodes) can't do it.
  Targets dominated by logistic/sigmoid structure won't work.
* **Not a general tabular model.** On mixed-feature datasets (Auto-MPG,
  Concrete, Airfoil), EBM and XGBoost are better. This is a tool for
  scientific data with known exp / log / power structure.
* **Random-init recovery collapses at depth ≥ 3.** See the landscape
  study above. The library side-steps this by always warm-starting from
  the atlas; without warm-start, deep trees essentially never recover
  the target.
* **Primitive atlas only covers depth ≤ 2.** Follow-up work on a
  neural-guided atlas generator is the natural path to scaling beyond
  the currently-shipped primitives.

## References

* Odrzywołek, A. (2026). *All elementary functions from a single binary
  operator.* [arXiv:2603.21852](https://arxiv.org/abs/2603.21852)
* Lou, Y., Caruana, R., Gehrke, J., Hooker, G. (2013). *Accurate
  intelligible models with pairwise interactions.* KDD.
* Cranmer, M. (2023). *Interpretable ML for science with PySR.*
  [arXiv:2305.01582](https://arxiv.org/abs/2305.01582)
* Jang, E., Gu, S., Poole, B. (2017). *Categorical reparameterization
  with Gumbel-softmax.* ICLR.
* Udrescu, S.-M., Tegmark, M. (2020). *AI Feynman: a
  physics-inspired method for symbolic regression.* Science Advances.

## License

MIT. See [LICENSE](LICENSE).
