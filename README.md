# EML-GA²M / Canonical Pools — breaking the depth wall of single-operator symbolic regression

Research codebase built on the single binary operator
`eml(x, y) = exp(x) − ln(y)` from
[Odrzywołek 2026](https://arxiv.org/abs/2603.21852), which generates —
together with the constant 1 — every function on a scientific
calculator. That paper turns symbolic regression into training a single
uniform tree architecture, and reports the central obstruction, the
**depth wall**: blind gradient recovery of EML trees collapses from
100% at depth 2 to <1% at depth 5 and 0/448 at depth 6, while the
syntactic search space explodes to 3.1×10¹¹ configurations at depth 4.

**Headline result of this repository: the depth wall is a mirage.**
The *function space* of (real-clamped) EML trees is more than five
orders of magnitude smaller than its syntax. It can be **enumerated
completely** up to depth 4 in about a minute on a laptop CPU, queried
**exhaustively**, and inverted **analytically** at the root for depth-5
recovery in seconds — turning a hopeless stochastic search into a
verified database lookup, with certificates on both success and
failure.

## The five contributions

### 1. The function-space collapse, quantified

Two mechanisms shrink the space. *Dead slots*: a slot snapped to a
terminal disconnects its entire subtree — only `S(d−1)²` live trees
matter (`S(k) = 2 + S(k−1)²` univariate). *Clamp degeneracy*: the
numerically unavoidable clamps of `exp`/`log` identify huge families
of trees as the same function. Measured on a 320-point grid:

| depth | raw snap space | live trees | **distinct functions** |
|------:|---------------:|-----------:|-----------------------:|
| 1 | 4 | 4 | 4 |
| 2 | 144 | 36 | 32 |
| 3 | 186 624 | 1 444 | **1 142** |
| 4 | 3.13×10¹¹ | 2 090 916 | **613 593** |

At depth 4 the reachable function space is **0.0002%** of the
syntactic one. It fits in memory.

### 2. Canonical pools: complete enumeration + analytic root inversion

`eml_gam/canonical_pool.py` builds the complete deduplicated atlas of
every distinct slot function to depth 4 (~60 s, 64-bit hashed value
vectors, minimal expression DAGs). On top of it:

* **Exhaustive recovery at depth ≤ 4** — vectorised OLS over *every*
  representable function; completeness by construction.
* **Root inversion at depth 5** — from `y = exp(clip(a)) −
  ln(clamp(b))` the exp-side child is *implied*:
  `clip(a) = ln(y + ln(clamp(b)))`. Enumerate `b` over the pool, hash-match
  the implied vector, and symmetrically solve for `b` given `a`.
  Matches ranked by informativeness (side-canonical run length), then
  Occam size; O(pool) per pass.
* **Recursive peeling beyond** — enumerate a shallow root child, invert,
  recurse on the implied deep child.
* **Verification-gated everything** — a recovery only counts if it
  matches the target on an independent dense grid to 1e−6 relative
  error. Grid-R²=1.0 ties are common at depth ≥ 4 and are *rejected*.

Recovery on **depth-critical** random targets (provably not
representable one level shallower — the honest test; recovery counts
only if the expression matches the target on an independent dense grid
to 1e−6):

| depth | syntactic space | canonical pool | gradient descent (8 restarts) | uniform sampling (30k) |
|------:|----------------:|:--------------:|:-----------------------------:|:----------------------:|
| 3 | 1.9×10⁵ | **40/40** (0.00 s) | 1/8 (4 MSE-hits) | 6/8 |
| 4 | 3.1×10¹¹ | **40/40** (0.03 s) | 1/8 (1 MSE-hit) | 4/8 |
| 5 | 8.8×10²³ | **38/40** (28 s avg) | 0/8 (2 MSE-hits) | 1/8 |

Wilson 95% CIs: 40/40 → [91.2%, 100%]; 38/40 → [83.5%, 98.6%].
**With the full escalation ladder, depth-5 recovery is 40/40**: the 2
default-budget misses (whose correct pairs rank deep inside
clamp-degenerate match classes — diagnosed in
`depth5_miss_diagnostics.json`) fall to the last rung, *unranked
streamed verification*, which checks every generated pair with no caps
(164 s / 11.7M pairs and 557 s / 42.3M pairs;
`depth_wall_results.json → depths.5.pool_streamed`). Zero unexplained
failures remain at any depth 3–5.

Two honest footnotes. (i) GD's "MSE-hits" reach low training MSE with
the *wrong formula* — dense verification rejects them; this
MSE-good/formula-wrong gap is exactly why unverified recovery claims
mislead. (ii) Uniform sampling is *not* hopeless at depths 3–4: dead
slots give small live trees millions of syntactic realisations, so
blind sampling finds syntactically abundant targets — another face of
the same collapse. It dies at depth 5 (space 8.8×10²³). The fraction
of random deep trees that are *not* depth-critical is itself
measured: 21/66 random depth-5 trees collapse to shallower functions.

### Operator generality: the same machinery on ψ

The pool never looks inside the operator beyond node evaluation and
two side-canonicalisations, so the entire pipeline runs verbatim on
`ψ(x,y) = sinh(x) − arsinh(y)` — the *harder* enumeration case: with
no clamps there is no functional collapse (1,446 distinct = all live
trees at level 3; 1.43M at level 4, 2.3× EML's). Depth-critical
recovery (`psi_depth_wall_results.json`):

| depth | ψ pool recovery | mean time |
|------:|:---------------:|----------:|
| 3 | **20/20** | 0.00 s |
| 4 | **20/20** | 0.01 s |
| 5 | **20/20** | 39.5 s |

Zero targets collapsed to shallower functions (no degeneracy ⇒ every
random ψ tree is depth-critical), and inversion is exact
(`a = arsinh(y + arsinh b)`). Reproduce:
`python -m eml_gam.benchmarks.psi_depth_wall`.

Coverage of unfiltered random deeper targets by the depth-5 solver
(their children collapse functionally into the depth-4 pool):
**depth 6: 14/30 (47%), depth 7: 6/30 (20%), depth 8: 4/30 (13%)** —
the uncovered remainder is the honest residual frontier.
Reproduce: `python -m eml_gam.benchmarks.depth_wall`.

**Showcase.** The source paper's own bivariate ML target
`ln(e − ln(eˣ − ln y))` — depth 5, the family its >1000 gradient runs
plateaued on — is recovered **exactly, symbolically, in 10.7 s** from
256 samples (bivariate level-3 pool: 19 407 distinct functions, 1 s
build; the solver returned the equivalent form
`E − log(exp(E)/(E − log(eˣ − log y)))`, verified to 1e−8 on 2000
independent points). `python -m eml_gam.benchmarks.showcase`.

**Minimal-depth certificates.** Completeness turns failed searches
into grid-relative theorems for the real-clamped semantics: `ln x`
needs exactly depth 3 as a pure tree (certifying the paper's identity
`ln z = eml(1, eml(eml(1,z),1))` optimal) and depth 1 with an output
affine; `−x` and `1/x` need affine + depth 2; **no** tree of depth ≤ 4
matches `x²`, `√x`, `sinh x`, `cosh x` (their calculator constructions
genuinely require the complex domain / more depth).

### 3. Why the wall exists: cliffs, plateaus, and the same degeneracy

`docs/depth_wall_theory.md` + `eml_gam/benchmarks/saturation.py`. A
mean-field recursion for activations at random init crosses the exp
clamp at the **fifth** composition level
(V₁≈3.1 → V₂≈5.7 → V₃≈14 → V₄≈242 → m₄≈82 ≫ 10). Measured over 300
inits/depth, the gradient pathology has two regimes:

* **Cliffs (depths 3–7):** median *max* bottom-gradient explodes from
  6.8 (d2) to 2×10⁵ (d3) and 4×10⁷ (d4);
* **Plateaus (depth ≥ 6):** the clamp's derivative is *exactly zero*,
  so dead bottom slots grow from 1% (d6) to 22% (d7) to 54% (d8), and
  the median surviving gradient collapses to 3×10⁻⁵.

The smooth sibling `ψ(x,y) = sinh(x) − arsinh(y)` contracts to a
stable fixed point (W*≈0.19, ρ≈0.21): no dead slots through depth 8,
max gradients ≤ 0.06, benign geometric decay ≈0.4/level — the
mechanistic explanation of the earlier cross-operator landscape (ψ
trains at 100% at depths 2–3 where EML is at 0%). And the punchline:
the saturation that kills gradients is exactly what *identifies*
functions and makes enumeration feasible. **The wall's cause and its
cure are the same phenomenon.**

### 4. SafePoolGAM: certified safe extrapolation on real data

The earlier honest 9-dataset UCI sweep showed the unguarded EML-GA²M
winning 1/9 with catastrophic failures elsewhere (R² −8245 … −5×10³;
re-run: down to −2.7×10⁷). `eml_gam/safe.py` converts the atlas into a
deployable additive model:

* complete depth-4 pool sweep per feature **plus** calibrated
  classical families (`exp(k·u)` with refined rate, `u^k`, `ln u`);
* selection by R² on held-out **tail slices** (where extrapolation
  behaviour lives), not in-sample fit;
* two-sided tail validation + backward elimination on the
  extrapolation frontier (kills shapes that merely re-explain a
  stronger feature's variance);
* **per-feature gates, earned not granted**: every component —
  including plain linear terms — extends beyond the training range
  only if its free extension beats the flat (tree-like) extension by
  a margin on the feature's tails, judged with the full model;
* **frontier family selection**: the symbolic model competes against
  structurally-flat challengers (boxed-linear, constant) on the
  extrapolation frontier; challengers deploy only on a clear win, and
  raw unbounded linear extrapolation is never deployed;
* a declared output envelope as the last line of defence.

Result — 9 UCI physical extrapolation splits, identical data and
splits for all models (extrapolation R²; full details in
`safe_uci_results.json`):

| dataset | linear | EBM | XGBoost | EML-GA²M unguarded | **SafePoolGAM** |
|---|---:|---:|---:|---:|---:|
| yacht | −0.82 | −1.07 | −1.07 | −0.95 | **+0.63** |
| concrete | −40.42 | **+0.54** | +0.58 | −2.7×10⁷ | +0.11 |
| superconductivity | +0.64 | +0.65 | **+0.74** | −3.8×10⁵ | −0.00 |
| auto_mpg | −3.00 | **+0.44** | −0.13 | −757.6 | +0.19 |
| energy_eff | +0.91 | **+0.93** | +0.93 | −1.1×10⁵ | +0.73 |
| abalone | **+0.14** | −0.24 | −0.20 | −3.3×10⁴ | −0.07 |
| ccpp | **+0.29** | −1.57 | −1.67 | −745.4 | −0.62 |
| airfoil | **+0.27** | +0.13 | +0.11 | −7.8×10⁵ | −0.47 |
| forest_fires | −0.22 | **−0.02** | −0.77 | −170.5 | −0.03 |
| **worst case** | −40.4 | −1.6 | −1.7 | −2.7×10⁷ | **−0.6** |

The three claims this table supports, exactly as stated:

1. **Best worst-case of any model.** SafePoolGAM's worst score across
   all nine datasets is −0.62 — versus −1.6 (EBM), −1.7 (XGBoost),
   −40.4 (linear), and −2.7×10⁷ (its own unguarded predecessor, which
   is catastrophically last on every single dataset).
2. **The closed-form win survives the safety machinery.** Yacht:
   `0.076·exp(13.3·Fr)`, R² +0.63, where every baseline is negative —
   a genuine beyond-the-training-range extrapolation no tree model can
   produce.
3. **Graceful degradation is structural, not lucky.** Every deployed
   prediction path is either flat beyond the training box (tree-like)
   or individually certified on held-out tails; raw unbounded linear
   extrapolation is never deployed (that is how linear regression
   earns its −40.4).

Reproduce: `python -m scripts.run_uci_safe`.

### 5. Retained theory: ψ-expressivity and transcendence monotonicity

Unchanged from the earlier iterations and complementary to the new
mechanistic picture (proofs in `docs/`):

* finite-order-of-vanishing theorem with sharp bound `ord(f) ≤ 3^k`
  closing the terminal-free ψ case negatively;
* Pure-sinh Non-representability Lemma over `{1, x}` rigorous at depth
  ≤ 4 (exact rational Taylor arithmetic); the infinite self-tree
  family closed at all depths via `c_{k+1} = c_k³/3`;
* two **unconditional** transcendence-monotonicity theorems via
  Ax–Schanuel (witness family and arbitrary-seed orbit family), the
  universal case reduced to one combinatorial genericity condition,
  PSLQ-verified at 200 digits through depth 4.

## Figures

| | |
|---|---|
| `figures/function_space_collapse.png` | the 5-orders-of-magnitude collapse |
| `figures/recovery_vs_depth.png` | verified recovery through the wall |
| `figures/gradient_pathology.png` | explode-and-die (eml) vs contraction (ψ) |
| `figures/landscape_heatmap.png` | earlier landscape study |

## Installation

```bash
pip install -e .[benchmarks,dev]
# core: numpy, sympy, torch; benchmarks add scikit-learn, xgboost,
# interpret-core, pandas, matplotlib
```

## Quick start — recover a formula through the wall

```python
import numpy as np
from eml_gam.canonical_pool import FunctionPool

x = np.sort(np.random.default_rng(0).uniform(0.3, 2.5, 256))
pool = FunctionPool(x, operator="eml")
pool.build(3)                     # complete atlas to depth 3 (instant)

y = np.e - np.log(np.log(x) + np.e)   # some depth-3-ish law
for cand in pool.solve(y, depth=3, top_k=3):
    print(cand["r2"], pool.nested_to_sympy(cand["nested"], ["x"]))
```

And the safe applied model:

```python
from eml_gam.safe import SafePoolGAM
model = SafePoolGAM(depth=4).fit(X_train, y_train)
print(model.formulas(feature_names))   # gated closed forms
y_hat = model.predict(X_new)           # certified-safe extrapolation
```

## Reproducing everything

One command replays the entire evidence base (~6–9 h, CPU only):

```bash
bash scripts/reproduce_all.sh
```

or piecewise:

```bash
python -m eml_gam.benchmarks.depth_wall     # headline: recovery through the wall (~hours)
python -m scripts.depth5_rescue             # extended-budget pass on the depth-5 misses
python -m eml_gam.benchmarks.showcase       # paper target + minimal-depth certificates
python -m eml_gam.benchmarks.saturation     # gradient-pathology measurements
python -m eml_gam.benchmarks.psi_depth_wall # operator-generality check (ψ)
python -m scripts.run_uci_safe              # 9-dataset UCI safe-extrapolation suite
python -m scripts.make_depth_wall_figures   # regenerate figures
python -m pytest tests/ -q                  # unit tests
```

Claim-by-claim artifact map: [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
Venue plan and cover summary for submitting the manuscript:
[SUBMISSION.md](SUBMISSION.md). Compiled paper:
[paper/paper.pdf](paper/paper.pdf).

Earlier-generation experiments (landscape, cross-operator, AEES,
multiseed, Nguyen/Feynman, transcendence checks) remain reproducible
via their original entry points; see `docs/` and `paper/paper.tex`
for the full map.

## Honest scope

* All completeness/certificate claims are relative to (i) the
  **real-clamped semantics** (`safe_eml`: exp clipped at ±10, log
  clamped at 1e−10) — the class every trainable implementation
  actually optimises — and (ii) **grid identity with dense
  verification**. The complex-domain identities of the source paper
  (e.g. for −x) live outside this class, and we say so explicitly.
* Pools materialise to univariate depth 4 / bivariate depth 3 on a
  4-core, 15 GB machine. Depth-6+ recovery is by partial peeling with
  measured coverage, not completeness.
* SafePoolGAM is a main-effects model (no pairwise terms yet); its
  never-catastrophic behaviour comes from validated gating + envelope,
  and it converges to flat/linear behaviour when no closed form
  certifies — by design.
* The transcendence-monotonicity results concern ψ; the universal
  conjecture remains open.

## Project layout

```
eml_gam/
    canonical_pool.py       # THE core: pools, inversion, peeling, certificates
    safe.py                 # SafePoolGAM (tail-validated, gated, enveloped)
    eml_tree.py, gam.py     # differentiable trees + GA²M (earlier generation)
    sheffer.py              # ψ trees
    transcendence*.py       # atc invariant + witness verification
    neural_beam.py          # earlier neural PoC (kept for comparison)
    atlas_expansion.py      # earlier AEES (kept for comparison)
    benchmarks/
        depth_wall.py       # headline benchmark
        showcase.py         # paper target + certificates
        saturation.py       # gradient pathology
        (landscape, cross_operator_landscape, multiseed, ... earlier suites)
scripts/
    run_uci_safe.py         # 9-dataset UCI suite
    make_depth_wall_figures.py
docs/
    depth_wall_theory.md    # mechanism: mean-field, propositions, measurements
    theory.md, sheffer_analysis.md, transcendence_theorem.md
paper/paper.tex             # manuscript
```

## References

* Odrzywołek, A. (2026). *All elementary functions from a single
  operator.* [arXiv:2603.21852](https://arxiv.org/abs/2603.21852)
* Lou, Y., Caruana, R., Gehrke, J., Hooker, G. (2013). *Accurate
  intelligible models with pairwise interactions.* KDD.
* Cranmer, M. (2023). *Interpretable ML for science with PySR.*
* Petersen, B. et al. (2021). *Deep symbolic regression.* ICLR.
* Ax, J. (1971). *On Schanuel's conjectures.* Annals of Mathematics.

## License

MIT. See [LICENSE](LICENSE).
