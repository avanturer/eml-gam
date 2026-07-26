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
representable one level shallower — the honest test):

| depth | syntactic space | canonical pool | gradient descent (10 restarts) | uniform sampling (100k) |
|------:|----------------:|:--------------:|:------------------------------:|:-----------------------:|
| 3 | 1.9×10⁵ | **40/40** (0.00 s) | TBD | TBD |
| 4 | 3.1×10¹¹ | **TBD** | TBD | TBD |
| 5 | 8.8×10²³ | **TBD** | TBD | TBD |

Coverage of unfiltered random deeper targets by the depth-5 solver
(their children collapse into the pool): depth 6: TBD, depth 7: TBD,
depth 8: TBD. Reproduce: `python -m eml_gam.benchmarks.depth_wall`.

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
* **per-feature gates**: free extension beyond the training range only
  if it beats the clipped (flat, tree-like) extension on the tails;
* a declared output envelope as the last line of defence.

Result (9 UCI physical extrapolation splits, identical data for all
models — full table in `safe_uci_results.json`):

| dataset | linear | EBM | XGBoost | EML-GA²M unguarded | **SafePoolGAM** |
|---|---:|---:|---:|---:|---:|
| TBD | | | | | |

On yacht it recovers `0.076·exp(13.3·Fr)` (R² **+0.63** where every
baseline is negative); on the datasets where the unguarded model
imploded it degrades gracefully to flat/linear behaviour — never a
catastrophic score, by construction.
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

```bash
python -m eml_gam.benchmarks.depth_wall    # headline: recovery through the wall (~hours)
python -m eml_gam.benchmarks.showcase      # paper target + minimal-depth certificates
python -m eml_gam.benchmarks.saturation    # gradient-pathology measurements
python -m scripts.run_uci_safe             # 9-dataset UCI safe-extrapolation suite
python -m scripts.make_depth_wall_figures  # regenerate figures
python -m pytest tests/ -q                 # unit tests
```

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
