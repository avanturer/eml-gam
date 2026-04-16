# EML-GA²M

Interpretable regression where every component is a formula, not a black box.

Built on the EML operator `eml(x, y) = exp(x) - ln(y)` from [Odrzywołek 2026](https://arxiv.org/abs/2603.21852) — a single binary op that can express exp, log, powers, and their compositions. We wrap small EML trees into a GA²M additive structure (like EBM, but symbolic), train them end-to-end with gradient descent, and read out closed-form expressions at the end.

The main use case is **scientific tabular data** where the underlying relationship is exponential, logarithmic, or power-law. On such targets, this approach extrapolates correctly beyond the training range — something tree-based models (EBM, XGBoost) fundamentally cannot do.

## Why this exists

Tree-based regressors are great interpolators but they predict a flat constant outside their training range. If you train XGBoost on reaction rates at temperatures 350-450K and then ask it to predict at 600K, you get the same value as at 450K. That's not a bug — it's how trees work.

For scientific data with known functional structure (Arrhenius, power laws, exponential decay...), we want a model that:
- recovers the right functional form, not just fits a curve
- extrapolates correctly because the formula is correct, not because we got lucky
- gives you an actual equation you can write in a paper

EML-GA²M does this for targets expressible through exp/log compositions.

## Results

### Extrapolation benchmark (synthetic scientific targets)

Train on a narrow range, test outside. Positive R² on the extrapolation set means the model actually learned the right function.

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

Best on 4/8 targets, positive on 6/8, beats all tree baselines on 6/8. gplearn wins on 3 (it has a richer operator set — add, mul, sqrt, etc. — which helps on targets like Arrhenius where the affine reparameterisation is tricky).

Logistic growth is an honest loss: sigmoid needs addition inside a logarithm, which requires ~19 EML operations (way beyond depth-2). See [Limitations](#limitations).

### UCI Yacht Hydrodynamics

Residuary resistance vs. Froude number. Train on slow regime (Fr < 0.3), test on planing regime (Fr >= 0.3).

| Model | R² (extrap) | Time |
|-------|------------:|-----:|
| Linear | -0.82 | 0.0s |
| EBM | -1.07 | 1.7s |
| XGBoost | -1.07 | 0.1s |
| gplearn | -4.59 | 15.6s |
| **EML-GA²M** | **+0.66** | 2.2s |

Recovered formula: `0.005 * exp(13.47 * Fr) - 0.30`

This is the only model with positive R² on the extrapolation set. The exponential fits the steep rise of resistance with Froude number, which is consistent with the physics (Michell integral / power-law behaviour).

### Scalability (20 features)

Target: `y = 3*log(x1) + 2*exp(x2) - 4*x3 + 5*(x4/x5) + noise` with 15 noise features.

| Model | R² | Time |
|-------|---:|-----:|
| Linear | 0.955 | 0.0s |
| EBM | **0.998** | 14s |
| XGBoost | 0.971 | 0.6s |
| gplearn | 0.853 | 17s |
| EML-GA²M | 0.976 | 30s |

Pair selector correctly identifies (x4, x5) as the top interaction. EBM wins here because it's pure interpolation — no extrapolation advantage for symbolic models.

### Ablation

Each component matters — see `benchmarks/ablation.py`:
- Warm-start from primitive atlas: biggest single contributor (fixes convergence from ~25% to ~100%)
- Scale normalisation: needed for features with extreme magnitudes (Arrhenius 1/T ~ 0.002)
- Two-sided holdout + adaptive simplicity tolerance: fine-tunes symbolic form selection

## Installation

```bash
pip install torch sympy numpy scikit-learn

# for running baselines (optional)
pip install interpret xgboost gplearn pandas openpyxl
```

Or in editable mode:
```bash
pip install -e .
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
# something like: 0.11 + 9.12*exp(-0.42*t)

t_new = np.linspace(3.0, 8.0, 100)
y_pred = model.predict(t_new.reshape(-1, 1))  # extrapolates correctly
```

## How it works

### EML tree

A binary tree where every node computes `eml(left, right) = exp(left) - log(right)`. Each input slot chooses between `{1, x, f_child}` via a softmax during training. At the end, argmax snaps it to a hard choice and you read out a symbolic expression.

A depth-2 univariate tree has 14 trainable parameters. That's tiny — the whole point is that the operator itself is expressive enough.

### Training

1. **Warm-start** — fit a primitive atlas (hand-coded configurations for exp, log, 1/x, etc.) against the residual via OLS. This gets you in the right ballpark before gradient descent starts.
2. **Adam warm-up** — standard gradient descent with soft softmax (temperature = 1).
3. **Hardening** — anneal temperature from 1 to 0.01; the softmax sharpens toward one-hot.
4. **Snap** — take argmax at every slot, read out the symbolic formula.

### GA²M structure

Prediction: `y = bias + sum_i w_i * tree_i(x_i) + sum_{i<j} w_ij * tree_ij(x_i, x_j)`

Each tree also has a learnable input affine `x' = scale*x + offset` and an output weight, so the actual shape function is `w * tree(a*x + b)`. This absorbs multiplicative/additive constants without changing the symbolic structure.

## Project layout

```
eml_gam/
    eml_tree.py           # the differentiable EML tree
    gam.py                # EMLGAM model
    primitives.py         # primitive atlas and warm-start logic
    train.py              # training loop
    symbolic.py           # sympy helpers
    interaction_select.py # pair selection for bivariate components
    utils.py              # safe_eml, tensor conversion
    benchmarks/
        scientific.py     # synthetic targets (exp decay, Arrhenius, etc.)
        extrapolation.py  # main benchmark runner
        multiseed.py      # multi-seed runs with error bars
        ablation.py       # ablation study
        scalability.py    # 20-feature experiment
        real_world.py     # UCI datasets (Yacht, Auto-MPG, Concrete, Airfoil)
tests/
    test_eml_tree.py      # unit tests
scripts/
    download_datasets.py  # fetch UCI data
    save_results.py       # reproduce all benchmarks
data/                     # downloaded CSVs (gitignored)
```

## Reproducing

```bash
# download UCI datasets first
python -m scripts.download_datasets

# run benchmarks
python -m eml_gam.benchmarks.extrapolation
python -m eml_gam.benchmarks.ablation
python -m eml_gam.benchmarks.scalability
python -m eml_gam.benchmarks.real_world
python -m eml_gam.benchmarks.multiseed   # multi-seed with error bars

# save everything to results.json
python -m scripts.save_results

# tests
python -m pytest tests/ -v
```

## Limitations

- **Sigmoid is out.** `1/(1+exp(-x))` needs addition, which costs ~19 EML operations. Depth-2 trees (3 nodes) can't do it. Targets dominated by logistic/sigmoid structure won't work. This could be fixed with depth-3+ trees but convergence gets harder.
- **Not a general tabular model.** On something like Auto-MPG (mixed features, no dominant physical law), EBM and XGBoost are better. This is a tool for scientific data with known exp/log/power structure, not a replacement for gradient boosting.
- **Convergence isn't guaranteed.** Warm-start helps a lot, but some targets still need multi-start. The primitive atlas only covers depth-2 forms; deeper trees fall back to random init.

## References

- Odrzywołek, A. (2026). *All elementary functions from a single binary operator.* [arXiv:2603.21852](https://arxiv.org/abs/2603.21852)
- Lou et al. (2013). *Accurate intelligible models with pairwise interactions.* KDD.
- Cranmer, M. (2023). *Interpretable ML for science with PySR.* [arXiv:2305.01582](https://arxiv.org/abs/2305.01582)

## License

MIT. See [LICENSE](LICENSE).
