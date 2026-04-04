"""Unit tests for EMLTree, EMLGAM, and the surrounding API."""

from __future__ import annotations

import numpy as np
import sympy as sp
import torch

from eml_gam import EMLGAM, EMLTree, TrainConfig, safe_eml, train_tree
from eml_gam.benchmarks.scientific import exponential_decay


def test_param_counts_univariate():
    """Univariate trees have 5 * 2**n - 6 free parameters."""
    for d in [1, 2, 3]:
        t = EMLTree(depth=d, n_inputs=1)
        assert t.n_params == 5 * 2 ** d - 6, (d, t.n_params)


def test_param_counts_bivariate():
    """Bivariate trees have 7 * 2**n - 8 free parameters."""
    for d in [1, 2, 3]:
        t = EMLTree(depth=d, n_inputs=2)
        assert t.n_params == 7 * 2 ** d - 8, (d, t.n_params)


def test_safe_eml_numeric():
    """safe_eml must return finite values for pathological inputs."""
    x = torch.tensor([0.0, 1.0, 100.0, -100.0], dtype=torch.float64)
    y = torch.tensor([1.0, 1.0, 1.0, 1e-20], dtype=torch.float64)
    out = safe_eml(x, y)
    assert torch.isfinite(out).all(), out


def test_manual_snap_reproduces_exp():
    """A manually-snapped depth-1 tree with choices (x, 1) must equal exp(x)."""
    tree = EMLTree(depth=1, n_inputs=1)
    with torch.no_grad():
        tree.level_logits[0][0, :] = torch.tensor([0.0, 10.0])
        tree.level_logits[0][1, :] = torch.tensor([10.0, 0.0])
    tree.snap()
    expr = tree.get_symbolic_expression(["x"])
    assert sp.simplify(expr - sp.exp(sp.Symbol("x"))) == 0, expr
    x = torch.tensor([[0.5]], dtype=torch.float64)
    assert torch.allclose(tree(x), torch.exp(torch.tensor([0.5], dtype=torch.float64)))


def test_depth1_recovery_exp_minus_log():
    """A depth-1 tree must recover ``exp(x1) - log(x2)`` to machine precision."""
    torch.manual_seed(42)
    n = 512
    x = torch.empty(n, 2, dtype=torch.float64)
    x[:, 0] = torch.empty(n, dtype=torch.float64).uniform_(-1.0, 1.0)
    x[:, 1] = torch.empty(n, dtype=torch.float64).uniform_(0.3, 2.5)
    y = torch.exp(x[:, 0]) - torch.log(x[:, 1])

    tree = EMLTree(depth=1, n_inputs=2)
    info = train_tree(
        tree, x, y, cfg=TrainConfig(n_epochs=1500, lr=5e-2, entropy_weight=1e-3)
    )
    assert info["final_mse"] < 1e-6, info
    with torch.no_grad():
        pred = tree(x)
    assert torch.allclose(pred, y, atol=1e-3, rtol=1e-3), (pred, y)


def test_emlgam_additive_recovery():
    """EMLGAM must numerically recover an additive EML target."""
    torch.manual_seed(7)
    np.random.seed(7)
    ds = exponential_decay(n_train=512, noise=0.0, seed=42)
    model = EMLGAM(
        n_features=1,
        univariate_depth=2,
        feature_names=ds.feature_names,
        standardize=False,
    )
    model.fit(
        ds.X_train, ds.y_train,
        cfg=TrainConfig(n_epochs=1500, lr=5e-2, entropy_weight=1e-3),
    )
    mse = float(np.mean((model.predict(ds.X_train) - ds.y_train) ** 2))
    assert mse < 1e-2, mse


def test_sklearn_api_fit_predict_numpy():
    """``fit`` / ``predict`` accept numpy arrays and return numpy arrays."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(128, 2))
    X[:, 1] = np.abs(X[:, 1]) + 0.5
    y = np.exp(X[:, 0]) - np.log(X[:, 1])
    model = EMLGAM(n_features=2, univariate_depth=1, standardize=False)
    model.fit(X, y, cfg=TrainConfig(n_epochs=600, lr=5e-2))
    pred = model.predict(X)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (128,)


def test_symbolic_expression_determinism():
    """Renaming the variable changes only the symbol name in the formula."""
    torch.manual_seed(0)
    tree = EMLTree(depth=2, n_inputs=1)
    tree.snap()
    e1 = tree.get_symbolic_expression(["x"])
    e2 = tree.get_symbolic_expression(["foo"])
    assert (
        str(e1).replace("x", "foo") == str(e2)
        or sp.simplify(e1.subs(sp.Symbol("x"), sp.Symbol("foo")) - e2) == 0
    )


def test_multi_start_runs_all():
    """multi-start reports per-run MSE and keeps the best snap."""
    torch.manual_seed(0)
    from eml_gam import train_with_multistart
    tree = EMLTree(depth=2, n_inputs=1)
    x = torch.linspace(0.5, 2.0, 128, dtype=torch.float64).unsqueeze(1)
    y = torch.log(x.squeeze())
    info = train_with_multistart(
        tree, x, y, n_starts=3,
        cfg=TrainConfig(n_epochs=400, lr=5e-2),
    )
    assert len(info["per_run"]) == 3
    assert info["final_mse"] == min(r["final_mse"] for r in info["per_run"])


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        print(f"== {fn.__name__} ==")
        fn()
        print("  ok")
    print(f"\nAll {len(fns)} tests passed.")
