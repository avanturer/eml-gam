"""Unit tests for the canonical pool, root inversion, and SafePoolGAM."""

import numpy as np
import pytest

from eml_gam.canonical_pool import (
    FunctionPool,
    nested_depth,
    sample_live_nested,
    syntactic_live_count,
    syntactic_snap_count,
)


@pytest.fixture(scope="module")
def pool3():
    g = np.random.default_rng(42)
    x = np.sort(g.uniform(0.3, 2.5, 96))
    p = FunctionPool(x, operator="eml")
    p.build(3)
    return p


def test_counts_match_known_values():
    assert syntactic_snap_count(2, 1) == 144
    assert syntactic_snap_count(3, 1) == 186_624
    assert syntactic_live_count(2, 1) == 36
    assert syntactic_live_count(3, 1) == 1_444
    assert syntactic_live_count(4, 1) == 2_090_916


def test_pool_level_sizes_monotone(pool3):
    # leaves + strictly growing distinct-function counts
    assert pool3.level_end[0] == 2
    assert pool3.level_end[1] > pool3.level_end[0]
    assert pool3.level_end[3] <= syntactic_live_count(3, 1) + 2 + 4 + 36
    # distinct never exceeds the cumulative live bound per level


def test_exact_recovery_depth2(pool3):
    x = pool3.x[:, 0]
    y = np.exp(x)  # eml(x, 1)
    res = pool3.solve(y, depth=2, top_k=4)
    assert res and res[0]["r2"] > 1 - 1e-12
    nested = res[0]["nested"]
    x_new = np.linspace(0.4, 2.4, 333)
    assert np.allclose(pool3.evaluate_nested(nested, x_new), np.exp(x_new))


def test_lookup_root_finds_random_depth4_target(pool3):
    rng = np.random.default_rng(3)
    for _ in range(5):
        nested = sample_live_nested(4, 1, rng)
        y = pool3.evaluate_nested(nested, pool3.x[:, 0])
        if not (np.all(np.isfinite(y)) and np.std(y) > 1e-8):
            continue
        pairs = pool3.lookup_root_pairs(y, 4, cap=5000)
        x_dense = np.linspace(0.3, 2.5, 501)
        y_dense = pool3.evaluate_nested(nested, x_dense)
        cache: dict = {}
        ok = False
        for ia, ib in pairs:
            h = pool3.evaluate_pair(ia, ib, x_dense, cache)
            if np.all(np.isfinite(h)) and np.max(np.abs(h - y_dense)) < 1e-6 * (
                1 + np.max(np.abs(y_dense))
            ):
                ok = True
                break
        assert ok, "verified lookup recovery failed on a depth-4 target"
        return
    pytest.skip("no non-degenerate target sampled")


def test_snap_config_roundtrip(pool3):
    """Pool expressions embed into EMLTree snaps and evaluate identically."""
    import torch

    from eml_gam.eml_tree import EMLTree

    rng = np.random.default_rng(11)
    nested = sample_live_nested(3, 1, rng)
    snap = pool3.nested_to_snap_config(nested, depth=3)
    tree = EMLTree(depth=3, n_inputs=1, use_input_affine=False)
    tree.set_snap_config(snap)
    x = np.linspace(0.4, 2.3, 77)
    with torch.no_grad():
        out_tree = tree(torch.tensor(x[:, None])).numpy()
    out_pool = pool3.evaluate_nested(nested, x)
    assert np.allclose(out_tree, out_pool, atol=1e-10)


def test_nested_depth_and_sampler():
    rng = np.random.default_rng(0)
    for d in (2, 3, 5):
        nested = sample_live_nested(d, 1, rng)
        assert nested_depth(nested) == d


def test_psi_pool_builds():
    g = np.random.default_rng(1)
    x = np.sort(g.uniform(0.3, 2.5, 64))
    p = FunctionPool(x, operator="psi")
    p.build(2)
    assert p.level_end[2] > p.level_end[1]


def test_safepoolgam_recovers_exp_decay_and_stays_bounded():
    from eml_gam.safe import SafePoolGAM

    rng = np.random.default_rng(0)
    n = 400
    X = np.column_stack([rng.uniform(0, 3, n), rng.uniform(-1, 1, n)])
    y = 4.0 * np.exp(-1.1 * X[:, 0]) + 0.3 * X[:, 1]
    tr = X[:, 0] < 2.0
    m = SafePoolGAM(depth=3, n_rounds=1, n_search=350)
    m.fit(X[tr], y[tr])
    pred = m.predict(X[~tr])
    ss = 1 - np.sum((y[~tr] - pred) ** 2) / np.sum(
        (y[~tr] - y[~tr].mean()) ** 2
    )
    assert ss > 0.5, f"extrapolation R2 too low: {ss}"
    # envelope: predictions are bounded regardless of input
    X_wild = np.array([[1e6, 0.0], [-1e6, 0.0]])
    p_wild = m.predict(X_wild)
    span = y[tr].max() - y[tr].min()
    assert np.all(np.abs(p_wild) < np.abs(y[tr]).max() + 25 * span)
