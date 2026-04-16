"""Heuristics for selecting which feature pairs deserve a bivariate tree.

For ``p`` features there are ``p (p-1) / 2`` candidate pairs, which is
expensive for ``p >= 20``. Three strategies are implemented:

- ``correlation``: rank pairs by absolute Pearson correlation between the
  product ``x_i * x_j`` and the residual of a main-effects fit. O(p^2).
- ``mutual_info``: scikit-learn's mutual information between ``x_i * x_j``
  and the residual. O(p^2), slower than correlation but catches non-linear
  associations.
- ``greedy``: fit a small bivariate EML tree on every pair and rank by the
  MSE reduction it achieves on the residual. Most accurate, most expensive.
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np

from .utils import to_tensor

PairKey = Tuple[int, int]


def select_correlation(
    X: np.ndarray, residual: np.ndarray, top_k: int = 20,
) -> List[PairKey]:
    """Top-k pairs by ``|corr(x_i * x_j, residual)|``."""
    p = X.shape[1]
    scores: list[tuple[float, PairKey]] = []
    for i, j in combinations(range(p), 2):
        inter = X[:, i] * X[:, j]
        s = np.abs(np.corrcoef(inter, residual)[0, 1])
        scores.append((float(np.nan_to_num(s)), (i, j)))
    scores.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in scores[:top_k]]


def select_mutual_info(
    X: np.ndarray,
    residual: np.ndarray,
    top_k: int = 20,
    random_state: Optional[int] = 0,
) -> List[PairKey]:
    """Top-k pairs by mutual information between ``x_i * x_j`` and residual."""
    from sklearn.feature_selection import mutual_info_regression

    p = X.shape[1]
    scores: list[tuple[float, PairKey]] = []
    for i, j in combinations(range(p), 2):
        inter = (X[:, i] * X[:, j]).reshape(-1, 1)
        mi = mutual_info_regression(inter, residual, random_state=random_state)[0]
        scores.append((float(mi), (i, j)))
    scores.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in scores[:top_k]]


def select_greedy(
    X: np.ndarray,
    residual: np.ndarray,
    top_k: int = 10,
    depth: int = 2,
    n_epochs: int = 300,
    lr: float = 5e-2,
) -> List[PairKey]:
    """Top-k pairs by the MSE drop achieved by an independent bivariate EML
    tree fitted to the residual."""
    from .eml_tree import EMLTree
    from .train import TrainConfig, train_tree

    p = X.shape[1]
    base_mse = float(np.mean(residual ** 2))
    scores: list[tuple[float, PairKey]] = []
    cfg = TrainConfig(
        n_epochs=n_epochs,
        lr=lr,
        warmup_frac=0.7,
        hardening_frac=0.3,
        entropy_weight=1e-3,
        verbose=False,
    )
    for i, j in combinations(range(p), 2):
        x_pair = to_tensor(X[:, [i, j]])
        y_res = to_tensor(residual)
        tree = EMLTree(depth=depth, n_inputs=2)
        info = train_tree(tree, x_pair, y_res, cfg=cfg)
        reduction = base_mse - float(info["final_mse"])
        scores.append((reduction, (i, j)))
    scores.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in scores[:top_k]]


def select_pairs(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "correlation",
    top_k: int = 20,
    residual: Optional[np.ndarray] = None,
    **kwargs,
) -> List[PairKey]:
    """Unified entry point. If ``residual`` is not provided, ``y`` is used."""
    r = y if residual is None else residual
    if method == "correlation":
        return select_correlation(X, r, top_k=top_k)
    if method == "mutual_info":
        return select_mutual_info(X, r, top_k=top_k, **kwargs)
    if method == "greedy":
        return select_greedy(X, r, top_k=top_k, **kwargs)
    raise ValueError(f"unknown method: {method}")
