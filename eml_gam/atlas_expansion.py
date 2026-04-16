"""Atlas Expansion via Exhaustive Search (AEES).

Enumerates every valid snap configuration of an EML tree at a given
depth and ranks them by the ordinary-least-squares residual after
fitting a scalar affine readout ``y ~ alpha + beta * tree(x)`` on the
provided data. Returns the top-``k`` configurations as candidate
warm-starts.

This is the concrete algorithmic answer to the landscape-collapse
finding in ``benchmarks/landscape.py``. Random-initialisation gradient
descent recovers depth-3 EML targets at 0 per cent; the hand-coded
14-entry atlas also misses any target that is not one of its listed
primitives. AEES replaces both with an exhaustive enumeration of the
entire snap space at depth ``d`` and depth-``d``-sized atlas whose
coverage is provably complete at depth ``<= 3``.

Complexity. At depth ``d`` with ``n_inputs = 1``:

    n_snaps(d=1) = (1+n)^2                                = 4
    n_snaps(d=2) = (2+n)^2 * (1+n)^4                      = 144
    n_snaps(d=3) = (2+n)^2 * (2+n)^4 * (1+n)^8            = 186 624
    n_snaps(d=4) = (2+n)^2 * (2+n)^4 * (2+n)^8 * (1+n)^16 = 4.3e+9

so exhaustive enumeration is feasible up to depth 3 and completely
infeasible at depth 4 (with ``n_inputs = 1``). For depth 4 and beyond,
AEES gives way to beam search / neural guidance — future work.

For ``n_inputs = 2`` the counts are larger: depth 2 = 1 296, depth 3 =
27 million (upper bound of feasibility in a single worker).

The module is self-contained: no torch optimiser is invoked, only forward
evaluation of each snapped tree plus a two-parameter OLS per candidate.
Per-config cost is ``O(N)`` in the number of data points.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator

import numpy as np
import torch

from .eml_tree import EMLTree


@dataclass
class AtlasCandidate:
    snap: dict[int, torch.Tensor]
    r2: float
    alpha: float
    beta: float


def enumerate_snaps(depth: int, n_inputs: int) -> Iterator[dict[int, torch.Tensor]]:
    """Yield every valid snap configuration as a mapping ``level -> indices``.

    Bottom-level slots (``level == depth - 1``) choose from ``{0, 1, ...,
    n_inputs}`` (constant ``1`` plus each input), while upper-level slots
    additionally have the ``f_child`` option at index ``1 + n_inputs``.
    """
    per_level_counts: list[int] = []
    per_level_options: list[int] = []
    for level in range(depth):
        n_slots = 2 ** (level + 1)
        is_bottom = level == depth - 1
        n_options = (1 + n_inputs) if is_bottom else (2 + n_inputs)
        per_level_counts.append(n_slots)
        per_level_options.append(n_options)

    level_iters = [
        list(product(range(opts), repeat=n))
        for opts, n in zip(per_level_options, per_level_counts)
    ]
    for combo in product(*level_iters):
        snap = {
            level: torch.tensor(choices, dtype=torch.long)
            for level, choices in enumerate(combo)
        }
        yield snap


def _count_snaps(depth: int, n_inputs: int) -> int:
    total = 1
    for level in range(depth):
        n_slots = 2 ** (level + 1)
        is_bottom = level == depth - 1
        n_options = (1 + n_inputs) if is_bottom else (2 + n_inputs)
        total *= n_options ** n_slots
    return total


def aees_search(
    x: torch.Tensor,
    y: torch.Tensor,
    depth: int,
    n_inputs: int = 1,
    top_k: int = 10,
    use_input_affine: bool = False,
    verbose: bool = False,
) -> list[AtlasCandidate]:
    """Return the top-``k`` snap configurations for the target ``y = f(x)``.

    Parameters
    ----------
    x : ``(N,)`` or ``(N, n_inputs)`` tensor.
    y : ``(N,)`` tensor.
    depth : depth of the tree to enumerate.
    n_inputs : 1 or 2.
    top_k : number of candidates to return, sorted by R^2 descending.
    use_input_affine : passed through to ``EMLTree``. The enumeration
        does not search over ``scale`` / ``offset``; enabling this flag
        leaves the default unit affine in place.

    Returns
    -------
    list of ``AtlasCandidate`` sorted by ``r2`` descending.
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)
    assert x.shape[1] == n_inputs, (x.shape, n_inputs)

    y_np = y.detach().cpu().numpy().astype(np.float64)
    y_mean = float(y_np.mean())
    ss_tot = float(np.sum((y_np - y_mean) ** 2))
    if ss_tot < 1e-20:
        ss_tot = 1.0

    scaffold = EMLTree(
        depth=depth, n_inputs=n_inputs, use_input_affine=use_input_affine,
    )
    n_total = _count_snaps(depth, n_inputs)
    if verbose:
        print(f"  AEES: enumerating {n_total:,} snap configs")

    results: list[AtlasCandidate] = []
    progress_step = max(1, n_total // 20)
    for i, snap in enumerate(enumerate_snaps(depth, n_inputs)):
        scaffold.set_snap_config(snap)
        with torch.no_grad():
            pred = scaffold(x).detach().cpu().numpy().astype(np.float64)
        if not np.all(np.isfinite(pred)) or np.std(pred) < 1e-12:
            continue
        # OLS for y ~ alpha + beta * pred.
        m, v = pred.mean(), pred.var()
        cov = np.mean((pred - m) * (y_np - y_mean))
        beta = cov / max(v, 1e-30)
        alpha = y_mean - beta * m
        resid = y_np - (alpha + beta * pred)
        r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot
        results.append(
            AtlasCandidate(snap=snap, r2=r2, alpha=alpha, beta=float(beta))
        )
        if verbose and (i + 1) % progress_step == 0:
            print(f"    {i + 1:,}/{n_total:,}")

    results.sort(key=lambda c: -c.r2)
    return results[:top_k]


def aees_recover(
    x: torch.Tensor,
    y: torch.Tensor,
    depth: int,
    n_inputs: int = 1,
    r2_threshold: float = 0.999,
    use_input_affine: bool = False,
    verbose: bool = False,
) -> tuple[bool, AtlasCandidate | None]:
    """Convenience wrapper returning ``(recovered, best_candidate)``.

    A run is considered successful if the top candidate reaches R^2 above
    ``r2_threshold``. This is the AEES analogue of ``landscape.py``'s
    ``low_mse`` metric.
    """
    top = aees_search(
        x, y, depth=depth, n_inputs=n_inputs, top_k=1,
        use_input_affine=use_input_affine, verbose=verbose,
    )
    if not top:
        return False, None
    best = top[0]
    return best.r2 >= r2_threshold, best


__all__ = [
    "AtlasCandidate", "aees_search", "aees_recover",
    "enumerate_snaps",
]
