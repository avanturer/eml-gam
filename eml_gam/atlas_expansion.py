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


def enumerate_unbranched_snaps_univariate(depth: int) -> Iterator[dict[int, torch.Tensor]]:
    """Yield every "unbranched" snap configuration for a univariate tree.

    A snap is unbranched when at every upper level the active node uses
    exactly one ``f_child`` option — i.e. there is a single active path
    from the root down to the bottom. Formally, the snap is parameterised
    by:

    * ``path[l] in {0, 1}`` for ``l in [0, depth - 2]`` — which slot
      within the active node's pair at level ``l`` picks ``f_child``.
    * ``non_active[l] in {0, 1}`` for ``l in [0, depth - 2]`` — the
      content (``0`` = constant ``1``, ``1`` = ``x``) of the sibling of
      the active slot at level ``l``.
    * ``bot[0], bot[1] in {0, 1}`` — contents of the two slots in the
      active pair at the bottom level.

    All other slots are filled with ``0`` (constant ``1``) since they do
    not appear on the active path and therefore do not affect the tree
    output. The total count is

        4^depth         (2^{d-1} paths x 2^{d-1} non-active x 4 bottom)

    so enumeration is feasible through depth 10+. This is the scalable
    analogue of :func:`aees_search` for targets that are representable
    by a single nested function chain such as ``e - exp^{d - 2}(x)``
    from the landscape benchmark.
    """
    assert depth >= 1
    for path in product(range(2), repeat=max(depth - 1, 0)):
        for non_active in product(range(2), repeat=max(depth - 1, 0)):
            for bot in product(range(2), repeat=2):
                snap: dict[int, torch.Tensor] = {
                    l: torch.zeros(2 ** (l + 1), dtype=torch.long)
                    for l in range(depth)
                }
                active_node = 0
                for l in range(depth - 1):
                    p_l = path[l]
                    active_slot = 2 * active_node + p_l
                    non_active_slot = 2 * active_node + (1 - p_l)
                    snap[l][active_slot] = 2  # f_child
                    snap[l][non_active_slot] = non_active[l]
                    active_node = active_slot
                bot_l = depth - 1
                snap[bot_l][2 * active_node] = bot[0]
                snap[bot_l][2 * active_node + 1] = bot[1]
                yield snap


def _count_unbranched_snaps(depth: int) -> int:
    return 4 ** depth


def aees_search_unbranched(
    x: torch.Tensor,
    y: torch.Tensor,
    depth: int,
    top_k: int = 10,
    use_input_affine: bool = False,
    verbose: bool = False,
) -> list[AtlasCandidate]:
    """Run AEES restricted to unbranched snaps (scales to large depth).

    Uses :func:`enumerate_unbranched_snaps_univariate` as the enumeration
    source. The function evaluation, OLS fitting and ranking are identical
    to :func:`aees_search`. Total cost is ``O(4^depth * N)``.
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)
    assert x.shape[1] == 1, "unbranched AEES is univariate"

    y_np = y.detach().cpu().numpy().astype(np.float64)
    y_mean = float(y_np.mean())
    ss_tot = float(np.sum((y_np - y_mean) ** 2))
    if ss_tot < 1e-20:
        ss_tot = 1.0

    scaffold = EMLTree(
        depth=depth, n_inputs=1, use_input_affine=use_input_affine,
    )
    n_total = _count_unbranched_snaps(depth)
    if verbose:
        print(f"  unbranched AEES: enumerating {n_total:,} snaps")

    results: list[AtlasCandidate] = []
    for snap in enumerate_unbranched_snaps_univariate(depth):
        scaffold.set_snap_config(snap)
        with torch.no_grad():
            pred = scaffold(x).detach().cpu().numpy().astype(np.float64)
        if not np.all(np.isfinite(pred)) or np.std(pred) < 1e-12:
            continue
        m, v = pred.mean(), pred.var()
        cov = np.mean((pred - m) * (y_np - y_mean))
        beta = cov / max(v, 1e-30)
        alpha = y_mean - beta * m
        resid = y_np - (alpha + beta * pred)
        r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot
        results.append(
            AtlasCandidate(snap=snap, r2=r2, alpha=alpha, beta=float(beta))
        )
    results.sort(key=lambda c: -c.r2)
    return results[:top_k]


__all__ = [
    "AtlasCandidate", "aees_search", "aees_recover",
    "enumerate_snaps",
    "enumerate_unbranched_snaps_univariate",
    "aees_search_unbranched",
]
