"""Cross-operator landscape: does the depth-collapse appear in psi too?

The headline finding in ``benchmarks/landscape.py`` is that EML trees
fail to recover a depth-d target from random initialisation at d >= 3.
Odrzywolek's paper conjectures that this is intrinsic to the iterated
exp/log structure of the operator; the natural follow-up is to check
whether the *smooth* sibling ``psi(x, y) = sinh(x) - arsinh(y)`` shares
the same failure mode or, more interestingly, breaks it.

Target construction mirrors ``landscape.py``: at every depth d we use a
known-good snap which produces a non-degenerate function of the input.
We train both an EML and a Psi tree from random initialisation on the
same dataset and report the fraction of trials that drove the final MSE
below ``1e-3``. The question of exact-snap match is not meaningful
across operators (Psi has a different set of snap configurations); we
therefore report only the MSE-based success rate.

This experiment is the operator-comparative analogue of Odrzywolek's
Figure 4 / Table 2 and the landscape study of Section 2 of our README.
A clean answer in either direction is publishable: if Psi shows the
same collapse, the phenomenon is intrinsic to the iterated-Sheffer
structure and not to exp/log specifically; if Psi breaks the collapse,
it is a positive resolution to the open problem #1 of Odrzywolek (2026).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from ..eml_tree import EMLTree
from ..sheffer import PsiTree
from ..utils import DTYPE
from .landscape import target_snap_elog_iterated_exp


@dataclass
class CrossResult:
    operator: str
    depth: int
    n_trials: int
    n_low_mse: int
    success_rate: float


def _target_data(depth: int, n: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(0)
    if depth == 1:
        x_np = rng.uniform(0.2, 3.0, size=n)
    elif depth == 2:
        x_np = rng.uniform(-1.5, 1.5, size=n)
    elif depth == 3:
        x_np = rng.uniform(-2.0, 0.5, size=n)
    elif depth == 4:
        x_np = rng.uniform(-3.0, -0.3, size=n)
    else:
        x_np = rng.uniform(-3.0, -0.8, size=n)
    x = torch.tensor(x_np, dtype=DTYPE).unsqueeze(1)
    tree = EMLTree(depth=depth, n_inputs=1, use_input_affine=False)
    tree.set_snap_config(target_snap_elog_iterated_exp(depth))
    with torch.no_grad():
        y = tree(x)
    return x, y


def _train_random(
    tree_cls, depth: int, x: torch.Tensor, y: torch.Tensor,
    n_epochs: int = 1000, seed: int = 0,
) -> float:
    torch.manual_seed(seed)
    kwargs = {"depth": depth, "n_inputs": 1}
    if tree_cls is EMLTree:
        kwargs["use_input_affine"] = False
    tree = tree_cls(**kwargs)
    optim = torch.optim.Adam(tree.parameters(), lr=5e-2)
    for _ in range(n_epochs):
        optim.zero_grad()
        pred = tree(x)
        mse = torch.mean((pred - y) ** 2)
        if not torch.isfinite(mse):
            return float("inf")
        mse.backward()
        nn.utils.clip_grad_norm_(tree.parameters(), 1.0)
        optim.step()
    with torch.no_grad():
        if hasattr(tree, "snap"):
            tree.snap()
        pred = tree(x)
        if not torch.isfinite(pred).all():
            return float("inf")
        return float(torch.mean((pred - y) ** 2).item())


def run_cross_operator(
    depths: tuple[int, ...] = (2, 3, 4, 5),
    n_trials: int = 20,
    n_epochs: int = 1000,
    mse_threshold: float = 1e-3,
    verbose: bool = True,
) -> list[CrossResult]:
    out: list[CrossResult] = []
    if verbose:
        print(f"{'=' * 60}")
        print("  Cross-operator landscape (random init)")
        print(f"{'=' * 60}")
    for depth in depths:
        x, y = _target_data(depth)
        for op_name, tree_cls in (("eml", EMLTree), ("psi", PsiTree)):
            n_low = 0
            for trial in range(n_trials):
                final_mse = _train_random(
                    tree_cls, depth, x, y,
                    n_epochs=n_epochs, seed=trial * 31 + 7,
                )
                if np.isfinite(final_mse) and final_mse < mse_threshold:
                    n_low += 1
            result = CrossResult(
                operator=op_name, depth=depth,
                n_trials=n_trials, n_low_mse=n_low,
                success_rate=n_low / n_trials,
            )
            out.append(result)
            if verbose:
                print(
                    f"  depth={depth}  {op_name:3s}: "
                    f"success={n_low:2d}/{n_trials}  ({result.success_rate:.0%})"
                )
    return out


def save(results: list[CrossResult], path: str) -> None:
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


if __name__ == "__main__":
    results = run_cross_operator()
    save(results, "cross_operator_landscape.json")
