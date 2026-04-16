"""Compare stability of the EML and smooth-Sheffer operators.

Two identically parameterised trees — one using ``eml(x, y) = exp(x) - ln(y)``
and the other using ``psi(x, y) = sinh(x) - arsinh(y)`` — are trained on the
same set of simple targets from random initialisation. We report:

* the fraction of trials that yielded a finite loss throughout training
  (a proxy for overflow robustness),
* the median final MSE,
* the fraction of trials that drove the loss below ``1e-3``.

The target set is deliberately small and narrow: ``exp``, ``e - log(x)``,
``exp(-x^2 / 2)`` evaluated on ``[-1, 1]`` or ``[0.3, 3]``. We are not
claiming expressivity parity between the two operators in finite depth
here — we are asking whether the smooth surrogate trains without the
clamp-induced pathologies that complicate deep EML trees. A clean
"yes" answer is sufficient evidence for pursuing ``psi`` in follow-up
work on the scaling wall problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..eml_tree import EMLTree
from ..sheffer import PsiTree
from ..utils import DTYPE


@dataclass
class OperatorResult:
    operator: str
    target: str
    depth: int
    n_trials: int
    n_finite: int
    n_low_mse: int
    median_final_mse: float


def _train_one(
    tree: nn.Module, x: torch.Tensor, y: torch.Tensor, n_epochs: int = 1500,
) -> dict:
    optim = torch.optim.Adam(tree.parameters(), lr=5e-2)
    history: list[float] = []
    nan_seen = False
    for epoch in range(n_epochs):
        optim.zero_grad()
        pred = tree(x)
        mse = torch.mean((pred - y) ** 2)
        if not torch.isfinite(mse):
            nan_seen = True
            break
        mse.backward()
        nn.utils.clip_grad_norm_(tree.parameters(), 1.0)
        optim.step()
        history.append(float(mse.item()))
    with torch.no_grad():
        if hasattr(tree, "snap"):
            tree.snap()
        pred = tree(x)
        final_mse = float(torch.mean((pred - y) ** 2).item()) if torch.isfinite(pred).all() else float("inf")
    return {"final_mse": final_mse, "nan_seen": nan_seen, "history": history}


def compare_operators(
    targets: dict[str, tuple],
    depths: tuple[int, ...] = (2, 3, 4),
    n_trials: int = 20,
    n_epochs: int = 1500,
    mse_threshold: float = 1e-3,
    verbose: bool = True,
) -> list[OperatorResult]:
    """Run parallel training of EML and Psi trees on the given targets.

    ``targets`` maps a name to ``(x_min, x_max, fn)`` where ``fn`` takes a
    numpy array and returns a numpy array.
    """
    out: list[OperatorResult] = []
    for target_name, (lo, hi, fn) in targets.items():
        x_np = np.linspace(lo, hi, 256)
        y_np = fn(x_np)
        x = torch.tensor(x_np, dtype=DTYPE).unsqueeze(1)
        y = torch.tensor(y_np, dtype=DTYPE)
        for depth in depths:
            for operator_name, tree_cls in (("eml", EMLTree), ("psi", PsiTree)):
                n_finite = 0
                n_low = 0
                finals: list[float] = []
                for trial in range(n_trials):
                    torch.manual_seed(trial * 31 + 7)
                    if operator_name == "eml":
                        tree = tree_cls(depth=depth, n_inputs=1, use_input_affine=False)
                    else:
                        tree = tree_cls(depth=depth, n_inputs=1)
                    info = _train_one(tree, x, y, n_epochs=n_epochs)
                    if np.isfinite(info["final_mse"]) and not info["nan_seen"]:
                        n_finite += 1
                    if np.isfinite(info["final_mse"]) and info["final_mse"] < mse_threshold:
                        n_low += 1
                    finals.append(info["final_mse"])
                finite_finals = [v for v in finals if np.isfinite(v)]
                median = float(np.median(finite_finals)) if finite_finals else float("inf")
                result = OperatorResult(
                    operator=operator_name, target=target_name, depth=depth,
                    n_trials=n_trials, n_finite=n_finite, n_low_mse=n_low,
                    median_final_mse=median,
                )
                out.append(result)
                if verbose:
                    print(
                        f"  {target_name:12s} d={depth} {operator_name:3s}: "
                        f"finite={n_finite:3d}/{n_trials}  "
                        f"low_mse={n_low:3d}/{n_trials}  "
                        f"median_mse={median:.3e}"
                    )
    return out


def default_targets() -> dict[str, tuple]:
    return {
        "exp":        (-1.0, 1.0, lambda x: np.exp(x)),
        "e_minuslog": (0.3, 3.0, lambda x: np.e - np.log(x)),
        "gauss":      (-1.5, 1.5, lambda x: np.exp(-x ** 2 / 2)),
    }


if __name__ == "__main__":
    compare_operators(default_targets())
