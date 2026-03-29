"""Training loop for standalone EML trees.

Three stages, following Odrzywołek (2026), Section 4.3:

1. Warm-up — Adam on a soft softmax at temperature 1.
2. Hardening — exponential annealing of the temperature toward 0.01.
3. Snap — ``argmax`` at every slot and symbolic read-out.

A multi-start wrapper is provided for cases in which random initialisation
does not reliably converge (depth-3 trees in the paper's reference
experiments report a 25 per cent success rate).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from .eml_tree import EMLTree
from .utils import DTYPE, to_tensor


@dataclass
class TrainConfig:
    n_epochs: int = 2000
    lr: float = 5e-2
    batch_size: Optional[int] = None
    warmup_frac: float = 0.7
    hardening_frac: float = 0.25
    temp_start: float = 1.0
    temp_end: float = 1e-2
    entropy_weight: float = 1e-3
    grad_clip: float = 1.0
    patience: int = 200
    verbose: bool = False


def _loss_fn(
    module: nn.Module, x: torch.Tensor, y: torch.Tensor, entropy_weight: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    pred = module(x)
    mse = torch.mean((pred - y) ** 2)
    entropy_term = torch.zeros((), dtype=DTYPE, device=x.device)
    if hasattr(module, "entropy"):
        entropy_term = module.entropy()
    loss = mse + entropy_weight * entropy_term
    return loss, mse


def train_tree(
    tree: EMLTree,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: TrainConfig = TrainConfig(),
    callback: Optional[Callable[[int, dict], None]] = None,
) -> dict:
    """Run a single Adam-hardening-snap training of an ``EMLTree``."""
    x = to_tensor(x)
    y = to_tensor(y).reshape(-1)
    assert x.shape[0] == y.shape[0]

    optim = torch.optim.Adam(tree.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=50
    )

    n_warmup = int(cfg.n_epochs * cfg.warmup_frac)
    n_harden = int(cfg.n_epochs * cfg.hardening_frac)
    history: list[dict] = []
    best_mse = float("inf")
    bad_epochs = 0

    tree.unsnap()
    tree.set_temperature(cfg.temp_start)

    for epoch in range(n_warmup + n_harden):
        in_hardening = epoch >= n_warmup
        if in_hardening:
            progress = (epoch - n_warmup) / max(n_harden - 1, 1)
            log_t = (
                math.log(cfg.temp_start) * (1 - progress)
                + math.log(cfg.temp_end) * progress
            )
            tree.set_temperature(math.exp(log_t))

        if cfg.batch_size and cfg.batch_size < x.shape[0]:
            idx = torch.randperm(x.shape[0], device=x.device)[: cfg.batch_size]
            xb, yb = x[idx], y[idx]
        else:
            xb, yb = x, y

        optim.zero_grad()
        loss, mse = _loss_fn(tree, xb, yb, cfg.entropy_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(tree.parameters(), cfg.grad_clip)
        optim.step()
        scheduler.step(mse.item())

        mse_val = mse.item()
        if mse_val < best_mse - 1e-12:
            best_mse = mse_val
            bad_epochs = 0
        else:
            bad_epochs += 1

        if callback is not None:
            callback(
                epoch,
                {"loss": loss.item(), "mse": mse_val, "temp": tree.temperature},
            )
        if cfg.verbose and (epoch % max(1, (n_warmup + n_harden) // 10) == 0):
            history.append(
                {
                    "epoch": epoch,
                    "loss": loss.item(),
                    "mse": mse_val,
                    "temp": tree.temperature,
                }
            )
        if (
            not in_hardening
            and bad_epochs > cfg.patience
            and epoch > n_warmup // 4
        ):
            break

    tree.snap()
    with torch.no_grad():
        pred = tree(x)
        final_mse = torch.mean((pred - y) ** 2).item()

    return {
        "final_mse": final_mse,
        "best_train_mse": best_mse,
        "history": history,
        "n_epochs_run": epoch + 1,
    }


def train_with_multistart(
    tree: EMLTree,
    x: torch.Tensor,
    y: torch.Tensor,
    n_starts: int = 5,
    cfg: TrainConfig = TrainConfig(),
    verbose: bool = False,
) -> dict:
    """Run ``train_tree`` ``n_starts`` times and keep the state with the
    smallest final MSE."""
    best_state: Optional[dict] = None
    best_mse = float("inf")
    best_info: dict = {}
    per_run: list[dict] = []

    for start in range(n_starts):
        tree.reset_parameters()
        info = train_tree(tree, x, y, cfg=cfg)
        per_run.append({"start": start, **info})
        if verbose:
            print(
                f"[multistart {start + 1}/{n_starts}] "
                f"final_mse={info['final_mse']:.3e} "
                f"best_train={info['best_train_mse']:.3e}"
            )
        if info["final_mse"] < best_mse:
            best_mse = info["final_mse"]
            best_state = copy.deepcopy(tree.state_dict())
            best_info = info

    assert best_state is not None
    tree.load_state_dict(best_state)
    tree.snap()
    return {
        "final_mse": best_mse,
        "best_info": best_info,
        "per_run": per_run,
    }
