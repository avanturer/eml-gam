"""Landscape study of EML-tree recovery.

This benchmark directly targets the open problem stated in Section "Scaling
gradient-based recovery" of Odrzywolek (2026): random-initialisation
recovery succeeds in 100 per cent of trials at depth 2 but drops to below
1 per cent at depth 5 and hits zero at depth 6, while local perturbation of
a correct snap always recovers the target. We reproduce both halves of that
phenomenon and expose them as an API that can be called from the multi-seed
runner and from the paper's figure-generating scripts.

Two experiments are exposed:

``random_init_recovery(depth, ...)``
    Generates data from a known snap of the requested depth and trains
    ``n_trials`` EML trees from random initialisation. Reports the fraction
    of trials that reach ``final_mse < mse_threshold`` and the fraction
    whose argmax snap matches the ground-truth one.

``perturbed_init_recovery(depth, sigma, ...)``
    Generates data from the same ground-truth snap and trains from the
    correct snap with Gaussian noise of standard deviation ``sigma`` added
    to every logit. Reports the same two metrics; as ``sigma`` grows, the
    recovery rate curves give a basin-width diagnostic around the optimum.

Target family: ``e - exp^{d-2}(x)`` for ``d >= 2`` and ``e - log(x)`` for
``d = 1``. This family is non-degenerate across depths (the target is a
genuinely different function at every depth), grows slowly for negative
``x`` so numerical clamping does not dominate at depth 5, and is
representable by a single EML tree of exactly the matching depth.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch

from ..eml_tree import EMLTree
from ..train import TrainConfig, train_tree
from ..utils import DTYPE


def target_snap_elog_iterated_exp(depth: int) -> dict[int, torch.Tensor]:
    """Return the snap whose output is ``e - exp^{d-2}(x)`` for ``d >= 2``.

    At depth 1 the function degenerates to ``e - log(x)``. The snap is
    chosen so the "active" path goes through the right child of the root
    and then through the node at position ``2**level`` at every subsequent
    level, leaving all other slots as the constant ``1``.
    """
    snap: dict[int, torch.Tensor] = {}
    for level in range(depth):
        n_slots = 2 ** (level + 1)
        is_bottom = level == depth - 1
        choices = [0] * n_slots
        if level == 0:
            if depth == 1:
                choices = [0, 1]
            else:
                choices = [0, 2]
        else:
            active = 2 ** level
            choices[active] = 1 if is_bottom else 2
        snap[level] = torch.tensor(choices, dtype=torch.long)
    return snap


def _generate_data(
    depth: int,
    n: int = 256,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
    rng = np.random.default_rng(seed)
    if depth == 1:
        # Depth 1 target is e - log(x); we need x > 0 for the log.
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
    target_snap = target_snap_elog_iterated_exp(depth)
    tree = EMLTree(depth=depth, n_inputs=1, use_input_affine=False)
    tree.set_snap_config(target_snap)
    with torch.no_grad():
        y = tree(x)
    return x, y, target_snap


def _snap_match(
    a: dict[int, torch.Tensor], b: dict[int, torch.Tensor]
) -> bool:
    keys = set(a.keys()) | set(b.keys())
    for k in keys:
        if k not in a or k not in b:
            return False
        if not torch.equal(a[k], b[k]):
            return False
    return True


@dataclass
class RecoveryResult:
    depth: int
    init_mode: str
    sigma: float
    n_trials: int
    n_exact_snap: int
    n_low_mse: int
    mse_threshold: float
    success_rate: float
    snap_rate: float


def _run_trials(
    depth: int,
    init_fn,
    n_trials: int,
    n_epochs: int,
    mse_threshold: float,
    data_seed: int = 0,
    verbose: bool = False,
) -> tuple[int, int]:
    x, y, target_snap = _generate_data(depth, seed=data_seed)
    n_match = 0
    n_low = 0
    for trial in range(n_trials):
        torch.manual_seed(trial * 97 + 13)
        tree = EMLTree(depth=depth, n_inputs=1, use_input_affine=False)
        init_fn(tree, target_snap, trial)
        cfg = TrainConfig(
            n_epochs=n_epochs, lr=5e-2,
            entropy_weight=1e-3, grad_clip=1.0,
            temp_start=1.0, temp_end=1e-2,
            warmup_frac=0.7, hardening_frac=0.3,
            patience=max(100, n_epochs // 10),
        )
        try:
            info = train_tree(tree, x, y, cfg=cfg)
            final_mse = info["final_mse"]
        except Exception:
            final_mse = float("inf")
        if np.isfinite(final_mse) and final_mse < mse_threshold:
            n_low += 1
        rec = tree.get_snap_config()
        if _snap_match(rec, target_snap):
            n_match += 1
        if verbose:
            print(
                f"  trial={trial:3d}  mse={final_mse:.3e}  "
                f"snap_match={_snap_match(rec, target_snap)}"
            )
    return n_match, n_low


def random_init_recovery(
    depth: int,
    n_trials: int = 50,
    n_epochs: int = 1500,
    mse_threshold: float = 1e-3,
    data_seed: int = 0,
    verbose: bool = False,
) -> RecoveryResult:
    def init_fn(tree: EMLTree, target_snap, trial: int) -> None:
        tree.reset_parameters()

    n_match, n_low = _run_trials(
        depth, init_fn, n_trials, n_epochs, mse_threshold,
        data_seed=data_seed, verbose=verbose,
    )
    return RecoveryResult(
        depth=depth, init_mode="random", sigma=float("inf"),
        n_trials=n_trials, n_exact_snap=n_match, n_low_mse=n_low,
        mse_threshold=mse_threshold,
        success_rate=n_low / n_trials,
        snap_rate=n_match / n_trials,
    )


def perturbed_init_recovery(
    depth: int,
    sigma: float,
    n_trials: int = 20,
    n_epochs: int = 1500,
    mse_threshold: float = 1e-3,
    data_seed: int = 0,
    verbose: bool = False,
) -> RecoveryResult:
    def init_fn(tree: EMLTree, target_snap, trial: int) -> None:
        tree.set_snap_config(target_snap, logit_strength=5.0)
        tree.unsnap()
        with torch.no_grad():
            for logits in tree.level_logits:
                noise = torch.randn_like(logits) * sigma
                logits.data.add_(noise)

    n_match, n_low = _run_trials(
        depth, init_fn, n_trials, n_epochs, mse_threshold,
        data_seed=data_seed, verbose=verbose,
    )
    return RecoveryResult(
        depth=depth, init_mode="perturbed", sigma=sigma,
        n_trials=n_trials, n_exact_snap=n_match, n_low_mse=n_low,
        mse_threshold=mse_threshold,
        success_rate=n_low / n_trials,
        snap_rate=n_match / n_trials,
    )


def run_landscape_experiment(
    depths: Sequence[int] = (1, 2, 3, 4, 5),
    sigmas: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 5.0),
    n_random: int = 40,
    n_perturbed: int = 20,
    n_epochs: int = 1500,
    verbose: bool = True,
) -> dict:
    results: dict[str, list[RecoveryResult]] = {
        "random": [], "perturbed": [],
    }
    if verbose:
        print(f"{'=' * 70}")
        print("  Landscape study (random-init vs perturbed-init recovery)")
        print(f"{'=' * 70}")
    for d in depths:
        r = random_init_recovery(
            d, n_trials=n_random, n_epochs=n_epochs,
        )
        results["random"].append(r)
        if verbose:
            print(
                f"  depth={d} [random]        mse_ok={r.n_low_mse:3d}/{r.n_trials}  "
                f"snap_ok={r.n_exact_snap:3d}/{r.n_trials}  "
                f"success={r.success_rate:.2%}"
            )
        for s in sigmas:
            rp = perturbed_init_recovery(
                d, sigma=s, n_trials=n_perturbed, n_epochs=n_epochs,
            )
            results["perturbed"].append(rp)
            if verbose:
                print(
                    f"  depth={d} [sigma={s:4.2f}]    mse_ok={rp.n_low_mse:3d}/{rp.n_trials}  "
                    f"snap_ok={rp.n_exact_snap:3d}/{rp.n_trials}  "
                    f"success={rp.success_rate:.2%}"
                )
    return results


def save_landscape_results(results: dict, path: str) -> None:
    payload = {
        "random": [asdict(r) for r in results["random"]],
        "perturbed": [asdict(r) for r in results["perturbed"]],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    import sys

    short = "--short" in sys.argv
    if short:
        results = run_landscape_experiment(
            depths=(1, 2, 3, 4),
            sigmas=(0.5, 2.0),
            n_random=10, n_perturbed=5, n_epochs=800,
        )
    else:
        results = run_landscape_experiment()
    save_landscape_results(results, "landscape_results.json")
