"""Why the depth wall exists: clamp saturation and exact zero gradients.

Two elementary observations turn the empirical depth wall into a
mechanism:

**P1 (exact gradient blocking).** ``safe_eml`` clamps the ``exp``
argument to ``[-C, C]`` and the ``log`` argument to ``[EPS, inf)``.
``torch.clamp`` has *identically zero* derivative outside the active
interval. Hence if, at the current parameters, the pre-clamp exp-side
input of some node exceeds ``C`` on **every** training point, then
every parameter whose only influence on the output flows through that
input has *exactly* zero gradient — not a small gradient, zero. No
optimiser that follows gradients can escape.

**P2 (mean-field saturation depth).** At standard random init the
softmax mixture is nearly uniform, so every slot value is close to the
mean of its options. With inputs on ``[0.3, 2.5]`` (mean ``x_bar``)
the layer-to-layer recursion of typical node values is

    m_0 = (1 + x_bar) / 2
    V_{k+1} = exp(min(m_k, C)) - ln(max(m_k, EPS))
    m_{k+1} = (1 + x_bar + V_{k+1}) / 3

which is doubly exponential: for ``x_bar = 1.4``, ``m`` crosses the
clamp ``C = 10`` at the **fifth** composition level. A random-init
EML tree of depth >= 5 therefore has its upper exp inputs saturated in
the mean-field picture — P1 kills the exp-side gradients outright, and
the surviving log-side path is suppressed by a factor ``1/m_k`` per
level, i.e. doubly exponentially in depth.

**P3 (the smooth sibling escapes).** The same recursion for
``psi(x, y) = sinh(x) - arsinh(y)`` is

    W_{k+1} = sinh(m_k) - arsinh(m_k),   m_k = (1 + x_bar + W_k)/3

which has a stable fixed point (``W* ~= 0.19``, ``m* ~= 0.86`` at
``x_bar = 1.4``) with contraction factor ``~0.21``: activations neither
explode nor die, and per-level gradient attenuation is a benign
geometric factor instead of a doubly-exponential collapse. This is
precisely the empirical cross-operator landscape: psi trains from
random init at depths where eml is at 0 per cent.

This module *measures* both mechanisms directly:

* fraction of random inits whose bottom-level logits receive exactly
  zero gradient (P1, via autograd);
* median finite gradient magnitude of bottom-level logits (P2/P3);
* the mean-field trajectories themselves.

Writes ``saturation_results.json``.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from ..eml_tree import EMLTree
from ..sheffer import PsiTree
from ..utils import CLAMP_VAL, EPS

X_LO, X_HI = 0.3, 2.5
N_POINTS = 256
N_TRIALS = 300
DEPTHS = (1, 2, 3, 4, 5, 6, 7, 8)


def mean_field_trajectory(operator: str, depth: int, x_bar: float = 1.4) -> list[float]:
    """Typical node values ``V_1 .. V_depth`` under uniform mixing."""
    m = (1.0 + x_bar) / 2.0
    out = []
    for _ in range(depth):
        if operator == "eml":
            v = float(np.exp(min(m, CLAMP_VAL)) - np.log(max(m, EPS)))
        else:
            v = float(np.sinh(m) - np.arcsinh(m))
        out.append(v)
        m = (1.0 + x_bar + v) / 3.0
    return out


def measure_operator(
    make_tree, depths=DEPTHS, n_trials: int = N_TRIALS, seed: int = 0
) -> dict:
    g = torch.Generator().manual_seed(seed)
    x = torch.linspace(X_LO, X_HI, N_POINTS, dtype=torch.float64).unsqueeze(1)
    rows = {}
    for depth in depths:
        n_all_zero = 0
        n_nan_fwd = 0
        dead_slot_fracs: list[float] = []
        live_medians: list[float] = []
        max_grads: list[float] = []
        t0 = time.time()
        for trial in range(n_trials):
            torch.manual_seed(int(torch.randint(0, 2**31, (1,), generator=g)))
            tree = make_tree(depth)
            # Random target from the same architecture (fresh init) so
            # the loss is non-degenerate.
            with torch.no_grad():
                target_tree = make_tree(depth)
                y = target_tree(x)
            if not torch.all(torch.isfinite(y)):
                y = torch.zeros(x.shape[0], dtype=torch.float64)
            pred = tree(x)
            if not torch.all(torch.isfinite(pred)):
                # NaN forward: gradient is unusable; count as blocked.
                n_nan_fwd += 1
                n_all_zero += 1
                dead_slot_fracs.append(1.0)
                continue
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            bottom = tree.level_logits[depth - 1].grad
            if bottom is None or torch.all(bottom == 0):
                n_all_zero += 1
                dead_slot_fracs.append(1.0)
                continue
            # A bottom "slot" is dead when its entire logit row gets a
            # zero gradient — the choice at that slot cannot move.
            row_dead = (bottom.abs().sum(dim=1) == 0).double().mean()
            dead_slot_fracs.append(float(row_dead))
            nonzero = bottom.abs()[bottom.abs() > 0]
            live_medians.append(float(nonzero.median()))
            max_grads.append(float(bottom.abs().max()))
        rows[str(depth)] = {
            "all_zero_fraction": n_all_zero / n_trials,
            "nan_forward_fraction": n_nan_fwd / n_trials,
            "mean_dead_slot_fraction": float(np.mean(dead_slot_fracs)),
            "median_live_grad": (
                float(np.median(live_medians)) if live_medians else None
            ),
            "median_max_grad": (
                float(np.median(max_grads)) if max_grads else None
            ),
            "n_trials": n_trials,
            "seconds": time.time() - t0,
        }
        r = rows[str(depth)]
        print(
            f"    depth {depth}: dead-slots {r['mean_dead_slot_fraction']:.1%}, "
            f"live |grad| median {r['median_live_grad']}, "
            f"max |grad| median {r['median_max_grad']}"
        )
    return rows


def main(out_path: str = "saturation_results.json") -> dict:
    results: dict = {
        "protocol": {
            "x_range": [X_LO, X_HI],
            "n_points": N_POINTS,
            "n_trials": N_TRIALS,
            "clamp": CLAMP_VAL,
            "eps": EPS,
        },
        "mean_field": {
            op: {
                str(d): mean_field_trajectory(op, d)
                for d in (5, 8)
            }
            for op in ("eml", "psi")
        },
    }
    print("  EML trees:")
    results["eml"] = measure_operator(
        lambda d: EMLTree(depth=d, n_inputs=1, use_input_affine=False)
    )
    print("  psi trees:")
    results["psi"] = measure_operator(lambda d: PsiTree(depth=d, n_inputs=1))
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")
    return results


if __name__ == "__main__":
    main()
