"""Numerical helpers for EML trees."""

from __future__ import annotations

import torch


CLAMP_VAL: float = 10.0
EPS: float = 1e-10
DTYPE = torch.float64


def safe_eml(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Numerically stable evaluation of ``eml(x, y) = exp(x) - log(y)``.

    The argument to ``exp`` is clamped to ``[-CLAMP_VAL, CLAMP_VAL]`` to avoid
    overflow in the upward direction and to keep gradients informative in the
    downward direction. The argument to ``log`` is clamped from below by
    ``EPS`` to avoid ``log(0)`` and ``log`` of negative values produced by
    round-off in deeply nested compositions.
    """
    x_c = torch.clamp(x, min=-CLAMP_VAL, max=CLAMP_VAL)
    y_c = torch.clamp(y, min=EPS)
    return torch.exp(x_c) - torch.log(y_c)


def to_tensor(x, dtype=DTYPE, device=None) -> torch.Tensor:
    """Convert numpy / list / tensor input to a float64 ``torch.Tensor``."""
    if isinstance(x, torch.Tensor):
        t = x.to(dtype=dtype)
    else:
        t = torch.as_tensor(x, dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t
