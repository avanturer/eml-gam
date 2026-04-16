"""Smooth Sheffer surrogate: ``psi(x, y) = sinh(x) - arsinh(y)``.

Odrzywolek (2026), Section "Open problems", explicitly calls for a binary
Sheffer-like operator that avoids the composed-exponential overflow that
plagues deep EML trees. ``psi`` is a natural candidate:

* ``sinh(x) = (e^x - e^{-x}) / 2`` — odd, smooth everywhere, asymptotically
  ``e^x / 2`` for large ``x`` and ``-e^{-x} / 2`` for large ``-x``.
* ``arsinh(y) = log(y + sqrt(y**2 + 1))`` — odd, smooth everywhere, even
  at ``y = 0`` where ``log`` of ``eml`` would need clamping.

Neither composant overflows under the dynamic range used by depth-5 trees.
``arsinh`` also accepts negative arguments, so the zero-crossing of
intermediate expressions does not force the ``log`` clamp. The functional
class spanned by iterated ``psi`` compositions is conjectured (but not
proven) to be the same class spanned by iterated ``eml`` in the deep-tree
limit; the empirical check in ``scripts/sheffer_stability.py`` exercises
this hypothesis on a set of eight targets.

This module provides a parallel to :mod:`eml_gam.eml_tree` in which the
``eml`` atom is replaced by ``psi``. It is intentionally minimal — only
what is needed for the stability experiment. Production-grade symbolic
read-out is not implemented; the intent is empirical landscape work, not
end-to-end interpretability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import DTYPE


def safe_psi(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Numerically stable ``psi(x, y) = sinh(x) - arsinh(y)``.

    ``sinh`` is clamped at 20 (well before IEEE overflow at ~87). ``arsinh``
    requires no clamping and handles the full real line including zero.
    """
    x_c = torch.clamp(x, min=-20.0, max=20.0)
    return torch.sinh(x_c) - torch.asinh(y)


class PsiTree(nn.Module):
    """Differentiable ``psi`` tree — same structure as ``EMLTree``."""

    def __init__(self, depth: int = 2, n_inputs: int = 1, temperature: float = 1.0):
        super().__init__()
        assert depth >= 1 and n_inputs in (1, 2)
        self.depth = depth
        self.n_inputs = n_inputs
        self._temperature = float(temperature)
        self.n_bottom_options = 1 + n_inputs
        self.n_upper_options = 2 + n_inputs
        self.level_logits = nn.ParameterList()
        for level in range(depth):
            n_slots = 2 ** (level + 1)
            is_bottom = level == depth - 1
            n_options = self.n_bottom_options if is_bottom else self.n_upper_options
            self.level_logits.append(
                nn.Parameter(torch.randn(n_slots, n_options, dtype=DTYPE) * 0.1)
            )
        self._is_snapped = False
        for level in range(depth):
            n_slots = 2 ** (level + 1)
            self.register_buffer(
                f"_snap_l{level}",
                torch.zeros(n_slots, dtype=torch.long),
                persistent=False,
            )

    def set_temperature(self, temperature: float) -> None:
        self._temperature = max(float(temperature), 1e-4)

    @property
    def temperature(self) -> float:
        return self._temperature

    def entropy(self) -> torch.Tensor:
        total, count = torch.zeros((), dtype=DTYPE), 0
        for level in range(self.depth):
            logits = self.level_logits[level]
            probs = F.softmax(logits / self._temperature, dim=1)
            log_probs = F.log_softmax(logits / self._temperature, dim=1)
            total = total - (probs * log_probs).sum()
            count += logits.shape[0]
        return total / max(count, 1)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for logits in self.level_logits:
                logits.uniform_(-0.1, 0.1)
        self._is_snapped = False

    def _slot_weights(self, level: int) -> torch.Tensor:
        logits = self.level_logits[level]
        return F.softmax(logits / self._temperature, dim=1)

    def _snapped_choices(self, level: int) -> torch.Tensor:
        return getattr(self, f"_snap_l{level}")

    def snap(self) -> None:
        with torch.no_grad():
            for level in range(self.depth):
                self._snapped_choices(level).copy_(
                    torch.argmax(self.level_logits[level], dim=1)
                )
            self._is_snapped = True

    def unsnap(self) -> None:
        self._is_snapped = False

    def set_snap_config(
        self,
        choices_per_level: dict[int, torch.Tensor] | list[torch.Tensor],
        logit_strength: float = 5.0,
    ) -> None:
        if isinstance(choices_per_level, (list, tuple)):
            choices_per_level = {k: v for k, v in enumerate(choices_per_level)}
        with torch.no_grad():
            for level in range(self.depth):
                choices = choices_per_level[level]
                if not isinstance(choices, torch.Tensor):
                    choices = torch.as_tensor(choices, dtype=torch.long)
                n_slots, _ = self.level_logits[level].shape
                self.level_logits[level].zero_()
                self.level_logits[level].scatter_(
                    1, choices.unsqueeze(1),
                    torch.full((n_slots, 1), logit_strength, dtype=DTYPE),
                )
                self._snapped_choices(level).copy_(choices)
        self._is_snapped = True

    def get_snap_config(self) -> dict[int, torch.Tensor]:
        if not self._is_snapped:
            self.snap()
        return {
            level: self._snapped_choices(level).clone()
            for level in range(self.depth)
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(1)
        assert x.shape[1] == self.n_inputs
        B = x.shape[0]

        bottom_level = self.depth - 1
        ones = torch.ones(B, 1, dtype=x.dtype, device=x.device)
        bottom_options = torch.cat([ones, x], dim=1)
        if self._is_snapped:
            choices = self._snapped_choices(bottom_level)
            slot_values = bottom_options[:, choices]
        else:
            weights = self._slot_weights(bottom_level)
            slot_values = bottom_options @ weights.T
        node_outputs = safe_psi(slot_values[:, 0::2], slot_values[:, 1::2])

        for level in range(self.depth - 2, -1, -1):
            n_slots = 2 ** (level + 1)
            options = torch.empty(
                B, n_slots, self.n_upper_options, dtype=x.dtype, device=x.device
            )
            options[:, :, 0] = 1.0
            options[:, :, 1 : 1 + self.n_inputs] = x.unsqueeze(1).expand(
                B, n_slots, self.n_inputs
            )
            options[:, :, -1] = node_outputs
            if self._is_snapped:
                choices = self._snapped_choices(level)
                slot_values = options[
                    torch.arange(B).unsqueeze(1), torch.arange(n_slots), choices
                ]
            else:
                weights = self._slot_weights(level)
                slot_values = (options * weights.unsqueeze(0)).sum(dim=2)
            node_outputs = safe_psi(slot_values[:, 0::2], slot_values[:, 1::2])

        return node_outputs.squeeze(1)
