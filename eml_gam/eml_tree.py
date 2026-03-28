"""Differentiable EML tree.

The tree follows equation (6) of Odrzywołek (2026). Each input of an internal
``eml`` node is a linear combination

    alpha * 1 + beta * x + gamma * f_child,

where ``f_child`` is the output of the corresponding child subtree. On the
deepest level there is no child and the ``gamma`` term is removed. Softmax
over ``(alpha, beta, gamma)`` is used during training; ``argmax`` provides
a deterministic snap that is read out as a SymPy expression.

Parameter counts:
  univariate tree of depth n: 5 * 2**n - 6
  bivariate  tree of depth n: 7 * 2**n - 8

A learnable per-input affine transform ``x' = scale * x + offset`` is applied
before the tree. It absorbs arbitrary scale and shift of the input without
changing the symbolic structure.
"""

from __future__ import annotations

from typing import List, Optional

import sympy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import DTYPE, safe_eml


class EMLTree(nn.Module):
    """EML tree for univariate or bivariate symbolic regression.

    Parameters
    ----------
    depth : int
        Tree depth. Recommended values are 1, 2, or 3; deeper trees do not
        converge from random initialisation in the paper's reference
        experiments.
    n_inputs : int
        1 for a univariate main-effect component, 2 for a bivariate
        interaction component.
    temperature : float
        Initial softmax temperature. Annealed toward zero during training.
    """

    def __init__(self, depth: int = 2, n_inputs: int = 2, temperature: float = 1.0):
        super().__init__()
        assert depth >= 1, "depth must be >= 1"
        assert n_inputs in (1, 2), "n_inputs must be 1 (univariate) or 2 (bivariate)"

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

        self._is_snapped: bool = False
        for level in range(depth):
            n_slots = 2 ** (level + 1)
            self.register_buffer(
                f"_snap_l{level}",
                torch.zeros(n_slots, dtype=torch.long),
                persistent=False,
            )

        self.input_scale = nn.Parameter(torch.ones(n_inputs, dtype=DTYPE))
        self.input_offset = nn.Parameter(torch.zeros(n_inputs, dtype=DTYPE))

    @property
    def temperature(self) -> float:
        return self._temperature

    def set_temperature(self, temperature: float) -> None:
        self._temperature = max(float(temperature), 1e-4)

    @property
    def is_snapped(self) -> bool:
        return self._is_snapped

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.level_logits)

    def _slot_weights(self, level: int) -> torch.Tensor:
        logits = self.level_logits[level]
        return F.softmax(logits / self._temperature, dim=1)

    def _snapped_choices(self, level: int) -> torch.Tensor:
        return getattr(self, f"_snap_l{level}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(1)
        assert x.shape[1] == self.n_inputs, (
            f"expected n_inputs={self.n_inputs}, got {x.shape[1]}"
        )
        x = x * self.input_scale + self.input_offset
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

        node_outputs = safe_eml(slot_values[:, 0::2], slot_values[:, 1::2])

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

            node_outputs = safe_eml(slot_values[:, 0::2], slot_values[:, 1::2])

        return node_outputs.squeeze(1)

    def entropy(self) -> torch.Tensor:
        """Mean entropy of the softmax distribution across all slots."""
        total, count = torch.zeros((), dtype=DTYPE), 0
        for level in range(self.depth):
            logits = self.level_logits[level]
            probs = F.softmax(logits / self._temperature, dim=1)
            log_probs = F.log_softmax(logits / self._temperature, dim=1)
            total = total - (probs * log_probs).sum()
            count += logits.shape[0]
        return total / max(count, 1)

    def snap(self) -> None:
        """Freeze the ``argmax`` choice at every slot."""
        with torch.no_grad():
            for level in range(self.depth):
                choices = torch.argmax(self.level_logits[level], dim=1)
                self._snapped_choices(level).copy_(choices)
            self._is_snapped = True

    def unsnap(self) -> None:
        self._is_snapped = False

    def set_snap_config(
        self,
        choices_per_level: dict[int, torch.Tensor] | list[torch.Tensor],
        logit_strength: float = 5.0,
    ) -> None:
        """Manually install a snap configuration and align the logits with it.

        Used for warm-starting: the argmax choices go into the snap buffers,
        and the logits are set so that softmax at ``tau = 1`` assigns almost
        all probability to the chosen options. Calling ``unsnap`` then starts
        training near the installed configuration.
        """
        if isinstance(choices_per_level, (list, tuple)):
            choices_per_level = {k: v for k, v in enumerate(choices_per_level)}
        with torch.no_grad():
            for level in range(self.depth):
                choices = choices_per_level[level]
                if not isinstance(choices, torch.Tensor):
                    choices = torch.as_tensor(choices, dtype=torch.long)
                n_slots, n_opts = self.level_logits[level].shape
                assert choices.shape == (n_slots,), (
                    f"level {level}: expected ({n_slots},), got {tuple(choices.shape)}"
                )
                self.level_logits[level].zero_()
                self.level_logits[level].scatter_(
                    1,
                    choices.unsqueeze(1),
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

    def reset_parameters(self) -> None:
        """Re-initialise the logits. Used for multi-start training."""
        with torch.no_grad():
            for logits in self.level_logits:
                logits.uniform_(-0.1, 0.1)
            self.input_scale.data.fill_(1.0)
            self.input_offset.data.zero_()
        self._is_snapped = False

    def get_symbolic_expression(
        self, feature_names: Optional[List[str]] = None, simplify: bool = True
    ) -> sp.Expr:
        """Return a SymPy expression for the current snap configuration.

        The input affine transform is folded into the symbol substitution: a
        feature ``x`` enters the formula as ``scale * x + offset``. When
        ``scale == 1`` and ``offset == 0`` the symbol is left unchanged.
        """
        if not self._is_snapped:
            self.snap()

        if feature_names is None:
            feature_names = [f"x{i + 1}" for i in range(self.n_inputs)]
        assert len(feature_names) == self.n_inputs

        const_one = sp.Integer(1)
        scale = self.input_scale.detach().cpu().tolist()
        offset = self.input_offset.detach().cpu().tolist()
        symbols: list[sp.Expr] = []
        for i, name in enumerate(feature_names):
            s = sp.Symbol(name)
            sc = float(scale[i])
            off = float(offset[i])
            if abs(sc - 1.0) < 1e-12 and abs(off) < 1e-12:
                expr_i: sp.Expr = s
            elif abs(sc + 1.0) < 1e-12 and abs(off) < 1e-12:
                expr_i = -s
            else:
                expr_i = sp.Float(sc) * s + sp.Float(off)
            symbols.append(expr_i)

        def _leaf_expr(choice: int) -> sp.Expr:
            if choice == 0:
                return const_one
            return symbols[choice - 1]

        def _upper_expr(choice: int, f_child: sp.Expr) -> sp.Expr:
            if choice == 0:
                return const_one
            if 1 <= choice <= self.n_inputs:
                return symbols[choice - 1]
            return f_child

        bottom_level = self.depth - 1
        bottom_choices = self._snapped_choices(bottom_level).tolist()
        slot_exprs: list[sp.Expr] = [_leaf_expr(c) for c in bottom_choices]
        node_exprs: list[sp.Expr] = [
            sp.exp(slot_exprs[i]) - sp.log(slot_exprs[i + 1])
            for i in range(0, len(slot_exprs), 2)
        ]

        for level in range(self.depth - 2, -1, -1):
            upper_choices = self._snapped_choices(level).tolist()
            n_slots = len(upper_choices)
            slot_exprs = [
                _upper_expr(upper_choices[j], node_exprs[j]) for j in range(n_slots)
            ]
            node_exprs = [
                sp.exp(slot_exprs[i]) - sp.log(slot_exprs[i + 1])
                for i in range(0, len(slot_exprs), 2)
            ]

        assert len(node_exprs) == 1
        expr = node_exprs[0]
        return sp.simplify(expr) if simplify else expr

    def extra_repr(self) -> str:
        return (
            f"depth={self.depth}, n_inputs={self.n_inputs}, "
            f"n_params={self.n_params}, temperature={self._temperature:.3f}, "
            f"snapped={self._is_snapped}"
        )
