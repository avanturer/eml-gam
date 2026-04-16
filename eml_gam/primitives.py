"""Primitive atlas and warm-start for EML trees.

The atlas contains hand-coded snap configurations for a small catalogue of
scientific primitives (``exp``, ``log``, ``1/x``, saturating, log-linear,
etc.). Before training, each tree is initialised by an ordinary-least-squares
projection of the target residual onto the atlas. The search is performed
over both signs of every input and over all atlas entries jointly; a
two-sided hold-out validator and an adaptive simplicity tolerance select the
best candidate.

For depth-2 univariate trees the slot layout is:
  level 0 (root)   : 2 input slots, options {0: 1, 1: x, 2: f_child}
  level 1 (bottom) : 4 input slots, options {0: 1, 1: x}

The forward computation is:
  childL = eml(v_0, v_1);  childR = eml(v_2, v_3)
  root   = eml(root_L, root_R) = exp(root_L) - log(root_R)

Bivariate trees add ``x2`` to every slot's option list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import sympy as sp
import torch

from .eml_tree import EMLTree
from .utils import DTYPE


@dataclass
class PrimitiveConfig:
    name: str
    depth: int
    n_inputs: int
    snap_choices: dict[int, torch.Tensor]
    description: str
    formula: Optional[sp.Expr] = None


def _mk(
    depth: int,
    n_inputs: int,
    name: str,
    desc: str,
    choices: dict[int, list[int]],
) -> PrimitiveConfig:
    snap = {k: torch.tensor(v, dtype=torch.long) for k, v in choices.items()}
    tree = EMLTree(depth=depth, n_inputs=n_inputs)
    tree.set_snap_config(snap)
    feat = ["x1"] if n_inputs == 1 else ["x1", "x2"]
    formula = tree.get_symbolic_expression(feat, simplify=True)
    return PrimitiveConfig(
        name=name,
        depth=depth,
        n_inputs=n_inputs,
        snap_choices=snap,
        description=desc,
        formula=formula,
    )


# ------------------------------------------------------------------- atlases


def univariate_atlas_depth1() -> list[PrimitiveConfig]:
    """Trivial atlas for depth-1 univariate trees (two slots, two options)."""
    return [
        _mk(1, 1, "const_e",        "e = eml(1, 1)",            {0: [0, 0]}),
        _mk(1, 1, "e_minus_logx",   "e - log(x) = eml(1, x)",   {0: [0, 1]}),
        _mk(1, 1, "exp",            "exp(x) = eml(x, 1)",       {0: [1, 0]}),
        _mk(1, 1, "exp_minus_logx", "exp(x) - log(x) = eml(x, x)",
            {0: [1, 1]}),
    ]


def univariate_atlas_depth2() -> list[PrimitiveConfig]:
    """Atlas for depth-2 univariate trees (14 parameters per tree)."""
    atlas: list[PrimitiveConfig] = []

    atlas.append(_mk(2, 1, "const_e",
                     "constant e",
                     {0: [0, 0], 1: [0, 0, 0, 0]}))

    atlas.append(_mk(2, 1, "exp",
                     "exp(x)",
                     {0: [1, 0], 1: [0, 0, 0, 0]}))
    atlas.append(_mk(2, 1, "e_minus_logx",
                     "e - log(x)",
                     {0: [0, 1], 1: [0, 0, 0, 0]}))
    atlas.append(_mk(2, 1, "exp_minus_logx",
                     "exp(x) - log(x)",
                     {0: [1, 1], 1: [0, 0, 0, 0]}))

    atlas.append(_mk(2, 1, "exp_of_eminuslogx",
                     "exp(e - log(x)) = e^e / x",
                     {0: [2, 0], 1: [0, 1, 0, 0]}))
    atlas.append(_mk(2, 1, "exp_exp",
                     "exp(exp(x))",
                     {0: [2, 0], 1: [1, 0, 0, 0]}))
    atlas.append(_mk(2, 1, "exp_of_exp_minus_logx",
                     "exp(exp(x) - log(x))",
                     {0: [2, 0], 1: [1, 1, 0, 0]}))

    atlas.append(_mk(2, 1, "e_minus_log_eminuslogx",
                     "e - log(e - log(x))",
                     {0: [0, 2], 1: [0, 0, 0, 1]}))
    atlas.append(_mk(2, 1, "exp_x_minus_1",
                     "exp(x) - 1",
                     {0: [1, 2], 1: [0, 0, 0, 0]}))
    atlas.append(_mk(2, 1, "exp_x_minus_x",
                     "exp(x) - x",
                     {0: [1, 2], 1: [0, 0, 1, 0]}))
    atlas.append(_mk(2, 1, "exp_x_minus_log_eminuslogx",
                     "exp(x) - log(e - log(x))",
                     {0: [1, 2], 1: [0, 0, 0, 1]}))
    atlas.append(_mk(2, 1, "exp_x_minus_log_expx_minus_logx",
                     "exp(x) - log(exp(x) - log(x))",
                     {0: [1, 2], 1: [0, 0, 1, 1]}))

    atlas.append(_mk(2, 1, "exp_exp_minus_log_eminuslogx",
                     "exp(exp(x)) - log(e - log(x))",
                     {0: [2, 2], 1: [1, 0, 0, 1]}))

    return atlas


def bivariate_atlas_depth2() -> list[PrimitiveConfig]:
    """Atlas for depth-2 bivariate trees (20 parameters per tree).

    Slot options:
      bottom (level=1): {0: 1, 1: x1, 2: x2}
      upper  (level=0): {0: 1, 1: x1, 2: x2, 3: f_child}
    """
    atlas: list[PrimitiveConfig] = []

    atlas.append(_mk(2, 2, "eml_x1_x2",
                     "exp(x1) - log(x2)",
                     {0: [1, 2], 1: [0, 0, 0, 0]}))
    atlas.append(_mk(2, 2, "eml_x2_x1",
                     "exp(x2) - log(x1)",
                     {0: [2, 1], 1: [0, 0, 0, 0]}))
    atlas.append(_mk(2, 2, "exp_x1_minus_x2",
                     "exp(x1) - x2 (via f_child)",
                     {0: [1, 2], 1: [0, 0, 2, 0]}))
    atlas.append(_mk(2, 2, "exp_x1",
                     "exp(x1)",
                     {0: [1, 0], 1: [0, 0, 0, 0]}))
    atlas.append(_mk(2, 2, "minuslog_x2",
                     "e - log(x2)",
                     {0: [0, 2], 1: [0, 0, 0, 0]}))

    atlas.append(_mk(2, 2, "mm_exp_ratio",
                     "exp(exp(x1) - log(x2)) = exp(exp(x1)) / x2",
                     {0: [3, 0], 1: [1, 2, 0, 0]}))
    atlas.append(_mk(2, 2, "exp_eml_minus_log_x2",
                     "exp(exp(x1) - log(x2)) - log(x2)",
                     {0: [3, 2], 1: [1, 2, 0, 0]}))

    atlas.append(_mk(2, 2, "exp_x1_minus_log_emlx1x2",
                     "exp(x1) - log(exp(x1) - log(x2))",
                     {0: [1, 3], 1: [0, 0, 1, 2]}))
    atlas.append(_mk(2, 2, "e_minus_log_emlx1x2",
                     "e - log(exp(x1) - log(x2))",
                     {0: [0, 3], 1: [0, 0, 1, 2]}))

    atlas.append(_mk(2, 2, "eml_of_emls",
                     "exp(exp(x1)) - log(exp(x2)) = exp(exp(x1)) - x2",
                     {0: [3, 3], 1: [1, 0, 2, 0]}))
    atlas.append(_mk(2, 2, "exp_x1_x2",
                     "exp(exp(x1)) - log(e - log(x2))",
                     {0: [3, 3], 1: [1, 0, 0, 2]}))

    return atlas


def default_atlas(depth: int, n_inputs: int) -> list[PrimitiveConfig]:
    if n_inputs == 1:
        if depth == 1:
            return univariate_atlas_depth1()
        if depth == 2:
            return univariate_atlas_depth2()
    if n_inputs == 2 and depth == 2:
        return bivariate_atlas_depth2()
    raise ValueError(f"no default atlas for depth={depth}, n_inputs={n_inputs}")


# -------------------------------------------------------------- scoring / ws


def _primitive_values(config: PrimitiveConfig, x: torch.Tensor) -> torch.Tensor:
    tree = EMLTree(depth=config.depth, n_inputs=config.n_inputs)
    tree.set_snap_config(config.snap_choices)
    with torch.no_grad():
        return tree(x.detach())


def score_primitive(
    config: PrimitiveConfig,
    x: torch.Tensor,
    y: torch.Tensor,
    outlier_ratio_max: float = 8.0,
    holdout_quantile: float = 0.8,
    use_holdout: bool = True,
) -> tuple[float, float, float]:
    """OLS fit ``y ~ alpha + beta * primitive(x)`` with two validation guards.

    A candidate is rejected if:
      - the primitive produces non-finite values,
      - its absolute values have ``max / median > outlier_ratio_max``
        (typically triggered by ``1/x`` near zero or ``log(x)`` of values
        clamped to ``EPS``),
      - its dynamic range ``(p95 - p05) / max(|median|, 1)`` is smaller than
        0.05 (near-constant primitive or primitive saturating against the
        clamp).

    The R² used for selection is computed on a two-sided hold-out: training
    on the central 80 per cent of the first feature's range and scoring on
    the outer 10 per cent on each side. This rewards primitives that have
    the correct asymptotic behaviour on both tails, which is much more
    predictive of extrapolation quality than standard in-sample R².
    """
    values = _primitive_values(config, x).cpu().numpy().astype(np.float64)
    y_np = y.detach().cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(values)) or np.std(values) < 1e-12:
        return -np.inf, 0.0, 0.0

    abs_v = np.abs(values)
    med = np.median(abs_v)
    if med > 1e-12 and abs_v.max() / med > outlier_ratio_max:
        return -np.inf, 0.0, 0.0

    p95, p05 = np.percentile(values, [95, 5])
    dyn_range = p95 - p05
    denom = max(abs(np.median(values)), 1.0)
    if dyn_range / denom < 0.05:
        return -np.inf, 0.0, 0.0

    if use_holdout:
        if x.dim() == 1:
            feat = x.detach().cpu().numpy()
        else:
            feat = x[:, 0].detach().cpu().numpy()
        order = np.argsort(feat)
        n = len(values)
        edge_frac = (1.0 - holdout_quantile) / 2.0
        n_edge = max(int(edge_frac * n), 4)
        if 2 * n_edge >= n - 8:
            n_edge = max((n - 8) // 2, 2)
        val_idx = np.concatenate([order[:n_edge], order[-n_edge:]])
        train_idx = order[n_edge : n - n_edge]

        v_train, y_train = values[train_idx], y_np[train_idx]
        v_val, y_val = values[val_idx], y_np[val_idx]
        if np.std(v_train) < 1e-12 or np.std(y_val) < 1e-12:
            return -np.inf, 0.0, 0.0

        design_train = np.stack([np.ones_like(v_train), v_train], axis=1)
        coef, *_ = np.linalg.lstsq(design_train, y_train, rcond=None)

        y_hat_val = coef[0] + coef[1] * v_val
        ss_res = float(np.sum((y_val - y_hat_val) ** 2))
        ss_tot = float(np.sum((y_val - y_val.mean()) ** 2))
        r2_val = 1.0 - ss_res / max(ss_tot, 1e-12)
    else:
        # In-sample R² (no holdout)
        design = np.stack([np.ones_like(values), values], axis=1)
        coef, *_ = np.linalg.lstsq(design, y_np, rcond=None)
        y_hat = coef[0] + coef[1] * values
        ss_res = float(np.sum((y_np - y_hat) ** 2))
        ss_tot = float(np.sum((y_np - y_np.mean()) ** 2))
        r2_val = 1.0 - ss_res / max(ss_tot, 1e-12)

    design_full = np.stack([np.ones_like(values), values], axis=1)
    coef_full, *_ = np.linalg.lstsq(design_full, y_np, rcond=None)
    return r2_val, float(coef_full[0]), float(coef_full[1])


def best_primitive_for_feature(
    atlas: list[PrimitiveConfig],
    x: torch.Tensor,
    y: torch.Tensor,
    simplicity_tol: float = 0.10,
) -> tuple[PrimitiveConfig, float]:
    """Return the atlas entry with the highest hold-out R²; ties go to the
    simpler entry (smaller index in the atlas)."""
    best_r2, best_cfg = -np.inf, atlas[0]
    for cfg in atlas:
        r2, _, _ = score_primitive(cfg, x, y)
        if r2 > best_r2 + simplicity_tol:
            best_r2 = r2
            best_cfg = cfg
        elif r2 > best_r2:
            best_r2 = max(r2, best_r2)
    return best_cfg, best_r2


def warm_start_tree(
    tree: EMLTree,
    atlas: list[PrimitiveConfig],
    x: torch.Tensor,
    y: torch.Tensor,
    try_signs: bool = True,
    try_offsets: bool = True,
    simplicity_tol: float = 0.10,
    use_holdout: bool = True,
) -> PrimitiveConfig:
    """Install the best atlas entry into the given tree.

    Every combination of ``(sign_vector, offset_vector, primitive)`` is
    scored with ``score_primitive``.  The offset search tries shifting each
    input by a few data-driven values (0, median, various percentiles).
    This is critical for rational-function targets like Michaelis–Menten
    ``v = Vmax * S / (Km + S)``: the primitive ``e^e / x`` evaluated at
    ``x + Km`` gives a near-exact fit, but at ``x`` alone it does not.

    An adaptive tolerance prefers simpler atlas entries when several
    candidates score within a small margin.
    """
    candidates = [
        c for c in atlas if c.depth == tree.depth and c.n_inputs == tree.n_inputs
    ]
    if not candidates:
        raise ValueError(
            f"no atlas entry for depth={tree.depth}, n_inputs={tree.n_inputs}"
        )

    if not try_signs:
        sign_combos: list[list[float]] = [[1.0] * tree.n_inputs]
    elif tree.n_inputs == 1:
        sign_combos = [[1.0], [-1.0]]
    else:
        sign_combos = [[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]]

    # Build offset candidates per input dimension.
    if try_offsets and tree.use_input_affine:
        x_np = x.detach().cpu().numpy()
        offset_candidates: list[list[float]] = []
        for dim in range(tree.n_inputs):
            col = x_np[:, dim] if x_np.ndim == 2 else x_np
            med = float(np.median(col))
            p25 = float(np.percentile(col, 25))
            p75 = float(np.percentile(col, 75))
            offset_candidates.append(
                sorted(set([0.0, med, -med, p25, -p25, p75, -p75]))
            )
        if tree.n_inputs == 1:
            offset_combos = [[o] for o in offset_candidates[0]]
        else:
            # For bivariate: only try offsets on one dim at a time
            offset_combos = [[0.0, 0.0]]
            for o in offset_candidates[0]:
                if o != 0.0:
                    offset_combos.append([o, 0.0])
            for o in offset_candidates[1]:
                if o != 0.0:
                    offset_combos.append([0.0, o])
    else:
        offset_combos = [[0.0] * tree.n_inputs]

    # Score all (sign, offset, primitive) combinations.
    Candidate = tuple[float, int, list[float], list[float], PrimitiveConfig]
    all_ranked: list[Candidate] = []
    for signs in sign_combos:
        sign_t = torch.tensor(signs, dtype=x.dtype, device=x.device)
        for offsets in offset_combos:
            off_t = torch.tensor(offsets, dtype=x.dtype, device=x.device)
            x_shifted = x * sign_t + off_t
            for idx, cfg in enumerate(candidates):
                r2, _, _ = score_primitive(
                    cfg, x_shifted, y, use_holdout=use_holdout
                )
                all_ranked.append((r2, idx, signs, offsets, cfg))

    # Step 1: find best among zero-offset candidates (original logic).
    zero_only = [c for c in all_ranked if all(abs(o) < 1e-8 for o in c[3])]
    best_zero_r2 = max(c[0] for c in zero_only) if zero_only else -np.inf
    zero_tol = max(0.05, min(simplicity_tol, (1.0 - best_zero_r2) * 0.5))
    zero_acceptable = [c for c in zero_only if c[0] >= best_zero_r2 - zero_tol]
    best_zero = min(zero_acceptable, key=lambda c: (c[1], -c[0]))

    # Step 2: check if any offset candidate beats the zero-winner.
    # Only consider offset for rational-form primitives (e^e/x family).
    # Log-based primitives with offset inflate holdout R² but extrapolate
    # poorly (log grows unboundedly, misleading the holdout scorer).
    _offset_ok = {"exp_of_eminuslogx"}
    offset_only = [
        c for c in all_ranked
        if any(abs(o) > 1e-8 for o in c[3]) and c[4].name in _offset_ok
    ]
    offset_winner = max(offset_only, key=lambda c: c[0]) if offset_only else None

    if offset_winner is not None and offset_winner[0] > best_zero[0] + 0.03:
        chosen = offset_winner
    else:
        chosen = best_zero
    _, _, best_signs, best_offsets, best_cfg = chosen

    tree.set_snap_config(best_cfg.snap_choices)
    with torch.no_grad():
        tree.input_scale.data.copy_(
            torch.tensor(best_signs, dtype=DTYPE, device=tree.input_scale.device)
        )
        if tree.use_input_affine:
            tree.input_offset.data.copy_(
                torch.tensor(best_offsets, dtype=DTYPE, device=tree.input_offset.device)
            )
    tree.unsnap()
    return best_cfg
