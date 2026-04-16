"""EMLGAM: generalized additive model with pairwise interactions.

Prediction:

    y_hat = bias + sum_i w_i * tree_i(x_i) + sum_{i<j} w_ij * tree_ij(x_i, x_j)

Each tree is an :class:`EMLTree`; the weights ``w_i`` and ``w_ij`` are scalar
learnable parameters that absorb multiplicative constants, while the trees
themselves encode the symbolic form. Feature scaling is optional and is
applied only when the empirical standard deviation of a feature falls
outside ``[0.1, 10]`` (this makes training feasible on inputs such as
``1/T`` in the Arrhenius law while leaving well-conditioned features
untouched).

The ``fit`` method follows the standard three-stage schedule: Adam warm-up,
hardening with exponential temperature annealing, final ``argmax`` snap.
A warm-start step initialised from the primitive atlas runs before the Adam
stage; it can be disabled via ``warm_start=False``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import sympy as sp
import torch
import torch.nn as nn

from .eml_tree import EMLTree
from .train import TrainConfig
from .utils import DTYPE, to_tensor


PairKey = Tuple[int, int]


def _pair_key(i: int, j: int) -> str:
    a, b = sorted((i, j))
    return f"{a}_{b}"


def _parse_pair_key(key: str) -> PairKey:
    a, b = key.split("_")
    return (int(a), int(b))


class EMLGAM(nn.Module):
    """Generalised additive model with pairwise EML components.

    Parameters
    ----------
    n_features : int
    interaction_pairs : iterable of ``(i, j)`` tuples or ``None``
        Pairs of feature indices for bivariate interaction components. If
        ``None``, only main effects are used.
    univariate_depth : int
        Depth of each univariate tree (defaults to 2).
    bivariate_depth : int
        Depth of each bivariate tree (defaults to 2).
    use_univariate : bool
        If ``False``, skip the main-effect components entirely.
    feature_names : list of str or ``None``
        Names used in the returned SymPy formulas. Defaults to
        ``[x1, x2, ...]``.
    standardize : bool
        If ``True``, features and target are standardised before training
        and the result is de-standardised at ``predict`` time.
    scale_normalize : bool
        If ``True``, divides features by their standard deviation without
        centring; otherwise standard z-score normalisation is used.
    """

    def __init__(
        self,
        n_features: int,
        interaction_pairs: Optional[Iterable[PairKey]] = None,
        univariate_depth: int = 2,
        bivariate_depth: int = 2,
        use_univariate: bool = True,
        use_input_affine: bool = True,
        feature_names: Optional[List[str]] = None,
        standardize: bool = True,
        scale_normalize: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.univariate_depth = univariate_depth
        self.bivariate_depth = bivariate_depth
        self.use_univariate = use_univariate
        self.use_input_affine = use_input_affine
        self.standardize = standardize
        self.scale_normalize = scale_normalize
        self.feature_names = feature_names or [
            f"x{i + 1}" for i in range(n_features)
        ]
        assert len(self.feature_names) == n_features

        self.bias = nn.Parameter(torch.zeros((), dtype=DTYPE))

        if use_univariate:
            self.univariate_trees = nn.ModuleList(
                [
                    EMLTree(
                        depth=univariate_depth,
                        n_inputs=1,
                        use_input_affine=use_input_affine,
                    )
                    for _ in range(n_features)
                ]
            )
            self.univariate_weights = nn.Parameter(
                torch.ones(n_features, dtype=DTYPE)
            )
        else:
            self.univariate_trees = nn.ModuleList()
            self.register_parameter(
                "univariate_weights", nn.Parameter(torch.zeros(0, dtype=DTYPE))
            )

        self.bivariate_trees = nn.ModuleDict()
        self._pair_order: list[PairKey] = []
        self.bivariate_weights = nn.ParameterList()
        if interaction_pairs:
            for pair in interaction_pairs:
                self.add_interaction(pair)

        self.register_buffer("x_mean", torch.zeros(n_features, dtype=DTYPE))
        self.register_buffer("x_std", torch.ones(n_features, dtype=DTYPE))
        self.register_buffer("y_mean", torch.zeros((), dtype=DTYPE))
        self.register_buffer("y_std", torch.ones((), dtype=DTYPE))
        self._fitted: bool = False

    def add_interaction(self, pair: PairKey) -> None:
        i, j = sorted(pair)
        assert 0 <= i < self.n_features and 0 <= j < self.n_features
        assert i != j, "pair indices must differ"
        key = _pair_key(i, j)
        if key in self.bivariate_trees:
            return
        self.bivariate_trees[key] = EMLTree(
            depth=self.bivariate_depth,
            n_inputs=2,
            use_input_affine=self.use_input_affine,
        )
        self._pair_order.append((i, j))
        self.bivariate_weights.append(nn.Parameter(torch.ones((), dtype=DTYPE)))

    def _all_trees(self) -> list[EMLTree]:
        trees: list[EMLTree] = list(self.univariate_trees)
        trees.extend(self.bivariate_trees.values())
        return trees

    def set_temperature(self, temperature: float) -> None:
        for tree in self._all_trees():
            tree.set_temperature(temperature)

    def snap_all(self) -> None:
        for tree in self._all_trees():
            tree.snap()

    def unsnap_all(self) -> None:
        for tree in self._all_trees():
            tree.unsnap()

    def entropy(self) -> torch.Tensor:
        terms: list[torch.Tensor] = [t.entropy() for t in self._all_trees()]
        if not terms:
            return torch.zeros((), dtype=DTYPE)
        return torch.stack(terms).mean()

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.standardize:
            return x
        if self.scale_normalize:
            return x / self.x_std
        return (x - self.x_mean) / self.x_std

    def forward(
        self, x: torch.Tensor, destandardize_y: bool = False
    ) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(1)
        x_z = self._standardize(x) if self.standardize else x
        out = self.bias.expand(x_z.shape[0]).clone()
        for i, tree in enumerate(self.univariate_trees):
            out = out + self.univariate_weights[i] * tree(x_z[:, i : i + 1])
        for idx, ((i, j), key) in enumerate(
            zip(self._pair_order, self.bivariate_trees.keys())
        ):
            out = out + self.bivariate_weights[idx] * self.bivariate_trees[key](
                x_z[:, [i, j]]
            )
        if destandardize_y and self.standardize:
            out = out * self.y_std + self.y_mean
        return out

    def fit(
        self,
        X,
        y,
        cfg: Optional[TrainConfig] = None,
        interaction_pairs: Optional[Iterable[PairKey]] = None,
        verbose: bool = False,
        warm_start: bool = True,
        use_holdout: bool = True,
        atlas_univariate: Optional[list] = None,
        atlas_bivariate: Optional[list] = None,
    ) -> "EMLGAM":
        """Run the three-stage training schedule on ``(X, y)``."""
        cfg = cfg or TrainConfig()
        x = to_tensor(X)
        y_t = to_tensor(y).reshape(-1)

        if self.standardize:
            with torch.no_grad():
                mean = x.mean(dim=0)
                std = x.std(dim=0).clamp(min=1e-6)
                self.x_mean.copy_(mean)
                self.x_std.copy_(std)
                if self.scale_normalize:
                    # Only normalise features whose scale is pathological.
                    keep_mask = (std >= 0.1) & (std <= 10.0)
                    effective_std = torch.where(
                        keep_mask, torch.ones_like(std), std
                    )
                    self.x_std.copy_(effective_std)
                self.y_mean.copy_(y_t.mean())
                self.y_std.copy_(y_t.std().clamp(min=1e-6))
            y_work = (y_t - self.y_mean) / self.y_std
        else:
            y_work = y_t

        with torch.no_grad():
            self.bias.fill_(y_work.mean().item())

        if interaction_pairs is not None:
            for p in interaction_pairs:
                self.add_interaction(p)

        if warm_start:
            self._warm_start_trees(
                x, y_work, atlas_univariate, atlas_bivariate,
                verbose=verbose, use_holdout=use_holdout,
            )

        optim = torch.optim.Adam(self.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=0.5, patience=50
        )

        n_warmup = int(cfg.n_epochs * cfg.warmup_frac)
        n_harden = int(cfg.n_epochs * cfg.hardening_frac)

        self.unsnap_all()
        self.set_temperature(cfg.temp_start)
        best_mse = float("inf")
        bad = 0

        for epoch in range(n_warmup + n_harden):
            in_hardening = epoch >= n_warmup
            if in_hardening:
                progress = (epoch - n_warmup) / max(n_harden - 1, 1)
                log_t = (
                    math.log(cfg.temp_start) * (1 - progress)
                    + math.log(cfg.temp_end) * progress
                )
                self.set_temperature(math.exp(log_t))

            if cfg.batch_size and cfg.batch_size < x.shape[0]:
                idx = torch.randperm(x.shape[0], device=x.device)[: cfg.batch_size]
                xb, yb = x[idx], y_work[idx]
            else:
                xb, yb = x, y_work

            optim.zero_grad()
            pred = self(xb)
            mse = torch.mean((pred - yb) ** 2)
            loss = mse + cfg.entropy_weight * self.entropy()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), cfg.grad_clip)
            optim.step()
            scheduler.step(mse.item())

            if mse.item() < best_mse - 1e-12:
                best_mse = mse.item()
                bad = 0
            else:
                bad += 1

            if verbose and epoch % max(1, (n_warmup + n_harden) // 20) == 0:
                temp = (
                    self._all_trees()[0].temperature if self._all_trees() else 1.0
                )
                phase = "harden" if in_hardening else "warmup"
                print(
                    f"epoch={epoch:5d} mse={mse.item():.3e} "
                    f"temp={temp:.3f} phase={phase}"
                )
            if (
                not in_hardening
                and bad > cfg.patience
                and epoch > n_warmup // 4
            ):
                break

        self.snap_all()
        self._fitted = True
        return self

    def _warm_start_trees(
        self,
        x: torch.Tensor,
        y_work: torch.Tensor,
        atlas_univariate,
        atlas_bivariate,
        verbose: bool = False,
        use_holdout: bool = True,
    ) -> None:
        """Install warm-start snap configurations by fitting a primitive per
        component against the current residual."""
        from .primitives import default_atlas, warm_start_tree

        x_z = self._standardize(x) if self.standardize else x

        if len(self.univariate_trees) > 0:
            atlas_uni = atlas_univariate or default_atlas(
                depth=self.univariate_depth, n_inputs=1
            )
            residual = y_work.clone()
            with torch.no_grad():
                self.bias.fill_(residual.mean().item())
            residual = residual - self.bias.detach()
            for i, tree in enumerate(self.univariate_trees):
                best = warm_start_tree(
                    tree, atlas_uni, x_z[:, i : i + 1], residual,
                    use_holdout=use_holdout,
                )
                with torch.no_grad():
                    values = tree(x_z[:, i : i + 1])
                    if values.std().item() > 1e-8:
                        cov = (
                            (values - values.mean())
                            * (residual - residual.mean())
                        ).sum()
                        var = ((values - values.mean()) ** 2).sum()
                        beta = (cov / var.clamp(min=1e-12)).item()
                    else:
                        beta = 0.0
                    self.univariate_weights.data[i] = beta
                    residual = residual - beta * values
                if verbose:
                    print(
                        f"  warm-start uni[{i}] <- '{best.name}'  "
                        f"beta={beta:+.3f}  sign={tree.input_scale.tolist()}"
                    )

        if len(self.bivariate_trees) > 0:
            atlas_bi = atlas_bivariate or default_atlas(
                depth=self.bivariate_depth, n_inputs=2
            )
            for idx, ((i, j), key) in enumerate(
                zip(self._pair_order, self.bivariate_trees.keys())
            ):
                tree = self.bivariate_trees[key]
                best = warm_start_tree(
                    tree, atlas_bi, x_z[:, [i, j]], residual,
                    use_holdout=use_holdout,
                )
                with torch.no_grad():
                    values = tree(x_z[:, [i, j]])
                    if values.std().item() > 1e-8:
                        cov = (
                            (values - values.mean())
                            * (residual - residual.mean())
                        ).sum()
                        var = ((values - values.mean()) ** 2).sum()
                        beta = (cov / var.clamp(min=1e-12)).item()
                    else:
                        beta = 0.0
                    self.bivariate_weights[idx].data.fill_(beta)
                    residual = residual - beta * values
                if verbose:
                    print(
                        f"  warm-start bi[{i},{j}] <- '{best.name}'  "
                        f"beta={beta:+.3f}"
                    )

        with torch.no_grad():
            self.bias.add_(residual.mean())

    @torch.no_grad()
    def predict(self, X, clip_factor: float = 0.0) -> np.ndarray:
        """Predict on new data.

        Parameters
        ----------
        clip_factor : float
            If positive, clip predictions to
            ``[y_min - factor * range, y_max + factor * range]``
            where min/max/range are computed from the training target.
            This prevents extrapolation explosions (e.g. exp(0.06 * age)
            on Concrete). Set to 0 (default) to disable.
        """
        x = to_tensor(X)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        out = self(x, destandardize_y=True).cpu().numpy()
        if clip_factor > 0.0 and self._fitted:
            y_lo = float(self.y_mean - 3.0 * self.y_std)
            y_hi = float(self.y_mean + 3.0 * self.y_std)
            y_range = y_hi - y_lo
            out = np.clip(
                out,
                y_lo - clip_factor * y_range,
                y_hi + clip_factor * y_range,
            )
        return out

    def get_formulas(self, simplify: bool = True) -> dict:
        """Return a dictionary mapping component names to SymPy expressions.

        Keys are ``'bias'``, each feature name for main effects, and
        ``'<feat_i>_x_<feat_j>'`` for each bivariate component.
        """
        result: dict[str, sp.Expr] = {
            "bias": sp.Float(float(self.bias.item()))
        }
        if self.standardize:
            subs: dict[sp.Symbol, sp.Expr] = {}
            for i, name in enumerate(self.feature_names):
                std_i = float(self.x_std[i].item())
                if self.scale_normalize:
                    subs[sp.Symbol(name)] = sp.Symbol(name) / std_i
                else:
                    mean_i = float(self.x_mean[i].item())
                    subs[sp.Symbol(name)] = (sp.Symbol(name) - mean_i) / std_i
        else:
            subs = None

        for i, tree in enumerate(self.univariate_trees):
            expr = tree.get_symbolic_expression(
                [self.feature_names[i]], simplify=simplify
            )
            weight = float(self.univariate_weights[i].item())
            expr = sp.Float(weight) * expr
            if subs is not None:
                expr = expr.subs(subs)
            if simplify:
                expr = sp.simplify(expr)
            result[self.feature_names[i]] = expr

        for idx, ((i, j), key) in enumerate(
            zip(self._pair_order, self.bivariate_trees.keys())
        ):
            tree = self.bivariate_trees[key]
            names = [self.feature_names[i], self.feature_names[j]]
            expr = tree.get_symbolic_expression(names, simplify=simplify)
            weight = float(self.bivariate_weights[idx].item())
            expr = sp.Float(weight) * expr
            if subs is not None:
                expr = expr.subs(subs)
            if simplify:
                expr = sp.simplify(expr)
            result[f"{names[0]}_x_{names[1]}"] = expr
        return result

    def total_formula(self, simplify: bool = True) -> sp.Expr:
        """Return the full prediction as a single SymPy expression.

        If ``standardize`` is enabled, the expression includes the
        de-standardisation of the target: ``y = y_std * (bias + sum trees) +
        y_mean``.
        """
        forms = self.get_formulas(simplify=False)
        total = forms["bias"]
        for k, v in forms.items():
            if k == "bias":
                continue
            total = total + v
        if self.standardize:
            total = sp.Float(float(self.y_std.item())) * total + sp.Float(
                float(self.y_mean.item())
            )
        return sp.simplify(total) if simplify else total
