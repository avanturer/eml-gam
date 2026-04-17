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

import copy
import math
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
        self.register_buffer("x_mean_eff", torch.zeros(n_features, dtype=DTYPE))
        self.register_buffer("x_lo", torch.zeros(n_features, dtype=DTYPE))
        self.register_buffer("x_hi", torch.zeros(n_features, dtype=DTYPE))
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
            # Features with extreme scale (very small / very large std) use
            # z-score: the mean is stored in ``x_mean_eff`` and subtracted.
            # Moderate-scale features keep ``x_mean_eff == 0`` and are only
            # divided by ``x_std``. This prevents the warm-start atlas from
            # trying to evaluate primitives on values outside the outlier-
            # filter acceptance band (see the Arrhenius regression fix).
            return (x - self.x_mean_eff) / self.x_std
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
        try_offsets: bool = False,
        n_restarts: int = 1,
        robust: bool = False,
        atlas_univariate: Optional[list] = None,
        atlas_bivariate: Optional[list] = None,
        warm_start_rank: int = 0,
    ) -> "EMLGAM":
        """Run the three-stage training schedule on ``(X, y)``.

        Parameters
        ----------
        n_restarts : int
            If > 1, run the full warm-start + training pipeline this many
            times and keep the result with the lowest training MSE. On
            consecutive restarts the warm-start rotates through the top-k
            atlas candidates (rank 0, 1, 2, ...) rather than picking the
            same primitive deterministically.
        robust : bool
            If ``True``, enable multi-start with ``max(3, n_restarts)``
            restarts and an automatic retry on NaN / Inf failures. Use when
            the target is sensitive to the warm-start choice (e.g.
            bivariate rational functions).
        warm_start_rank : int
            Index into the ranked atlas-candidate list. ``0`` selects the
            overall best, higher values try progressively less well-scoring
            candidates. Primarily used internally by multi-start.
        """
        if robust and n_restarts < 3:
            n_restarts = 3
        if n_restarts > 1:
            return self._fit_multistart(
                X, y, cfg=cfg, interaction_pairs=interaction_pairs,
                verbose=verbose, warm_start=warm_start,
                use_holdout=use_holdout, try_offsets=try_offsets,
                n_restarts=n_restarts, robust=robust,
                atlas_univariate=atlas_univariate,
                atlas_bivariate=atlas_bivariate,
            )
        cfg = cfg or TrainConfig()
        x = to_tensor(X)
        y_t = to_tensor(y).reshape(-1)

        if self.standardize:
            with torch.no_grad():
                mean = x.mean(dim=0)
                std = x.std(dim=0).clamp(min=1e-6)
                self.x_mean.copy_(mean)
                self.x_std.copy_(std)
                self.x_lo.copy_(x.min(dim=0).values)
                self.x_hi.copy_(x.max(dim=0).values)
                if self.scale_normalize:
                    # Three regimes, chosen by empirical stability:
                    #   std in [0.1, 100] — no transformation. Keeps features
                    #     positive where they started positive, so log- and
                    #     reciprocal-primitives remain valid. This is the
                    #     common case (Cobb-Douglas, power law, Michaelis-
                    #     Menten, exp decay, combined gas law).
                    #   std < 0.1 or std > 100 — full z-score. Required for
                    #     Arrhenius-class targets where the native input
                    #     scale of the feature forces any absorbable
                    #     coefficient to span several orders of magnitude,
                    #     which warm-start cannot navigate if the primitive
                    #     values fall outside the outlier-filter band.
                    moderate = (std >= 0.1) & (std <= 100.0)
                    effective_std = torch.where(
                        moderate, torch.ones_like(std), std
                    )
                    effective_mean = torch.where(
                        moderate, torch.zeros_like(mean), mean
                    )
                    self.x_std.copy_(effective_std)
                    self.x_mean_eff.copy_(effective_mean)
                else:
                    self.x_mean_eff.copy_(mean)
                self.y_mean.copy_(y_t.mean())
                self.y_std.copy_(y_t.std().clamp(min=1e-6))
            y_work = (y_t - self.y_mean) / self.y_std
        else:
            y_work = y_t
            with torch.no_grad():
                self.x_lo.copy_(x.min(dim=0).values)
                self.x_hi.copy_(x.max(dim=0).values)

        with torch.no_grad():
            self.bias.fill_(y_work.mean().item())

        if interaction_pairs is not None:
            for p in interaction_pairs:
                self.add_interaction(p)

        if warm_start:
            self._warm_start_trees(
                x, y_work, atlas_univariate, atlas_bivariate,
                verbose=verbose, use_holdout=use_holdout,
                try_offsets=try_offsets, rank=warm_start_rank,
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
        self._last_fit_failed = False
        self._last_final_mse = float("inf")

        extrap_w = float(getattr(cfg, "extrap_penalty_weight", 0.0))
        extrap_margin = float(getattr(cfg, "extrap_margin", 0.5))
        extrap_n = int(getattr(cfg, "extrap_n_samples", 32))
        extrap_max = float(getattr(cfg, "extrap_max_std", 5.0))
        entropy_stop = float(getattr(cfg, "entropy_stop", 0.0))

        if extrap_w > 0.0:
            with torch.no_grad():
                x_lo = x.min(dim=0).values.detach()
                x_hi = x.max(dim=0).values.detach()
                x_rg = (x_hi - x_lo).clamp(min=1e-12)

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

            if extrap_w > 0.0:
                u = torch.rand(
                    extrap_n, x.shape[1], dtype=x.dtype, device=x.device
                )
                lo = x_lo - extrap_margin * x_rg
                hi = x_hi + extrap_margin * x_rg
                x_synth = lo + u * (hi - lo)
                pred_synth = self(x_synth)
                excess = torch.relu(pred_synth.abs() - extrap_max)
                loss = loss + extrap_w * torch.mean(excess ** 2)

            if not torch.isfinite(loss):
                self._last_fit_failed = True
                if verbose:
                    print(f"  [abort] loss non-finite at epoch={epoch}")
                break

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
            if (
                in_hardening
                and entropy_stop > 0.0
                and float(self.entropy().item()) < entropy_stop
            ):
                # Logits already concentrated on a single option at every
                # slot; further hardening steps cannot change the snap.
                break

        self.snap_all()
        with torch.no_grad():
            pred_final = self(x)
            if torch.isfinite(pred_final).all():
                self._last_final_mse = float(
                    torch.mean((pred_final - y_work) ** 2).item()
                )
            else:
                self._last_fit_failed = True
                self._last_final_mse = float("inf")
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
        try_offsets: bool = False,
        rank: int = 0,
    ) -> None:
        """Install warm-start snap configurations by fitting a primitive per
        component against the current residual.

        ``rank == 0`` matches the default best-primitive behaviour;
        ``rank > 0`` rotates bivariate components through alternative
        candidates on multi-start retries. Univariate components are
        intentionally left at rank 0 because their atlas is well-ordered
        by simplicity and rotating them tends to regress more than it
        helps; bivariate rotation is where the multi-start win comes
        from on rational / interaction targets.
        """
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
            uni_rank = rank // 2 if rank > 0 else 0
            for i, tree in enumerate(self.univariate_trees):
                best = warm_start_tree(
                    tree, atlas_uni, x_z[:, i : i + 1], residual,
                    use_holdout=use_holdout, try_offsets=try_offsets,
                    rank=uni_rank,
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
                        f"  warm-start uni[{i}] <- '{best.name}' "
                        f"rank={uni_rank} beta={beta:+.3f} "
                        f"sign={tree.input_scale.tolist()}"
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
                    use_holdout=use_holdout, try_offsets=try_offsets,
                    rank=rank,
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
                        f"  warm-start bi[{i},{j}] <- '{best.name}' "
                        f"rank={rank} beta={beta:+.3f}"
                    )

        with torch.no_grad():
            self.bias.add_(residual.mean())

    def _fit_multistart(
        self, X, y, *, n_restarts, robust=False, **kwargs
    ) -> "EMLGAM":
        """Run ``fit`` multiple times and keep the best result.

        Consecutive restarts rotate the bivariate warm-start rank
        (0, 1, 2, ...), so each restart explores a different region of
        primitive space instead of deterministically re-picking the same
        candidate. Failed runs (NaN / Inf loss, or final MSE non-finite)
        are skipped. When ``robust`` is true, an additional retry with a
        fresh random seed is spawned for every failure, up to
        ``2 * n_restarts`` attempts in total.
        """
        best_state = None
        best_mse = float("inf")
        verbose = kwargs.get("verbose", False)
        x_t = to_tensor(X)
        y_t = to_tensor(y).reshape(-1)
        max_attempts = 2 * n_restarts if robust else n_restarts
        attempt = 0
        successful = 0
        while attempt < max_attempts and successful < n_restarts:
            attempt += 1
            # Reset all tree parameters for a fresh start.
            for tree in self._all_trees():
                tree.reset_parameters()
            self.bias.data.zero_()
            if self.use_univariate:
                self.univariate_weights.data.fill_(1.0)
            for w in self.bivariate_weights:
                w.data.fill_(1.0)
            self._fitted = False

            kwargs_single = dict(kwargs)
            kwargs_single["n_restarts"] = 1
            kwargs_single["robust"] = False
            kwargs_single["warm_start_rank"] = successful
            self.fit(X, y, **kwargs_single)

            failed = bool(getattr(self, "_last_fit_failed", False))
            with torch.no_grad():
                pred = self(x_t)
                if not torch.isfinite(pred).all():
                    failed = True
                    mse = float("inf")
                else:
                    y_work = (
                        (y_t - self.y_mean) / self.y_std
                        if self.standardize else y_t
                    )
                    mse = float(torch.mean((pred - y_work) ** 2).item())
            if verbose:
                status = "FAIL" if failed else "ok"
                print(
                    f"  [restart {attempt}/{max_attempts}] "
                    f"rank={successful} mse={mse:.3e} [{status}]"
                )
            if failed:
                continue
            successful += 1
            if mse < best_mse:
                best_mse = mse
                best_state = copy.deepcopy(self.state_dict())
        if best_state is not None:
            self.load_state_dict(best_state)
        else:
            if verbose:
                print("  [multistart] all attempts failed; keeping last state")
        self.snap_all()
        self._fitted = True
        self._last_final_mse = best_mse
        return self

    def param_summary(self) -> dict:
        """Return a breakdown of EMLGAM parameters relevant for complexity
        analysis and the effective-DoF discussion in the theory section.

        Keys:
          ``n_raw_logits`` : total softmax logits (only live before snap)
          ``n_slot_choices`` : integer slot DoF that vanish after snap
          ``n_continuous_post_snap`` : scale + offset + output weights + bias
        """
        trees = self._all_trees()
        n_out_weights = len(self.univariate_trees) + len(self.bivariate_trees)
        return {
            "n_trees": len(trees),
            "n_features": self.n_features,
            "n_pairs": len(self.bivariate_trees),
            "n_raw_logits": sum(t.n_params for t in trees),
            "n_slot_choices": sum(t.n_slots for t in trees),
            "n_continuous_post_snap": (
                sum(t.n_continuous_post_snap for t in trees)
                + n_out_weights
                + 1  # bias
            ),
        }

    @torch.no_grad()
    def predict(
        self,
        X,
        clip_factor: float = 0.0,
        input_clip_factor: float = 0.0,
    ) -> np.ndarray:
        """Predict on new data.

        Parameters
        ----------
        clip_factor : float
            If positive, clip predictions to
            ``[y_min - factor * range, y_max + factor * range]``
            where min/max/range are computed from the training target.
            This prevents extrapolation explosions (for example
            ``exp(0.06 * age)`` on the UCI Concrete dataset).
        input_clip_factor : float
            If positive, clamp inputs to
            ``[x_lo - factor * range, x_hi + factor * range]`` before
            evaluating the tree. With ``factor = 2`` the model is allowed
            to extrapolate by up to two training-range widths beyond the
            observed min/max; anything further is clamped to the clip
            boundary. Set to 0 (default) to disable.
        """
        x = to_tensor(X)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if input_clip_factor > 0.0 and self._fitted:
            x_range = (self.x_hi - self.x_lo).clamp(min=1e-12)
            lo = self.x_lo - input_clip_factor * x_range
            hi = self.x_hi + input_clip_factor * x_range
            x = torch.max(torch.min(x, hi), lo)
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
