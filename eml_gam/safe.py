"""SafePoolGAM: pool-backfitting additive model with certified
extrapolation gating.

The two chronic failure modes of symbolic-GAM regressors on real
tabular data are (i) a weak shape-function search (a 14-entry
hand-coded atlas, or gradient descent that dies against the depth
wall) and (ii) *uncertified extrapolation*: an exp-composition that
fits the training range well can be off by four orders of magnitude
one standard deviation outside it — the ``R^2 = -8245`` failures of
the earlier 9-dataset UCI sweep.

SafePoolGAM addresses both with machinery already in this repository:

1. **Shape functions come from the canonical pool** — the complete
   deduplicated atlas of every function an EML tree of the configured
   depth can express (:mod:`eml_gam.canonical_pool`). Selecting a
   shape is a vectorised OLS sweep over the entire class: no gradient
   descent, no depth wall, exact and fast.
2. **Tail-validated selection.** Candidate shapes are ranked not by
   in-sample fit but by R^2 on held-out *tail* slices (the outer
   quantiles of each feature), which is where extrapolation behaviour
   lives. A shape that only wins in the bulk is worth nothing at the
   extrapolation frontier.
3. **Per-feature extrapolation gates, earned not granted.** *Every*
   component — including plain linear terms — is evaluated at the
   clipped (flat-beyond-the-training-box) input unless its free
   extension beats the clipped one by a margin on the feature's tail
   slices, judged with the full additive model fit on the inner rows.
   Flat extension is exactly what a tree ensemble would produce; free
   extension must be earned with evidence.
4. **Frontier family selection.** The final symbolic model competes
   against two structurally-flat challengers — boxed-linear (linear
   inside the training box, flat outside) and the constant mean — on
   the extrapolation frontier (tail rows), each scored with weights
   fit on inner rows. A challenger deploys only if it wins by a clear
   margin; raw unbounded linear extrapolation is never deployed.
5. **A declared output envelope.** Predictions are clamped to the
   training-target range widened by ``envelope_margin`` target spans
   per side — the last line of defence, catching only
   orders-of-magnitude insanity.

The result keeps the closed-form wins where structure exists (the
model can genuinely leave the training range, unlike any tree model)
while being *constructionally incapable* of the catastrophic losses of
its unguarded predecessor: every deployed prediction path is either
flat beyond the box or individually certified on held-out tails.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .canonical_pool import FunctionPool

U_LO, U_HI = 0.3, 2.5  # canonical input range (the pool grammar's habitat)


@dataclass
class _Component:
    feature: int
    kind: str  # "pool" | "linear" | "exp" | "pow" | "log" | "dropped"
    nested: Optional[tuple] = None  # pool expression when kind == "pool"
    param: float = 0.0  # rate/exponent for the parametric families
    tail_r2: float = -np.inf
    gate_free: bool = False  # True: evaluate beyond the training range


def _shape_values(kind: str, param: float, nested, pool, u: np.ndarray) -> np.ndarray:
    if kind == "linear":
        return u.astype(np.float64)
    if kind == "exp":
        return np.exp(np.clip(param * u, -60.0, 60.0))
    if kind == "pow":
        return np.power(np.maximum(u, 1e-9), param)
    if kind == "log":
        return np.log(np.maximum(u, 1e-9))
    if kind == "pool":
        return pool.evaluate_nested(nested, u)
    return np.zeros_like(u)


class SafePoolGAM:
    """Additive regression with complete-atlas shape functions and
    certified extrapolation gating.

    Parameters
    ----------
    depth : root depth of candidate shape functions (the pool is built
        to ``depth - 1``; depth 4 keeps the sweep at ~1.4M candidates
        per feature per round, a few seconds each).
    n_rounds : backfitting rounds.
    tail_quantile : per-feature fraction (each side) held out for
        tail validation.
    n_search : max points used during shape search (full data is
        always used for the final joint refit).
    envelope_margin : output clamp width in units of the training
        target span, added on each side of the observed range.
    min_gain : minimal tail-R^2 advantage over the linear term
        required to accept a pool shape (Occam guard).
    """

    #: decay/growth-rate grid for the parametric ``exp(k u)`` family and
    #: exponent grid for the ``u**k`` family. The pool grammar has no
    #: input affine, so calibrated-rate classical shapes are offered as
    #: explicit candidates alongside the pool exotics.
    EXP_RATES = tuple(
        s * r
        for s in (1.0, -1.0)
        for r in (0.25, 0.4, 0.6, 0.8, 1.0, 1.25, 1.6, 2.0, 2.6, 3.4, 4.5, 6.0, 8.0)
    )
    POW_EXPONENTS = (-3.0, -2.0, -1.0, -0.5, 0.5, 2.0, 3.0, 4.0)

    def __init__(
        self,
        depth: int = 4,
        n_rounds: int = 2,
        tail_quantile: float = 0.1,
        n_search: int = 900,
        envelope_margin: float = 20.0,
        min_gain: float = 0.03,
        drop_threshold: float = 0.10,
        gate_margin: float = 0.02,
        free_min_tail: float = 0.30,
        max_active_features: int = 12,
        top_k_candidates: int = 128,
        random_state: int = 0,
        verbose: bool = False,
    ):
        self.depth = depth
        self.n_rounds = n_rounds
        self.tail_quantile = tail_quantile
        self.n_search = n_search
        self.envelope_margin = envelope_margin
        self.min_gain = min_gain
        self.drop_threshold = drop_threshold
        self.gate_margin = gate_margin
        self.free_min_tail = free_min_tail
        self.max_active_features = max_active_features
        self.top_k_candidates = top_k_candidates
        self.random_state = random_state
        self.verbose = verbose

    # -- canonical input map -------------------------------------------------

    def _to_canonical(self, X: np.ndarray) -> np.ndarray:
        U = np.empty_like(X, dtype=np.float64)
        for j in range(X.shape[1]):
            lo, hi = self._x_lo[j], self._x_hi[j]
            span = hi - lo
            if span <= 0:
                U[:, j] = 0.5 * (U_LO + U_HI)
            else:
                U[:, j] = U_LO + (U_HI - U_LO) * (X[:, j] - lo) / span
        return U

    # -- fitting -------------------------------------------------------------

    def fit(self, X, y) -> "SafePoolGAM":
        t_start = time.time()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, p = X.shape
        rng = np.random.default_rng(self.random_state)

        self._x_lo = X.min(axis=0)
        self._x_hi = X.max(axis=0)
        self._y_lo, self._y_hi = float(y.min()), float(y.max())
        U = self._to_canonical(X)

        # Search subset.
        if n > self.n_search:
            sub = rng.choice(n, self.n_search, replace=False)
        else:
            sub = np.arange(n)
        U_s, y_s = U[sub], y[sub]
        ns = len(sub)

        # Feature screening: on wide datasets only the strongest
        # univariate signals get a symbolic search (the rest are
        # dropped, exactly as an L1 path would do — documented).
        active = list(range(p))
        if p > self.max_active_features:
            scores = []
            yc = y_s - y_s.mean()
            for j in range(p):
                uj = U_s[:, j]
                s = np.std(uj)
                scores.append(
                    0.0 if s < 1e-12 else abs(np.mean((uj - uj.mean()) * yc)) / s
                )
            order = np.argsort(scores)[::-1]
            active = sorted(int(j) for j in order[: self.max_active_features])

        # Per-feature pools and tail masks on the search subset.
        pools: list[Optional[FunctionPool]] = [None] * p
        tails: list[np.ndarray] = [np.zeros(ns, dtype=bool)] * p
        for j in active:
            uj = U_s[:, j]
            if np.std(uj) < 1e-12:
                continue
            pool = FunctionPool(uj, operator="eml")
            pool.build(self.depth - 1)
            pool.compact()
            pools[j] = pool
            lo_q = np.quantile(uj, self.tail_quantile)
            hi_q = np.quantile(uj, 1.0 - self.tail_quantile)
            tails[j] = (uj <= lo_q) | (uj >= hi_q)

        # Backfitting on the search subset.
        comp: list[_Component] = [
            _Component(feature=j, kind="dropped") for j in range(p)
        ]
        contrib = np.zeros((ns, p))
        bias = float(np.mean(y_s))

        for rnd in range(self.n_rounds):
            for j in active:
                pool = pools[j]
                if pool is None:
                    continue
                r = y_s - bias - contrib.sum(axis=1) + contrib[:, j]
                tail = tails[j]
                inner = ~tail
                if tail.sum() < 8 or inner.sum() < 16:
                    tail = np.zeros(ns, dtype=bool)
                    inner = ~tail
                uj = U_s[:, j]

                # (tail_r2, kind, nested, param, values)
                cands: list[tuple] = []
                lin_score, lin_fit = self._tail_score(uj, r, inner, tail)
                cands.append((lin_score, "linear", None, 0.0, lin_fit))
                # Parametric classical shapes with calibrated rates —
                # the pool grammar has no input affine, so these carry
                # the scale burden for exp/power/log laws.
                for k in self.EXP_RATES:
                    v = _shape_values("exp", k, None, None, uj)
                    s, f = self._tail_score(v, r, inner, tail)
                    cands.append((s, "exp", None, k, f))
                for k in self.POW_EXPONENTS:
                    v = _shape_values("pow", k, None, None, uj)
                    s, f = self._tail_score(v, r, inner, tail)
                    cands.append((s, "pow", None, k, f))
                v = _shape_values("log", 0.0, None, None, uj)
                s, f = self._tail_score(v, r, inner, tail)
                cands.append((s, "log", None, 0.0, f))
                # Pool exotics: pre-rank by full-subset OLS, then
                # tail-validate the top slice.
                top = pool.exhaustive_root(
                    r, self.depth, top_k=self.top_k_candidates
                )
                for cand in top:
                    vals = pool.evaluate_nested(cand["nested"], uj)
                    if not np.all(np.isfinite(vals)) or np.std(vals) < 1e-12:
                        continue
                    score, fitted = self._tail_score(vals, r, inner, tail)
                    cands.append((score, "pool", cand["nested"], 0.0, fitted))

                cands.sort(key=lambda t: -t[0])
                best = cands[0]
                # Occam guard: exotic pool shapes must beat the best
                # classical (linear/exp/pow/log) by min_gain.
                if best[1] == "pool":
                    best_classical = max(
                        (c for c in cands if c[1] != "pool"),
                        key=lambda c: c[0],
                    )
                    if best[0] < best_classical[0] + self.min_gain:
                        best = best_classical
                # Local rate refinement for the exp family: the grid is
                # log-spaced, the tails are sensitive to the exact rate.
                if best[1] == "exp":
                    k_lo, k_hi = best[3] / 1.35, best[3] * 1.35
                    for _ in range(10):
                        k_m1 = k_lo + (k_hi - k_lo) / 3
                        k_m2 = k_hi - (k_hi - k_lo) / 3
                        s1, f1 = self._tail_score(
                            _shape_values("exp", k_m1, None, None, uj), r, inner, tail
                        )
                        s2, f2 = self._tail_score(
                            _shape_values("exp", k_m2, None, None, uj), r, inner, tail
                        )
                        if s1 >= s2:
                            k_hi = k_m2
                            if s1 > best[0]:
                                best = (s1, "exp", None, k_m1, f1)
                        else:
                            k_lo = k_m1
                            if s2 > best[0]:
                                best = (s2, "exp", None, k_m2, f2)
                # Drop rule: a component must explain the combined
                # tails AND not be actively harmful on either side.
                if best[0] < self.drop_threshold or not self._both_sides_ok(
                    best[4], r, uj, tail
                ):
                    comp[j] = _Component(feature=j, kind="dropped")
                    contrib[:, j] = 0.0
                    continue
                comp[j] = _Component(
                    feature=j, kind=best[1], nested=best[2],
                    param=best[3], tail_r2=best[0],
                )
                contrib[:, j] = best[4]
                if self.verbose:
                    print(
                        f"  round {rnd} feat {j}: {best[1]} "
                        f"param={best[3]:+.2f} (tail R2 {best[0]:+.3f})"
                    )
            bias = float(np.mean(y_s - contrib.sum(axis=1)))

        self.components_ = comp
        self._pools = pools

        # Backward elimination on the frontier: a component stays only
        # if removing it hurts the additive model's R^2 on the union
        # of all tail slices. Kills shapes that merely re-explain
        # variance already owned by a stronger feature.
        frontier = np.zeros(ns, dtype=bool)
        for j in active:
            frontier |= tails[j]
        if frontier.sum() >= 12:
            inner_all = ~frontier

            def frontier_r2(active_idx: list[int]) -> float:
                cols = [np.ones(ns)] + [contrib[:, j] for j in active_idx]
                Phi_s = np.stack(cols, axis=1)
                w, *_ = np.linalg.lstsq(
                    Phi_s[inner_all], y_s[inner_all], rcond=None
                )
                pred = Phi_s[frontier] @ w
                ss_tot = np.sum((y_s[frontier] - y_s[frontier].mean()) ** 2)
                if ss_tot < 1e-16:
                    return 0.0
                return 1.0 - np.sum((y_s[frontier] - pred) ** 2) / ss_tot

            live = [j for j in active if comp[j].kind != "dropped"]
            improved = True
            while improved and len(live) > 1:
                improved = False
                base = frontier_r2(live)
                scores = [
                    (frontier_r2([k for k in live if k != j]), j) for j in live
                ]
                best_without, j_worst = max(scores)
                if best_without >= base - 0.01:
                    live.remove(j_worst)
                    comp[j_worst] = _Component(feature=j_worst, kind="dropped")
                    contrib[:, j_worst] = 0.0
                    improved = True
                    if self.verbose:
                        print(f"  pruned feature {j_worst} (frontier)")

        # Extrapolation gates, decided on ALL rows with the full
        # additive model: feature j keeps its free extension beyond the
        # training range only if (i) its shape was strongly
        # tail-validated during selection and (ii) the full model with
        # j free beats the full model with j clipped by ``gate_margin``
        # on j's full-data tail slice. Borderline cases default to the
        # clipped (flat, tree-like) extension — conservatism is the
        # point.
        def _predict_variant(U_var: np.ndarray, gates: list[bool],
                             fit_mask: np.ndarray) -> np.ndarray:
            for jj, c2 in enumerate(comp):
                c2.gate_free = gates[jj]
            Phi_g = self._design_matrix(U_var)
            w_g, *_ = np.linalg.lstsq(
                Phi_g[fit_mask], y[fit_mask], rcond=None
            )
            return Phi_g @ w_g

        gates = [False for _ in comp]
        candidates_free = [
            j for j in active
            if comp[j].kind != "dropped"
            and pools[j] is not None
            and comp[j].tail_r2 >= self.free_min_tail
        ]
        for j in candidates_free:
            uj_full = U[:, j]
            lo_q = np.quantile(uj_full, self.tail_quantile)
            hi_q = np.quantile(uj_full, 1.0 - self.tail_quantile)
            tail_full = (uj_full <= lo_q) | (uj_full >= hi_q)
            inner_full = ~tail_full
            if tail_full.sum() < 12 or inner_full.sum() < 24:
                continue
            # Simulate flat-beyond-inner-range behaviour for feature j
            # and let both variants predict j's actual tail data from a
            # fit on the inner rows only.
            g = list(gates)
            g[j] = True
            U_clipped = U.copy()
            U_clipped[:, j] = np.clip(
                uj_full, float(uj_full[inner_full].min()),
                float(uj_full[inner_full].max()),
            )
            pred_free = _predict_variant(U, g, inner_full)
            pred_clip = _predict_variant(U_clipped, g, inner_full)
            ss_tot = np.sum((y[tail_full] - y[tail_full].mean()) ** 2)
            if ss_tot < 1e-16:
                continue
            r2_free = 1 - np.sum((y[tail_full] - pred_free[tail_full]) ** 2) / ss_tot
            r2_clip = 1 - np.sum((y[tail_full] - pred_clip[tail_full]) ** 2) / ss_tot
            if r2_free >= r2_clip + self.gate_margin:
                gates[j] = True

        for jj, c2 in enumerate(comp):
            c2.gate_free = gates[jj]

        # Frontier model selection: the symbolic model must *earn* its
        # deployment by beating plain linear regression and the
        # constant predictor on the extrapolation frontier (union of
        # all feature tails), each scored with weights fit on the inner
        # rows only. Whichever family wins is refit on all rows. This
        # is the final never-catastrophic guarantee: when no closed
        # form certifies, the model degrades to linear or to the mean
        # — never to a confidently wrong formula.
        def tails_of(js: list[int]) -> np.ndarray:
            mask = np.zeros(n, dtype=bool)
            for j2 in js:
                ujf = U[:, j2]
                if np.std(ujf) < 1e-12:
                    continue
                lo_q = np.quantile(ujf, self.tail_quantile)
                hi_q = np.quantile(ujf, 1.0 - self.tail_quantile)
                mask |= (ujf <= lo_q) | (ujf >= hi_q)
            return mask

        frontier_full = tails_of(active)
        if (~frontier_full).sum() < 24 and len(active) > 1:
            # The union of every feature's tails swallowed the data set;
            # fall back to the tails of the strongest single feature.
            yc_all = y - y.mean()
            strength = []
            for j2 in active:
                s2 = np.std(U[:, j2])
                strength.append(
                    0.0 if s2 < 1e-12
                    else abs(np.mean((U[:, j2] - U[:, j2].mean()) * yc_all)) / s2
                )
            top = active[int(np.argmax(strength))]
            frontier_full = tails_of([top])
        inner_rows = ~frontier_full
        self.model_family_ = "symbolic"
        if frontier_full.sum() >= 12 and inner_rows.sum() >= 24:
            Phi_sym = self._design_matrix(U)
            U_boxed = np.clip(U, U_LO, U_HI)
            X_lin = np.column_stack([np.ones(n), U_boxed])

            def frontier_score(Phi_m: np.ndarray) -> float:
                w_m, *_ = np.linalg.lstsq(
                    Phi_m[inner_rows], y[inner_rows], rcond=None
                )
                pred = Phi_m[frontier_full] @ w_m
                ss_tot = np.sum(
                    (y[frontier_full] - y[frontier_full].mean()) ** 2
                )
                if ss_tot < 1e-16:
                    return 0.0
                return 1.0 - np.sum((y[frontier_full] - pred) ** 2) / ss_tot

            s_sym = frontier_score(Phi_sym)
            s_lin = frontier_score(X_lin)
            s_const = 0.0  # the constant predictor scores R^2 = 0 - eps
            # Symbolic keeps deployment unless a structurally-flat
            # challenger beats it by a clear margin. Every challenger
            # is flat beyond the training box by construction — raw
            # unbounded linear extrapolation is never deployed blind.
            best_family = "symbolic"
            if s_lin > max(s_sym, s_const) + 0.02:
                best_family = "linear"
            elif s_const > s_sym + 0.02:
                best_family = "mean"
            self.model_family_ = best_family
            self.frontier_scores_ = {
                "symbolic": s_sym, "linear": s_lin, "mean": s_const,
            }

        # Deploy the winning family, refit on ALL rows.
        if self.model_family_ == "symbolic":
            Phi = self._design_matrix(U)
            lam = 1e-8 * Phi.shape[0]
            self._weights = np.linalg.solve(
                Phi.T @ Phi + lam * np.eye(Phi.shape[1]), Phi.T @ y
            )
        elif self.model_family_ == "linear":
            U_boxed = np.clip(U, U_LO, U_HI)
            X_lin = np.column_stack([np.ones(n), U_boxed])
            self._lin_weights, *_ = np.linalg.lstsq(X_lin, y, rcond=None)
        else:
            self._mean_value = float(y.mean())
        self.fit_seconds_ = time.time() - t_start
        return self

    @staticmethod
    def _tail_score(vals, r, inner, tail) -> tuple[float, np.ndarray]:
        """Fit ``r ~ a + b*vals`` on ``inner``; score R^2 on ``tail``
        (falls back to overall R^2 when the tail is empty). Returns
        ``(score, fitted values on all rows)``."""
        v_i, r_i = vals[inner], r[inner]
        v_var = np.var(v_i)
        if v_var < 1e-16:
            return -np.inf, np.zeros_like(vals)
        beta = np.cov(v_i, r_i, bias=True)[0, 1] / v_var
        alpha = r_i.mean() - beta * v_i.mean()
        fitted = alpha + beta * vals
        mask = tail if tail.any() else np.ones_like(tail, dtype=bool)
        ss_res = np.sum((r[mask] - fitted[mask]) ** 2)
        ss_tot = np.sum((r[mask] - r[mask].mean()) ** 2)
        if ss_tot < 1e-16:
            return -np.inf, fitted
        return 1.0 - ss_res / ss_tot, fitted

    @staticmethod
    def _both_sides_ok(fitted, r, u, tail, tol: float = 1.10) -> bool:
        """A shape must not be *worse than its own absence* on either
        tail side separately (junk shapes often look fine on the
        combined tails while bending the wrong way on one side).

        The comparison is against the no-component prediction (zero
        residual contribution), not against the side's own mean — a
        side-mean constant is an oracle fitted on the very slice being
        scored, and rejecting shapes for losing to it vetoes perfectly
        good components whenever the slice's residual variance is
        dominated by other features' noise.
        """
        if not tail.any():
            return True
        med = np.median(u)
        for side in (tail & (u <= med), tail & (u > med)):
            if side.sum() < 4:
                continue
            ss_shape = np.sum((r[side] - fitted[side]) ** 2)
            ss_none = np.sum(r[side] ** 2)
            if ss_shape > tol * ss_none + 1e-12:
                return False
        return True

    @staticmethod
    def _const_score(r, tail) -> float:
        mask = tail if tail.any() else np.ones_like(tail, dtype=bool)
        ss_tot = np.sum((r[mask] - r[mask].mean()) ** 2)
        ss_res = np.sum(r[mask] ** 2)
        if ss_tot < 1e-16:
            return -np.inf
        return 1.0 - ss_res / ss_tot

    # -- prediction ----------------------------------------------------------

    def _component_values(self, U: np.ndarray, j: int) -> np.ndarray:
        c = self.components_[j]
        if c.kind == "dropped":
            return np.zeros(U.shape[0])
        u = U[:, j]
        if not c.gate_free:
            u = np.clip(u, U_LO, U_HI)
        else:
            # Even a certified formula is not trusted arbitrarily far:
            # inputs are confined to 3 canonical spans beyond the box.
            span = U_HI - U_LO
            u = np.clip(u, U_LO - 3 * span, U_HI + 3 * span)
        return _shape_values(c.kind, c.param, c.nested, self._pools[j], u)

    def _design_matrix(self, U: np.ndarray) -> np.ndarray:
        cols = [np.ones(U.shape[0])]
        for j in range(U.shape[1]):
            vals = self._component_values(U, j)
            vals = np.where(np.isfinite(vals), vals, 0.0)
            cols.append(vals)
        return np.stack(cols, axis=1)

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        U = self._to_canonical(X)
        family = getattr(self, "model_family_", "symbolic")
        if family == "symbolic":
            yhat = self._design_matrix(U) @ self._weights
        elif family == "linear":
            U_boxed = np.clip(U, U_LO, U_HI)
            yhat = np.column_stack(
                [np.ones(U_boxed.shape[0]), U_boxed]
            ) @ self._lin_weights
        else:
            yhat = np.full(U.shape[0], self._mean_value)
        span = self._y_hi - self._y_lo
        m = self.envelope_margin * max(span, 1e-12)
        return np.clip(yhat, self._y_lo - m, self._y_hi + m)

    # -- readout -------------------------------------------------------------

    def formulas(self, feature_names: Optional[list[str]] = None) -> dict:
        import sympy as sp

        names = feature_names or [
            f"x{j+1}" for j in range(len(self.components_))
        ]
        out = {}
        family = getattr(self, "model_family_", "symbolic")
        if family != "symbolic":
            return {"__deployed__": {
                "expr": family, "gate": "n/a",
                "tail_r2": float("nan"),
            }}
        for j, c in enumerate(self.components_):
            w = float(self._weights[1 + j])
            if c.kind == "dropped" or abs(w) < 1e-12:
                continue
            xj = sp.Symbol(names[j])
            lo, hi = float(self._x_lo[j]), float(self._x_hi[j])
            span = hi - lo if hi > lo else 1.0
            u_expr = U_LO + (U_HI - U_LO) * (xj - lo) / span
            if c.kind == "linear":
                shape = u_expr
            elif c.kind == "exp":
                shape = sp.exp(sp.Float(c.param) * u_expr)
            elif c.kind == "pow":
                shape = u_expr ** sp.Float(c.param)
            elif c.kind == "log":
                shape = sp.log(u_expr)
            else:
                shape = self._pools[j].nested_to_sympy(
                    c.nested, feature_names=["__u__"]
                ).subs(sp.Symbol("__u__"), u_expr)
            gate = "free" if c.gate_free else "clipped"
            out[names[j]] = {
                "expr": sp.nsimplify(w, rational=False) * shape,
                "gate": gate,
                "tail_r2": c.tail_r2,
            }
        return out


__all__ = ["SafePoolGAM"]
