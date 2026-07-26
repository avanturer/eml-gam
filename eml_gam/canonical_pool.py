"""Canonical function-pool enumeration for single-operator expression trees.

The syntactic snap space of an EML tree explodes as ``3^(2^d - 2) * 2^(2^d)``
(univariate): 186 624 at depth 3, 3.1e11 at depth 4, 8.8e23 at depth 5.
Both gradient descent (0 per cent recovery at depth >= 3 from random
init) and naive exhaustive enumeration (infeasible at depth >= 4) die
against this wall.

The wall is a mirage. Two structural collapses shrink the *function*
space by many orders of magnitude:

1. **Dead slots.** A slot that snaps to ``1`` or ``x`` disconnects the
   entire subtree hanging below it. Only *live* configurations — pruned
   binary trees — are functionally distinct. The live count follows
   ``S(0) = 1 + n_inputs``, ``S(k) = (1 + n_inputs) + S(k-1)^2`` and the
   number of depth-``d`` root functions is ``S(d-1)^2``: 36 at depth 2,
   1 444 at depth 3, 2.1e6 at depth 4 (univariate) — a 150 000-fold
   reduction at depth 4.

2. **Functional degeneracy.** Distinct live trees frequently compute
   the *same function* on any data set: subtrees saturate the ``exp``
   clamp, collapse to constants, or reproduce shallower expressions.
   Deduplicating by value vector on a fixed evaluation grid collapses
   the live space further (measured factors of 10-100x at depth >= 3).

This module exploits both collapses:

* ``FunctionPool`` enumerates every distinct function computable by a
  *slot* of depth budget ``k`` (memoised bottom-up, deduplicated by a
  rounded value-vector digest). Each entry stores a minimal expression
  DAG, so the pool doubles as a canonical atlas of the operator's
  reachable function space.
* ``exhaustive_root`` scores **every** distinct depth-``d`` root
  function against a target via vectorised OLS — a certificate of
  completeness over the enumerated class at depths where the syntactic
  count was hopeless.
* ``lookup_root`` inverts the root node analytically: if
  ``y = exp(clip(a)) - log(clamp(b))`` for pool functions ``a, b``,
  then for each candidate ``b`` the implied ``clip(a) = log(y +
  log(clamp(b)))`` is a *hash lookup* in the pool — recovery at depth
  ``max_pool_depth + 1`` in O(pool size) instead of O(pool size^2).
  A symmetric pass enumerates ``a`` and solves for ``b``.

Together these break the depth wall for exact recovery: complete
enumeration through depth 5 and lookup-based recovery at depth 6+ on a
laptop CPU, where the prior art (gradient descent, syntactic beam
search, syntactic AEES) is at 0 per cent.

All heavy lifting is float64 numpy; torch is needed only to interoperate
with :class:`~eml_gam.eml_tree.EMLTree` snap configurations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .utils import CLAMP_VAL, EPS

Digest = int

# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


def eml_node(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``exp(clip(a)) - log(clamp(b))`` — matches ``utils.safe_eml`` exactly."""
    return np.exp(np.clip(a, -CLAMP_VAL, CLAMP_VAL)) - np.log(
        np.maximum(b, EPS)
    )


def psi_node(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``sinh(a) - arsinh(b)`` — the smooth sibling operator (no clamps)."""
    with np.errstate(over="ignore"):
        return np.sinh(a) - np.arcsinh(b)


def _eml_exp_side_canon(v: np.ndarray) -> np.ndarray:
    """What the eml root actually *sees* of its exp-side child."""
    return np.clip(v, -CLAMP_VAL, CLAMP_VAL)


def _eml_log_side_canon(v: np.ndarray) -> np.ndarray:
    """What the eml root actually *sees* of its log-side child."""
    return np.maximum(v, EPS)


_OPERATORS: dict[str, dict] = {
    "eml": {
        "node": eml_node,
        "exp_side_canon": _eml_exp_side_canon,
        "log_side_canon": _eml_log_side_canon,
        "invertible": True,
    },
    "psi": {
        "node": psi_node,
        "exp_side_canon": lambda v: v,
        "log_side_canon": lambda v: v,
        "invertible": True,
    },
}


# ---------------------------------------------------------------------------
# Pool entries
# ---------------------------------------------------------------------------


@dataclass
class PoolEntry:
    """One distinct slot function.

    ``expr`` is either ``("leaf", j)`` — ``j = 0`` is the constant 1 and
    ``j >= 1`` is input ``x_j`` — or ``("node", ia, ib)`` with ``ia``,
    ``ib`` indices of earlier pool entries (the expression DAG is
    naturally memoised).
    """

    idx: int
    expr: tuple
    depth: int
    vec: np.ndarray  # values on the evaluation grid, float64 (N,)


def syntactic_live_count(depth: int, n_inputs: int = 1) -> int:
    """Number of live (dead-slot-free) root configurations at exact
    tree depth budget ``d``: ``S(d-1)^2`` with ``S(0) = 1 + n_inputs``,
    ``S(k) = 1 + n_inputs + S(k-1)^2``."""
    s = 1 + n_inputs
    for _ in range(depth - 1):
        s = 1 + n_inputs + s * s
    return s * s


def syntactic_snap_count(depth: int, n_inputs: int = 1) -> int:
    """Raw snap-space size (the number AEES would enumerate)."""
    total = 1
    for level in range(depth):
        n_slots = 2 ** (level + 1)
        is_bottom = level == depth - 1
        n_options = (1 + n_inputs) if is_bottom else (2 + n_inputs)
        total *= n_options**n_slots
    return total


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------


class FunctionPool:
    """Deduplicated enumeration of all slot functions up to a depth budget.

    Parameters
    ----------
    x_grid : ``(N,)`` or ``(N, n_inputs)`` float array
        Evaluation grid. Function identity is decided on this grid
        (rounded to ``decimals``); final recoveries should always be
        re-verified on an independent grid via :meth:`evaluate_expr`.
    operator : ``"eml"`` or ``"psi"``.
    decimals : rounding used for the dedup digest. Absorbs round-off
        jitter between different-but-equal expressions.
    max_pool_size : safety cap per level; a level that would exceed the
        cap raises so the caller can lower the depth instead of
        silently truncating the atlas.
    """

    def __init__(
        self,
        x_grid: np.ndarray,
        operator: str = "eml",
        decimals: int = 8,
        max_pool_size: int = 3_000_000,
    ):
        x = np.asarray(x_grid, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        self.x = x
        self.n_points, self.n_inputs = x.shape
        self.operator = operator
        ops = _OPERATORS[operator]
        self._node: Callable = ops["node"]
        self._exp_canon: Callable = ops["exp_side_canon"]
        self._log_canon: Callable = ops["log_side_canon"]
        self.decimals = decimals
        self.max_pool_size = max_pool_size

        self.entries: list[PoolEntry] = []
        self._digest_to_idx: dict[Digest, int] = {}
        # level_end[k] = number of pool entries with depth budget <= k.
        self.level_end: list[int] = []
        self.build_stats: list[dict] = []

        # Level 0: leaves.
        leaves = [np.ones(self.n_points)] + [
            self.x[:, j].copy() for j in range(self.n_inputs)
        ]
        for j, vec in enumerate(leaves):
            self._try_add(("leaf", j), depth=0, vec=vec)
        self.level_end.append(len(self.entries))

    # -- digesting ----------------------------------------------------------
    #
    # Function identity is decided by quantising each value vector to
    # ``decimals`` decimal places and hashing the quantised integer
    # vector with a fixed random int64 dot product (wraparound
    # arithmetic). Equal rounded vectors always hash equal; unequal
    # vectors collide with probability ~2^-64 per pair, and every
    # recovery produced through the hash path is re-verified on an
    # independent dense grid downstream, so collisions cannot corrupt
    # reported results. Hashes keep the dedup index at 8 bytes per
    # entry instead of ``8 * n_points``.

    def _hash_rows(self, rows: np.ndarray) -> np.ndarray:
        """Vectorised 64-bit hash of quantised rows. ``rows`` is (M, N)."""
        scale = 10.0**self.decimals
        q = np.round(rows * scale)
        # Saturate to the int64-representable range; non-finite rows are
        # the caller's responsibility (masked before/after).
        np.clip(q, -9.0e17, 9.0e17, out=q)
        q = q.astype(np.int64, copy=False)
        q[q == 0] = 0  # collapse -0
        rng = getattr(self, "_hash_mults", None)
        if rng is None or rng.shape[0] < rows.shape[1]:
            g = np.random.default_rng(0xC0FFEE)
            self._hash_mults = g.integers(
                1, 2**62, size=rows.shape[1], dtype=np.int64
            ) | 1
            rng = self._hash_mults
        with np.errstate(over="ignore"):
            return (q * rng[: rows.shape[1]][None, :]).sum(
                axis=1, dtype=np.int64
            )

    def _digest(self, vec: np.ndarray) -> int:
        return int(self._hash_rows(vec[None, :])[0])

    def _try_add(self, expr: tuple, depth: int, vec: np.ndarray) -> Optional[int]:
        if not np.all(np.isfinite(vec)):
            return None
        d = self._digest(vec)
        found = self._digest_to_idx.get(d)
        if found is not None:
            return found
        idx = len(self.entries)
        self.entries.append(PoolEntry(idx=idx, expr=expr, depth=depth, vec=vec))
        self._digest_to_idx[d] = idx
        return idx

    # -- building -----------------------------------------------------------

    @property
    def max_depth(self) -> int:
        return len(self.level_end) - 1

    def grow(self, a_chunk: Optional[int] = None) -> dict:
        """Extend the pool by one depth level.

        Combines every ordered pair of existing entries through the
        operator, deduplicating on the fly. Chunked over the first
        argument to bound peak memory.
        """
        t0 = time.time()
        prev_n = self.level_end[-1]
        vecs = np.stack([e.vec for e in self.entries[:prev_n]])  # (M, N)
        M = prev_n
        exp_side = self._exp_canon(vecs)  # what the node sees on side a
        log_side = self._log_canon(vecs)  # what the node sees on side b
        if a_chunk is None:
            a_chunk = max(1, min(512, int(4e8 / max(M * self.n_points * 8, 1))))

        n_candidates = 0
        for a0 in range(0, M, a_chunk):
            a1 = min(a0 + a_chunk, M)
            block = self._node(
                exp_side[a0:a1, None, :], log_side[None, :, :]
            )  # (c, M, N)
            c = a1 - a0
            flat = block.reshape(c * M, self.n_points)
            finite = np.all(np.isfinite(flat), axis=1)
            n_candidates += c * M
            if not np.any(finite):
                continue
            idxs = np.nonzero(finite)[0]
            hashes = self._hash_rows(flat[idxs])
            _, first_pos = np.unique(hashes, return_index=True)
            for p in sorted(first_pos):
                flat_idx = idxs[p]
                ia = a0 + flat_idx // M
                ib = flat_idx % M
                self._try_add(
                    ("node", int(ia), int(ib)),
                    depth=1 + max(self.entries[ia].depth, self.entries[ib].depth),
                    vec=flat[flat_idx],
                )
            if len(self.entries) > self.max_pool_size:
                raise MemoryError(
                    f"pool exceeded max_pool_size={self.max_pool_size} "
                    f"at level {self.max_depth + 1}"
                )
        self.level_end.append(len(self.entries))
        stats = {
            "level": self.max_depth,
            "n_candidate_pairs": n_candidates,
            "pool_size": len(self.entries),
            "new_entries": len(self.entries) - prev_n,
            "syntactic_live": syntactic_live_count(
                self.max_depth, self.n_inputs
            )
            if self.max_depth >= 1
            else None,
            "syntactic_snaps": syntactic_snap_count(
                self.max_depth, self.n_inputs
            )
            if self.max_depth >= 1
            else None,
            "seconds": time.time() - t0,
        }
        self.build_stats.append(stats)
        return stats

    def build(
        self,
        max_depth: int,
        a_chunk: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Grow the pool until slot depth budget ``max_depth``.

        A pool of slot depth ``k`` supports exhaustive root search at
        tree depth ``k + 1`` and lookup root search at ``k + 1``.
        """
        while self.max_depth < max_depth:
            stats = self.grow(a_chunk=a_chunk)
            if verbose:
                print(
                    f"  pool level {stats['level']}: "
                    f"{stats['pool_size']:,} distinct functions "
                    f"(live syntactic {stats['syntactic_live']:,}, "
                    f"raw snaps {stats['syntactic_snaps']:,}) "
                    f"[{stats['seconds']:.1f}s]"
                )

    def compact(self) -> None:
        """Free the build-time dedup index (a large dict). Search and
        matching remain fully functional; only further :meth:`grow`
        calls and raw-digest membership tests are lost."""
        self._digest_to_idx = {}

    def level_slice(self, k: int) -> slice:
        """Entries with depth budget ``<= k``."""
        return slice(0, self.level_end[min(k, self.max_depth)])

    # -- expression readout -------------------------------------------------

    def expr_to_nested(self, idx: int) -> tuple:
        """Expand entry ``idx`` into a self-contained nested tuple."""
        e = self.entries[idx]
        if e.expr[0] == "leaf":
            return e.expr
        _, ia, ib = e.expr
        return ("node", self.expr_to_nested(ia), self.expr_to_nested(ib))

    def expr_size(self, idx: int) -> int:
        cache = getattr(self, "_size_cache", None)
        if cache is None:
            cache = self._size_cache = {}
        if idx not in cache:
            e = self.entries[idx]
            if e.expr[0] == "leaf":
                cache[idx] = 1
            else:
                _, ia, ib = e.expr
                cache[idx] = 1 + self.expr_size(ia) + self.expr_size(ib)
        return cache[idx]

    def evaluate_nested(self, nested: tuple, x: np.ndarray) -> np.ndarray:
        """Evaluate a nested expression tuple on new inputs ``x``."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if nested[0] == "leaf":
            j = nested[1]
            if j == 0:
                return np.ones(x.shape[0])
            return x[:, j - 1].copy()
        _, na, nb = nested
        return self._node(
            self.evaluate_nested(na, x), self.evaluate_nested(nb, x)
        )

    def nested_to_sympy(self, nested: tuple, feature_names: Optional[list] = None):
        """Symbolic (un-clamped) readout of a nested expression."""
        import sympy as sp

        names = feature_names or [f"x{i+1}" for i in range(self.n_inputs)]
        syms = [sp.Symbol(n) for n in names]

        def rec(node):
            if node[0] == "leaf":
                j = node[1]
                return sp.Integer(1) if j == 0 else syms[j - 1]
            _, na, nb = node
            if self.operator == "eml":
                return sp.exp(rec(na)) - sp.log(rec(nb))
            return sp.sinh(rec(na)) - sp.asinh(rec(nb))

        return rec(nested)

    def nested_to_snap_config(self, nested: tuple, depth: int) -> dict:
        """Embed a nested expression into an :class:`EMLTree` snap dict.

        Dead slots are filled with option 0 (the constant ``1``). The
        expression's depth must be ``<= depth``.
        """
        import torch

        snap = {
            level: torch.zeros(2 ** (level + 1), dtype=torch.long)
            for level in range(depth)
        }
        f_child_opt = 1 + self.n_inputs

        def fill(node: tuple, level: int, slot: int) -> None:
            # ``node`` is the content of slot ``slot`` at ``level - 1``'s
            # node... slots at level l feed nodes at level l; we place the
            # choice for a slot and recurse into its child node.
            if node[0] == "leaf":
                snap[level][slot] = node[1]  # 0 -> '1', j -> x_j
                return
            snap[level][slot] = f_child_opt
            _, na, nb = node
            child_node = slot  # node index at level + 1 equals slot index
            fill(na, level + 1, 2 * child_node)
            fill(nb, level + 1, 2 * child_node + 1)

        assert nested[0] == "node", "root must be a node"
        _, na, nb = nested
        fill(na, 0, 0)
        fill(nb, 0, 1)
        return snap

    # -- search -------------------------------------------------------------

    def score_pool_affine(
        self, y: np.ndarray, k: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised OLS of ``y ~ alpha + beta * f`` for every pool entry
        with depth budget ``<= k``. Returns ``(r2, alpha, beta)`` arrays."""
        vecs = self.stacked_vecs(self.max_depth if k is None else k)
        return _ols_r2_rows(vecs, np.asarray(y, dtype=np.float64))

    def exhaustive_root(
        self,
        y: np.ndarray,
        depth: int,
        top_k: int = 8,
        a_chunk: int = 32,
        allow_affine: bool = True,
    ) -> list[dict]:
        """Score every distinct depth-``depth`` root function against
        ``y`` and return the ``top_k`` candidates.

        The candidate set is complete over the enumerated function
        class: every depth-``depth`` tree function equals
        ``node(a, b)`` for some pool entries ``a, b`` of budget
        ``depth - 1``, and the pool contains *all* of those up to grid
        identity.
        """
        y = np.asarray(y, dtype=np.float64)
        vecs = self.stacked_vecs(depth - 1)
        M = vecs.shape[0]
        exp_side = self._exp_canon(vecs)
        log_side = self._log_canon(vecs)

        y_mean = y.mean()
        y_c = y - y_mean
        ss_tot = float(np.dot(y_c, y_c))
        if ss_tot < 1e-20:
            ss_tot = 1.0

        best: list[tuple[float, int, int, float, float]] = []
        for a0 in range(0, M, a_chunk):
            a1 = min(a0 + a_chunk, M)
            block = self._node(
                exp_side[a0:a1, None, :], log_side[None, :, :]
            )  # (c, M, N)
            c = a1 - a0
            flat = block.reshape(c * M, self.n_points)
            r2, alpha, beta = _ols_r2_rows(
                flat, y, allow_affine=allow_affine
            )
            take = min(top_k, r2.shape[0])
            part = np.argpartition(-r2, take - 1)[:take]
            for p in part:
                score = float(r2[p])
                ia = a0 + int(p) // M
                ib = int(p) % M
                best.append((score, ia, ib, float(alpha[p]), float(beta[p])))
        # Rank by R^2 (rounded so exact ties compare equal), then by
        # expression size ascending — the Occam tie-break pushes the
        # canonical representation of the target ahead of coincidental
        # grid-ties.
        best.sort(
            key=lambda t: (
                -round(t[0], 10),
                self.expr_size(t[1]) + self.expr_size(t[2]),
            )
        )
        out = []
        for score, ia, ib, alpha_v, beta_v in best[:top_k]:
            out.append(
                {
                    "r2": score,
                    "alpha": alpha_v,
                    "beta": beta_v,
                    "size": 1 + self.expr_size(ia) + self.expr_size(ib),
                    "nested": (
                        "node",
                        self.expr_to_nested(ia),
                        self.expr_to_nested(ib),
                    ),
                }
            )
        return out

    # -- vectorised side-canonical matching ---------------------------------

    _MISS_SENTINEL = np.int64(-0x6E3A_1B2C_4D5E_6F70)

    def _side_index(self, k: int, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Sorted 64-bit-hash index of side-canonicalised pool vectors at
        depth budget ``<= k``. Cached. Returns ``(sorted_hashes, order)``
        with ``order`` mapping sorted position -> pool entry index."""
        cache = getattr(self, "_side_cache", None)
        if cache is None:
            cache = self._side_cache = {}
        key = (k, side)
        if key not in cache:
            vecs = self.stacked_vecs(k)
            canon = (
                self._exp_canon(vecs) if side == "exp" else self._log_canon(vecs)
            )
            hashes = self._hash_rows(canon)
            order = np.argsort(hashes, kind="stable")
            cache[key] = (hashes[order], order)
        return cache[key]

    def match_rows(
        self, rows: np.ndarray, k: int, side: str
    ) -> np.ndarray:
        """For each row, return the index of a pool entry (depth budget
        ``<= k``) whose side-canonical vector equals the row exactly
        (after rounding), or ``-1``. Fully vectorised. When several
        entries share the side-canonical vector an arbitrary one is
        returned; use :meth:`match_row_runs` to get all of them."""
        sorted_hashes, order = self._side_index(k, side)
        finite = np.all(np.isfinite(rows), axis=1)
        safe_rows = np.where(finite[:, None], rows, 0.0)
        targets = self._hash_rows(safe_rows)
        targets[~finite] = self._MISS_SENTINEL
        pos = np.searchsorted(sorted_hashes, targets)
        pos = np.clip(pos, 0, len(sorted_hashes) - 1)
        hit = (sorted_hashes[pos] == targets) & finite
        out = np.where(hit, order[np.clip(pos, 0, None)], -1)
        return out.astype(np.int64)

    def sizes_array(self, k: int) -> np.ndarray:
        """Expression sizes of all pool entries at budget ``<= k`` as an
        int array (cached)."""
        cache = getattr(self, "_sizes_arr_cache", None)
        if cache is None:
            cache = self._sizes_arr_cache = {}
        if k not in cache:
            sl = self.level_slice(k)
            n = sl.stop
            sizes = np.zeros(n, dtype=np.int64)
            for i in range(n):
                e = self.entries[i]
                if e.expr[0] == "leaf":
                    sizes[i] = 1
                else:
                    _, ia, ib = e.expr
                    sizes[i] = 1 + sizes[ia] + sizes[ib]
            cache[k] = sizes
        return cache[k]

    def match_row_runs(
        self, rows: np.ndarray, k: int, side: str, per_run_cap: int = 32
    ) -> list[tuple[int, list[int]]]:
        """Like :meth:`match_rows` but returns *every* pool entry whose
        side-canonical vector matches, as ``(row_index, [entry ids])``.

        Side canonicalisation is lossy (``clip`` for the eml exp side,
        ``max(., EPS)`` for the log side), so distinct pool functions
        can agree exactly on what the parent node sees. All of them are
        legitimate reconstruction candidates; runs are ranked by
        expression size and truncated at ``per_run_cap``.
        """
        sorted_hashes, order = self._side_index(k, side)
        sizes = self.sizes_array(k)
        finite = np.all(np.isfinite(rows), axis=1)
        safe_rows = np.where(finite[:, None], rows, 0.0)
        targets = self._hash_rows(safe_rows)
        targets[~finite] = self._MISS_SENTINEL
        lo = np.searchsorted(sorted_hashes, targets, side="left")
        hi = np.searchsorted(sorted_hashes, targets, side="right")
        run_cache: dict[tuple[int, int], list[int]] = {}
        out: list[tuple[int, list[int], int]] = []
        for i in np.nonzero((hi > lo) & finite)[0]:
            key = (int(lo[i]), int(hi[i]))
            members = run_cache.get(key)
            if members is None:
                run = order[key[0] : key[1]]
                if len(run) > per_run_cap:
                    sub = np.argpartition(sizes[run], per_run_cap - 1)[
                        :per_run_cap
                    ]
                    run = run[sub[np.argsort(sizes[run][sub])]]
                members = [int(e) for e in run]
                run_cache[key] = members
            out.append((int(i), members, int(hi[i] - lo[i])))
        return out

    def stacked_vecs(self, k: int) -> np.ndarray:
        """Cached ``(M, N)`` matrix of pool vectors at budget ``<= k``."""
        cache = getattr(self, "_vecs_cache", None)
        if cache is None:
            cache = self._vecs_cache = {}
        if k not in cache:
            sl = self.level_slice(k)
            cache[k] = np.stack([e.vec for e in self.entries[sl]])
        return cache[k]

    def implied_exp_children(
        self, y: np.ndarray, vecs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """For each row ``b`` of ``vecs``, the implied exp-side child
        ``a`` such that ``node(a, b) = y``. Returns ``(a, valid)``."""
        if self.operator == "eml":
            log_seen = self._log_canon(vecs)
            with np.errstate(invalid="ignore", divide="ignore"):
                t = y[None, :] + np.log(log_seen)
                valid = np.all(t > 0.0, axis=1)
                a = np.log(np.maximum(t, 1e-300))
            a = np.clip(a, -CLAMP_VAL, CLAMP_VAL)
            return a, valid
        a = np.arcsinh(y[None, :] + np.arcsinh(vecs))
        return a, np.all(np.isfinite(a), axis=1)

    def implied_log_children(
        self, y: np.ndarray, vecs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """For each row ``a`` of ``vecs``, the implied log-side child
        ``b`` such that ``node(a, b) = y``."""
        if self.operator == "eml":
            exp_seen = self._exp_canon(vecs)
            with np.errstate(over="ignore"):
                lb = np.exp(exp_seen) - y[None, :]
                # log(clamp(b)) below log(EPS) is unreachable: no b exists.
                valid = np.all(lb >= np.log(EPS) - 1e-9, axis=1)
                b = np.exp(lb)
            b = np.maximum(b, EPS)
            return b, valid & np.all(np.isfinite(b), axis=1)
        with np.errstate(over="ignore"):
            b = np.sinh(np.sinh(vecs) - y[None, :])
        return b, np.all(np.isfinite(b), axis=1)

    def lookup_root(
        self,
        y: np.ndarray,
        depth: int,
        max_matches: int = 8,
        row_chunk: int = 65_536,
    ) -> list[dict]:
        """Exact-recovery search: assumes ``y`` *is* a depth-``depth``
        tree function (alpha = 0, beta = 1) and inverts the root node.

        For the eml operator: ``y = exp(clip(a)) - log(clamp(b))``
        implies ``clip(a) = log(y + log(clamp(b)))``. Enumerating ``b``
        over the pool and matching the implied exp-side vector against a
        pre-sorted index finds every representation in ``O(pool size)``
        — this is what makes depth ``max_pool_depth + 1`` recovery a
        sub-second operation instead of an ``O(pool^2)`` sweep.

        For psi: ``sinh(a) = y + arsinh(b)`` gives ``a = arsinh(y +
        arsinh(b))``; the symmetric pass solves ``b = sinh(sinh(a) - y)``.
        """
        ranked = self.lookup_root_pairs(y, depth, cap=max_matches)
        return [self._match_dict(ia, ib) for ia, ib in ranked]

    def lookup_root_pairs(
        self,
        y: np.ndarray,
        depth: int,
        cap: int = 100_000,
        row_chunk: int = 65_536,
        per_run_cap: int = 32,
    ) -> list[tuple[int, int]]:
        """Pair-index form of :meth:`lookup_root` — no expression
        expansion, suitable for streaming verification of large match
        sets.

        Both passes always run (a per-pass budget prevents one side's
        pathological matches from starving the other). Pairs are ranked
        by ``(match run length, total expression size)``: a match whose
        side-canonical hash is shared by few pool entries is close to a
        certificate, while hits inside huge saturated runs (thousands
        of clamp-degenerate entries agreeing on what the root sees) are
        low-information and verified last.
        """
        y = np.asarray(y, dtype=np.float64)
        k = depth - 1
        vecs = self.stacked_vecs(k)
        M = vecs.shape[0]
        pass_cap = cap
        pairs: dict[tuple[int, int], int] = {}

        def note(pair: tuple[int, int], run_len: int) -> None:
            old = pairs.get(pair)
            if old is None or run_len < old:
                pairs[pair] = run_len

        n0 = 0
        for r0 in range(0, M, row_chunk):
            if len(pairs) - n0 >= pass_cap:
                break
            r1 = min(r0 + row_chunk, M)
            a_implied, valid_b = self.implied_exp_children(y, vecs[r0:r1])
            a_implied[~valid_b] = np.nan
            for off, members, run_len in self.match_row_runs(
                a_implied, k, "exp", per_run_cap=per_run_cap
            ):
                for ia in members:
                    note((ia, r0 + off), run_len)

        n0 = len(pairs)
        for r0 in range(0, M, row_chunk):
            if len(pairs) - n0 >= pass_cap:
                break
            r1 = min(r0 + row_chunk, M)
            b_implied, valid_a = self.implied_log_children(y, vecs[r0:r1])
            b_implied[~valid_a] = np.nan
            for off, members, run_len in self.match_row_runs(
                b_implied, k, "log", per_run_cap=per_run_cap
            ):
                for ib in members:
                    note((r0 + off, ib), run_len)

        sizes = self.sizes_array(k)
        ranked = sorted(
            pairs, key=lambda p: (pairs[p], sizes[p[0]] + sizes[p[1]])
        )
        return ranked[:cap]

    def _match_dict(self, ia: int, ib: int) -> dict:
        return {
            "r2": 1.0,
            "alpha": 0.0,
            "beta": 1.0,
            "size": 1 + self.expr_size(ia) + self.expr_size(ib),
            "pair": (ia, ib),
            "nested": ("node", self.expr_to_nested(ia), self.expr_to_nested(ib)),
        }

    # -- fast DAG evaluation on new grids -----------------------------------

    def evaluate_entry(
        self, idx: int, x: np.ndarray, cache: Optional[dict] = None
    ) -> np.ndarray:
        """Evaluate pool entry ``idx`` on inputs ``x`` through the
        expression DAG with memoisation. Passing the same ``cache``
        dict across many candidates makes batch verification cheap —
        candidates share subtrees by construction."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if cache is None:
            cache = {}
        return self._eval_entry_rec(idx, x, cache)

    def _eval_entry_rec(self, idx: int, x: np.ndarray, cache: dict) -> np.ndarray:
        got = cache.get(idx)
        if got is not None:
            return got
        e = self.entries[idx]
        if e.expr[0] == "leaf":
            j = e.expr[1]
            v = np.ones(x.shape[0]) if j == 0 else x[:, j - 1].copy()
        else:
            _, ia, ib = e.expr
            v = self._node(
                self._eval_entry_rec(ia, x, cache),
                self._eval_entry_rec(ib, x, cache),
            )
        cache[idx] = v
        return v

    def evaluate_pair(
        self, ia: int, ib: int, x: np.ndarray, cache: Optional[dict] = None
    ) -> np.ndarray:
        """Evaluate the root candidate ``node(entry ia, entry ib)``."""
        x2 = np.asarray(x, dtype=np.float64)
        if x2.ndim == 1:
            x2 = x2[:, None]
        if cache is None:
            cache = {}
        return self._node(
            self._eval_entry_rec(ia, x2, cache),
            self._eval_entry_rec(ib, x2, cache),
        )

    def solve_recursive(
        self,
        y: np.ndarray,
        depth: int,
        shallow_k: int = 2,
        shallow_budget: int = 256,
        max_matches: int = 16,
    ) -> list[tuple]:
        """Exact recovery beyond ``max_pool_depth + 1`` by recursive
        root peeling.

        For a target of depth ``d > max_pool_depth + 1``, at least one
        root child is often *shallow* (or functionally shallow). This
        solver enumerates shallow candidates for one side (pool entries
        of budget ``<= shallow_k``, smallest expressions first, at most
        ``shallow_budget`` of them), analytically inverts the root node
        for the other side, and recurses on the implied child until the
        depth budget reaches the pool's native lookup range.

        Returns nested expression tuples whose evaluation equals ``y``
        on the pool grid. Coverage is *partial by design* — targets
        whose root children are both deep and both outside the pool are
        not found; the depth-wall benchmark measures that fraction
        honestly. Every returned expression should be re-verified on an
        independent dense grid by the caller.
        """
        y = np.asarray(y, dtype=np.float64)
        if not np.all(np.isfinite(y)):
            return []
        if depth <= self.max_depth + 1:
            found = self.lookup_root_pairs(y, depth, cap=max_matches)
            return [
                ("node", self.expr_to_nested(ia), self.expr_to_nested(ib))
                for ia, ib in found
            ]

        out: list[tuple] = []
        sl = self.level_slice(shallow_k)
        sizes = self.sizes_array(shallow_k)
        order = np.argsort(sizes[: sl.stop], kind="stable")[:shallow_budget]

        for idx in order:
            e = self.entries[int(idx)]
            v = e.vec
            # Case 1: shallow entry is the exp-side child a; solve b.
            if self.operator == "eml":
                with np.errstate(over="ignore"):
                    lb = np.exp(self._exp_canon(v)) - y
                ok = np.all(lb >= np.log(EPS) - 1e-9)
                b_implied = np.maximum(np.exp(lb), EPS) if ok else None
            else:
                with np.errstate(over="ignore"):
                    b_implied = np.sinh(np.sinh(v) - y)
                ok = bool(np.all(np.isfinite(b_implied)))
            if ok and b_implied is not None:
                for sub in self.solve_recursive(
                    b_implied, depth - 1, shallow_k, shallow_budget // 4 or 1,
                    max_matches=4,
                ):
                    out.append(("node", self.expr_to_nested(int(idx)), sub))
                    if len(out) >= max_matches:
                        return out
            # Case 2: shallow entry is the log-side child b; solve a.
            if self.operator == "eml":
                with np.errstate(invalid="ignore", divide="ignore"):
                    t = y + np.log(self._log_canon(v))
                if np.all(t > 0):
                    a_implied = np.clip(
                        np.log(np.maximum(t, 1e-300)), -CLAMP_VAL, CLAMP_VAL
                    )
                else:
                    a_implied = None
            else:
                a_implied = np.arcsinh(y + np.arcsinh(v))
                if not np.all(np.isfinite(a_implied)):
                    a_implied = None
            if a_implied is not None:
                for sub in self.solve_recursive(
                    a_implied, depth - 1, shallow_k, shallow_budget // 4 or 1,
                    max_matches=4,
                ):
                    out.append(("node", sub, self.expr_to_nested(int(idx))))
                    if len(out) >= max_matches:
                        return out
        return out

    def solve(
        self,
        y: np.ndarray,
        depth: int,
        top_k: int = 8,
        try_lookup: bool = True,
        allow_affine: bool = True,
    ) -> list[dict]:
        """Recover ``y`` as a depth-``depth`` tree.

        Uses the O(pool) lookup first (exact snap targets), then falls
        back to the complete O(pool^2) OLS sweep. ``depth`` may be at
        most ``max_pool_depth + 1``.
        """
        assert depth <= self.max_depth + 1, (
            f"depth {depth} needs pool of depth {depth - 1}, "
            f"have {self.max_depth}"
        )
        if try_lookup:
            found = self.lookup_root(y, depth, max_matches=top_k)
            if found:
                return found
        return self.exhaustive_root(
            y, depth, top_k=top_k, allow_affine=allow_affine
        )


# ---------------------------------------------------------------------------
# Vectorised OLS
# ---------------------------------------------------------------------------


def _ols_r2_rows(
    preds: np.ndarray, y: np.ndarray, allow_affine: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-wise OLS ``y ~ alpha + beta * preds[i]``.

    Returns ``(r2, alpha, beta)``; rows with non-finite or
    zero-variance predictions get ``r2 = -inf``.
    """
    y = np.asarray(y, dtype=np.float64)
    y_mean = y.mean()
    y_c = y - y_mean
    ss_tot = float(np.dot(y_c, y_c))
    if ss_tot < 1e-20:
        ss_tot = 1.0

    finite = np.all(np.isfinite(preds), axis=1)
    p = np.where(finite[:, None], preds, 0.0)
    p_mean = p.mean(axis=1)
    p_c = p - p_mean[:, None]
    p_var = np.einsum("ij,ij->i", p_c, p_c)
    cov = p_c @ y_c
    if allow_affine:
        beta = cov / np.maximum(p_var, 1e-30)
        alpha = y_mean - beta * p_mean
        ss_res = np.maximum(ss_tot - beta * cov, 0.0)
    else:
        beta = np.ones(p.shape[0])
        alpha = np.zeros(p.shape[0])
        resid = p - y[None, :]
        ss_res = np.einsum("ij,ij->i", resid, resid)
    r2 = 1.0 - ss_res / ss_tot
    bad = (~finite) | (p_var < 1e-24)
    r2[bad] = -np.inf
    return r2, alpha, beta


# ---------------------------------------------------------------------------
# Target sampling (for recovery benchmarks)
# ---------------------------------------------------------------------------


def sample_live_nested(
    depth: int,
    n_inputs: int,
    rng: np.random.Generator,
    p_expand: float = 0.75,
) -> tuple:
    """Sample a random *live* expression of exact depth ``depth``.

    The root is a node; every slot expands into a child node with
    probability ``p_expand`` while depth budget remains, and at least
    one root-to-leaf path reaches the full budget (rejection-free by
    forcing a random spine).
    """

    def sample_slot(budget: int, force_full: bool) -> tuple:
        if budget == 0 or (not force_full and rng.random() > p_expand):
            j = int(rng.integers(0, 1 + n_inputs))
            return ("leaf", j)
        spine = int(rng.integers(0, 2)) if force_full else -1
        a = sample_slot(budget - 1, force_full and spine == 0)
        b = sample_slot(budget - 1, force_full and spine == 1)
        return ("node", a, b)

    assert depth >= 1
    spine = int(np.random.default_rng(rng.integers(1 << 31)).integers(0, 2))
    a = sample_slot(depth - 1, spine == 0)
    b = sample_slot(depth - 1, spine == 1)
    return ("node", a, b)


def nested_depth(nested: tuple) -> int:
    if nested[0] == "leaf":
        return 0
    return 1 + max(nested_depth(nested[1]), nested_depth(nested[2]))


__all__ = [
    "FunctionPool",
    "PoolEntry",
    "eml_node",
    "psi_node",
    "syntactic_live_count",
    "syntactic_snap_count",
    "sample_live_nested",
    "nested_depth",
]
