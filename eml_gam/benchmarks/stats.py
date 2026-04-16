"""Statistical comparison helpers for benchmark results.

The benchmark scripts in this package produce a list of per-target,
per-model records. Downstream analyses (README tables, paper figures)
need two kinds of summary from the raw records:

1. Multi-seed aggregates: for every ``(target, model)`` pair we compute
   mean, standard deviation and a 95 per cent bootstrap confidence
   interval from the per-seed scores.
2. Paired hypothesis tests: for every ``(target, model_a, model_b)``
   triple we run the Wilcoxon signed-rank test across seeds. The output
   is a table of ``p``-values and effect sizes which can be used to
   populate a "best per row in bold" column in the paper.

The code is intentionally dependency-light: ``numpy`` for everything,
and ``scipy.stats`` only if it is available (falls back to an analytic
Wilcoxon approximation otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


def bootstrap_ci(
    values: Iterable[float], n_boot: int = 2000, q: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, arr.size, size=arr.size)
        boots[i] = arr[idx].mean()
    lo = float(np.percentile(boots, (1.0 - q) / 2.0 * 100.0))
    hi = float(np.percentile(boots, (1.0 + q) / 2.0 * 100.0))
    return lo, hi


@dataclass
class SeedSummary:
    target: str
    model: str
    n_seeds: int
    mean: float
    std: float
    median: float
    ci_lo: float
    ci_hi: float


def summarise_per_seed(
    values_by_pair: dict[tuple[str, str], list[float]],
) -> list[SeedSummary]:
    out: list[SeedSummary] = []
    for (target, model), vals in sorted(values_by_pair.items()):
        arr = np.asarray(vals, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            mean = std = med = ci_lo = ci_hi = float("nan")
        else:
            mean = float(finite.mean())
            std = float(finite.std(ddof=0))
            med = float(np.median(finite))
            ci_lo, ci_hi = bootstrap_ci(finite)
        out.append(SeedSummary(
            target=target, model=model, n_seeds=int(arr.size),
            mean=mean, std=std, median=med, ci_lo=ci_lo, ci_hi=ci_hi,
        ))
    return out


@dataclass
class PairedTest:
    target: str
    model_a: str
    model_b: str
    n_seeds: int
    median_delta: float
    p_value: float
    effect: str  # "a_better", "b_better", or "tie"


def _wilcoxon(deltas: np.ndarray) -> Optional[float]:
    """Return two-sided Wilcoxon signed-rank p-value, or ``None`` on failure."""
    finite = deltas[np.isfinite(deltas)]
    if finite.size < 4:
        return None
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(finite, zero_method="wilcox", alternative="two-sided")
        return float(res.pvalue)
    except Exception:
        # Normal approximation fallback.
        ranks = np.argsort(np.argsort(np.abs(finite))) + 1
        signs = np.sign(finite)
        w = float((signs * ranks).sum())
        n = finite.size
        var = n * (n + 1) * (2 * n + 1) / 6.0
        if var <= 0:
            return None
        z = w / np.sqrt(var)
        from math import erfc, sqrt
        return float(erfc(abs(z) / sqrt(2)))


def paired_tests(
    values_by_pair: dict[tuple[str, str], list[float]],
    models: list[str],
    alpha: float = 0.05,
) -> list[PairedTest]:
    """Run pairwise Wilcoxon signed-rank tests on per-seed score lists.

    ``values_by_pair[(target, model)]`` is a list of per-seed scores; the
    tests are paired across seeds, so each list must be the same length
    for all models of a given target.
    """
    out: list[PairedTest] = []
    targets = sorted({t for (t, _) in values_by_pair.keys()})
    for target in targets:
        for a, b in [(i, j) for i in models for j in models if i < j]:
            va = values_by_pair.get((target, a), [])
            vb = values_by_pair.get((target, b), [])
            if len(va) != len(vb) or len(va) < 4:
                continue
            arr_a = np.asarray(va, dtype=np.float64)
            arr_b = np.asarray(vb, dtype=np.float64)
            delta = arr_a - arr_b
            median_delta = float(np.median(delta[np.isfinite(delta)]))
            p = _wilcoxon(delta)
            if p is None:
                effect = "tie"
            elif p > alpha:
                effect = "tie"
            elif median_delta > 0:
                effect = "a_better"
            else:
                effect = "b_better"
            out.append(PairedTest(
                target=target, model_a=a, model_b=b,
                n_seeds=len(va),
                median_delta=median_delta,
                p_value=p if p is not None else float("nan"),
                effect=effect,
            ))
    return out


def format_summary_table(
    summaries: list[SeedSummary], models: list[str],
) -> str:
    by_target: dict[str, dict[str, SeedSummary]] = {}
    for s in summaries:
        by_target.setdefault(s.target, {})[s.model] = s
    targets = sorted(by_target.keys())
    header = (
        f"  {'target':18s}  "
        + "  ".join(f"{m:>20s}" for m in models)
    )
    lines = [header, "  " + "-" * (len(header) - 2)]
    for t in targets:
        cells = []
        for m in models:
            s = by_target[t].get(m)
            if s is None or not np.isfinite(s.mean):
                cells.append(" " * 20)
            else:
                cells.append(
                    f"{s.mean:+7.3f}+/-{s.std:5.3f} (n={s.n_seeds})".rjust(20)
                )
        lines.append(f"  {t:18s}  " + "  ".join(cells))
    return "\n".join(lines)
