"""Multi-seed benchmark: run experiments N times with different seeds and report mean +/- std.

This provides confidence intervals for all benchmark results, addressing
the single-run criticism by showing statistical robustness.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .extrapolation import ExtrapResult, bench_dataset
from .real_world import run_yacht
from .scientific import ALL_SCIENTIFIC, ScientificDataset


def _set_seeds(seed: int) -> None:
    """Set all global random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def _generate_datasets_with_seed(seed: int) -> list[ScientificDataset]:
    """Generate all scientific datasets using a specific data-generation seed."""
    return [fn(seed=seed) for fn in ALL_SCIENTIFIC]


def run_multiseed_synthetic(
    n_seeds: int = 5,
    base_seed: int = 0,
    verbose: bool = True,
) -> dict[str, dict[str, dict[str, float | list[float]]]]:
    """Run scientific benchmarks across multiple seeds and report statistics.

    For each seed the data is regenerated (different noise draws) and
    all models are retrained (different weight initialisation), so the
    resulting variance captures both data and optimisation randomness.

    Returns
    -------
    dict
        ``{dataset_name: {model_name: {"mean": float, "std": float, "values": list[float]}}}``
        where the metric is R^2 on the extrapolation test set.
    """
    # {dataset: {model: [r2_extrap, ...]}}
    collector: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for i, seed in enumerate(range(base_seed, base_seed + n_seeds)):
        if verbose:
            print(f"\n--- seed {seed} ({i + 1}/{n_seeds}) ---")

        _set_seeds(seed)
        datasets = _generate_datasets_with_seed(seed)

        for ds in datasets:
            results: list[ExtrapResult] = bench_dataset(ds)
            for r in results:
                collector[r.dataset][r.model].append(r.r2_extrap)
                if verbose:
                    print(
                        f"  {ds.name:25s} | {r.model:8s} | "
                        f"R2_extrap = {r.r2_extrap:+.4f}"
                    )

    # Aggregate
    summary: dict[str, dict[str, dict[str, float | list[float]]]] = {}
    for ds_name, models in sorted(collector.items()):
        summary[ds_name] = {}
        for model_name, values in sorted(models.items()):
            arr = np.array(values)
            summary[ds_name][model_name] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "values": values,
            }

    if verbose:
        _print_summary_table(summary, n_seeds)

    return summary


def run_multiseed_yacht(
    n_seeds: int = 5,
    base_seed: int = 0,
    verbose: bool = True,
    isolated_physics: bool = True,
) -> dict[str, dict[str, dict[str, float | list[float]]]]:
    """Run yacht benchmark across multiple seeds and report statistics.

    Each seed changes the torch / numpy random state before training so
    that model weight initialisation differs across runs.  The dataset
    itself is fixed (UCI Yacht Hydrodynamics), but the model variance is
    captured.

    Returns
    -------
    dict
        ``{split: {model_name: {"mean": float, "std": float, "values": list[float]}}}``
        where *split* is ``"interp"`` or ``"extrap"``.
    """
    collector: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for i, seed in enumerate(range(base_seed, base_seed + n_seeds)):
        if verbose:
            print(f"\n--- yacht seed {seed} ({i + 1}/{n_seeds}) ---")

        _set_seeds(seed)
        results = run_yacht(verbose=False, isolated_physics=isolated_physics)

        for r in results:
            key = r.split  # "interp" or "extrap"
            collector[key][r.model].append(r.r2)
            if verbose:
                print(
                    f"  yacht ({r.split:6s}) | {r.model:8s} | "
                    f"R2 = {r.r2:+.4f}"
                )

    summary: dict[str, dict[str, dict[str, float | list[float]]]] = {}
    for split_name, models in sorted(collector.items()):
        summary[split_name] = {}
        for model_name, values in sorted(models.items()):
            arr = np.array(values)
            summary[split_name][model_name] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "values": values,
            }

    if verbose:
        print(f"\n{'=' * 65}")
        print(f"  Multi-seed yacht results (N={n_seeds})")
        print(f"{'=' * 65}")
        print(f"  {'Split':<8s} | {'Model':<8s} | R2 (mean +/- std)")
        print(f"  {'-' * 8} | {'-' * 8} | {'-' * 22}")
        for split_name in sorted(summary):
            for model_name in sorted(summary[split_name]):
                m = summary[split_name][model_name]["mean"]
                s = summary[split_name][model_name]["std"]
                print(
                    f"  {split_name:<8s} | {model_name:<8s} | "
                    f"{m:+.4f} +/- {s:.4f}"
                )

    return summary


def _print_summary_table(
    summary: dict[str, dict[str, dict[str, float | list[float]]]],
    n_seeds: int,
) -> None:
    """Print a formatted table of multi-seed results."""
    print(f"\n{'=' * 65}")
    print(f"  Multi-seed synthetic results (N={n_seeds})")
    print(f"{'=' * 65}")
    print(
        f"  {'Dataset':<25s} | {'Model':<8s} | "
        f"R2 extrap (mean +/- std)"
    )
    print(f"  {'-' * 25} | {'-' * 8} | {'-' * 22}")
    for ds_name in sorted(summary):
        for model_name in sorted(summary[ds_name]):
            m = summary[ds_name][model_name]["mean"]
            s = summary[ds_name][model_name]["std"]
            print(
                f"  {ds_name:<25s} | {model_name:<8s} | "
                f"{m:+.4f} +/- {s:.4f}"
            )


if __name__ == "__main__":
    run_multiseed_synthetic(n_seeds=5)
    run_multiseed_yacht(n_seeds=5)
