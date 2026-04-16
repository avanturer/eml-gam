"""Generate paper-ready landscape figures from ``landscape_results.json``.

Produces two PNGs in ``figures/``:

    landscape_heatmap.png
        Perturbed-init MSE-success heatmap — x axis sigma, y axis depth.
        Reads the 30-trials-per-cell "success_rate" from the landscape
        results file.
    landscape_basin_curves.png
        Success rate vs sigma, one curve per depth. Overlays the
        random-init baseline (sigma = infinity) as a horizontal tick
        at the left edge.

Run with:
    python -m scripts.make_landscape_figure

The figures are saved headlessly (matplotlib Agg backend), so this
script works on CI machines without a display.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_heatmap(results: dict, out_path: str) -> None:
    depths = sorted({r["depth"] for r in results["perturbed"]})
    sigmas = sorted({r["sigma"] for r in results["perturbed"]})
    grid = np.full((len(depths), len(sigmas)), np.nan)
    for r in results["perturbed"]:
        i = depths.index(r["depth"])
        j = sigmas.index(r["sigma"])
        grid[i, j] = r["success_rate"]

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    im = ax.imshow(
        grid, aspect="auto", cmap="RdYlGn",
        vmin=0.0, vmax=1.0, origin="lower",
    )
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels([f"{s:g}" for s in sigmas])
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([str(d) for d in depths])
    ax.set_xlabel(r"perturbation $\sigma$ (logit stddev)")
    ax.set_ylabel("tree depth")
    ax.set_title(
        "EML-tree basin width — MSE success rate from perturbed init"
    )
    # Cell annotations.
    for i in range(len(depths)):
        for j in range(len(sigmas)):
            v = grid[i, j]
            if np.isfinite(v):
                ax.text(
                    j, i, f"{v:.0%}",
                    ha="center", va="center", fontsize=8,
                    color="black" if 0.3 < v < 0.8 else "white",
                )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("fraction of trials with MSE < 1e-3")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_curves(results: dict, out_path: str) -> None:
    depths = sorted({r["depth"] for r in results["perturbed"]})
    by_depth: dict[int, list[tuple[float, float]]] = {d: [] for d in depths}
    for r in results["perturbed"]:
        by_depth[r["depth"]].append((r["sigma"], r["success_rate"]))
    for d in depths:
        by_depth[d].sort()

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for d in depths:
        xs = [s for s, _ in by_depth[d]]
        ys = [p for _, p in by_depth[d]]
        ax.plot(xs, ys, marker="o", label=f"depth {d}")
    # Random-init baseline markers at sigma=0 proxy on the left edge.
    for r in results["random"]:
        ax.scatter(
            [0.05], [r["success_rate"]], marker="x", s=60,
            color=f"C{depths.index(r['depth'])}", zorder=5,
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"perturbation $\sigma$ (log scale)")
    ax.set_ylabel("MSE-success rate")
    ax.set_title("Basin width of the correct snap, per depth")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, linestyle="--", color="grey", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "landscape_results.json"
    if not os.path.exists(src):
        raise SystemExit(f"results file not found: {src}")
    results = load_results(src)
    os.makedirs("figures", exist_ok=True)
    save_heatmap(results, "figures/landscape_heatmap.png")
    save_curves(results, "figures/landscape_basin_curves.png")
    print("saved figures/landscape_heatmap.png and landscape_basin_curves.png")


if __name__ == "__main__":
    main()
