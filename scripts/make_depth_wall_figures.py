"""Figures for the canonical-pool depth-wall results.

Reads ``depth_wall_results.json``, ``saturation_results.json`` and the
pool build stats, writes three paper-ready PNGs into ``figures/``:

* ``function_space_collapse.png`` — raw snap space vs live trees vs
  distinct functions per depth (log scale): the combinatorial wall is
  a mirage.
* ``recovery_vs_depth.png`` — verified exact-recovery rate per method
  per depth, with the source paper's reported blind-recovery curve for
  context.
* ``gradient_pathology.png`` — median live gradient magnitude and
  dead-slot fraction for eml vs psi trees at random init.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)


def fig_collapse():
    with open(os.path.join(ROOT, "depth_wall_results.json")) as f:
        res = json.load(f)
    stats = res["protocol"]["pool_build_stats"]
    depths = [s["level"] for s in stats]
    raw = [s["syntactic_snaps"] for s in stats]
    live = [s["syntactic_live"] for s in stats]
    distinct = [
        s["pool_size"] for s in stats
    ]  # cumulative distinct functions <= level

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(depths, raw, "o-", label="raw snap space (AEES enumerates)")
    ax.semilogy(depths, live, "s-", label="live trees (dead slots removed)")
    ax.semilogy(
        depths, distinct, "d-",
        label="distinct functions (canonical pool)",
    )
    for d, r, dv in zip(depths, raw, distinct):
        if d >= 3:
            ax.annotate(
                f"×{r / dv:,.0f}",
                (d, dv), textcoords="offset points", xytext=(8, -14),
                fontsize=8, color="tab:green",
            )
    ax.set_xlabel("slot depth budget")
    ax.set_ylabel("count (log scale)")
    ax.set_xticks(depths)
    ax.set_title(
        "The combinatorial wall is a mirage:\n"
        "the EML function space collapses by 5+ orders of magnitude"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "function_space_collapse.png"), dpi=160)
    print("wrote function_space_collapse.png")


def fig_recovery():
    with open(os.path.join(ROOT, "depth_wall_results.json")) as f:
        res = json.load(f)
    depths = sorted(int(d) for d in res["depths"])
    pool_rate, gd_rate, uni_rate = [], [], []
    for d in depths:
        row = res["depths"][str(d)]
        pool_rate.append(row["pool"]["recovered"] / row["pool"]["n"])
        gd_rate.append(row["gd"]["recovered"] / max(row["gd"]["n"], 1))
        uni_rate.append(
            row["uniform"]["recovered"] / max(row["uniform"]["n"], 1)
        )
    cov_depths = sorted(int(d) for d in res.get("coverage", {}))
    cov_rate = [
        res["coverage"][str(d)]["coverage_fraction"] for d in cov_depths
    ]

    # Source-paper blind-recovery context (Odrzywolek 2026, Sect. 4.3).
    paper_d = [2, 3, 4, 5, 6]
    paper_r = [1.0, 0.25, 0.25, 0.01, 0.0]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(
        depths, pool_rate, "o-", lw=2.5, color="tab:green",
        label="canonical pool (this work), depth-critical targets",
    )
    if cov_depths:
        ax.plot(
            cov_depths, cov_rate, "o--", lw=1.8, color="tab:olive",
            label="pool coverage of random deeper targets",
        )
    ax.plot(depths, gd_rate, "s-", color="tab:red",
            label="gradient descent (8 restarts)")
    ax.plot(depths, uni_rate, "v-", color="tab:orange",
            label="uniform syntactic sampling (30k)")
    ax.plot(paper_d, paper_r, "^:", color="gray",
            label="blind recovery reported in the source paper")
    ax.set_xlabel("target depth")
    ax.set_ylabel("verified exact-recovery rate")
    ax.set_ylim(-0.04, 1.06)
    ax.set_xticks(sorted(set(depths + cov_depths + paper_d)))
    ax.set_title("Exact recovery through the depth wall")
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "recovery_vs_depth.png"), dpi=160)
    print("wrote recovery_vs_depth.png")


def fig_gradients():
    with open(os.path.join(ROOT, "saturation_results.json")) as f:
        res = json.load(f)
    depths = sorted(int(d) for d in res["eml"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))

    for op, color in (("eml", "tab:red"), ("psi", "tab:blue")):
        med = [res[op][str(d)]["median_live_grad"] for d in depths]
        mx = [res[op][str(d)]["median_max_grad"] for d in depths]
        ax1.semilogy(depths, med, "o-", color=color, label=f"{op}: median live |grad|")
        ax1.semilogy(depths, mx, "s--", color=color, alpha=0.55,
                     label=f"{op}: median max |grad|")
        dead = [res[op][str(d)]["mean_dead_slot_fraction"] for d in depths]
        ax2.plot(depths, dead, "o-", color=color, label=op)

    ax1.set_xlabel("tree depth")
    ax1.set_ylabel("bottom-level logit gradient")
    ax1.set_title("Gradient scale at random init")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)
    ax2.set_xlabel("tree depth")
    ax2.set_ylabel("fraction of dead bottom slots")
    ax2.set_title("Exactly-zero gradients (clamp saturation)")
    ax2.set_ylim(-0.03, 1.03)
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "gradient_pathology.png"), dpi=160)
    print("wrote gradient_pathology.png")


if __name__ == "__main__":
    fig_collapse()
    fig_recovery()
    fig_gradients()
