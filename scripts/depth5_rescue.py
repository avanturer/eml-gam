"""Extended-budget rescue pass for the depth-5 misses.

Regenerates the same 40 depth-critical depth-5 targets (same seed as
the main benchmark), identifies the ones the default budget missed,
and re-runs them with escalated caps (1M candidate pairs, per-run cap
256, 20-minute budget). Merges a ``pool_extended`` row into
``depth_wall_results.json``.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eml_gam.benchmarks.depth_wall import (  # noqa: E402
    Verifier,
    make_grid,
    sample_critical_targets,
)
from eml_gam.canonical_pool import FunctionPool  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    x = make_grid()
    pool = FunctionPool(x, operator="eml")
    pool.build(4, verbose=True)
    pool.compact()

    targets, _ = sample_critical_targets(pool, 5, 40, seed=1005)
    print(f"regenerated {len(targets)} depth-5 critical targets")

    results = []
    for i, nested in enumerate(targets):
        y = pool.evaluate_nested(nested, pool.x[:, 0])
        ver = Verifier(pool, nested)
        t0 = time.time()
        # default budget first (fast path, mirrors the main benchmark)
        ok = False
        n_checked = 0
        for ia, ib in pool.lookup_root_pairs(y, 5, cap=300_000, per_run_cap=64):
            n_checked += 1
            if ver.check_pair(ia, ib):
                ok = True
                break
            if time.time() - t0 > 300:
                break
        via = "default"
        if not ok:
            print(f"  target {i}: default budget missed, escalating ...",
                  flush=True)
            via = "extended"
            for ia, ib in pool.lookup_root_pairs(
                y, 5, cap=1_000_000, per_run_cap=256,
                match_decimals=(8, 6, 5),
            ):
                n_checked += 1
                if ver.check_pair(ia, ib):
                    ok = True
                    break
                if time.time() - t0 > 1200:
                    break
        dt = time.time() - t0
        results.append(
            {"target": i, "recovered": bool(ok), "via": via,
             "seconds": dt, "n_checked": n_checked}
        )
        if via == "extended":
            print(f"    -> recovered={ok} in {dt:.0f}s", flush=True)

    n_rec = sum(r["recovered"] for r in results)
    n_ext = sum(1 for r in results if r["via"] == "extended")
    print(f"depth-5 with escalation: {n_rec}/40 recovered "
          f"({n_ext} needed the extended budget)")

    path = os.path.join(ROOT, "depth_wall_results.json")
    with open(path) as f:
        data = json.load(f)
    data["depths"]["5"]["pool_extended"] = {
        "n": len(results),
        "recovered": n_rec,
        "n_needed_extension": n_ext,
        "mean_seconds": float(np.mean([r["seconds"] for r in results])),
        "max_seconds": float(np.max([r["seconds"] for r in results])),
        "budget": "cap 1e6 pairs, per-run 256, 1200 s",
        "per_target": results,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print("merged pool_extended into depth_wall_results.json")


if __name__ == "__main__":
    main()
