"""Characterise the four residual depth-5 misses.

For each miss (targets 0, 5, 26, 38 of the seed-1005 critical set):
check whether the target's true root children exist in the pool as
functions (grid identity), whether the stored representatives agree
with the true children off-grid, and whether the true pair is
discoverable by either inversion pass. Writes
``depth5_miss_diagnostics.json`` — the factual basis for the paper's
failure-mode paragraph.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eml_gam.benchmarks.depth_wall import (  # noqa: E402
    make_grid,
    sample_critical_targets,
)
from eml_gam.canonical_pool import FunctionPool  # noqa: E402

MISSES = (0, 5, 26, 38)


def main() -> None:
    x = make_grid()
    pool = FunctionPool(x, operator="eml")
    pool.build(4)
    targets, _ = sample_critical_targets(pool, 5, 40, seed=1005)
    x_dense = np.linspace(0.3, 2.5, 1111)

    out = []
    for i in MISSES:
        nested = targets[i]
        _, na, nb = nested
        va = pool.evaluate_nested(na, x)
        vb = pool.evaluate_nested(nb, x)
        ya = pool.evaluate_nested(na, x_dense)
        yb = pool.evaluate_nested(nb, x_dense)
        row: dict = {"target": i}
        for name, v, y_dense_child in (("a", va, ya), ("b", vb, yb)):
            idx = pool._digest_to_idx.get(pool._digest(v))
            row[f"{name}_in_pool"] = idx is not None
            if idx is not None:
                rep = pool.evaluate_entry(idx, x_dense)
                dev = float(np.max(np.abs(rep - y_dense_child)))
                row[f"{name}_rep_offgrid_dev"] = dev
                row[f"{name}_rep_size"] = int(pool.expr_size(idx))
        # Is the true pair itself producible by matching?
        if row.get("a_in_pool") and row.get("b_in_pool"):
            ia = pool._digest_to_idx[pool._digest(va)]
            ib = pool._digest_to_idx[pool._digest(vb)]
            pair_dense = pool.evaluate_pair(ia, ib, x_dense)
            y_dense = pool.evaluate_nested(nested, x_dense)
            row["rep_pair_offgrid_dev"] = float(
                np.max(np.abs(pair_dense - y_dense))
            )
        out.append(row)
        print(row, flush=True)

    with open("depth5_miss_diagnostics.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote depth5_miss_diagnostics.json")


if __name__ == "__main__":
    main()
