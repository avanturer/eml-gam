"""Run every benchmark and write the aggregated results to ``results.json``."""

import dataclasses
import json

import numpy as np
import torch

from eml_gam.benchmarks.extrapolation import run_all as run_synthetic
from eml_gam.benchmarks.real_world import (
    run_airfoil,
    run_auto_mpg,
    run_concrete,
    run_yacht,
)
from eml_gam.benchmarks.scalability import run_scalability


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    def _reseed():
        torch.manual_seed(0)
        np.random.seed(0)

    synthetic = [dataclasses.asdict(r) for r in run_synthetic(verbose=False)]
    _reseed()
    yacht = [
        dataclasses.asdict(r)
        for r in run_yacht(verbose=False, isolated_physics=True)
    ]
    _reseed()
    auto_mpg = [
        dataclasses.asdict(r)
        for r in run_auto_mpg(verbose=False, isolated_physics=False)
    ]
    _reseed()
    concrete = [
        dataclasses.asdict(r)
        for r in run_concrete(verbose=False, isolated_physics=True)
    ]
    _reseed()
    airfoil = [
        dataclasses.asdict(r)
        for r in run_airfoil(verbose=False, isolated_physics=True)
    ]
    _reseed()
    scale = [dataclasses.asdict(r) for r in run_scalability(verbose=False)]

    out = {
        "synthetic_extrapolation": synthetic,
        "real_world_yacht": yacht,
        "real_world_auto_mpg": auto_mpg,
        "real_world_concrete": concrete,
        "real_world_airfoil": airfoil,
        "scalability_20_features": scale,
    }
    out_path = "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"saved all benchmarks -> {out_path}")
    print(f"  synthetic:   {len(synthetic)} rows")
    print(f"  yacht:       {len(yacht)} rows")
    print(f"  auto_mpg:    {len(auto_mpg)} rows")
    print(f"  concrete:    {len(concrete)} rows")
    print(f"  airfoil:     {len(airfoil)} rows")
    print(f"  scalability: {len(scale)} rows")


if __name__ == "__main__":
    main()
