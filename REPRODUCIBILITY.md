# Reproducibility

Every quantitative claim in the paper and README traces to a versioned
JSON produced by a deterministic (seeded) entry point in this
repository. `bash scripts/reproduce_all.sh` regenerates everything
end-to-end (~6–9 h on a 4-core CPU, 15 GB RAM, no GPU).

## Claim → artifact map

| claim | artifact | entry point | seeds |
|---|---|---|---|
| function-space collapse (Table 1 / Fig 1) | `depth_wall_results.json → protocol.pool_build_stats` | `eml_gam.benchmarks.depth_wall` | grid seed 12345 |
| ψ zero-degeneracy control | `psi_pool_stats.json`, `psi_depth_wall_results.json` | `eml_gam.benchmarks.psi_depth_wall` | 12345 / 3000+d |
| recovery table (Table 2 / Fig 2) | `depth_wall_results.json → depths` | `eml_gam.benchmarks.depth_wall` | targets 1000+d |
| depth-5 extended-budget rescue | `depth_wall_results.json → depths.5.pool_extended` | `scripts.depth5_rescue` | 1005 |
| coverage at depths 6–8 (Table 3) | `depth_wall_results.json → coverage` | `eml_gam.benchmarks.depth_wall` | 2000+d |
| gradient pathology (Fig 3) | `saturation_results.json` | `eml_gam.benchmarks.saturation` | torch gen 0 |
| showcase target + certificates | `showcase_results.json` | `eml_gam.benchmarks.showcase` | 777 / 12345 |
| UCI table (Table 4) | `safe_uci_results.json` | `scripts.run_uci_safe` | SEED 0 |
| earlier-generation results (landscape, cross-operator, AEES, transcendence) | respective JSONs at repo root | see `paper/README.md` | as recorded |

## Environment

Python ≥ 3.10. Reference versions used for the reported numbers
(`requirements-repro.txt`): numpy 2.4.6, torch 2.13.0 (CPU), sympy
1.14.0, scikit-learn 1.9.0, pandas 3.0.5, xgboost 3.2.0,
interpret-core 0.7.8, matplotlib 3.11.1. Loose ranges in
`pyproject.toml` are sufficient — no result depends on version-specific
behaviour beyond ordinary floating-point reproducibility.

## Determinism and verification discipline

* All target sampling, grids, and subsamples are seeded (table above);
  re-runs reproduce the same targets and the same accept/reject
  decisions.
* A "recovery" is **never** a training-grid R² claim: every reported
  recovery re-evaluates the candidate expression on an independent
  dense grid (1111 points in-range) at relative tolerance 1e−6, plus a
  64-point probe prefilter. Failures are reported as failures.
* Function identity inside the pool uses 64-bit hashes of quantised
  value vectors; hash collisions (~2⁻⁶⁴) cannot corrupt results
  because every recovery is re-verified as above.
* The UCI comparison runs every model on identical arrays and splits;
  superconductivity is subsampled to 5000 train rows *for all models
  alike*. Data files come from public GitHub mirrors (loaders record
  URLs) because `archive.ics.uci.edu` is not reachable from every
  network; all models consume the same mirror files, so the comparison
  is internally consistent.

## Known reproduction sensitivities

* Wall-clock numbers (pool build ≈ 60–90 s, per-target solve times)
  vary with CPU; recovery *counts* do not.
* `interpret`'s EBM has internal parallelism; its R² can shift in the
  third decimal across machines. No claim depends on that precision.
* The four depth-5 default-budget misses are deterministic given the
  seeds; the rescue pass documents their behaviour under an escalated
  budget.
