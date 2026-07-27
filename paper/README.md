# Paper draft

`paper.tex` is a self-contained LaTeX manuscript covering every
scientific contribution of this repository. Compile with

```
pdflatex paper.tex
pdflatex paper.tex
```

(bibliography is inline; two passes resolve references) or any
equivalent workflow (Overleaf, TeX Live, MiKTeX).

## Section-to-code map

| paper section | source of numbers |
|---|---|
| §3 function-space collapse (Table 1, Fig 1) | `eml_gam/canonical_pool.py` → `depth_wall_results.json` (protocol → pool_build_stats); ψ control in `psi_pool_stats.json` |
| §4 pool / root inversion / peeling | `eml_gam/canonical_pool.py` |
| §5 recovery tables (Tables 2–3, Fig 2) | `eml_gam/benchmarks/depth_wall.py` → `depth_wall_results.json` |
| §5 showcase target | `eml_gam/benchmarks/showcase.py` → `showcase_results.json` |
| §6 mechanism (Propositions, Fig 3) | `eml_gam/benchmarks/saturation.py` → `saturation_results.json`; derivation in `docs/depth_wall_theory.md` |
| §6.1 cross-operator landscape | `eml_gam/benchmarks/cross_operator_landscape.py` → `cross_operator_landscape.json` |
| §7 minimal-depth certificates | `eml_gam/benchmarks/showcase.py` → `showcase_results.json` |
| §8 SafePoolGAM UCI table | `scripts/run_uci_safe.py` → `safe_uci_results.json` |
| §9 retained theory | `docs/sheffer_analysis.md`, `docs/transcendence_theorem.md`, `eml_gam/transcendence_witness.py` |

All numerical claims can be reproduced by running the relevant Python
entry point from the repository root. Figures regenerate via
`python -m scripts.make_depth_wall_figures`.
