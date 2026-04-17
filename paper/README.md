# Paper draft

`paper.tex` is a self-contained LaTeX manuscript summarising every
scientific contribution of this repository. Compile with

```
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

or any equivalent workflow (Overleaf, TeX Live, MiKTeX).

## Section-to-code map

| paper section | source of numbers |
|---|---|
| Section 4 landscape tables | `eml_gam/benchmarks/landscape.py` -> `landscape_results.json` |
| Section 5 cross-operator table | `eml_gam/benchmarks/cross_operator_landscape.py` -> `cross_operator_landscape.json` |
| Section 5 theorem 1 | `docs/sheffer_analysis.md` |
| Section 5 conjecture (transcendence) | `eml_gam/transcendence.py`, `docs/transcendence_analysis.md` |
| Section 6 AEES | `eml_gam/atlas_expansion.py`, `eml_gam/benchmarks/aees_vs_gradient.py` |
| Section 7 neural beam search | `eml_gam/neural_beam.py` |
| Section 8 synthetic table | `eml_gam/benchmarks/extrapolation.py` -> `results.json` |
| Section 8 Yacht table | `eml_gam/benchmarks/real_world.py` (`run_yacht`) |
| Section 8 scalability paragraph | `eml_gam/benchmarks/scalability.py` |

All numerical claims can be reproduced by running the relevant Python
entry point from the repository root.
