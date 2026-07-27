#!/usr/bin/env bash
# End-to-end reproduction of every number in the paper and README.
#
# Hardware assumption: 4+ CPU cores, 15 GB RAM, no GPU needed.
# Total wall-clock: roughly 6-9 hours, dominated by the depth-wall
# benchmark's gradient-descent baselines and the UCI suite.
#
# Every step writes a versioned JSON at the repository root; the paper
# tables and figures are generated from those files only.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== [1/7] unit tests + lint (sanity) ==="
python -m pytest tests/ -q
python -m ruff check eml_gam tests scripts

echo "=== [2/7] depth-wall benchmark (pool vs GD vs uniform + coverage) ==="
python -m eml_gam.benchmarks.depth_wall          # ~3-4 h

echo "=== [3/7] extended-budget rescue pass for depth-5 misses ==="
python -m scripts.depth5_rescue                  # ~30-60 min

echo "=== [4/7] mechanism measurements (saturation / gradients) ==="
python -m eml_gam.benchmarks.saturation          # ~25 min

echo "=== [5/7] showcase: paper target + minimal-depth certificates ==="
python -m eml_gam.benchmarks.showcase            # ~5 min

echo "=== [6/7] psi-operator generality benchmark ==="
python -m eml_gam.benchmarks.psi_depth_wall      # ~1-2 h

echo "=== [7/7] 9-dataset UCI safe-extrapolation suite ==="
# Data files are fetched from public GitHub mirrors (see loaders in
# scripts/run_uci_safe.py); place them under data/ first if the
# original archive.ics.uci.edu is unreachable from your network.
python -m scripts.run_uci_safe                   # ~1.5-2 h

echo "=== figures ==="
python -m scripts.make_depth_wall_figures

echo "=== paper ==="
if command -v pdflatex >/dev/null; then
  (cd paper && pdflatex -interaction=nonstopmode paper.tex >/dev/null \
            && pdflatex -interaction=nonstopmode paper.tex >/dev/null)
  echo "paper/paper.pdf rebuilt"
fi

echo "ALL DONE. Result JSONs at repo root; figures in figures/."
