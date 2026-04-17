#!/bin/bash
# Run all three UCI benchmarks sequentially to avoid CPU contention.
# Each produces a JSON result file; combine via aggregate_uci_wins.py.

set -e
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1

echo "[1/3] Running tuned UCI (yacht, auto_mpg, concrete, airfoil)..."
python -u scripts/run_tuned_real_world.py 2>&1 | tee /tmp/tuned_uci.log

echo ""
echo "[2/3] Running new UCI (CCPP, energy_eff, superconductivity)..."
python -u scripts/run_new_uci_datasets.py 2>&1 | tee /tmp/new_uci.log

echo ""
echo "[3/3] Running abalone + forest_fires..."
python -u scripts/run_abalone_forestfires.py 2>&1 | tee /tmp/abalone_ff.log

echo ""
echo "=== Aggregating wins ==="
python -u scripts/aggregate_uci_wins.py 2>&1 | tee /tmp/uci_wins.log
