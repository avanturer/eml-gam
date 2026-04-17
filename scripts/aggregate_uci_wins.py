"""Aggregate all real-UCI extrapolation benchmark results into one
table and count EML-GA²M wins.

Sources consumed (skipped if missing):
  - results.json (legacy yacht/auto_mpg/concrete/airfoil — old run)
  - tuned_real_world.json (this session, robust=True + extrap_penalty)
  - new_uci_results.json (ccpp, energy_eff, superconductivity)
  - abalone_forest_results.json (abalone, forest_fires)

Output: ``uci_wins_summary.json`` + printed Markdown-ready table.
"""
from __future__ import annotations

import json
import os


def load_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"error loading {path}: {e}")
        return None


def _r2_extrap(row):
    """Accept either {'r2_extrap': ...} or legacy 'r2' with 'split'=='extrap'."""
    if "r2_extrap" in row:
        return row["r2_extrap"]
    if row.get("split") == "extrap":
        return row.get("r2")
    return None


def extract_entries(name: str, rows, model_key: str):
    """Given list of rows from a benchmark JSON, extract extrap R² per model."""
    if not isinstance(rows, list):
        return {}
    out = {}
    for r in rows:
        m = r.get("model")
        if m is None:
            continue
        r2 = _r2_extrap(r)
        if r2 is None:
            continue
        out[m] = r2
    return out


def main():
    # Existing results: legacy
    legacy = load_json("results.json")
    legacy_uci = {}
    if legacy:
        for key in ("real_world_yacht", "real_world_auto_mpg",
                    "real_world_concrete", "real_world_airfoil"):
            rows = legacy.get(key, [])
            ds_name = key.replace("real_world_", "")
            legacy_uci[ds_name] = extract_entries(ds_name, rows, "model")

    tuned = load_json("tuned_real_world.json") or {}
    new_uci = load_json("new_uci_results.json") or {}
    abalone_ff = load_json("abalone_forest_results.json") or {}

    all_rows = []

    # For each dataset, combine legacy + tuned (tuned takes priority for emlgam)
    datasets = set(legacy_uci) | set(tuned) | set(new_uci) | set(abalone_ff)
    for ds in sorted(datasets):
        row = {"dataset": ds}
        # Baselines come from whichever source has them
        for src in [legacy_uci.get(ds, {}), extract_entries(ds, tuned.get(ds), "model"),
                    extract_entries(ds, new_uci.get(ds), "model"),
                    extract_entries(ds, abalone_ff.get(ds), "model")]:
            for m in ["linear", "ebm", "xgboost", "gplearn"]:
                if m in src and m not in row:
                    row[m] = src[m]
        # EML-GA²M: prefer tuned version
        em_tuned = None
        for src in [extract_entries(ds, tuned.get(ds), "model"),
                    extract_entries(ds, new_uci.get(ds), "model"),
                    extract_entries(ds, abalone_ff.get(ds), "model")]:
            if "emlgam_tuned" in src:
                em_tuned = src["emlgam_tuned"]
                break
        em_legacy = legacy_uci.get(ds, {}).get("emlgam")
        row["emlgam_tuned"] = em_tuned
        row["emlgam_legacy"] = em_legacy

        # Best baseline
        baselines = [row.get(m) for m in ["linear", "ebm", "xgboost", "gplearn"] if row.get(m) is not None]
        row["best_baseline"] = max(baselines) if baselines else None

        # Win? (use best of tuned/legacy for emlgam)
        em = em_tuned if em_tuned is not None else em_legacy
        if em is not None and row["best_baseline"] is not None:
            row["emlgam_best"] = em
            row["win"] = em > row["best_baseline"]
        else:
            row["emlgam_best"] = em
            row["win"] = None

        all_rows.append(row)

    # Summary
    wins = [r for r in all_rows if r.get("win")]
    print(f"\n=== EML-GA²M Real-UCI Extrapolation Wins ({len(wins)}/{len(all_rows)}) ===\n")
    print(f"{'dataset':<22s} {'linear':>8} {'ebm':>8} {'xgboost':>8} {'gplearn':>8} {'emlgam':>9}  verdict")
    for r in all_rows:
        ds = r["dataset"]
        def fmt(k):
            v = r.get(k)
            return f"{v:+.2f}" if isinstance(v, (int, float)) else "   -  "
        verdict = "WIN" if r.get("win") else ("loss" if r.get("win") is False else "-")
        print(f"{ds:<22s} {fmt('linear'):>8} {fmt('ebm'):>8} {fmt('xgboost'):>8} "
              f"{fmt('gplearn'):>8} {fmt('emlgam_best'):>9}  {verdict}")

    print(f"\nTotal wins: {len(wins)} / {len(all_rows)}")
    for w in wins:
        print(f"  WIN: {w['dataset']} — EML={w['emlgam_best']:.3f} vs best baseline={w['best_baseline']:.3f}")

    with open("uci_wins_summary.json", "w") as f:
        json.dump({
            "n_wins": len(wins),
            "n_total": len(all_rows),
            "datasets": all_rows,
            "wins_list": [w["dataset"] for w in wins],
        }, f, indent=2)
    print("\nSaved uci_wins_summary.json")


if __name__ == "__main__":
    main()
