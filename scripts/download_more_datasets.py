"""Download additional UCI / scientific datasets with natural exp/log
structure and extrapolation axis:

- Combined Cycle Power Plant (CCPP): 4 features (T, V, AP, RH) → net
  electrical output. Power output has strong thermodynamic dependence
  on temperature (exponential in Carnot efficiency). Extrap split: T.

- Energy Efficiency: 8 features (building geometry, orientation,
  glazing area) → heating load, cooling load. Heating load is
  approximately exp-decaying in surface area ratio. Extrap: glazing
  area.

- Superconductivity Dataset: 81 features → critical temperature Tc.
  BCS theory: Tc ∝ exp(-1/(lambda - mu*)), so exp structure. Extrap:
  feature quantile split.

All datasets are saved as CSV under data/.
"""
from __future__ import annotations

import io
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)


def _download_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def fetch_ccpp():
    out_path = os.path.join(DATA_DIR, "ccpp.csv")
    if os.path.exists(out_path):
        print(f"skip: {out_path} already present")
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    # UCI hosts CCPP.zip with excel file
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip"
    try:
        data = _download_bytes(url)
        z = zipfile.ZipFile(io.BytesIO(data))
        # Look for xlsx file inside
        for n in z.namelist():
            if n.endswith(".xlsx") and "Folds5x2" in n:
                raw = z.read(n)
                df = pd.read_excel(io.BytesIO(raw), sheet_name="Sheet1")
                df.to_csv(out_path, index=False)
                print(f"saved {df.shape} -> {out_path}")
                return
        print(f"CCPP xlsx not found in zip: {z.namelist()}")
    except Exception as e:
        print(f"CCPP download failed: {e}")


def fetch_energy_efficiency():
    out_path = os.path.join(DATA_DIR, "energy_eff.csv")
    if os.path.exists(out_path):
        print(f"skip: {out_path} already present")
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    try:
        data = _download_bytes(url)
        df = pd.read_excel(io.BytesIO(data))
        # Drop any rows that are all NaN
        df = df.dropna(how='all')
        df.columns = [
            "X1_rel_compactness", "X2_surface_area", "X3_wall_area",
            "X4_roof_area", "X5_overall_height", "X6_orientation",
            "X7_glazing_area", "X8_glazing_distribution",
            "Y1_heating_load", "Y2_cooling_load",
        ][:df.shape[1]]
        df.to_csv(out_path, index=False)
        print(f"saved {df.shape} -> {out_path}")
    except Exception as e:
        print(f"Energy Efficiency download failed: {e}")


def fetch_superconductivity():
    out_path = os.path.join(DATA_DIR, "superconductivity.csv")
    if os.path.exists(out_path):
        print(f"skip: {out_path} already present")
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip"
    try:
        data = _download_bytes(url)
        z = zipfile.ZipFile(io.BytesIO(data))
        for n in z.namelist():
            if n.endswith("train.csv"):
                raw = z.read(n)
                df = pd.read_csv(io.BytesIO(raw))
                df.to_csv(out_path, index=False)
                print(f"saved {df.shape} -> {out_path}")
                return
    except Exception as e:
        print(f"Superconductivity download failed: {e}")


def fetch_abalone():
    out_path = os.path.join(DATA_DIR, "abalone.csv")
    if os.path.exists(out_path):
        print(f"skip: {out_path} already present")
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "abalone/abalone.data"
    )
    cols = [
        "sex", "length", "diameter", "height",
        "whole_weight", "shucked_weight", "viscera_weight", "shell_weight",
        "rings",
    ]
    try:
        data = _download_bytes(url)
        df = pd.read_csv(io.BytesIO(data), header=None, names=cols)
        # Drop sex (categorical) for numeric benchmark
        df = df.drop(columns=["sex"])
        df.to_csv(out_path, index=False)
        print(f"saved {df.shape} -> {out_path}")
    except Exception as e:
        print(f"Abalone download failed: {e}")


def fetch_forest_fires():
    out_path = os.path.join(DATA_DIR, "forest_fires.csv")
    if os.path.exists(out_path):
        print(f"skip: {out_path} already present")
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "forest-fires/forestfires.csv"
    )
    try:
        data = _download_bytes(url)
        df = pd.read_csv(io.BytesIO(data))
        # Drop categorical month, day
        df = df.drop(columns=[c for c in ["month", "day"] if c in df.columns])
        # Target: area is log1p-distributed, log1p transform y
        df["area_log1p"] = np.log1p(df["area"])
        df.to_csv(out_path, index=False)
        print(f"saved {df.shape} -> {out_path}")
    except Exception as e:
        print(f"Forest Fires download failed: {e}")


def main():
    fetch_ccpp()
    fetch_energy_efficiency()
    fetch_superconductivity()
    fetch_abalone()
    fetch_forest_fires()


if __name__ == "__main__":
    main()
