"""Download the UCI datasets used by the real-world benchmarks.

Running this script places ``data/auto_mpg.csv`` and ``data/yacht.csv``
under the repository root. The datasets are redistributed by UCI under
their standard licence.
"""

from __future__ import annotations

import io
import os
import urllib.request

import pandas as pd

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)


AUTO_MPG_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "auto-mpg/auto-mpg.data"
)
YACHT_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00243/yacht_hydrodynamics.data"
)
CONCRETE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "concrete/compressive/Concrete_Data.xls"
)
AIRFOIL_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00291/airfoil_self_noise.dat"
)


def _download(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode()


def _download_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch_auto_mpg() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "auto_mpg.csv")
    if os.path.exists(out_path):
        print(f"skip: {out_path} already present")
        return
    text = _download(AUTO_MPG_URL)
    cols = [
        "mpg", "cylinders", "displacement", "horsepower", "weight",
        "acceleration", "model_year", "origin", "car_name",
    ]
    df = (
        pd.read_csv(io.StringIO(text), sep=r"\s+", header=None, names=cols, na_values="?")
        .drop(columns=["car_name"])
        .dropna()
    )
    df.to_csv(out_path, index=False)
    print(f"saved {df.shape} -> {out_path}")


def fetch_yacht() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "yacht.csv")
    if os.path.exists(out_path):
        print(f"skip: {out_path} already present")
        return
    text = _download(YACHT_URL)
    cols = [
        "longitudinal_pos", "prismatic_coef", "length_disp_ratio",
        "beam_draught_ratio", "length_beam_ratio", "froude_number",
        "residuary_resistance",
    ]
    df = pd.read_csv(io.StringIO(text), sep=r"\s+", header=None, names=cols)
    df.to_csv(out_path, index=False)
    print(f"saved {df.shape} -> {out_path}")


def fetch_concrete() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "concrete.csv")
    if os.path.exists(out_path):
        print(f"skip: {out_path} already present")
        return
    raw = _download_bytes(CONCRETE_URL)
    cols = [
        "cement", "blast_furnace_slag", "fly_ash", "water",
        "superplasticizer", "coarse_aggregate", "fine_aggregate",
        "age", "compressive_strength",
    ]
    df = pd.read_excel(io.BytesIO(raw), header=0, names=cols)
    df.to_csv(out_path, index=False)
    print(f"saved {df.shape} -> {out_path}")


def fetch_airfoil() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "airfoil.csv")
    if os.path.exists(out_path):
        print(f"skip: {out_path} already present")
        return
    text = _download(AIRFOIL_URL)
    cols = [
        "frequency", "angle_of_attack", "chord_length",
        "free_stream_velocity", "suction_side_displacement",
        "sound_pressure_level",
    ]
    df = pd.read_csv(io.StringIO(text), sep="\t", header=None, names=cols)
    df.to_csv(out_path, index=False)
    print(f"saved {df.shape} -> {out_path}")


if __name__ == "__main__":
    fetch_auto_mpg()
    fetch_yacht()
    fetch_concrete()
    fetch_airfoil()
