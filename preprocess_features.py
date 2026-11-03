"""Build the reduced feature set from the labelled wearable CSVs.

Usage:
    python preprocess_features.py <input_dir> <output_dir>

Reads ALL_PATIENTS_LABELED_{12H,24H,48H}.csv from <input_dir>, derives the
tri-axial magnitude features, keeps only the selected features plus the
identifier and label columns, and writes FEATURES_{12H,24H,48H}.csv to
<output_dir>.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = ["12H", "24H", "48H"]

MAGNITUDE_SOURCES = {
    "acticounts_magnitude": ["acticounts_x_axis", "acticounts_y_axis", "acticounts_z_axis"],
    "counts_magnitude": ["counts_x_axis", "counts_y_axis", "counts_z_axis"],
}

FEATURES = [
    "acticounts_magnitude",
    "counts_magnitude",
    "accelerometers_std_g",
    "activity_counts",
    "eda_scl_usiemens",
    "met",
    "pulse_rate_bpm",
    "step_counts",
    "temperature_celsius",
]

ID_COLUMNS = ["participant_full_id", "timestamp_unix", "timestamp_iso"]
LABEL_COLUMNS = ["state", "state_name", "migraine_type", "migraine_type_name"]


def add_magnitudes(df: pd.DataFrame) -> pd.DataFrame:
    """Add Euclidean-norm columns sqrt(x^2 + y^2 + z^2) for each axis group."""
    df = df.copy()
    for name, axes in MAGNITUDE_SOURCES.items():
        df[name] = np.sqrt((df[axes] ** 2).sum(axis=1, min_count=len(axes)))
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_magnitudes(df)
    out = df[ID_COLUMNS + FEATURES + LABEL_COLUMNS]
    return out.sort_values(["participant_full_id", "timestamp_unix"]).reset_index(drop=True)


def main(input_dir: str, output_dir: str) -> None:
    in_dir, out_dir = Path(input_dir), Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for horizon in HORIZONS:
        raw = pd.read_csv(in_dir / f"ALL_PATIENTS_LABELED_{horizon}.csv")
        feats = build_features(raw)
        feats.to_csv(out_dir / f"FEATURES_{horizon}.csv", index=False)
        print(f"{horizon}: {feats.shape[0]} rows x {len(FEATURES)} features")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
