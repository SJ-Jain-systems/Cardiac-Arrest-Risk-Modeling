"""Create a processed cardiac patient dataset with engineered features.

This script keeps the raw CSV unchanged, appends deterministic clinical feature
engineering columns, and writes the processed dataset to ``data/processed``.
Run from the repository root with:

    python src/create_processed_dataset.py
"""

from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_PATH
from src.features import add_clinical_features

OUTPUT_PATH = PROCESSED_DATA_DIR / "cardiac_patient_processed.csv"


def create_processed_dataset() -> pd.DataFrame:
    """Load raw patient data, append engineered features, and save a CSV."""
    raw_df = pd.read_csv(RAW_DATA_PATH)
    raw_columns = list(raw_df.columns)

    processed_df = add_clinical_features(raw_df)
    new_features = [
        column for column in processed_df.columns if column not in raw_columns
    ]

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Raw dataset shape: {raw_df.shape}")
    print(f"Processed dataset shape: {processed_df.shape}")
    print(f"Output path: {OUTPUT_PATH}")
    print("New engineered features created:")
    for feature in new_features:
        print(f"- {feature}")

    return processed_df


if __name__ == "__main__":
    create_processed_dataset()
