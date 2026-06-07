"""Basic regression tests for the cardiac arrest risk modeling data pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features import add_clinical_features, add_pulse_pressure, add_shock_index
from src.preprocessing import create_train_validation_test_splits, load_raw_dataset


REQUIRED_RAW_COLUMNS = {
    "ID",
    "SBP",
    "DBP",
    "HR",
    "RR",
    "BT",
    "SpO2",
    "Age",
    "Gender",
    "GCS",
    "Na",
    "K",
    "Cl",
    "Urea",
    "Ceratinine",
    "Alcoholic",
    "Smoke",
    "FHCD",
    "TriageScore",
    "Outcome",
}

EXPECTED_ENGINEERED_COLUMNS = {
    "pulse_pressure",
    "shock_index",
    "age_band",
    "hypoxemia_flag",
    "gcs_severity",
    "sodium_abnormal_flag",
    "potassium_abnormal_flag",
    "chloride_abnormal_flag",
    "urea_abnormal_flag",
    "creatinine_abnormal_flag",
}


def _minimal_raw_dataframe() -> pd.DataFrame:
    """Create enough raw columns to exercise all clinical feature functions."""
    return pd.DataFrame(
        {
            "ID": [1, 2],
            "SBP": [120, 80],
            "DBP": [80, 40],
            "HR": [60, 100],
            "RR": [16, 20],
            "BT": [98.6, 99.1],
            "SpO2": [98, 89],
            "Age": [45, 82],
            "Gender": [1, 0],
            "GCS": [15, 7],
            "Na": [140, 130],
            "K": [4.2, 5.6],
            "Cl": [101, 111],
            "Urea": [20, 50],
            "Ceratinine": [90, 150],
            "Alcoholic": [0, 1],
            "Smoke": [0, 1],
            "FHCD": [0, 1],
            "TriageScore": [3, 1],
            "Outcome": [0, 1],
        }
    )


def test_load_raw_dataset_returns_non_empty_dataframe() -> None:
    """The configured raw CSV should load into a non-empty dataframe."""
    raw_df = load_raw_dataset()

    assert isinstance(raw_df, pd.DataFrame)
    assert not raw_df.empty


def test_raw_dataset_contains_required_columns() -> None:
    """The raw dataset should expose all columns used by preprocessing/features."""
    raw_df = load_raw_dataset()

    assert REQUIRED_RAW_COLUMNS.issubset(raw_df.columns)


def test_add_clinical_features_creates_expected_columns() -> None:
    """Feature engineering should append all expected derived columns."""
    featured_df = add_clinical_features(_minimal_raw_dataframe())

    assert EXPECTED_ENGINEERED_COLUMNS.issubset(featured_df.columns)


def test_pulse_pressure_calculation() -> None:
    """Pulse pressure should equal systolic blood pressure minus diastolic."""
    df = pd.DataFrame({"SBP": [120, 95], "DBP": [80, 60]})

    result = add_pulse_pressure(df)

    assert result["pulse_pressure"].tolist() == [40, 35]


def test_shock_index_calculation() -> None:
    """Shock index should equal heart rate divided by systolic pressure."""
    df = pd.DataFrame({"HR": [60, 100], "SBP": [120, 80]})

    result = add_shock_index(df)

    assert result["shock_index"].tolist() == pytest.approx([0.5, 1.25])


def test_raw_dataset_contains_outcome_target_column() -> None:
    """The modeling target should be present in the raw dataset."""
    raw_df = load_raw_dataset()

    assert "Outcome" in raw_df.columns


def test_grouped_split_does_not_leak_repeated_patient_ids() -> None:
    """Repeated patient IDs must not appear in more than one split."""
    patient_ids = list(range(1, 11))
    df = pd.DataFrame(
        {
            "ID": [patient_id for patient_id in patient_ids for _ in range(2)],
            "SBP": [120] * 20,
            "Outcome": [patient_id % 2 for patient_id in patient_ids for _ in range(2)],
        }
    )

    train_df, validation_df, test_df = create_train_validation_test_splits(
        df,
        validation_size=0.2,
        test_size=0.2,
        random_state=7,
    )

    train_ids = set(train_df["ID"])
    validation_ids = set(validation_df["ID"])
    test_ids = set(test_df["ID"])

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
