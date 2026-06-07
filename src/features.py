"""Clinically meaningful feature engineering for cardiac arrest risk modeling.

Feature functions in this module are deterministic row-wise transformations.
They do not learn parameters from data, so they can be applied independently to
train, validation, and test splits after splitting. Any fitted transformations
such as imputation, encoding, scaling, or feature selection should still be fit
on training data only.
"""

from __future__ import annotations

from typing import Hashable

import numpy as np
import pandas as pd


LAB_ABNORMALITY_SPECS = {
    "Na": (135.0, 145.0, "sodium_abnormal_flag"),
    "K": (3.5, 5.0, "potassium_abnormal_flag"),
    "Cl": (98.0, 107.0, "chloride_abnormal_flag"),
    "Urea": (7.0, 45.0, "urea_abnormal_flag"),
    # The source data uses the misspelled raw column name "Ceratinine".
    "Ceratinine": (60.0, 110.0, "creatinine_abnormal_flag"),
}


def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived clinical features to a copy of the input dataframe.

    Parameters
    ----------
    df:
        Split-specific dataframe containing raw vitals, age, GCS, and laboratory
        values. Pass train/validation/test data separately to preserve the
        leakage-safe workflow.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with derived features appended.
    """
    featured = df.copy()
    featured = add_pulse_pressure(featured)
    featured = add_shock_index(featured)
    featured = add_age_band(featured)
    featured = add_hypoxemia_flag(featured)
    featured = add_gcs_severity(featured)
    featured = add_lab_abnormality_flags(featured)
    return featured


def add_pulse_pressure(
    df: pd.DataFrame,
    sbp_col: Hashable = "SBP",
    dbp_col: Hashable = "DBP",
    output_col: Hashable = "pulse_pressure",
) -> pd.DataFrame:
    """Add pulse pressure, calculated as systolic minus diastolic pressure."""
    _require_columns(df, [sbp_col, dbp_col])
    result = df.copy()
    result[output_col] = result[sbp_col] - result[dbp_col]
    return result


def add_shock_index(
    df: pd.DataFrame,
    hr_col: Hashable = "HR",
    sbp_col: Hashable = "SBP",
    output_col: Hashable = "shock_index",
) -> pd.DataFrame:
    """Add shock index, calculated as heart rate divided by systolic pressure."""
    _require_columns(df, [hr_col, sbp_col])
    result = df.copy()
    result[output_col] = result[hr_col] / result[sbp_col].replace(0, np.nan)
    return result


def add_age_band(
    df: pd.DataFrame,
    age_col: Hashable = "Age",
    output_col: Hashable = "age_band",
) -> pd.DataFrame:
    """Add clinically interpretable age categories.

    Bands are represented as ordered labels: ``<18``, ``18-39``, ``40-64``,
    ``65-79``, and ``80+``.
    """
    _require_columns(df, [age_col])
    result = df.copy()
    result[output_col] = pd.cut(
        result[age_col],
        bins=[-np.inf, 17, 39, 64, 79, np.inf],
        labels=["<18", "18-39", "40-64", "65-79", "80+"],
        right=True,
        ordered=True,
    )
    return result


def add_hypoxemia_flag(
    df: pd.DataFrame,
    spo2_col: Hashable = "SpO2",
    output_col: Hashable = "hypoxemia_flag",
    threshold: float = 90.0,
) -> pd.DataFrame:
    """Add an indicator for hypoxemia based on oxygen saturation.

    Values below ``threshold`` are flagged as 1. Missing values remain missing so
    downstream imputers can handle them using training data only.
    """
    _require_columns(df, [spo2_col])
    result = df.copy()
    result[output_col] = _flag_below_threshold(result[spo2_col], threshold)
    return result


def add_gcs_severity(
    df: pd.DataFrame,
    gcs_col: Hashable = "GCS",
    output_col: Hashable = "gcs_severity",
) -> pd.DataFrame:
    """Add Glasgow Coma Scale severity categories.

    Categories follow common clinical groupings: severe (3-8), moderate (9-12),
    mild (13-15), and unknown for values outside the expected range.
    """
    _require_columns(df, [gcs_col])
    result = df.copy()
    result[output_col] = pd.cut(
        result[gcs_col],
        bins=[-np.inf, 2, 8, 12, 15, np.inf],
        labels=["unknown", "severe", "moderate", "mild", "unknown"],
        right=True,
        ordered=False,
    )
    return result


def add_lab_abnormality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add abnormality flags for electrolytes and renal function markers.

    The flags use broad adult reference intervals and are intended as clinically
    interpretable screening features, not diagnosis-specific cutoffs. Missing lab
    values remain missing instead of being forced to normal or abnormal.
    """
    result = add_sodium_abnormality_flag(df)
    result = add_potassium_abnormality_flag(result)
    result = add_chloride_abnormality_flag(result)
    result = add_urea_abnormality_flag(result)
    result = add_creatinine_abnormality_flag(result)
    return result


def add_sodium_abnormality_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add a flag for sodium values outside the normal reference interval."""
    return _add_lab_abnormality_flag(df, raw_col="Na")


def add_potassium_abnormality_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add a flag for potassium values outside the normal reference interval."""
    return _add_lab_abnormality_flag(df, raw_col="K")


def add_chloride_abnormality_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add a flag for chloride values outside the normal reference interval."""
    return _add_lab_abnormality_flag(df, raw_col="Cl")


def add_urea_abnormality_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add a flag for urea values outside the normal reference interval."""
    return _add_lab_abnormality_flag(df, raw_col="Urea")


def add_creatinine_abnormality_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add a creatinine flag using the raw misspelled ``Ceratinine`` column."""
    return _add_lab_abnormality_flag(df, raw_col="Ceratinine")


def _add_lab_abnormality_flag(df: pd.DataFrame, raw_col: Hashable) -> pd.DataFrame:
    """Add one lab abnormality flag from ``LAB_ABNORMALITY_SPECS``."""
    lower, upper, output_col = LAB_ABNORMALITY_SPECS[raw_col]
    _require_columns(df, [raw_col])
    result = df.copy()
    result[output_col] = _flag_outside_range(result[raw_col], lower, upper)
    return result


def _flag_below_threshold(values: pd.Series, threshold: float) -> pd.Series:
    """Return nullable integer flags for values below a threshold."""
    flags = values.lt(threshold).astype("Int64")
    flags[values.isna()] = pd.NA
    return flags


def _flag_outside_range(values: pd.Series, lower: float, upper: float) -> pd.Series:
    """Return nullable integer flags for values outside a reference range."""
    flags = (values.lt(lower) | values.gt(upper)).astype("Int64")
    flags[values.isna()] = pd.NA
    return flags


def _require_columns(df: pd.DataFrame, columns: list[Hashable]) -> None:
    """Raise a clear error if expected columns are absent."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")
