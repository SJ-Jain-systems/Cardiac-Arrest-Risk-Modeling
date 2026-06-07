"""Leakage-safe preprocessing utilities for cardiac arrest risk modeling.

This module only handles raw data loading, target separation, and dataset
splitting. Feature engineering, imputation, encoding, scaling, and model fitting
should be performed *after* these split functions are called so that no
transformation is fit using validation or test data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Hashable

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.config import RAW_DATA_PATH


def load_raw_dataset(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw cardiac patient dataset.

    Parameters
    ----------
    path:
        CSV file to load. Defaults to ``data/CardiacPatientData.csv`` from the
        project configuration. The raw file is read only; this function does not
        clean, transform, or persist derived data.

    Returns
    -------
    pandas.DataFrame
        Raw clinical observations exactly as stored in the CSV.
    """
    return pd.read_csv(path)


def has_repeated_ids(df: pd.DataFrame, id_col: Hashable = "ID") -> bool:
    """Check whether any patient identifier appears more than once.

    Repeated identifiers indicate that rows may be repeated measurements or
    encounters for the same patient. In that case, train/validation/test splits
    must keep all rows for a given ``ID`` in the same split to avoid patient-level
    leakage.
    """
    _require_columns(df, [id_col])
    return bool(df[id_col].duplicated().any())


def check_id_repeats(df: pd.DataFrame, id_col: Hashable = "ID") -> bool:
    """Alias for :func:`has_repeated_ids` using user-facing wording."""
    return has_repeated_ids(df, id_col=id_col)


def separate_features_target(
    df: pd.DataFrame, target_col: Hashable = "Outcome"
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the target label.

    Parameters
    ----------
    df:
        Input data containing both predictors and the target column.
    target_col:
        Name of the target column. Defaults to ``Outcome``.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series]
        ``X`` containing all columns except ``target_col`` and ``y`` containing
        the target values.
    """
    _require_columns(df, [target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    return X, y


def create_train_validation_test_splits(
    df: pd.DataFrame,
    target_col: Hashable = "Outcome",
    id_col: Hashable = "ID",
    validation_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create leakage-safe train, validation, and test splits.

    If any ``ID`` values repeat, this function uses
    :class:`sklearn.model_selection.GroupShuffleSplit` so every row belonging to
    a patient stays entirely in one split. If IDs do not repeat, it uses standard
    stratified row-level splitting by ``Outcome``.

    The split is intentionally performed on the raw dataframe before feature
    engineering or estimator preprocessing. Fit imputers, encoders, scalers,
    feature selectors, and models on the returned training split only, then apply
    the fitted objects to validation/test data.

    Parameters
    ----------
    df:
        Raw dataframe containing the identifier, target, and predictor columns.
    target_col:
        Target label column used for stratification when patient grouping is not
        needed. Defaults to ``Outcome``.
    id_col:
        Patient identifier column used for grouped splitting when values repeat.
        Defaults to ``ID``.
    validation_size:
        Fraction of the full dataset assigned to validation.
    test_size:
        Fraction of the full dataset assigned to test.
    random_state:
        Seed for reproducible split assignments.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        ``train_df``, ``validation_df``, and ``test_df`` as independent copies.
    """
    _require_columns(df, [target_col, id_col])
    _validate_split_sizes(validation_size=validation_size, test_size=test_size)

    if has_repeated_ids(df, id_col=id_col):
        train_df, temp_df = _grouped_train_temp_split(
            df=df,
            id_col=id_col,
            temp_size=validation_size + test_size,
            random_state=random_state,
        )
        validation_df, test_df = _grouped_validation_test_split(
            temp_df=temp_df,
            id_col=id_col,
            validation_size=validation_size,
            test_size=test_size,
            random_state=random_state,
        )
    else:
        train_df, temp_df = train_test_split(
            df,
            test_size=validation_size + test_size,
            random_state=random_state,
            stratify=df[target_col],
        )
        relative_test_size = test_size / (validation_size + test_size)
        validation_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_size,
            random_state=random_state,
            stratify=temp_df[target_col],
        )

    # Reset indices so downstream code can align rows cleanly within each split.
    return (
        train_df.reset_index(drop=True).copy(),
        validation_df.reset_index(drop=True).copy(),
        test_df.reset_index(drop=True).copy(),
    )


def create_train_validation_test_split(
    df: pd.DataFrame,
    target_col: Hashable = "Outcome",
    id_col: Hashable = "ID",
    validation_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Alias for :func:`create_train_validation_test_splits`."""
    return create_train_validation_test_splits(
        df=df,
        target_col=target_col,
        id_col=id_col,
        validation_size=validation_size,
        test_size=test_size,
        random_state=random_state,
    )


def split_features_and_target(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: Hashable = "Outcome",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Separate ``Outcome`` from each already-created split.

    Call this after :func:`create_train_validation_test_splits`. Keeping target
    separation downstream of splitting makes it clear that no feature transform
    should be fit on the full raw dataset.
    """
    X_train, y_train = separate_features_target(train_df, target_col=target_col)
    X_validation, y_validation = separate_features_target(
        validation_df, target_col=target_col
    )
    X_test, y_test = separate_features_target(test_df, target_col=target_col)
    return X_train, y_train, X_validation, y_validation, X_test, y_test


def _grouped_train_temp_split(
    df: pd.DataFrame,
    id_col: Hashable,
    temp_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split raw data into train and temporary sets by patient ID."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=temp_size,
        random_state=random_state,
    )
    train_idx, temp_idx = next(splitter.split(df, groups=df[id_col]))
    return df.iloc[train_idx], df.iloc[temp_idx]


def _grouped_validation_test_split(
    temp_df: pd.DataFrame,
    id_col: Hashable,
    validation_size: float,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the temporary dataframe into validation and test sets by ID."""
    relative_test_size = test_size / (validation_size + test_size)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_test_size,
        random_state=random_state,
    )
    validation_idx, test_idx = next(splitter.split(temp_df, groups=temp_df[id_col]))
    return temp_df.iloc[validation_idx], temp_df.iloc[test_idx]


def _require_columns(df: pd.DataFrame, columns: list[Hashable]) -> None:
    """Raise a clear error if expected columns are absent."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")


def _validate_split_sizes(validation_size: float, test_size: float) -> None:
    """Validate split fractions before calling scikit-learn."""
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be greater than 0 and less than 1")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be less than 1")
