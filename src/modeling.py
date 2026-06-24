"""Reusable modeling utilities for cardiac arrest risk prediction.

This module contains the leakage-safe modeling logic that was previously
implemented inline in ``notebooks/05_predictive_modeling.ipynb`` and
``notebooks/06_final_model_evaluation.ipynb``. Keeping it here makes the logic
importable, testable, and shared so the notebooks can stay thin wrappers.

All randomness flows from the single :data:`RANDOM_STATE` constant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import joblib

from src.features import add_clinical_features

# Single source of truth for reproducible randomness across the project.
RANDOM_STATE = 42

# Engineered/raw columns that should be treated as categorical by the
# preprocessor. Columns not present in a given matrix are ignored.
CATEGORICAL_FEATURES = [
    "Gender",
    "Alcoholic",
    "Smoke",
    "FHCD",
    "TriageScore",
    "age_band",
    "gcs_severity",
    "hypoxemia_flag",
    "sodium_abnormal_flag",
    "potassium_abnormal_flag",
    "chloride_abnormal_flag",
    "urea_abnormal_flag",
    "creatinine_abnormal_flag",
]

# Cross-validation scorers shared by model comparison.
_CV_SCORING = {
    "auroc": "roc_auc",
    "auprc": "average_precision",
    "f1": "f1",
    "sensitivity": "recall",
    "precision": "precision",
}


def make_leakage_safe_holdout(
    df: pd.DataFrame,
    target_col: str = "Outcome",
    id_col: str = "ID",
    test_size: float = 0.20,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, str]:
    """Create a single leakage-safe train/test holdout split.

    When patient identifiers repeat, the first fold of a
    :class:`~sklearn.model_selection.StratifiedGroupKFold` is used so every row
    for a patient stays entirely in train or test. Otherwise a stratified
    row-level split is used. This mirrors the split used in notebooks 05 and 06
    so the held-out test set is identical across both.

    Returns
    -------
    tuple
        ``(train_raw, test_raw, groups_train, split_strategy)`` where
        ``groups_train`` is the training-split patient id series.
    """
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split

    ids_repeat = bool((df[id_col].value_counts() > 1).any())
    X_raw = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    if ids_repeat:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        train_idx, test_idx = next(splitter.split(X_raw, y, groups=df[id_col]))
        split_strategy = "StratifiedGroupKFold by patient ID"
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )
        split_strategy = "Stratified train/test split"

    train_raw = df.iloc[train_idx].copy().reset_index(drop=True)
    test_raw = df.iloc[test_idx].copy().reset_index(drop=True)
    groups_train = train_raw[id_col]
    return train_raw, test_raw, groups_train, split_strategy


def build_feature_matrix(
    raw_features: pd.DataFrame,
    categorical_features: list[str] | None = None,
    id_col: str = "ID",
) -> pd.DataFrame:
    """Engineer clinical features and return a model-ready feature matrix.

    Applies :func:`src.features.add_clinical_features`, drops the patient
    identifier so it cannot be memorized, and casts categorical columns to
    ``object`` (with ``NaN`` preserved) so the preprocessor one-hot encodes
    them rather than scaling them numerically.
    """
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    featured = add_clinical_features(raw_features)
    if id_col in featured.columns:
        featured = featured.drop(columns=[id_col])

    for col in categorical_features:
        if col in featured.columns:
            featured[col] = (
                featured[col].astype("object").where(featured[col].notna(), np.nan)
            )
    return featured


def build_preprocessor(
    X: pd.DataFrame, categorical_features: list[str] | None = None
) -> ColumnTransformer:
    """Build the standard leakage-safe preprocessing ``ColumnTransformer``.

    Numeric columns are median-imputed and standardized; categorical columns
    are most-frequent-imputed and one-hot encoded. All steps are fit inside the
    pipeline, so they are refit independently on each training fold.
    """
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    categorical = [col for col in categorical_features if col in X.columns]
    numeric = [col for col in X.columns if col not in categorical]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        sparse_threshold=0.0,
    )


def build_pipeline(model: BaseEstimator, preprocessor: ColumnTransformer) -> Pipeline:
    """Wrap a scikit-learn estimator in a pipeline with the standard preprocessor."""
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_groups: pd.Series | np.ndarray | None,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Cross-validate ``pipeline`` with patient-level ``GroupKFold``.

    Grouping by patient id keeps all rows for a patient within the same fold,
    preventing leakage between train and validation folds. If ``cv_groups`` is
    ``None``, plain K-fold (no grouping) is used.

    Returns
    -------
    dict
        ``{"n_splits", "cv", "metrics"}`` where ``metrics`` maps each scorer to
        ``{"mean", "std", "scores"}``.
    """
    if cv_groups is not None:
        cv = GroupKFold(n_splits=n_splits)
        cv_name = "GroupKFold"
        groups = np.asarray(cv_groups)
    else:
        from sklearn.model_selection import StratifiedKFold

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        cv_name = "StratifiedKFold"
        groups = None

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        groups=groups,
        cv=cv,
        scoring=_CV_SCORING,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )

    metrics = {}
    for name in _CV_SCORING:
        scores = cv_results[f"test_{name}"]
        metrics[name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "scores": scores.tolist(),
        }
    return {"n_splits": n_splits, "cv": cv_name, "metrics": metrics}


def save_model(pipeline: Pipeline, path: str | Path) -> Path:
    """Persist a fitted pipeline with joblib, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(path: str | Path) -> Pipeline:
    """Load a pipeline saved with :func:`save_model`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No saved model at: {path}")
    return joblib.load(path)
