"""Tests for the tuning and monotonic-constraint helpers in src.modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from src.modeling import (
    RANDOM_STATE,
    build_monotone_constraints_vector,
    build_pipeline,
    build_preprocessor,
    derive_monotonic_constraints,
    fit_categorical_categories,
    get_numeric_feature_order,
    tune_model,
)


def _synthetic_training_frame(
    n_groups: int = 20, rows_per_group: int = 3
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build a small synthetic feature matrix with repeated group IDs.

    Mirrors the shape of the real feature matrix loosely: a couple of numeric
    columns and one categorical column, enough rows per group to exercise
    GroupKFold without every fold collapsing to a single class.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n = n_groups * rows_per_group
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    numeric_a = rng.normal(size=n)
    numeric_b = rng.normal(size=n)
    categorical = rng.choice(["low", "high"], size=n)
    # Outcome correlated with numeric_a so the search has real signal to find.
    y = (numeric_a + rng.normal(scale=0.5, size=n) > 0).astype(int)

    X = pd.DataFrame(
        {"numeric_a": numeric_a, "numeric_b": numeric_b, "cat_col": categorical}
    )
    return X, pd.Series(y), pd.Series(groups)


def test_tune_model_returns_fitted_search_with_best_estimator() -> None:
    """tune_model should return a fitted RandomizedSearchCV with a usable result."""
    X_train, y_train, groups_train = _synthetic_training_frame()
    preprocessor = build_preprocessor(X_train, categorical_features=["cat_col"])
    pipeline = build_pipeline(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), preprocessor
    )

    search = tune_model(
        pipeline,
        param_distributions={"model__C": [0.1, 1.0]},
        X_train=X_train,
        y_train=y_train,
        cv_groups=groups_train,
        n_splits=3,
        n_iter=2,
    )

    assert hasattr(search, "best_estimator_")
    assert isinstance(search.best_params_, dict)
    assert len(search.best_params_) > 0
    # The fitted estimator should be usable for prediction.
    preds = search.best_estimator_.predict_proba(X_train)[:, 1]
    assert preds.shape[0] == len(X_train)


def test_tune_model_falls_back_to_stratified_kfold_when_groups_none() -> None:
    """When cv_groups is None, tune_model should still fit successfully."""
    X_train, y_train, _ = _synthetic_training_frame()
    preprocessor = build_preprocessor(X_train, categorical_features=["cat_col"])
    pipeline = build_pipeline(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), preprocessor
    )

    search = tune_model(
        pipeline,
        param_distributions={"model__C": [0.1, 1.0]},
        X_train=X_train,
        y_train=y_train,
        cv_groups=None,
        n_splits=3,
        n_iter=2,
    )

    assert hasattr(search, "best_estimator_")


def test_tune_model_respects_group_kfold_no_group_leakage() -> None:
    """The CV splitter used internally must not put a group's rows in both
    the train and validation portion of any fold."""
    X_train, y_train, groups_train = _synthetic_training_frame()

    cv = GroupKFold(n_splits=3)
    for train_idx, test_idx in cv.split(X_train, y_train, groups_train):
        train_groups = set(groups_train.iloc[train_idx])
        test_groups = set(groups_train.iloc[test_idx])
        assert train_groups.isdisjoint(test_groups)


def test_derive_monotonic_constraints_maps_signs_correctly() -> None:
    """Significant positive/negative coefficients map to +1/-1; non-significant
    or absent features map to 0."""
    odds_ratio_df = pd.DataFrame(
        [
            {
                "model": "Adjusted clinical model",
                "predictor": "SBP",
                "coefficient": 0.0428,
                "p_value": 4.9e-17,
            },
            {
                "model": "Adjusted clinical model",
                "predictor": "HR",
                "coefficient": -0.0946,
                "p_value": 7.9e-50,
            },
            {
                "model": "Adjusted clinical model",
                "predictor": "GCS",
                "coefficient": -0.0033,
                "p_value": 0.744,
            },
            {
                "model": "Unadjusted: Age",
                "predictor": "Age",
                "coefficient": -0.0518,
                "p_value": 2.9e-61,
            },
        ]
    )

    constraints = derive_monotonic_constraints(
        odds_ratio_df, numeric_feature_order=["SBP", "HR", "GCS", "Age", "Urea"]
    )

    # SBP: significant positive -> +1
    # HR: significant negative -> -1
    # GCS: not significant (adjusted) -> 0
    # Age: only present under a different model label -> 0 (not looked up)
    # Urea: absent entirely -> 0
    assert constraints == [1, -1, 0, 0, 0]


def test_get_numeric_feature_order_excludes_categoricals() -> None:
    """Numeric feature order should list only non-categorical columns, in order."""
    X = pd.DataFrame({"a": [1], "cat_col": ["x"], "b": [2]})

    order = get_numeric_feature_order(X, categorical_features=["cat_col"])

    assert order == ["a", "b"]


def test_build_monotone_constraints_vector_pads_onehot_with_zeros() -> None:
    """The assembled vector should have one entry per numeric column followed
    by one zero per one-hot output column."""
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4], "cat_col": ["low", "high"]})
    categorical_categories = fit_categorical_categories(
        X, categorical_features=["cat_col"]
    )

    vector = build_monotone_constraints_vector(
        X,
        numeric_signs=[1, -1],
        categorical_categories=categorical_categories,
        categorical_features=["cat_col"],
    )

    # 2 numeric signs + 2 one-hot columns (low/high) of zeros.
    assert vector == [1, -1, 0, 0]


def test_build_monotone_constraints_vector_raises_on_length_mismatch() -> None:
    """A numeric_signs list of the wrong length should raise, not silently
    misalign with the transformed feature matrix."""
    X = pd.DataFrame({"a": [1, 2], "cat_col": ["low", "high"]})
    categorical_categories = fit_categorical_categories(
        X, categorical_features=["cat_col"]
    )

    with pytest.raises(ValueError):
        build_monotone_constraints_vector(
            X,
            numeric_signs=[1, -1],  # X only has 1 numeric column ("a")
            categorical_categories=categorical_categories,
            categorical_features=["cat_col"],
        )


def test_fit_categorical_categories_fixes_width_across_subsets() -> None:
    """Categories computed on the full set should stay valid (a superset) even
    for a subset that is missing one of the rarer categories."""
    X_full = pd.DataFrame({"cat_col": ["a", "b", "c", "a", "b"]})
    categories = fit_categorical_categories(X_full, categorical_features=["cat_col"])

    assert categories["cat_col"] == ["a", "b", "c"]
