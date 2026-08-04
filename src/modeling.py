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

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

# Cross-validation scorers shared by model comparison. Public so notebooks and
# tune_model callers can request the same multi-metric scoring.
CV_SCORING = {
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


def get_numeric_feature_order(
    X: pd.DataFrame, categorical_features: list[str] | None = None
) -> list[str]:
    """Return the numeric column order that :func:`build_preprocessor` will use.

    Needed to align a monotonic-constraint vector (one entry per numeric
    column, in this order) with LightGBM's ``monotone_constraints`` input,
    which is a plain list keyed by transformed-feature position. See
    :func:`build_monotone_constraints_vector`.
    """
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES
    categorical = [col for col in categorical_features if col in X.columns]
    return [col for col in X.columns if col not in categorical]


def fit_categorical_categories(
    X: pd.DataFrame, categorical_features: list[str] | None = None
) -> dict[str, list[Any]]:
    """Compute fixed category levels for each categorical column from ``X``.

    Passing the result to :func:`build_preprocessor` keeps the one-hot output
    width identical across every cross-validation fold (rather than each
    fold's ``OneHotEncoder`` independently discovering whichever categories
    happen to appear in that fold's training partition). A fixed width is
    required so a fixed-length ``monotone_constraints`` vector (see
    :func:`build_monotone_constraints_vector`) stays valid regardless of which
    fold LightGBM is fit on.
    """
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES
    categories: dict[str, list[Any]] = {}
    for col in categorical_features:
        if col in X.columns:
            values = X[col].dropna().unique().tolist()
            categories[col] = sorted(values, key=str)
    return categories


def build_monotone_constraints_vector(
    X: pd.DataFrame,
    numeric_signs: list[int],
    categorical_categories: dict[str, list[Any]],
    categorical_features: list[str] | None = None,
) -> list[int]:
    """Assemble a full transformed-feature-width monotonic-constraint vector.

    ``numeric_signs`` must already be ordered to match
    :func:`get_numeric_feature_order`. Every one-hot categorical output column
    gets constraint ``0`` (dummy columns have no meaningful ordering).
    ``categorical_categories`` must be the same mapping passed to
    :func:`build_preprocessor` so the one-hot width computed here matches what
    the fitted preprocessor will actually produce.
    """
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES
    categorical = [col for col in categorical_features if col in X.columns]
    numeric = [col for col in X.columns if col not in categorical]
    if len(numeric_signs) != len(numeric):
        raise ValueError(
            f"numeric_signs has {len(numeric_signs)} entries but X has "
            f"{len(numeric)} numeric columns after excluding categoricals."
        )
    n_onehot = sum(len(categorical_categories[col]) for col in categorical)
    return list(numeric_signs) + [0] * n_onehot


def derive_monotonic_constraints(
    odds_ratio_df: pd.DataFrame,
    numeric_feature_order: list[str],
    model_label: str = "Adjusted clinical model",
    alpha: float = 0.05,
) -> list[int]:
    """Map adjusted odds-ratio evidence to monotonic-constraint signs.

    For each feature in ``numeric_feature_order``, looks up its row (if any)
    in the ``model_label`` subset of ``odds_ratio_df`` (expects columns
    ``model``, ``predictor``, ``coefficient``, ``p_value``, matching
    ``reports/Report CSV's/odds_ratio_results.csv``). A statistically
    significant (``p_value < alpha``) positive coefficient maps to ``+1``, a
    significant negative coefficient maps to ``-1``. Any feature that is
    either absent from ``model_label`` or not statistically significant maps
    to ``0`` (no constraint) rather than guessing a clinical direction for it.
    """
    subset = odds_ratio_df[odds_ratio_df["model"] == model_label]
    lookup = subset.set_index("predictor")

    constraints: list[int] = []
    for feature in numeric_feature_order:
        if feature not in lookup.index:
            constraints.append(0)
            continue
        row = lookup.loc[feature]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if row["p_value"] >= alpha:
            constraints.append(0)
            continue
        constraints.append(1 if row["coefficient"] > 0 else -1)
    return constraints


def build_preprocessor(
    X: pd.DataFrame,
    categorical_features: list[str] | None = None,
    categorical_categories: dict[str, list[Any]] | None = None,
) -> ColumnTransformer:
    """Build the standard leakage-safe preprocessing ``ColumnTransformer``.

    Numeric columns are median-imputed and standardized; categorical columns
    are most-frequent-imputed and one-hot encoded. All steps are fit inside the
    pipeline, so they are refit independently on each training fold.

    ``categorical_categories`` is optional; pass the output of
    :func:`fit_categorical_categories` (computed once on the full training
    set) to fix the one-hot category levels across every cross-validation
    fold. This is required whenever a fixed-length ``monotone_constraints``
    vector is used downstream (see :func:`build_monotone_constraints_vector`),
    since otherwise each fold's ``OneHotEncoder`` could independently discover
    a different number of categories and produce a different transformed
    width. Omit it (the default) to preserve the original per-fit behavior.
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
    if categorical_categories is not None:
        categories = [categorical_categories[col] for col in categorical]
        onehot = OneHotEncoder(
            categories=categories, handle_unknown="ignore", sparse_output=False
        )
    else:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", onehot),
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


def _build_cv(
    cv_groups: pd.Series | np.ndarray | None,
    n_splits: int,
    random_state: int = RANDOM_STATE,
) -> tuple[Any, np.ndarray | None, str]:
    """Build the leakage-safe CV splitter shared by :func:`train_model` and
    :func:`tune_model`.

    Returns ``(cv, groups, cv_name)``. When ``cv_groups`` is given, uses
    patient-level :class:`~sklearn.model_selection.GroupKFold`; otherwise falls
    back to :class:`~sklearn.model_selection.StratifiedKFold`.
    """
    if cv_groups is not None:
        cv = GroupKFold(n_splits=n_splits)
        cv_name = "GroupKFold"
        groups = np.asarray(cv_groups)
    else:
        from sklearn.model_selection import StratifiedKFold

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_name = "StratifiedKFold"
        groups = None
    return cv, groups, cv_name


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
    cv, groups, cv_name = _build_cv(cv_groups, n_splits)

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        groups=groups,
        cv=cv,
        scoring=CV_SCORING,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )

    metrics = {}
    for name in CV_SCORING:
        scores = cv_results[f"test_{name}"]
        metrics[name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "scores": scores.tolist(),
        }
    return {"n_splits": n_splits, "cv": cv_name, "metrics": metrics}


def tune_model(
    pipeline: Pipeline,
    param_distributions: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_groups: pd.Series | np.ndarray | None,
    n_splits: int = 5,
    n_iter: int = 25,
    scoring: str | dict[str, str] = "roc_auc",
    refit: str | bool | None = None,
    random_state: int = RANDOM_STATE,
) -> RandomizedSearchCV:
    """Hyperparameter-search ``pipeline`` with the same leakage-safe CV as
    :func:`train_model`.

    Uses patient-level ``GroupKFold`` when ``cv_groups`` is given (falls back
    to ``StratifiedKFold`` otherwise), exactly mirroring ``train_model``'s
    splitting behavior so tuning never leaks a patient's rows across the
    search's internal train/validation folds.

    ``scoring`` may be a single scorer name (the default) or a multi-metric
    dict such as :data:`CV_SCORING`, so the search's ``cv_results_`` can report
    AUPRC/F1/sensitivity for the winning configuration alongside AUROC without
    a second cross-validation pass. When ``scoring`` is a dict and ``refit`` is
    not given, it defaults to ``"auroc"`` if present in ``scoring``, otherwise
    ``True`` is required from the caller.

    Returns
    -------
    RandomizedSearchCV
        The fitted search object; use ``.best_estimator_``, ``.best_params_``,
        and ``.cv_results_`` to read the result.
    """
    cv, groups, _ = _build_cv(cv_groups, n_splits, random_state)
    if refit is None:
        refit = "auroc" if isinstance(scoring, dict) and "auroc" in scoring else True
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        refit=refit,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        error_score="raise",
    )
    search.fit(X_train, y_train, groups=groups)
    return search


def get_candidate_model_specs(
    preprocessor: ColumnTransformer,
    monotone_constraints: list[int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return the shared candidate-model and search-grid definitions.

    Used by both ``notebooks/05_predictive_modeling.ipynb`` (primary training
    path) and ``notebooks/07_model_interpretability.ipynb`` (fallback
    retraining path used only when ``models/best_model.joblib`` is missing) so
    the two notebooks cannot silently drift apart on which candidate models or
    search grids are evaluated.

    ``monotone_constraints`` (if given) is passed to the LightGBM candidate
    only; build it with :func:`derive_monotonic_constraints` and
    :func:`build_monotone_constraints_vector`.

    Returns
    -------
    dict
        Maps each candidate name to
        ``{"pipeline", "param_distributions", "n_iter"}``, ready to pass to
        :func:`tune_model`.
    """
    from lightgbm import LGBMClassifier
    from scipy.stats import loguniform
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    # Inner estimators use n_jobs=1 deliberately: RandomizedSearchCV already
    # parallelizes across (hyperparameter, CV fold) combinations with its own
    # n_jobs=-1 (see tune_model). Letting RandomForestClassifier/LGBMClassifier
    # *also* spawn n_jobs=-1 threads inside each of those parallel workers
    # oversubscribes the machine's cores many times over and makes tuning far
    # slower in wall-clock terms, not faster.
    specs: dict[str, dict[str, Any]] = {
        "Logistic Regression": {
            "pipeline": build_pipeline(
                LogisticRegression(
                    solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE
                ),
                preprocessor,
            ),
            "param_distributions": {"model__C": loguniform(1e-2, 1e2)},
            "n_iter": 10,
        },
        "Regularized Logistic Regression": {
            "pipeline": build_pipeline(
                LogisticRegression(
                    solver="saga",
                    penalty="elasticnet",
                    l1_ratio=0.5,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                preprocessor,
            ),
            "param_distributions": {"model__C": loguniform(1e-2, 1e2)},
            "n_iter": 5,
        },
        "Random Forest": {
            "pipeline": build_pipeline(
                RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
                preprocessor,
            ),
            "param_distributions": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 4, 8, 12],
                "model__min_samples_leaf": [1, 2, 5, 10],
                "model__max_features": ["sqrt", "log2", None],
            },
            "n_iter": 10,
        },
        "Gradient Boosting": {
            "pipeline": build_pipeline(
                GradientBoostingClassifier(random_state=RANDOM_STATE), preprocessor
            ),
            "param_distributions": {
                "model__n_estimators": [100, 150, 200],
                "model__learning_rate": loguniform(0.01, 0.3),
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.7, 0.85, 1.0],
            },
            "n_iter": 10,
        },
        "LightGBM (monotonic)": {
            "pipeline": build_pipeline(
                LGBMClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    verbose=-1,
                    monotone_constraints=monotone_constraints,
                ),
                preprocessor,
            ),
            "param_distributions": {
                "model__num_leaves": [15, 31, 63],
                "model__learning_rate": loguniform(0.01, 0.2),
                "model__n_estimators": [100, 200, 400],
                "model__min_child_samples": [5, 10, 20],
            },
            "n_iter": 12,
        },
    }
    return specs


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
