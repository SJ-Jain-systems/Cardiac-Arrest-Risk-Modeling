"""Inference layer for the cardiac arrest risk model.

Loads the serialized pipeline saved by ``notebooks/05_predictive_modeling.ipynb``
(``models/best_model.joblib``) and scores new patient records. Inference routes
through :func:`src.modeling.build_feature_matrix` — the *same* function used at
training time — so the engineered clinical features, dropped ``ID`` column, and
categorical casting are guaranteed identical between train and serve. Scoring any
other way risks silent train/serve skew.

Programmatic use
----------------
>>> from src.predict import predict_record
>>> predict_record({"SBP": 120, "DBP": 80, "HR": 88, "RR": 18, "BT": 98,
...                  "SpO2": 97, "Age": 66, "Gender": 1, "GCS": 15, "Na": 139,
...                  "K": 4.0, "Cl": 105, "Urea": 41, "Ceratinine": 91,
...                  "Alcoholic": 1, "Smoke": 1, "FHCD": 0, "TriageScore": 3})
{'risk_score': 0.83..., 'threshold': 0.45, 'high_risk': True}

Command-line use
----------------
    # Single record from a JSON file (or '-' to read JSON from stdin)
    python -m src.predict --input patient.json

    # Batch scoring: a CSV in, a CSV of scores out
    python -m src.predict --csv new_patients.csv --out scored.csv

    # Override the model artifact or operating threshold
    python -m src.predict --input patient.json --threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.config import MODELS_DIR, REPORTS_DIR
from src.modeling import build_feature_matrix, load_model

# The primary artifact produced by notebook 05. The ``.joblib`` binary is
# git-ignored and regenerated, so this path may not exist on a fresh checkout.
DEFAULT_MODEL_PATH = MODELS_DIR / "best_model.joblib"

# Fallback operating threshold. The project's recommended threshold is recorded
# in ``reports/final_model_metrics.csv`` and read from there when available; this
# constant matches the README's documented default and is used only if that file
# is missing or unreadable.
DEFAULT_THRESHOLD = 0.45

_METRICS_PATH = REPORTS_DIR / "final_model_metrics.csv"

# Raw predictor columns required to engineer features. ``ID`` is optional (it is
# dropped during feature building); ``Outcome`` is a label and never an input.
REQUIRED_INPUT_COLUMNS = [
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
]


def load_operating_threshold(metrics_path: str | Path = _METRICS_PATH) -> float:
    """Return the recommended operating threshold.

    Reads the ``threshold`` column from ``reports/final_model_metrics.csv`` when
    present so inference always uses the same threshold the evaluation notebook
    selected. Falls back to :data:`DEFAULT_THRESHOLD` if the file is missing,
    empty, or lacks the column.
    """
    path = Path(metrics_path)
    if not path.exists():
        return DEFAULT_THRESHOLD
    try:
        metrics = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return DEFAULT_THRESHOLD
    if "threshold" not in metrics.columns or metrics.empty:
        return DEFAULT_THRESHOLD
    return float(metrics["threshold"].iloc[0])


def _require_input_columns(df: pd.DataFrame) -> None:
    """Raise a clear error if any raw predictor column is absent."""
    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required input column(s): {missing}")


def predict_dataframe(
    raw: pd.DataFrame,
    model=None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Score a dataframe of raw patient records.

    Parameters
    ----------
    raw:
        One row per record, containing at least :data:`REQUIRED_INPUT_COLUMNS`.
        An ``ID`` column, if present, is carried through to the output but not
        used as a predictor. An ``Outcome`` column, if present, is ignored.
    model:
        A pre-loaded pipeline. When ``None``, the artifact at ``model_path`` is
        loaded. Pass a model to score many batches without reloading.
    model_path:
        Path to the ``joblib`` artifact, used only when ``model`` is ``None``.
    threshold:
        Operating threshold for the ``high_risk`` decision. When ``None``, the
        recommended threshold is read via :func:`load_operating_threshold`.

    Returns
    -------
    pandas.DataFrame
        The input index plus ``risk_score`` (calibrated probability of the
        positive outcome), ``threshold``, and ``high_risk`` (bool). An ``ID``
        column is preserved as the first column when it was supplied.
    """
    _require_input_columns(raw)
    if model is None:
        model = load_model(model_path)
    if threshold is None:
        threshold = load_operating_threshold()

    # build_feature_matrix drops ID and Outcome-adjacent handling itself, but we
    # drop Outcome defensively so a labelled batch scores identically to an
    # unlabelled one.
    features_input = raw.drop(columns=["Outcome"], errors="ignore")
    matrix = build_feature_matrix(features_input)

    risk_score = model.predict_proba(matrix)[:, 1]
    result = pd.DataFrame(index=raw.index)
    if "ID" in raw.columns:
        result["ID"] = raw["ID"].to_numpy()
    result["risk_score"] = risk_score
    result["threshold"] = threshold
    result["high_risk"] = result["risk_score"] >= threshold
    return result


def predict_record(
    record: dict,
    model=None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    threshold: float | None = None,
) -> dict:
    """Score a single patient record given as a mapping of column -> value.

    Returns a dict with ``risk_score``, ``threshold``, and ``high_risk``. Any
    ``ID`` in the record is echoed back under ``id``.
    """
    scored = predict_dataframe(
        pd.DataFrame([record]),
        model=model,
        model_path=model_path,
        threshold=threshold,
    )
    row = scored.iloc[0]
    output = {
        "risk_score": float(row["risk_score"]),
        "threshold": float(row["threshold"]),
        "high_risk": bool(row["high_risk"]),
    }
    if "ID" in scored.columns:
        output["id"] = _json_safe(row["ID"])
    return output


def _json_safe(value):
    """Coerce a numpy scalar to a plain Python type for JSON serialization."""
    if hasattr(value, "item"):
        return value.item()
    return value


def _read_record(source: str) -> dict:
    """Read a single JSON record from a file path or '-' for stdin."""
    text = sys.stdin.read() if source == "-" else Path(source).read_text()
    record = json.loads(text)
    if not isinstance(record, dict):
        raise TypeError("--input JSON must be a single object of column: value")
    return record


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.predict",
        description="Score patient records with the saved cardiac risk model.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input",
        metavar="PATH",
        help="JSON file with one record (a column: value object); '-' for stdin.",
    )
    group.add_argument(
        "--csv",
        metavar="PATH",
        help="CSV file of records to score in batch (one row per record).",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        help="Write batch scores to this CSV (default: stdout). Ignored for --input.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help=f"Path to the joblib model artifact (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the high-risk decision threshold (default: recommended).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _build_parser().parse_args(argv)

    try:
        model = load_model(args.model)
    except FileNotFoundError as exc:
        print(
            f"Error: {exc}\nRun notebooks/05_predictive_modeling.ipynb to "
            "regenerate models/best_model.joblib.",
            file=sys.stderr,
        )
        return 1

    try:
        if args.input is not None:
            record = _read_record(args.input)
            result = predict_record(record, model=model, threshold=args.threshold)
            print(json.dumps(result, indent=2))
        else:
            raw = pd.read_csv(args.csv)
            scored = predict_dataframe(raw, model=model, threshold=args.threshold)
            if args.out:
                scored.to_csv(args.out, index=False)
                print(f"Wrote {len(scored)} scored rows to {args.out}")
            else:
                scored.to_csv(sys.stdout, index=False)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
