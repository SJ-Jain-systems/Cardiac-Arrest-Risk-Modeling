"""Schema validation for the raw cardiac patient dataset.

Defines a declarative contract for ``data/CardiacPatientData.csv`` using
`pandera <https://pandera.readthedocs.io/>`_ so malformed data fails loudly at
load time — with a precise, column-level error — instead of surfacing as a
confusing traceback deep inside a notebook or as a silently wrong model.

The schema encodes the column set, dtypes, allowed categorical codes, and broad
clinical plausibility ranges. Ranges are deliberately wide screening bounds (not
diagnostic cutoffs); their job is to catch data errors such as unit mistakes,
swapped columns, or impossible values, not to judge clinical normality.
Laboratory and vital columns are ``nullable`` because genuine missingness is
expected and is handled downstream by leakage-safe imputation.

Usage
-----
>>> from src.validation import validate_raw_dataframe
>>> from src.preprocessing import load_raw_dataset
>>> df = validate_raw_dataframe(load_raw_dataset())   # raises on contract breach

To validate without an ``Outcome`` label (e.g. incoming records for inference):

>>> validate_raw_dataframe(new_records, require_outcome=False)

Requires ``pandera`` (add ``pandera==0.22.1`` to the project dependencies).
"""

from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

# Binary 0/1 indicator columns.
_BINARY_CODES = [0, 1]


def _binary_column(nullable: bool = False) -> Column:
    """A 0/1 coded integer indicator column."""
    return Column(
        int,
        checks=Check.isin(_BINARY_CODES),
        nullable=nullable,
        coerce=True,
    )


def _ranged_column(
    low: float,
    high: float,
    *,
    nullable: bool = True,
    as_int: bool = False,
) -> Column:
    """A numeric column bounded to a plausible clinical range."""
    return Column(
        int if as_int else float,
        checks=Check.in_range(low, high),
        nullable=nullable,
        coerce=True,
    )


# Column-level contract. Ranges are broad plausibility bounds; laboratory and
# vital columns are nullable because real missingness is expected. ``BT`` bounds
# are intentionally wide (30-115) so the schema accepts either Celsius or the
# Fahrenheit-scale values present in the source file without false failures.
_RAW_COLUMNS = {
    "ID": Column(int, checks=Check.ge(0), coerce=True),
    "SBP": _ranged_column(0, 300),
    "DBP": _ranged_column(0, 220),
    "HR": _ranged_column(0, 350),
    "RR": _ranged_column(0, 90),
    "BT": _ranged_column(30, 115),
    "SpO2": _ranged_column(0, 100),
    "Age": _ranged_column(0, 120),
    "Gender": _binary_column(),
    "GCS": _ranged_column(3, 15, as_int=True, nullable=True),
    "Na": _ranged_column(80, 200),
    "K": _ranged_column(1, 12),
    "Cl": _ranged_column(60, 160),
    "Urea": _ranged_column(0, 400),
    # Raw column keeps the source misspelling "Ceratinine" (see README).
    "Ceratinine": _ranged_column(0, 3000),
    "Alcoholic": _binary_column(nullable=True),
    "Smoke": _binary_column(nullable=True),
    "FHCD": _binary_column(nullable=True),
    "TriageScore": _ranged_column(1, 5, as_int=True, nullable=True),
}

# Outcome is required when validating labelled training data. It is declared
# separately so inference inputs can be validated with require_outcome=False.
_OUTCOME_COLUMN = {"Outcome": _binary_column()}

#: Schema for the labelled raw training CSV (Outcome required).
RAW_SCHEMA = DataFrameSchema(
    {**_RAW_COLUMNS, **_OUTCOME_COLUMN},
    strict=False,  # tolerate extra columns; enforce the ones we know
    coerce=True,
    name="CardiacPatientData",
)

#: Schema for unlabelled inference inputs (Outcome absent or ignored).
INFERENCE_SCHEMA = DataFrameSchema(
    _RAW_COLUMNS,
    strict=False,
    coerce=True,
    name="CardiacPatientInference",
)


def validate_raw_dataframe(
    df: pd.DataFrame,
    require_outcome: bool = True,
    lazy: bool = True,
) -> pd.DataFrame:
    """Validate a raw cardiac patient dataframe against the schema.

    Parameters
    ----------
    df:
        Dataframe to validate.
    require_outcome:
        When ``True`` (default), the ``Outcome`` label must be present and 0/1.
        Set ``False`` to validate records intended for inference.
    lazy:
        When ``True`` (default), collect *all* schema violations and report them
        together via :class:`pandera.errors.SchemaErrors`, rather than raising on
        the first failure. This makes data-quality triage far faster.

    Returns
    -------
    pandas.DataFrame
        The validated (and dtype-coerced) dataframe.

    Raises
    ------
    pandera.errors.SchemaError or pandera.errors.SchemaErrors
        If the dataframe violates the contract.
    """
    schema = RAW_SCHEMA if require_outcome else INFERENCE_SCHEMA
    return schema.validate(df, lazy=lazy)


__all__ = [
    "RAW_SCHEMA",
    "INFERENCE_SCHEMA",
    "validate_raw_dataframe",
    "pa",
]
