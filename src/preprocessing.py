from __future__ import annotations

import logging
from collections.abc import Hashable
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.config import RAW_DATA_PATH

logger = logging.getLogger(__name__)


def load_raw_dataset(
    path: str | Path = RAW_DATA_PATH,
    *,
    validate: str = "warn",
    require_outcome: bool = True,
) -> pd.DataFrame:
    """Load the raw cardiac patient dataset, checking it against the schema.

    Parameters
    ----------
    path:
        CSV file to load. Defaults to ``data/CardiacPatientData.csv`` from the
        project configuration. The raw file is read only; this function does not
        clean, transform, or persist derived data.
    validate:
        Schema-validation behaviour via :func:`src.validation.validate_raw_dataframe`:

        - ``"warn"`` (default): log any contract violations and return the data
          unchanged, so an already-known-dirty file still loads.
        - ``"raise"``: fail loudly on the first violation. Use this once the raw
          data is expected to be clean.
        - ``"skip"``: do not validate at all.
    require_outcome:
        Passed through to the validator. Set ``False`` for inference inputs that
        have no ``Outcome`` label.

    Returns
    -------
    pandas.DataFrame
        Raw clinical observations. On successful validation the returned frame is
        dtype-coerced to the schema; in ``"warn"`` mode after a failure, or in
        ``"skip"`` mode, it is returned exactly as read from the CSV.
    """
    df = pd.read_csv(path)
    if validate == "skip":
        return df

    # Imported here so the pandera dependency is only needed when validating.
    import pandera.errors as pa_errors

    from src.validation import validate_raw_dataframe

    try:
        return validate_raw_dataframe(df, require_outcome=require_outcome)
    except (pa_errors.SchemaError, pa_errors.SchemaErrors) as exc:
        if validate == "raise":
            raise
        logger.warning("Raw data failed schema validation:\n%s", exc)
        return df
