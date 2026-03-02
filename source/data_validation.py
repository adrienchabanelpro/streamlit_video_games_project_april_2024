"""Data validation with pandera for the video game sales dataset.

Defines a pandera DataFrameSchema for ``Ventes_jeux_video_final.csv`` and
exposes a lightweight ``validate_dataframe`` helper that returns advisory
validation results without raising exceptions.
"""

from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

SALES_SCHEMA: DataFrameSchema = DataFrameSchema(
    columns={
        "Name": Column(pa.String, nullable=True, required=True),
        "Platform": Column(pa.String, nullable=False, required=True),
        "Year": Column(
            pa.Float,
            checks=[
                Check.greater_than_or_equal_to(1970),
                Check.less_than_or_equal_to(2030),
            ],
            nullable=True,
            required=True,
        ),
        "Genre": Column(pa.String, nullable=False, required=True),
        "Publisher": Column(pa.String, nullable=True, required=True),
        "NA_Sales": Column(
            pa.Float,
            checks=[Check.greater_than_or_equal_to(0)],
            nullable=False,
            required=True,
        ),
        "EU_Sales": Column(
            pa.Float,
            checks=[Check.greater_than_or_equal_to(0)],
            nullable=False,
            required=True,
        ),
        "JP_Sales": Column(
            pa.Float,
            checks=[Check.greater_than_or_equal_to(0)],
            nullable=False,
            required=True,
        ),
        "Other_Sales": Column(
            pa.Float,
            checks=[Check.greater_than_or_equal_to(0)],
            nullable=False,
            required=True,
        ),
        "Global_Sales": Column(
            pa.Float,
            checks=[Check.greater_than_or_equal_to(0)],
            nullable=False,
            required=True,
        ),
        "meta_score": Column(
            pa.Float,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(10),
            ],
            nullable=True,
            required=True,
        ),
        "user_review": Column(
            pa.Float,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(10),
            ],
            nullable=True,
            required=True,
        ),
        "shipped": Column(
            pa.Float,
            checks=[Check.greater_than_or_equal_to(0)],
            nullable=True,
            required=False,
        ),
    },
    # Allow extra columns (e.g. Rank) that are not part of the core schema
    strict=False,
    coerce=True,
)


def validate_dataframe(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate *df* against the sales dataset schema.

    This function is **advisory**: it never raises and always returns a
    result so the caller can decide how to handle validation failures.

    Parameters
    ----------
    df:
        The DataFrame to validate.

    Returns
    -------
    tuple[bool, list[str]]
        ``(is_valid, error_messages)`` where *is_valid* is ``True`` when
        validation passes and *error_messages* is an empty list, or
        ``False`` with a list of human-readable error descriptions.
    """
    try:
        SALES_SCHEMA.validate(df, lazy=True)
        return True, []
    except pa.errors.SchemaErrors as exc:
        messages: list[str] = []
        for _, row in exc.failure_cases.iterrows():
            col = row.get("column", "?")
            check = row.get("check", "?")
            idx = row.get("index", "?")
            messages.append(f"Colonne '{col}' — verification '{check}' echouee (ligne {idx})")
        return False, messages
