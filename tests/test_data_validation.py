"""Tests for the data validation module (source/data_validation.py)."""

import numpy as np
import pandas as pd
from data_validation import SALES_SCHEMA, validate_dataframe


def _make_valid_df() -> pd.DataFrame:
    """Create a minimal valid DataFrame matching the schema."""
    return pd.DataFrame(
        {
            "Name": ["Game A", "Game B"],
            "Platform": ["PS4", "PC"],
            "Year": [2015.0, 2020.0],
            "Genre": ["Action", "RPG"],
            "Publisher": ["EA", "Ubisoft"],
            "NA_Sales": [1.0, 0.5],
            "EU_Sales": [0.5, 0.3],
            "JP_Sales": [0.1, 0.2],
            "Other_Sales": [0.1, 0.1],
            "Global_Sales": [1.7, 1.1],
            "meta_score": [85.0, 90.0],
            "user_review": [8.5, 9.0],
        }
    )


class TestValidateDataframe:
    def test_valid_df_passes(self):
        df = _make_valid_df()
        is_valid, errors = validate_dataframe(df)
        assert is_valid is True
        assert errors == []

    def test_missing_required_column_fails(self):
        df = _make_valid_df().drop(columns=["Genre"])
        is_valid, errors = validate_dataframe(df)
        assert is_valid is False
        assert len(errors) > 0

    def test_negative_sales_fails(self):
        df = _make_valid_df()
        df.loc[0, "Global_Sales"] = -1.0
        is_valid, errors = validate_dataframe(df)
        assert is_valid is False

    def test_meta_score_out_of_range_fails(self):
        df = _make_valid_df()
        df.loc[0, "meta_score"] = 150.0
        is_valid, errors = validate_dataframe(df)
        assert is_valid is False

    def test_year_out_of_range_fails(self):
        df = _make_valid_df()
        df.loc[0, "Year"] = 1800.0
        is_valid, errors = validate_dataframe(df)
        assert is_valid is False

    def test_nullable_year_passes(self):
        df = _make_valid_df()
        df.loc[0, "Year"] = np.nan
        is_valid, errors = validate_dataframe(df)
        assert is_valid is True

    def test_nullable_name_passes(self):
        df = _make_valid_df()
        df.loc[0, "Name"] = None
        is_valid, errors = validate_dataframe(df)
        assert is_valid is True

    def test_extra_columns_allowed(self):
        df = _make_valid_df()
        df["Rank"] = [1, 2]
        is_valid, errors = validate_dataframe(df)
        assert is_valid is True

    def test_returns_human_readable_errors(self):
        df = _make_valid_df()
        df.loc[0, "Global_Sales"] = -5.0
        _, errors = validate_dataframe(df)
        assert any("Global_Sales" in e for e in errors)


class TestSalesSchema:
    def test_schema_is_not_strict(self):
        assert SALES_SCHEMA.strict is False

    def test_schema_coerces_types(self):
        assert SALES_SCHEMA.coerce is True

    def test_required_columns_count(self):
        assert len(SALES_SCHEMA.columns) == 12
