"""Tests for the prediction pipeline (source/prediction.py)."""

import pandas as pd
import pytest
from prediction import _lookup_cumulative, get_features


# ---------------------------------------------------------------------------
# _lookup_cumulative (mirrors train_model version)
# ---------------------------------------------------------------------------
class TestLookupCumulative:
    def test_normal_lookup(self):
        cumsum = {"Action": {2012: 1.7, 2014: 2.3}}
        assert _lookup_cumulative(cumsum, "Action", 2014) == 2.3

    def test_category_not_found(self):
        assert _lookup_cumulative({}, "RPG", 2014) == 0.0

    def test_year_before_data(self):
        cumsum = {"Action": {2012: 1.7}}
        assert _lookup_cumulative(cumsum, "Action", 2010) == 0.0


# ---------------------------------------------------------------------------
# get_features
# ---------------------------------------------------------------------------
class TestGetFeatures:
    def test_returns_one_row_dataframe(self, train_stats_fixture):
        input_data = {"Year": 2015, "meta_score": 85.0, "user_review": 8.5}
        result = get_features(input_data, train_stats_fixture, "Action", "PS4")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_has_all_engineered_columns(self, train_stats_fixture):
        input_data = {"Year": 2015, "meta_score": 85.0, "user_review": 8.5}
        result = get_features(input_data, train_stats_fixture, "Action", "PS4")
        expected = [
            "Year",
            "meta_score",
            "user_review",
            "Global_Sales_mean_genre",
            "Global_Sales_mean_platform",
            "Year_Global_Sales_mean_genre",
            "Year_Global_Sales_mean_platform",
            "Cumulative_Sales_Genre",
            "Cumulative_Sales_Platform",
        ]
        for col in expected:
            assert col in result.columns

    def test_genre_mean_from_stats(self, train_stats_fixture):
        input_data = {"Year": 2015, "meta_score": 85.0, "user_review": 8.5}
        result = get_features(input_data, train_stats_fixture, "Action", "PS4")
        assert result["Global_Sales_mean_genre"].iloc[0] == pytest.approx(
            train_stats_fixture["genre_means"]["Action"]
        )

    def test_unknown_genre_uses_global_mean(self, train_stats_fixture):
        input_data = {"Year": 2015, "meta_score": 85.0, "user_review": 8.5}
        result = get_features(input_data, train_stats_fixture, "UnknownGenre", "PS4")
        assert result["Global_Sales_mean_genre"].iloc[0] == pytest.approx(
            train_stats_fixture["global_sales_mean"]
        )

    def test_interaction_feature_correct(self, train_stats_fixture):
        input_data = {"Year": 2015, "meta_score": 85.0, "user_review": 8.5}
        result = get_features(input_data, train_stats_fixture, "Action", "PS4")
        expected = 2015 * train_stats_fixture["genre_means"]["Action"]
        assert result["Year_Global_Sales_mean_genre"].iloc[0] == pytest.approx(expected)

    def test_cumulative_sales_lookup(self, train_stats_fixture):
        input_data = {"Year": 2014, "meta_score": 85.0, "user_review": 8.5}
        result = get_features(input_data, train_stats_fixture, "Action", "PS4")
        # Action cumsum at 2014 = 2.3
        assert result["Cumulative_Sales_Genre"].iloc[0] == pytest.approx(2.3)
