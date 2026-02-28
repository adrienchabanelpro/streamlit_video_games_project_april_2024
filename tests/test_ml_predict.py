"""Tests for the ml.predict module (source/ml/predict.py)."""

import pandas as pd
import pytest
from ml.predict import NUMERICAL_FEATURES, get_features, lookup_cumulative


class TestNumericalFeatures:
    def test_has_10_features(self):
        assert len(NUMERICAL_FEATURES) == 10

    def test_contains_expected_features(self):
        expected = {
            "Year",
            "meta_score",
            "user_review",
            "Global_Sales_mean_genre",
            "Global_Sales_mean_platform",
            "Year_Global_Sales_mean_genre",
            "Year_Global_Sales_mean_platform",
            "Cumulative_Sales_Genre",
            "Cumulative_Sales_Platform",
            "Publisher_encoded",
        }
        assert set(NUMERICAL_FEATURES) == expected


class TestLookupCumulative:
    def test_normal_lookup(self):
        cumsum = {"Action": {2012: 1.7, 2014: 2.3}}
        assert lookup_cumulative(cumsum, "Action", 2014) == 2.3

    def test_category_not_found(self):
        assert lookup_cumulative({}, "RPG", 2014) == 0.0

    def test_year_before_all_data(self):
        cumsum = {"Action": {2012: 1.7}}
        assert lookup_cumulative(cumsum, "Action", 2010) == 0.0

    def test_year_between_entries(self):
        cumsum = {"Action": {2012: 1.7, 2014: 2.3}}
        assert lookup_cumulative(cumsum, "Action", 2013) == 1.7

    def test_year_after_all_data(self):
        cumsum = {"Action": {2012: 1.7, 2014: 2.3}}
        assert lookup_cumulative(cumsum, "Action", 2020) == 2.3


class TestGetFeatures:
    def test_returns_dataframe(self, train_stats_fixture):
        inp = {"Year": 2015, "meta_score": 85.0, "user_review": 8.5}
        result = get_features(inp, train_stats_fixture, "Action", "PS4")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_all_engineered_columns_present(self, train_stats_fixture):
        inp = {"Year": 2015, "meta_score": 85.0, "user_review": 8.5}
        result = get_features(inp, train_stats_fixture, "Action", "PS4")
        for col in NUMERICAL_FEATURES:
            if col != "Publisher_encoded":
                assert col in result.columns

    def test_unknown_genre_falls_back(self, train_stats_fixture):
        inp = {"Year": 2015, "meta_score": 85.0, "user_review": 8.5}
        result = get_features(inp, train_stats_fixture, "UnknownGenre", "PS4")
        assert result["Global_Sales_mean_genre"].iloc[0] == pytest.approx(
            train_stats_fixture["global_sales_mean"]
        )

    def test_interaction_feature_correct(self, train_stats_fixture):
        inp = {"Year": 2015, "meta_score": 85.0, "user_review": 8.5}
        result = get_features(inp, train_stats_fixture, "Action", "PS4")
        expected = 2015 * train_stats_fixture["genre_means"]["Action"]
        assert result["Year_Global_Sales_mean_genre"].iloc[0] == pytest.approx(expected)
