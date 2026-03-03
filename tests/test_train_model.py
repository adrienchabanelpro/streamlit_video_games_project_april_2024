"""Tests for the training pipeline (scripts/training/)."""

import numpy as np
import pandas as pd
import pytest

from scripts.training.data_prep import (
    _lookup_cumulative,
    clean_data,
    compute_train_stats,
    engineer_features,
    load_dataset,
    temporal_split,
)
from scripts.training.evaluation import compute_metrics


# ---------------------------------------------------------------------------
# load_dataset + clean_data
# ---------------------------------------------------------------------------
class TestLoadAndCleanData:
    def test_drops_regional_sales(self, sample_raw_df, tmp_path):
        csv_path = tmp_path / "data.csv"
        sample_raw_df.to_csv(csv_path, index=False)
        df = clean_data(load_dataset(csv_path))

        for col in ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]:
            assert col not in df.columns

    def test_drops_rank_and_name(self, sample_raw_df, tmp_path):
        csv_path = tmp_path / "data.csv"
        sample_raw_df.to_csv(csv_path, index=False)
        df = clean_data(load_dataset(csv_path))

        assert "Rank" not in df.columns
        assert "Name" not in df.columns

    def test_drops_rows_with_missing_publisher_or_year(self, sample_raw_df, tmp_path):
        csv_path = tmp_path / "data.csv"
        sample_raw_df.to_csv(csv_path, index=False)
        df = clean_data(load_dataset(csv_path))

        # Row 6 (index 5) has NaN Year → dropped
        # Row with None Publisher may survive CSV round-trip as empty string
        # At minimum, the NaN-Year row is gone
        assert len(df) < len(sample_raw_df)
        assert df["Publisher"].notna().all()
        assert df["Year"].notna().all()

    def test_year_is_int(self, sample_raw_df, tmp_path):
        csv_path = tmp_path / "data.csv"
        sample_raw_df.to_csv(csv_path, index=False)
        df = clean_data(load_dataset(csv_path))

        assert df["Year"].dtype in (np.int64, np.int32, int)


# ---------------------------------------------------------------------------
# temporal_split
# ---------------------------------------------------------------------------
class TestTemporalSplit:
    def test_split_at_2013(self, clean_df):
        train, test = temporal_split(clean_df, 2013)
        assert (train["Year"] <= 2013).all()
        assert (test["Year"] > 2013).all()

    def test_no_overlap(self, clean_df):
        train, test = temporal_split(clean_df, 2014)
        common = set(train.index) & set(test.index)
        assert len(common) == 0

    def test_all_data_preserved(self, clean_df):
        train, test = temporal_split(clean_df, 2014)
        assert len(train) + len(test) == len(clean_df)

    def test_split_before_all_data(self, clean_df):
        train, test = temporal_split(clean_df, 2000)
        assert len(train) == 0
        assert len(test) == len(clean_df)

    def test_split_after_all_data(self, clean_df):
        train, test = temporal_split(clean_df, 2050)
        assert len(train) == len(clean_df)
        assert len(test) == 0


# ---------------------------------------------------------------------------
# compute_train_stats
# ---------------------------------------------------------------------------
class TestComputeTrainStats:
    def test_returns_required_keys(self, clean_df):
        stats = compute_train_stats(clean_df)
        required_keys = [
            "genre_means",
            "platform_means",
            "cumsum_genre",
            "cumsum_platform",
            "publishers",
            "genres",
            "platforms",
            "meta_score_mean",
            "user_review_mean",
            "global_sales_mean",
        ]
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

    def test_genre_means_correct(self, clean_df):
        stats = compute_train_stats(clean_df)
        # Action: (1.7 + 0.6 + 0.34) / 3 = 0.88
        assert stats["genre_means"]["Action"] == pytest.approx(0.88, abs=0.01)

    def test_lists_are_sorted(self, clean_df):
        stats = compute_train_stats(clean_df)
        assert stats["publishers"] == sorted(stats["publishers"])
        assert stats["genres"] == sorted(stats["genres"])
        assert stats["platforms"] == sorted(stats["platforms"])

    def test_cumsum_is_monotonic(self, clean_df):
        stats = compute_train_stats(clean_df)
        for genre, yearly in stats["cumsum_genre"].items():
            values = [yearly[y] for y in sorted(yearly.keys())]
            for i in range(1, len(values)):
                assert values[i] >= values[i - 1]

    def test_global_sales_mean_is_float(self, clean_df):
        stats = compute_train_stats(clean_df)
        assert isinstance(stats["global_sales_mean"], float)
        assert stats["global_sales_mean"] > 0


# ---------------------------------------------------------------------------
# _lookup_cumulative
# ---------------------------------------------------------------------------
class TestLookupCumulative:
    def test_normal_lookup(self):
        cumsum = {"Action": {2012: 1.7, 2014: 2.3}}
        assert _lookup_cumulative(cumsum, "Action", 2014) == 2.3

    def test_category_not_found(self):
        cumsum = {"Action": {2012: 1.7}}
        assert _lookup_cumulative(cumsum, "RPG", 2014) == 0.0

    def test_year_before_all_data(self):
        cumsum = {"Action": {2012: 1.7, 2014: 2.3}}
        assert _lookup_cumulative(cumsum, "Action", 2010) == 0.0

    def test_year_between_entries(self):
        cumsum = {"Action": {2012: 1.7, 2014: 2.3}}
        # Year 2013 should return 2012's value
        assert _lookup_cumulative(cumsum, "Action", 2013) == 1.7

    def test_empty_dict(self):
        assert _lookup_cumulative({}, "Action", 2014) == 0.0

    def test_year_after_all_data(self):
        cumsum = {"Action": {2012: 1.7, 2014: 2.3}}
        assert _lookup_cumulative(cumsum, "Action", 2020) == 2.3


# ---------------------------------------------------------------------------
# engineer_features
# ---------------------------------------------------------------------------
class TestEngineerFeatures:
    def test_adds_expected_columns(self, clean_df, train_stats_fixture):
        result = engineer_features(clean_df, train_stats_fixture)
        expected_cols = [
            "Global_Sales_mean_genre",
            "Global_Sales_mean_platform",
            "Year_Global_Sales_mean_genre",
            "Year_Global_Sales_mean_platform",
            "Cumulative_Sales_Genre",
            "Cumulative_Sales_Platform",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_does_not_modify_input(self, clean_df, train_stats_fixture):
        original_cols = list(clean_df.columns)
        engineer_features(clean_df, train_stats_fixture)
        assert list(clean_df.columns) == original_cols

    def test_interaction_features_correct(self, clean_df, train_stats_fixture):
        result = engineer_features(clean_df, train_stats_fixture)
        row = result.iloc[0]  # Year=2012, Genre=Action
        expected = row["Year"] * row["Global_Sales_mean_genre"]
        assert row["Year_Global_Sales_mean_genre"] == pytest.approx(expected)

    def test_unknown_genre_falls_back_to_global_mean(self, train_stats_fixture):
        df = pd.DataFrame(
            {
                "Platform": ["PS4"],
                "Year": [2015],
                "Genre": ["UnknownGenre"],
                "Publisher": ["EA"],
                "Global_Sales": [1.0],
                "meta_score": [80.0],
                "user_review": [8.0],
            }
        )
        result = engineer_features(df, train_stats_fixture)
        assert result["Global_Sales_mean_genre"].iloc[0] == pytest.approx(
            train_stats_fixture["global_sales_mean"]
        )


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------
class TestComputeMetrics:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        metrics = compute_metrics(y, y, log_transform=False)
        assert metrics["r2"] == pytest.approx(1.0)
        assert metrics["mse"] == pytest.approx(0.0)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["rmse"] == pytest.approx(0.0)

    def test_returns_all_keys(self):
        y = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 2.9])
        metrics = compute_metrics(y, y_pred, log_transform=False)
        assert {"r2", "mse", "rmse", "mae"}.issubset(set(metrics.keys()))

    def test_all_values_are_float(self):
        y = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        metrics = compute_metrics(y, y_pred, log_transform=False)
        for v in metrics.values():
            assert isinstance(v, float)

    def test_rmse_is_sqrt_mse(self):
        y = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        metrics = compute_metrics(y, y_pred, log_transform=False)
        assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]))
