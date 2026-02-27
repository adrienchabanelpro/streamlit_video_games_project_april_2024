"""Tests for the sentiment analysis pipeline (source/analyse_avis_utilisateurs.py)."""

import io

import pandas as pd
import pytest
from analyse_avis_utilisateurs import predict_user_reviews


class TestPredictUserReviews:
    def _make_csv_file(self, reviews: list[str]) -> io.BytesIO:
        """Create an in-memory CSV file with a user_review column."""
        df = pd.DataFrame({"user_review": reviews})
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return buf

    def test_returns_none_for_none_input(self):
        data, pos, neg = predict_user_reviews(None)
        assert data is None
        assert pos is None
        assert neg is None

    def test_positive_review_detected(self):
        f = self._make_csv_file(["This game is absolutely amazing and fun!"])
        data, pos, neg = predict_user_reviews(f)
        assert data is not None
        assert pos == pytest.approx(100.0)
        assert neg == pytest.approx(0.0)

    def test_negative_review_detected(self):
        f = self._make_csv_file(["Terrible game, awful graphics, worst game ever."])
        data, pos, neg = predict_user_reviews(f)
        assert data is not None
        assert neg == pytest.approx(100.0)
        assert pos == pytest.approx(0.0)

    def test_mixed_reviews(self):
        f = self._make_csv_file(
            [
                "Best game I have ever played!",
                "Horrible and boring experience.",
            ]
        )
        data, pos, neg = predict_user_reviews(f)
        assert data is not None
        assert pos + neg == pytest.approx(100.0)

    def test_output_has_expected_columns(self):
        f = self._make_csv_file(["Great game!"])
        data, _, _ = predict_user_reviews(f)
        assert "sentiment" in data.columns
        assert "confidence" in data.columns
        assert "predictions" in data.columns

    def test_confidence_in_range(self):
        f = self._make_csv_file(["Amazing!", "Terrible!"])
        data, _, _ = predict_user_reviews(f)
        assert (data["confidence"] >= 0).all()
        assert (data["confidence"] <= 1).all()

    def test_predictions_are_binary(self):
        f = self._make_csv_file(["Great!", "Bad!"])
        data, _, _ = predict_user_reviews(f)
        assert set(data["predictions"].unique()).issubset({0, 1})

    def test_missing_column_returns_none(self):
        buf = io.BytesIO()
        pd.DataFrame({"wrong_column": ["text"]}).to_csv(buf, index=False)
        buf.seek(0)
        data, pos, neg = predict_user_reviews(buf)
        assert data is None
