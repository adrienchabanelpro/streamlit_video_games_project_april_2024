"""Unit tests for the data collection pipeline.

Tests normalization, parsing, platform mapping, and merge logic
without requiring external API access.
"""

from __future__ import annotations

import pandas as pd

from scripts.data_collection.collect_steamspy import compute_review_pct, parse_owners
from scripts.data_collection.download_kaggle import extract_year, normalize_platform
from scripts.data_collection.merge_datasets import normalize_name

# ---------------------------------------------------------------------------
# normalize_platform
# ---------------------------------------------------------------------------


class TestNormalizePlatform:
    def test_known_platform(self):
        assert normalize_platform("PlayStation 4") == "PS4"

    def test_case_insensitive(self):
        assert normalize_platform("PLAYSTATION 4") == "PS4"

    def test_whitespace_stripped(self):
        assert normalize_platform("  Xbox 360  ") == "X360"

    def test_unknown_platform_passthrough(self):
        assert normalize_platform("Atari 2600") == "Atari 2600"

    def test_pc(self):
        assert normalize_platform("PC") == "PC"

    def test_nintendo_switch(self):
        assert normalize_platform("Nintendo Switch") == "NS"

    def test_wii(self):
        assert normalize_platform("Wii") == "Wii"

    def test_ps5(self):
        assert normalize_platform("PlayStation 5") == "PS5"


# ---------------------------------------------------------------------------
# extract_year
# ---------------------------------------------------------------------------


class TestExtractYear:
    def test_iso_date(self):
        assert extract_year("2020-03-15") == 2020.0

    def test_us_date(self):
        assert extract_year("03/15/2020") == 2020.0

    def test_year_only(self):
        assert extract_year("2015") == 2015.0

    def test_none(self):
        assert extract_year(None) is None

    def test_nan(self):
        assert extract_year(float("nan")) is None

    def test_empty_string(self):
        assert extract_year("") is None

    def test_garbage(self):
        assert extract_year("no date here") is None

    def test_embedded_year(self):
        assert extract_year("Released in 2018 Q3") == 2018.0


# ---------------------------------------------------------------------------
# parse_owners
# ---------------------------------------------------------------------------


class TestParseOwners:
    def test_typical_range(self):
        assert parse_owners("1,000,000 .. 2,000,000") == 1_500_000.0

    def test_small_range(self):
        assert parse_owners("0 .. 20,000") == 10_000.0

    def test_single_value_format(self):
        # Not a range — should return None
        assert parse_owners("50000") is None

    def test_none(self):
        assert parse_owners(None) is None

    def test_empty_string(self):
        assert parse_owners("") is None

    def test_non_string(self):
        assert parse_owners(12345) is None

    def test_exact_range(self):
        assert parse_owners("500,000 .. 1,000,000") == 750_000.0


# ---------------------------------------------------------------------------
# compute_review_pct
# ---------------------------------------------------------------------------


class TestComputeReviewPct:
    def test_all_positive(self):
        assert compute_review_pct(100, 0) == 100.0

    def test_all_negative(self):
        assert compute_review_pct(0, 100) == 0.0

    def test_mixed(self):
        assert compute_review_pct(75, 25) == 75.0

    def test_zero_reviews(self):
        assert compute_review_pct(0, 0) is None

    def test_none_values(self):
        assert compute_review_pct(None, 50) is None

    def test_string_numbers(self):
        assert compute_review_pct("80", "20") == 80.0


# ---------------------------------------------------------------------------
# normalize_name
# ---------------------------------------------------------------------------


class TestNormalizeName:
    def test_lowercase(self):
        assert normalize_name("GRAND THEFT AUTO V") == "grand theft auto v"

    def test_remove_punctuation(self):
        result = normalize_name("The Elder Scrolls V: Skyrim")
        assert ":" not in result

    def test_strip_edition(self):
        result = normalize_name("Skyrim Special Edition")
        assert "special edition" not in result
        assert "skyrim" in result

    def test_strip_remastered(self):
        result = normalize_name("Dark Souls Remastered")
        assert "remastered" not in result
        assert "dark souls" in result

    def test_strip_goty(self):
        result = normalize_name("Fallout 3 GOTY")
        assert "goty" not in result.lower()

    def test_strip_leading_article(self):
        result = normalize_name("The Witcher 3")
        assert not result.startswith("the ")

    def test_collapse_whitespace(self):
        result = normalize_name("  Super   Mario   Bros  ")
        assert "  " not in result
        assert result == "super mario bros"

    def test_none(self):
        assert normalize_name(None) == ""

    def test_empty_string(self):
        assert normalize_name("") == ""

    def test_non_string(self):
        assert normalize_name(12345) == ""

    def test_complex_name(self):
        result = normalize_name("The Legend of Zelda: Breath of the Wild - Deluxe Edition")
        assert "deluxe edition" not in result
        assert "legend" in result


# ---------------------------------------------------------------------------
# merge logic (integration-style, with small DataFrames)
# ---------------------------------------------------------------------------


class TestMergeLogic:
    """Test the merge by calling internal functions with small test data."""

    def _make_vg_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Name": ["Grand Theft Auto V", "The Witcher 3", "Unknown Game XYZ"],
                "Platform": ["PS4", "PC", "XOne"],
                "Year": [2013.0, 2015.0, 2020.0],
                "Genre": ["Action", "RPG", "Adventure"],
                "Publisher": ["Rockstar", "CDPR", "Unknown"],
                "NA_Sales": [10.0, 5.0, 0.01],
                "EU_Sales": [8.0, 4.0, 0.01],
                "JP_Sales": [1.0, 0.5, 0.0],
                "Other_Sales": [2.0, 1.0, 0.0],
                "Global_Sales": [21.0, 10.5, 0.02],
                "meta_score": [97.0, 92.0, float("nan")],
                "user_review": [float("nan")] * 3,
            }
        )

    def _make_steam_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "appid": [271590, 292030, 999999],
                "name": [
                    "Grand Theft Auto V",
                    "The Witcher 3: Wild Hunt",
                    "Some Other Steam Game",
                ],
                "owners": [
                    "50,000,000 .. 100,000,000",
                    "10,000,000 .. 20,000,000",
                    "0 .. 20,000",
                ],
                "owners_midpoint": [75_000_000.0, 15_000_000.0, 10_000.0],
                "positive": [500000, 600000, 100],
                "negative": [50000, 30000, 50],
                "review_pct": [90.9, 95.2, 66.7],
                "average_forever": [5000, 3000, 100],
                "median_forever": [2000, 1500, 50],
                "price": [29.99, 39.99, 9.99],
                "initialprice": [59.99, 59.99, 9.99],
                "ccu": [100000, 50000, 10],
                "tags": ["Action;Open World", "RPG;Open World", "Indie"],
            }
        )

    def test_exact_match(self):
        """Normalized names that match exactly should get score=100."""
        vg = self._make_vg_df()
        steam = self._make_steam_df()

        # GTA V should match exactly
        vg_norm = normalize_name(vg.at[0, "Name"])
        steam_norm = normalize_name(steam.at[0, "name"])
        assert vg_norm == steam_norm

    def test_fuzzy_match_witcher(self):
        """Witcher 3 vs Witcher 3: Wild Hunt should fuzzy-match above threshold."""
        from rapidfuzz import fuzz

        vg_norm = normalize_name("The Witcher 3")
        steam_norm = normalize_name("The Witcher 3: Wild Hunt")
        score = fuzz.WRatio(vg_norm, steam_norm)
        assert score >= 85

    def test_no_match_unknown(self):
        """A game with no close match should not get steam columns."""
        from rapidfuzz import fuzz

        vg_norm = normalize_name("Unknown Game XYZ")
        steam_names = [
            normalize_name("Grand Theft Auto V"),
            normalize_name("The Witcher 3: Wild Hunt"),
            normalize_name("Some Other Steam Game"),
        ]
        scores = [fuzz.WRatio(vg_norm, s) for s in steam_names]
        assert all(s < 85 for s in scores)

    def test_backward_compatible_columns(self):
        """Merged output should keep all original columns."""
        vg = self._make_vg_df()
        original_cols = set(vg.columns)
        # After merge, all original columns must still be present
        assert "Name" in original_cols
        assert "Global_Sales" in original_cols
        assert "meta_score" in original_cols

    def test_steam_prefix(self):
        """All SteamSpy columns should be prefixed with steam_."""
        from scripts.data_collection.merge_datasets import STEAM_COLUMNS

        for col in STEAM_COLUMNS:
            assert not col.startswith("steam_"), f"{col} shouldn't have prefix in source"
