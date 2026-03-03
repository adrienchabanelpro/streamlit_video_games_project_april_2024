"""Shared pytest fixtures for the video game sales prediction project."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root and source to sys.path so tests can import project modules
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "source" / "ml"))


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Minimal raw dataframe resembling Ventes_jeux_video_final.csv."""
    return pd.DataFrame(
        {
            "Rank": [1, 2, 3, 4, 5, 6],
            "Name": ["Game A", "Game B", "Game C", "Game D", "Game E", "Game F"],
            "Platform": ["PS4", "PS4", "XOne", "XOne", "PC", "PC"],
            "Year": [2012.0, 2013.0, 2014.0, 2015.0, 2016.0, np.nan],
            "Genre": ["Action", "RPG", "Action", "RPG", "Action", "RPG"],
            "Publisher": ["EA", "Ubisoft", "EA", "Ubisoft", "EA", None],
            "NA_Sales": [1.0, 0.5, 0.3, 0.8, 0.2, 0.1],
            "EU_Sales": [0.5, 0.3, 0.2, 0.4, 0.1, 0.05],
            "JP_Sales": [0.1, 0.2, 0.05, 0.1, 0.02, 0.01],
            "Other_Sales": [0.1, 0.1, 0.05, 0.1, 0.02, 0.01],
            "Global_Sales": [1.7, 1.1, 0.6, 1.4, 0.34, 0.17],
            "meta_score": [85.0, 90.0, np.nan, 78.0, 82.0, 70.0],
            "user_review": [8.5, 9.0, 7.5, np.nan, 8.0, 6.0],
        }
    )


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """Cleaned dataframe (after load_and_clean_data-like processing)."""
    return pd.DataFrame(
        {
            "Platform": ["PS4", "PS4", "XOne", "XOne", "PC"],
            "Year": [2012, 2013, 2014, 2015, 2016],
            "Genre": ["Action", "RPG", "Action", "RPG", "Action"],
            "Publisher": ["EA", "Ubisoft", "EA", "Ubisoft", "EA"],
            "Global_Sales": [1.7, 1.1, 0.6, 1.4, 0.34],
            "meta_score": [85.0, 90.0, 85.0, 78.0, 82.0],
            "user_review": [8.5, 9.0, 7.5, 8.5, 8.0],
        }
    )


@pytest.fixture
def train_stats_fixture() -> dict:
    """Pre-computed training statistics for feature engineering."""
    return {
        "genre_means": {"Action": 0.88, "RPG": 1.25},
        "platform_means": {"PS4": 1.4, "XOne": 1.0, "PC": 0.34},
        "cumsum_genre": {
            "Action": {2012: 1.7, 2014: 2.3},
            "RPG": {2013: 1.1, 2015: 2.5},
        },
        "cumsum_platform": {
            "PS4": {2012: 1.7, 2013: 2.8},
            "XOne": {2014: 0.6, 2015: 2.0},
            "PC": {2016: 0.34},
        },
        "publishers": ["EA", "Ubisoft"],
        "genres": ["Action", "RPG"],
        "platforms": ["PC", "PS4", "XOne"],
        "meta_score_mean": 84.0,
        "user_review_mean": 8.3,
        "global_sales_mean": 1.028,
    }
