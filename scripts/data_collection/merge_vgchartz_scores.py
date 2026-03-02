"""Merge VGChartz scores (user, critic, shipped) into the main dataset.

Fills missing user_review and meta_score values from a secondary VGChartz
download that contains critic/user scores and shipped units. Matches rows
by normalized (name, platform) key.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

MAIN_PATH = DATA_DIR / "Ventes_jeux_video_final.csv"
SCORES_PATH = DATA_DIR / "vgchartz_scores.csv"


def _normalize_key(name: str, platform: str) -> str:
    """Build a normalized (name|platform) lookup key."""
    n = str(name).strip().lower() if pd.notna(name) else ""
    p = str(platform).strip().lower() if pd.notna(platform) else ""
    return f"{n}|{p}"


def merge_scores() -> None:
    """Merge user/critic scores and shipped data into the main dataset."""
    if not MAIN_PATH.exists():
        raise FileNotFoundError(f"Main dataset not found: {MAIN_PATH}")
    if not SCORES_PATH.exists():
        raise FileNotFoundError(f"Scores dataset not found: {SCORES_PATH}")

    main = pd.read_csv(MAIN_PATH)
    scores = pd.read_csv(SCORES_PATH)

    print(f"[merge-scores] Main dataset: {len(main):,} rows")
    print(f"[merge-scores] Scores dataset: {len(scores):,} rows")

    # Build lookup from scores dataset: key → row index
    scores["_key"] = scores.apply(
        lambda r: _normalize_key(r["name"], r["platform"]), axis=1
    )
    # Keep first occurrence for duplicate keys
    scores_lookup: dict[str, int] = {}
    for idx, key in scores["_key"].items():
        if key and key not in scores_lookup:
            scores_lookup[key] = idx

    # Build keys for main dataset
    main["_key"] = main.apply(
        lambda r: _normalize_key(r["Name"], r["Platform"]), axis=1
    )

    # Track stats
    user_filled = 0
    critic_filled = 0
    shipped_filled = 0
    matched = 0

    # Initialize shipped column if not present
    if "shipped" not in main.columns:
        main["shipped"] = float("nan")

    for main_idx, key in main["_key"].items():
        if key not in scores_lookup:
            continue
        matched += 1
        scores_idx = scores_lookup[key]

        # Fill user_review from user where currently NaN
        if pd.isna(main.at[main_idx, "user_review"]):
            user_val = scores.at[scores_idx, "user"]
            if pd.notna(user_val):
                main.at[main_idx, "user_review"] = float(user_val)
                user_filled += 1

        # Fill meta_score from critic where currently NaN
        if pd.isna(main.at[main_idx, "meta_score"]):
            critic_val = scores.at[scores_idx, "critic"]
            if pd.notna(critic_val):
                main.at[main_idx, "meta_score"] = float(critic_val)
                critic_filled += 1

        # Fill shipped where currently NaN
        if pd.isna(main.at[main_idx, "shipped"]):
            shipped_val = scores.at[scores_idx, "shipped"]
            if pd.notna(shipped_val):
                main.at[main_idx, "shipped"] = float(shipped_val)
                shipped_filled += 1

    # Drop temp key column
    main = main.drop(columns=["_key"])

    # Save
    main.to_csv(MAIN_PATH, index=False)

    print(f"[merge-scores] Matched rows: {matched:,}")
    print(f"[merge-scores] user_review filled: {user_filled}")
    print(f"[merge-scores] meta_score filled: {critic_filled}")
    print(f"[merge-scores] shipped filled: {shipped_filled}")
    print(f"[merge-scores] Total user_review non-NaN: {main['user_review'].notna().sum()}")
    print(f"[merge-scores] Total meta_score non-NaN: {main['meta_score'].notna().sum()}")
    print(f"[merge-scores] Total shipped non-NaN: {main['shipped'].notna().sum()}")
    print(f"[merge-scores] Saved → {MAIN_PATH}")


if __name__ == "__main__":
    merge_scores()
