"""Step 3: Fuzzy-merge VGChartz 64K with SteamSpy data.

Produces a unified, backward-compatible CSV with steam_ prefixed columns
for matched games. All original columns are preserved.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

VGCHARTZ_PATH = RAW_DIR / "vgchartz_2024.csv"
STEAMSPY_PATH = RAW_DIR / "steamspy_all.csv"
ORIGINAL_BACKUP = RAW_DIR / "Ventes_jeux_video_original.csv"
OUTPUT_PATH = DATA_DIR / "Ventes_jeux_video_final.csv"

# SteamSpy columns to include in the merged output (prefixed with steam_)
STEAM_COLUMNS = [
    "appid",
    "owners",
    "owners_midpoint",
    "positive",
    "negative",
    "review_pct",
    "average_forever",
    "median_forever",
    "price",
    "initialprice",
    "ccu",
    "tags",
]

# Suffixes stripped during normalization
EDITION_SUFFIXES = re.compile(
    r"\b(remastered|remaster|remake|goty|game of the year|definitive edition|"
    r"deluxe edition|gold edition|ultimate edition|complete edition|"
    r"special edition|enhanced edition|hd|collection|anthology|"
    r"directors cut|director's cut)\b",
    re.IGNORECASE,
)

# Leading articles to strip
LEADING_ARTICLES = re.compile(r"^(the|a|an|le|la|les|l')\s+", re.IGNORECASE)


def normalize_name(name: str) -> str:
    """Normalize a game name for matching.

    Lowercases, removes punctuation, strips edition suffixes and leading
    articles, then collapses whitespace.

    Examples
    --------
    >>> normalize_name("The Elder Scrolls V: Skyrim - Special Edition")
    'elder scrolls v skyrim'
    >>> normalize_name("GRAND THEFT AUTO V")
    'grand theft auto v'
    """
    if not name or not isinstance(name, str):
        return ""
    s = name.lower().strip()
    # Remove edition suffixes
    s = EDITION_SUFFIXES.sub("", s)
    # Remove punctuation
    s = re.sub(r"[^\w\s]", " ", s)
    # Strip leading articles
    s = LEADING_ARTICLES.sub("", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def merge_datasets(match_threshold: int = 85, force: bool = False) -> Path:
    """Merge VGChartz and SteamSpy datasets.

    Parameters
    ----------
    match_threshold:
        Minimum fuzzy match score (0-100) for accepting a match.
    force:
        Overwrite output even if it already exists.

    Returns
    -------
    Path
        Path to the merged CSV.
    """
    if OUTPUT_PATH.exists() and not force:
        # Check if it already has steam_ columns
        sample = pd.read_csv(OUTPUT_PATH, nrows=1)
        if any(c.startswith("steam_") for c in sample.columns):
            print(f"[merge] Already merged: {OUTPUT_PATH} (use --force to redo)")
            return OUTPUT_PATH

    # Backup original data
    if OUTPUT_PATH.exists() and not ORIGINAL_BACKUP.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUTPUT_PATH, ORIGINAL_BACKUP)
        print(f"[merge] Backed up original → {ORIGINAL_BACKUP}")

    # Load VGChartz
    if not VGCHARTZ_PATH.exists():
        raise FileNotFoundError(
            f"VGChartz data not found: {VGCHARTZ_PATH}\n"
            "Run download_kaggle.py first."
        )
    vg = pd.read_csv(VGCHARTZ_PATH)
    print(f"[merge] VGChartz: {len(vg):,} rows")

    # Load SteamSpy (optional — merge still works without it)
    steam: pd.DataFrame | None = None
    if STEAMSPY_PATH.exists():
        steam = pd.read_csv(STEAMSPY_PATH)
        print(f"[merge] SteamSpy: {len(steam):,} rows")
    else:
        print("[merge] SteamSpy data not found — skipping enrichment.")

    # Build output
    if steam is not None and len(steam) > 0:
        merged = _fuzzy_merge(vg, steam, match_threshold)
    else:
        merged = vg.copy()
        # Add empty steam_ columns for schema consistency
        for col in STEAM_COLUMNS:
            merged[f"steam_{col}"] = float("nan")
        merged["steam_match_score"] = float("nan")

    # Recompute Rank
    merged = merged.sort_values("Global_Sales", ascending=False).reset_index(drop=True)
    merged["Rank"] = range(1, len(merged) + 1)

    # Ensure backward-compatible types
    _enforce_types(merged)

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"[merge] Saved {len(merged):,} rows ({_count_matched(merged)} matched) → {OUTPUT_PATH}")
    return OUTPUT_PATH


def _fuzzy_merge(
    vg: pd.DataFrame,
    steam: pd.DataFrame,
    threshold: int,
) -> pd.DataFrame:
    """Perform exact + fuzzy name matching between VGChartz and SteamSpy."""
    from rapidfuzz import fuzz

    # Normalize names
    vg = vg.copy()
    steam = steam.copy()

    vg["_norm_name"] = vg["Name"].astype(str).apply(normalize_name)
    steam_name_col = "name" if "name" in steam.columns else "Name"
    steam["_norm_name"] = steam[steam_name_col].astype(str).apply(normalize_name)

    # Build lookup dict for exact matches (steam side)
    steam_lookup: dict[str, int] = {}
    for idx, norm in steam["_norm_name"].items():
        if norm and norm not in steam_lookup:
            steam_lookup[norm] = idx

    # Prepare steam_ columns on the VG dataframe
    for col in STEAM_COLUMNS:
        vg[f"steam_{col}"] = float("nan")
    vg["steam_match_score"] = float("nan")

    # Phase 1: Exact matches
    exact_count = 0
    unmatched_indices = []
    for vg_idx, vg_norm in vg["_norm_name"].items():
        if not vg_norm:
            unmatched_indices.append(vg_idx)
            continue
        if vg_norm in steam_lookup:
            steam_idx = steam_lookup[vg_norm]
            _copy_steam_cols(vg, vg_idx, steam, steam_idx, score=100)
            exact_count += 1
        else:
            unmatched_indices.append(vg_idx)

    print(f"[merge] Phase 1 (exact): {exact_count:,} matches")

    # Phase 2: Fuzzy matches for remaining rows using optimized extractOne
    if unmatched_indices:
        from rapidfuzz import process

        steam_names = list(steam_lookup.keys())
        steam_indices = list(steam_lookup.values())
        # Build name→steam_idx mapping for O(1) lookup after fuzzy match
        name_to_steam_idx = dict(zip(steam_names, steam_indices))
        fuzzy_count = 0
        total_unmatched = len(unmatched_indices)

        # Collect all unmatched names for batch-style processing
        unmatched_names = [vg.at[idx, "_norm_name"] for idx in unmatched_indices]

        print(f"[merge] Phase 2 (fuzzy): matching {total_unmatched:,} names...")
        for i, (vg_idx, vg_norm) in enumerate(zip(unmatched_indices, unmatched_names)):
            if (i + 1) % 10000 == 0:
                print(f"[merge] Phase 2 (fuzzy): {i + 1:,}/{total_unmatched:,}...")

            if not vg_norm:
                continue

            result = process.extractOne(
                vg_norm, steam_names, scorer=fuzz.WRatio, score_cutoff=threshold
            )
            if result is not None:
                match_name, score, _ = result
                steam_idx = name_to_steam_idx[match_name]
                _copy_steam_cols(vg, vg_idx, steam, steam_idx, score=int(score))
                fuzzy_count += 1

        print(f"[merge] Phase 2 (fuzzy): {fuzzy_count:,} matches (threshold={threshold})")

    # Clean up temp column
    vg = vg.drop(columns=["_norm_name"])

    return vg


def _copy_steam_cols(
    vg: pd.DataFrame,
    vg_idx: int,
    steam: pd.DataFrame,
    steam_idx: int,
    score: int,
) -> None:
    """Copy SteamSpy columns from steam row to vg row."""
    for col in STEAM_COLUMNS:
        if col in steam.columns:
            vg.at[vg_idx, f"steam_{col}"] = steam.at[steam_idx, col]
    vg.at[vg_idx, "steam_match_score"] = score


def _count_matched(df: pd.DataFrame) -> int:
    """Count rows with a non-NaN steam_match_score."""
    if "steam_match_score" in df.columns:
        return int(df["steam_match_score"].notna().sum())
    return 0


def _enforce_types(df: pd.DataFrame) -> None:
    """Ensure backward-compatible types for the original columns."""
    float_cols = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales", "meta_score", "user_review"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge VGChartz + SteamSpy datasets")
    parser.add_argument("--threshold", type=int, default=85, help="Fuzzy match threshold (default: 85)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing merged file")
    args = parser.parse_args()

    merge_datasets(match_threshold=args.threshold, force=args.force)
