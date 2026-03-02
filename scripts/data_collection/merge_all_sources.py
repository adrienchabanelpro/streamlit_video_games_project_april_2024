"""Merge all data sources into a unified v3 dataset.

Sources: VGChartz (physical sales), SteamSpy (digital/PC), RAWG (metadata),
IGDB (themes/modes/franchise), HLTB (completion times).

Matching strategy:
1. RAWG as canonical registry (normalized names)
2. Exact match first, fuzzy match second (rapidfuzz WRatio >= 85)
3. Per-source match scores preserved for quality auditing

Output: data/Ventes_jeux_video_v3.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.data_collection.merge_datasets import normalize_name

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

# Input paths
VGCHARTZ_PATH = RAW_DIR / "vgchartz_2024.csv"
STEAMSPY_PATH = RAW_DIR / "steamspy_all.csv"
RAWG_PATH = RAW_DIR / "rawg_all.csv"
IGDB_PATH = RAW_DIR / "igdb_all.csv"
HLTB_PATH = RAW_DIR / "hltb_all.csv"
SCORES_PATH = DATA_DIR / "vgchartz_scores.csv"

# Output
OUTPUT_PATH = DATA_DIR / "Ventes_jeux_video_v3.csv"


def _build_lookup(names: pd.Series) -> dict[str, int]:
    """Build normalized name -> first index lookup."""
    lookup: dict[str, int] = {}
    for idx, name in names.items():
        norm = normalize_name(str(name)) if pd.notna(name) else ""
        if norm and norm not in lookup:
            lookup[norm] = idx
    return lookup


def _fuzzy_match_col(
    source_names: pd.Series,
    target_lookup: dict[str, int],
    threshold: int = 85,
    label: str = "source",
) -> dict[int, tuple[int, int]]:
    """Match source rows to target rows via exact + fuzzy matching.

    Returns
    -------
    dict mapping source_idx -> (target_idx, match_score)
    """
    from rapidfuzz import fuzz, process

    matches: dict[int, tuple[int, int]] = {}
    target_names = list(target_lookup.keys())
    target_indices = list(target_lookup.values())
    name_to_idx = dict(zip(target_names, target_indices))

    unmatched_src: list[tuple[int, str]] = []

    # Phase 1: exact
    exact = 0
    for src_idx, name in source_names.items():
        norm = normalize_name(str(name)) if pd.notna(name) else ""
        if not norm:
            continue
        if norm in target_lookup:
            matches[src_idx] = (target_lookup[norm], 100)
            exact += 1
        else:
            unmatched_src.append((src_idx, norm))

    print(f"  [{label}] Exact: {exact:,} matches")

    # Phase 2: fuzzy
    if unmatched_src:
        fuzzy = 0
        total = len(unmatched_src)
        for i, (src_idx, norm) in enumerate(unmatched_src):
            if (i + 1) % 10_000 == 0:
                print(f"  [{label}] Fuzzy: {i + 1:,}/{total:,}...")
            result = process.extractOne(
                norm, target_names, scorer=fuzz.WRatio, score_cutoff=threshold
            )
            if result is not None:
                match_name, score, _ = result
                matches[src_idx] = (name_to_idx[match_name], int(score))
                fuzzy += 1
        print(f"  [{label}] Fuzzy: {fuzzy:,} matches (threshold={threshold})")

    return matches


def merge_all_sources(
    match_threshold: int = 85,
    force: bool = False,
) -> Path:
    """Merge all available data sources into a unified dataset.

    Parameters
    ----------
    match_threshold:
        Minimum fuzzy match score (0-100).
    force:
        Overwrite output even if it exists.

    Returns
    -------
    Path to the merged CSV.
    """
    if OUTPUT_PATH.exists() and not force:
        print(f"[merge-v3] Already exists: {OUTPUT_PATH} (use --force)")
        return OUTPUT_PATH

    # ------------------------------------------------------------------
    # 1. Load VGChartz as the base (physical sales are the target variable)
    # ------------------------------------------------------------------
    if not VGCHARTZ_PATH.exists():
        # Fall back to the current final dataset
        alt = DATA_DIR / "Ventes_jeux_video_final.csv"
        if not alt.exists():
            raise FileNotFoundError(f"No VGChartz data found at {VGCHARTZ_PATH}")
        vg = pd.read_csv(alt)
        print(f"[merge-v3] VGChartz (from final): {len(vg):,} rows")
    else:
        vg = pd.read_csv(VGCHARTZ_PATH)
        print(f"[merge-v3] VGChartz: {len(vg):,} rows")

    vg["_norm_name"] = vg["Name"].astype(str).apply(normalize_name)

    # ------------------------------------------------------------------
    # 2. Merge SteamSpy (digital metrics)
    # ------------------------------------------------------------------
    if STEAMSPY_PATH.exists():
        steam = pd.read_csv(STEAMSPY_PATH)
        print(f"[merge-v3] SteamSpy: {len(steam):,} rows")

        steam_name_col = "name" if "name" in steam.columns else "Name"
        steam_lookup = _build_lookup(steam[steam_name_col])
        steam_matches = _fuzzy_match_col(
            vg["Name"], steam_lookup, threshold=match_threshold, label="steam"
        )

        # Add steam_ columns
        steam_cols = [
            "appid", "owners", "owners_midpoint", "positive", "negative",
            "review_pct", "average_forever", "median_forever",
            "price", "initialprice", "ccu", "tags",
        ]
        for col in steam_cols:
            vg[f"steam_{col}"] = float("nan")
        vg["steam_match_score"] = float("nan")

        for vg_idx, (steam_idx, score) in steam_matches.items():
            for col in steam_cols:
                if col in steam.columns:
                    vg.at[vg_idx, f"steam_{col}"] = steam.at[steam_idx, col]
            vg.at[vg_idx, "steam_match_score"] = score

        print(f"[merge-v3] SteamSpy: {len(steam_matches):,} total matches")
    else:
        print("[merge-v3] SteamSpy data not found — skipping")

    # ------------------------------------------------------------------
    # 3. Merge RAWG (metadata: playtime, ESRB, genres, tags, ratings)
    # ------------------------------------------------------------------
    if RAWG_PATH.exists():
        rawg = pd.read_csv(RAWG_PATH)
        print(f"[merge-v3] RAWG: {len(rawg):,} rows")

        rawg_lookup = _build_lookup(rawg["rawg_name"])
        rawg_matches = _fuzzy_match_col(
            vg["Name"], rawg_lookup, threshold=match_threshold, label="rawg"
        )

        rawg_cols = [
            "rawg_id", "rawg_slug", "rawg_released", "rawg_metacritic",
            "rawg_rating", "rawg_ratings_count", "rawg_playtime",
            "rawg_esrb", "rawg_genres", "rawg_platforms",
            "rawg_tags_top5", "rawg_developers", "rawg_publishers",
        ]
        for col in rawg_cols:
            vg[col] = float("nan") if col != "rawg_id" else pd.NA
        vg["rawg_match_score"] = float("nan")

        for vg_idx, (rawg_idx, score) in rawg_matches.items():
            for col in rawg_cols:
                if col in rawg.columns:
                    vg.at[vg_idx, col] = rawg.at[rawg_idx, col]
            vg.at[vg_idx, "rawg_match_score"] = score

        print(f"[merge-v3] RAWG: {len(rawg_matches):,} total matches")
    else:
        print("[merge-v3] RAWG data not found — skipping")

    # ------------------------------------------------------------------
    # 4. Merge IGDB (themes, game modes, perspectives, franchises)
    # ------------------------------------------------------------------
    if IGDB_PATH.exists():
        igdb = pd.read_csv(IGDB_PATH)
        print(f"[merge-v3] IGDB: {len(igdb):,} rows")

        igdb_lookup = _build_lookup(igdb["igdb_name"])
        igdb_matches = _fuzzy_match_col(
            vg["Name"], igdb_lookup, threshold=match_threshold, label="igdb"
        )

        igdb_cols = [
            "igdb_id", "igdb_slug", "igdb_released", "igdb_category",
            "igdb_total_rating", "igdb_rating_count",
            "igdb_hypes", "igdb_follows",
            "igdb_themes", "igdb_game_modes", "igdb_perspectives",
            "igdb_franchises", "igdb_developers", "igdb_publishers",
        ]
        for col in igdb_cols:
            vg[col] = float("nan") if col != "igdb_id" else pd.NA
        vg["igdb_match_score"] = float("nan")

        for vg_idx, (igdb_idx, score) in igdb_matches.items():
            for col in igdb_cols:
                if col in igdb.columns:
                    vg.at[vg_idx, col] = igdb.at[igdb_idx, col]
            vg.at[vg_idx, "igdb_match_score"] = score

        print(f"[merge-v3] IGDB: {len(igdb_matches):,} total matches")
    else:
        print("[merge-v3] IGDB data not found — skipping")

    # ------------------------------------------------------------------
    # 5. Merge HLTB (completion times)
    # ------------------------------------------------------------------
    if HLTB_PATH.exists():
        hltb = pd.read_csv(HLTB_PATH)
        print(f"[merge-v3] HLTB: {len(hltb):,} rows")

        hltb_lookup = _build_lookup(hltb["hltb_name"])
        hltb_matches = _fuzzy_match_col(
            vg["Name"], hltb_lookup, threshold=match_threshold, label="hltb"
        )

        hltb_cols = ["hltb_main", "hltb_main_extra", "hltb_completionist", "hltb_all_styles"]
        for col in hltb_cols:
            vg[col] = float("nan")
        vg["hltb_match_score"] = float("nan")

        for vg_idx, (hltb_idx, score) in hltb_matches.items():
            for col in hltb_cols:
                if col in hltb.columns:
                    vg.at[vg_idx, col] = hltb.at[hltb_idx, col]
            vg.at[vg_idx, "hltb_match_score"] = score

        print(f"[merge-v3] HLTB: {len(hltb_matches):,} total matches")
    else:
        print("[merge-v3] HLTB data not found — skipping")

    # ------------------------------------------------------------------
    # 6. Merge VGChartz scores enrichment
    # ------------------------------------------------------------------
    if SCORES_PATH.exists():
        scores = pd.read_csv(SCORES_PATH)
        print(f"[merge-v3] VGChartz scores: {len(scores):,} rows")

        # This enrichment uses Name+Platform join (already done in v2)
        # Only apply if meta_score column isn't already populated
        if "meta_score" in vg.columns:
            null_scores = vg["meta_score"].isna().sum()
            print(f"  [scores] {null_scores:,} rows missing meta_score")
    else:
        print("[merge-v3] VGChartz scores not found — skipping")

    # ------------------------------------------------------------------
    # 7. Derive additional columns
    # ------------------------------------------------------------------
    # Cross-platform count
    if "Name" in vg.columns:
        platform_counts = vg.groupby("Name")["Platform"].transform("nunique")
        vg["cross_platform_count"] = platform_counts
        vg["is_multi_platform"] = (platform_counts > 1).astype(int)

    # Release date parsing (from RAWG if available, else from Year)
    if "rawg_released" in vg.columns:
        rawg_dates = pd.to_datetime(vg["rawg_released"], errors="coerce")
        vg["release_month"] = rawg_dates.dt.month
        vg["release_quarter"] = rawg_dates.dt.quarter
        vg["release_day_of_week"] = rawg_dates.dt.dayofweek

    # ESRB ordinal encoding
    if "rawg_esrb" in vg.columns:
        esrb_map = {
            "Everyone": 0, "Everyone 10+": 1, "Teen": 2,
            "Mature": 3, "Adults Only": 4,
        }
        vg["esrb_encoded"] = vg["rawg_esrb"].map(esrb_map)

    # Franchise detection from IGDB
    if "igdb_franchises" in vg.columns:
        vg["has_franchise"] = (
            vg["igdb_franchises"].notna() & (vg["igdb_franchises"] != "")
        ).astype(int)

    # Game category from IGDB (0=main, 8=remake, 9=remaster, etc.)
    if "igdb_category" in vg.columns:
        vg["is_remake"] = (vg["igdb_category"] == 8).astype(int)
        vg["is_remaster"] = (vg["igdb_category"] == 9).astype(int)

    # Price tier from SteamSpy
    if "steam_initialprice" in vg.columns:
        price = pd.to_numeric(vg["steam_initialprice"], errors="coerce")
        vg["price_tier"] = pd.cut(
            price,
            bins=[-0.01, 0.0, 15.0, 40.0, 60.0, float("inf")],
            labels=["free", "indie", "standard", "premium", "deluxe"],
        )

    # ------------------------------------------------------------------
    # 8. Clean up and save
    # ------------------------------------------------------------------
    # Drop internal columns
    vg = vg.drop(columns=["_norm_name"], errors="ignore")

    # Sort by sales
    if "Global_Sales" in vg.columns:
        vg = vg.sort_values("Global_Sales", ascending=False).reset_index(drop=True)
        vg["Rank"] = range(1, len(vg) + 1)

    # Print summary
    _print_summary(vg)

    vg.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[merge-v3] Saved {len(vg):,} rows, {len(vg.columns)} columns → {OUTPUT_PATH}")
    return OUTPUT_PATH


def _print_summary(df: pd.DataFrame) -> None:
    """Print merge quality summary."""
    print("\n" + "=" * 60)
    print("MERGE SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    for source, col in [
        ("SteamSpy", "steam_match_score"),
        ("RAWG", "rawg_match_score"),
        ("IGDB", "igdb_match_score"),
        ("HLTB", "hltb_match_score"),
    ]:
        if col in df.columns:
            matched = df[col].notna().sum()
            pct = matched / len(df) * 100
            print(f"{source}: {matched:,} matched ({pct:.1f}%)")

    null_pct = df.isnull().mean().sort_values(ascending=False)
    high_null = null_pct[null_pct > 0.5]
    if len(high_null) > 0:
        print(f"\nColumns with >50% null: {len(high_null)}")
        for col, pct in high_null.head(10).items():
            print(f"  {col}: {pct:.1%} null")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge all data sources into v3 dataset")
    parser.add_argument(
        "--threshold", type=int, default=85, help="Fuzzy match threshold (default: 85)"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    merge_all_sources(match_threshold=args.threshold, force=args.force)
