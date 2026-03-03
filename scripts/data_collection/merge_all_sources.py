"""Merge all data sources into a unified v3 dataset.

Sources (9):
- VGChartz (physical sales) — base/target variable
- SteamSpy (digital/PC owner estimates, reviews, playtime)
- RAWG (metadata: ratings, tags, genres, platforms)
- IGDB (themes, modes, franchises, perspectives)
- HLTB (completion times)
- Wikipedia (verified official sales figures)
- Steam Store (pricing, DLC, categories, review counts)
- OpenCritic (aggregated critic scores from 100+ outlets)
- Gamedatacrunch (revenue estimates, CCU, regional pricing)

Matching strategy:
1. Exact match first, fuzzy match second (rapidfuzz WRatio >= 85)
2. Per-source match scores preserved for quality auditing

Output: data/Ventes_jeux_video_v3.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.data_collection.merge_datasets import normalize_name

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

# Input paths — original 5 sources
VGCHARTZ_PATH = RAW_DIR / "vgchartz_2024.csv"
STEAMSPY_PATH = RAW_DIR / "steamspy_all.csv"
RAWG_PATH = RAW_DIR / "rawg_all.csv"
IGDB_PATH = RAW_DIR / "igdb_all.csv"
HLTB_PATH = RAW_DIR / "hltb_all.csv"
SCORES_PATH = DATA_DIR / "vgchartz_scores.csv"

# Input paths — new 4 sources
WIKIPEDIA_PATH = RAW_DIR / "wikipedia_sales.csv"
STEAM_STORE_PATH = RAW_DIR / "steam_store.csv"
OPENCRITIC_PATH = RAW_DIR / "opencritic.csv"
GAMEDATACRUNCH_PATH = RAW_DIR / "gamedatacrunch.csv"

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
    # 6. Merge Wikipedia (verified official sales figures)
    # ------------------------------------------------------------------
    if WIKIPEDIA_PATH.exists():
        wiki = pd.read_csv(WIKIPEDIA_PATH)
        print(f"[merge-v3] Wikipedia: {len(wiki):,} rows")

        wiki_lookup = _build_lookup(wiki["wiki_name"])
        wiki_matches = _fuzzy_match_col(
            vg["Name"], wiki_lookup, threshold=match_threshold, label="wiki"
        )

        wiki_cols = [
            "wiki_sales_millions", "wiki_platform", "wiki_publisher",
            "wiki_developer", "wiki_release_date", "wiki_source_page",
            "wiki_sales_type",
        ]
        for col in wiki_cols:
            vg[col] = float("nan") if col == "wiki_sales_millions" else ""
        vg["wiki_match_score"] = float("nan")

        for vg_idx, (wiki_idx, score) in wiki_matches.items():
            for col in wiki_cols:
                if col in wiki.columns:
                    vg.at[vg_idx, col] = wiki.at[wiki_idx, col]
            vg.at[vg_idx, "wiki_match_score"] = score

        print(f"[merge-v3] Wikipedia: {len(wiki_matches):,} total matches")
    else:
        print("[merge-v3] Wikipedia data not found — skipping")

    # ------------------------------------------------------------------
    # 7. Merge Steam Store (pricing, DLC, categories, reviews)
    # ------------------------------------------------------------------
    if STEAM_STORE_PATH.exists():
        sstore = pd.read_csv(STEAM_STORE_PATH)
        print(f"[merge-v3] Steam Store: {len(sstore):,} rows")

        sstore_lookup = _build_lookup(sstore["steam_store_name"])
        sstore_matches = _fuzzy_match_col(
            vg["Name"], sstore_lookup, threshold=match_threshold, label="steam_store"
        )

        # Map source columns to target columns (rename steam_appid to avoid
        # collision with SteamSpy's steam_appid column)
        sstore_col_map = {
            "steam_appid": "steam_store_appid",
            "steam_store_price_usd": "steam_store_price_usd",
            "steam_store_is_free": "steam_store_is_free",
            "steam_store_release_date": "steam_store_release_date",
            "steam_store_coming_soon": "steam_store_coming_soon",
            "steam_store_recommendations": "steam_store_recommendations",
            "steam_store_categories": "steam_store_categories",
            "steam_store_genres": "steam_store_genres",
            "steam_store_dlc_count": "steam_store_dlc_count",
            "steam_store_metacritic": "steam_store_metacritic",
            "steam_store_platforms_win": "steam_store_platforms_win",
            "steam_store_platforms_mac": "steam_store_platforms_mac",
            "steam_store_platforms_linux": "steam_store_platforms_linux",
            "steam_store_developer": "steam_store_developer",
            "steam_store_publisher": "steam_store_publisher",
        }
        str_cols = {"steam_store_categories", "steam_store_genres", "steam_store_developer", "steam_store_publisher", "steam_store_release_date"}
        for target in sstore_col_map.values():
            vg[target] = "" if target in str_cols else float("nan")
        vg["steam_store_match_score"] = float("nan")

        for vg_idx, (sstore_idx, score) in sstore_matches.items():
            for src_col, tgt_col in sstore_col_map.items():
                if src_col in sstore.columns:
                    vg.at[vg_idx, tgt_col] = sstore.at[sstore_idx, src_col]
            vg.at[vg_idx, "steam_store_match_score"] = score

        print(f"[merge-v3] Steam Store: {len(sstore_matches):,} total matches")
    else:
        print("[merge-v3] Steam Store data not found — skipping")

    # ------------------------------------------------------------------
    # 8. Merge OpenCritic (aggregated critic scores)
    # ------------------------------------------------------------------
    if OPENCRITIC_PATH.exists():
        oc = pd.read_csv(OPENCRITIC_PATH)
        print(f"[merge-v3] OpenCritic: {len(oc):,} rows")

        oc_lookup = _build_lookup(oc["oc_name"])
        oc_matches = _fuzzy_match_col(
            vg["Name"], oc_lookup, threshold=match_threshold, label="opencritic"
        )

        oc_cols = [
            "oc_id", "oc_top_critic_score", "oc_percent_recommended",
            "oc_num_reviews", "oc_num_top_critic_reviews", "oc_tier",
            "oc_first_release_date",
        ]
        for col in oc_cols:
            vg[col] = float("nan") if col not in ("oc_tier", "oc_first_release_date") else ""
        vg["oc_match_score"] = float("nan")

        for vg_idx, (oc_idx, score) in oc_matches.items():
            for col in oc_cols:
                if col in oc.columns:
                    vg.at[vg_idx, col] = oc.at[oc_idx, col]
            vg.at[vg_idx, "oc_match_score"] = score

        print(f"[merge-v3] OpenCritic: {len(oc_matches):,} total matches")
    else:
        print("[merge-v3] OpenCritic data not found — skipping")

    # ------------------------------------------------------------------
    # 9. Merge Gamedatacrunch (revenue estimates, CCU, pricing)
    # ------------------------------------------------------------------
    if GAMEDATACRUNCH_PATH.exists():
        gdc = pd.read_csv(GAMEDATACRUNCH_PATH)
        print(f"[merge-v3] Gamedatacrunch: {len(gdc):,} rows")

        gdc_lookup = _build_lookup(gdc["gdc_name"])
        gdc_matches = _fuzzy_match_col(
            vg["Name"], gdc_lookup, threshold=match_threshold, label="gdc"
        )

        gdc_cols = [
            "gdc_appid", "gdc_revenue_estimate", "gdc_owners_estimate",
            "gdc_ccu_max", "gdc_price_usd", "gdc_review_score",
            "gdc_review_count", "gdc_release_date",
            "gdc_developer", "gdc_publisher", "gdc_tags", "gdc_genres",
        ]
        for col in gdc_cols:
            if col in ("gdc_developer", "gdc_publisher", "gdc_tags", "gdc_genres", "gdc_release_date"):
                vg[col] = ""
            else:
                vg[col] = float("nan")
        vg["gdc_match_score"] = float("nan")

        for vg_idx, (gdc_idx, score) in gdc_matches.items():
            for col in gdc_cols:
                if col in gdc.columns:
                    vg.at[vg_idx, col] = gdc.at[gdc_idx, col]
            vg.at[vg_idx, "gdc_match_score"] = score

        print(f"[merge-v3] Gamedatacrunch: {len(gdc_matches):,} total matches")
    else:
        print("[merge-v3] Gamedatacrunch data not found — skipping")

    # ------------------------------------------------------------------
    # 10. Merge VGChartz scores enrichment
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
    # 11. Derive additional columns
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
    # 12. Derive columns from new sources
    # ------------------------------------------------------------------
    # Verified digital sales from Wikipedia (highest reliability tier)
    if "wiki_sales_millions" in vg.columns:
        vg["has_verified_sales"] = vg["wiki_sales_millions"].notna().astype(int)

    # Revenue estimate from Gamedatacrunch
    if "gdc_revenue_estimate" in vg.columns:
        vg["has_revenue_estimate"] = vg["gdc_revenue_estimate"].notna().astype(int)

    # Critic consensus score: prefer OpenCritic, fallback to Steam Store metacritic
    if "oc_top_critic_score" in vg.columns:
        vg["critic_score_combined"] = vg["oc_top_critic_score"]
        if "steam_store_metacritic" in vg.columns:
            mask = vg["critic_score_combined"].isna()
            vg.loc[mask, "critic_score_combined"] = vg.loc[mask, "steam_store_metacritic"]

    # DLC availability from Steam Store
    if "steam_store_dlc_count" in vg.columns:
        vg["has_dlc"] = (
            pd.to_numeric(vg["steam_store_dlc_count"], errors="coerce").fillna(0) > 0
        ).astype(int)

    # ------------------------------------------------------------------
    # 13. Clean up and save
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
        ("Wikipedia", "wiki_match_score"),
        ("Steam Store", "steam_store_match_score"),
        ("OpenCritic", "oc_match_score"),
        ("Gamedatacrunch", "gdc_match_score"),
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
