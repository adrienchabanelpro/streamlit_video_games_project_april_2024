"""Estimate digital sales using the review-multiplier method (Boxleiter).

The core idea: `Estimated Sales = Total Reviews × Genre Multiplier`.
Industry average is ~50-63 sales per review, but varies by genre.

Calibration:
1. Find games with BOTH Wikipedia verified sales AND Steam reviews
2. Compute empirical multiplier per game
3. Average by Genre (trim outliers at 10th/90th percentile)
4. Apply genre-specific multipliers to all games with review data
5. Cross-validate against known VGChartz sales

Input:  data/Ventes_jeux_video_v3.csv (merged dataset)
        data/raw/steam_reviews_summary.csv (optional, authoritative reviews)
Output: data/raw/estimated_sales.csv
        reports/sales_estimation_report.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

V3_PATH = DATA_DIR / "Ventes_jeux_video_v3.csv"
STEAM_REVIEWS_PATH = RAW_DIR / "steam_reviews_summary.csv"
OUTPUT_PATH = RAW_DIR / "estimated_sales.csv"
REPORT_PATH = REPORTS_DIR / "sales_estimation_report.json"

# Fallback multiplier when genre has too few calibration samples
DEFAULT_MULTIPLIER = 55.0
MIN_REVIEWS_FOR_ESTIMATE = 10
MIN_CALIBRATION_SAMPLES = 5


def _load_and_enrich_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Merge authoritative Steam review counts if available.

    If steam_reviews_summary.csv exists, use those (more accurate).
    Otherwise fall back to SteamSpy review counts already in v3.
    """
    df = df.copy()

    # Compute total reviews from SteamSpy data already in v3
    if "steam_positive" in df.columns and "steam_negative" in df.columns:
        df["total_reviews_steamspy"] = (
            pd.to_numeric(df["steam_positive"], errors="coerce").fillna(0)
            + pd.to_numeric(df["steam_negative"], errors="coerce").fillna(0)
        )
    else:
        df["total_reviews_steamspy"] = 0

    # Merge authoritative Steam Reviews if collected
    if STEAM_REVIEWS_PATH.exists():
        reviews = pd.read_csv(STEAM_REVIEWS_PATH)
        print(f"[estimate] Loaded {len(reviews):,} authoritative Steam review summaries")

        # Match via appid
        appid_col = "steam_appid" if "steam_appid" in df.columns else "steam_store_appid"
        if appid_col in df.columns:
            df[appid_col] = pd.to_numeric(df[appid_col], errors="coerce")
            reviews["appid"] = pd.to_numeric(reviews["appid"], errors="coerce")

            reviews_lookup = reviews.set_index("appid")["review_total"].to_dict()
            df["total_reviews_api"] = df[appid_col].map(reviews_lookup)
            api_count = df["total_reviews_api"].notna().sum()
            print(f"[estimate] Matched {api_count:,} games via appid to authoritative reviews")
        else:
            df["total_reviews_api"] = np.nan
    else:
        print("[estimate] No steam_reviews_summary.csv found, using SteamSpy reviews only")
        df["total_reviews_api"] = np.nan

    # Use authoritative API reviews where available, fall back to SteamSpy
    df["total_reviews"] = df["total_reviews_api"].fillna(df["total_reviews_steamspy"])

    return df


def calibrate_multipliers(df: pd.DataFrame) -> tuple[dict[str, float], dict]:
    """Calibrate genre-specific review multipliers using Wikipedia verified sales.

    Returns (genre_multipliers, calibration_stats).
    """
    # Find calibration set: games with Wikipedia sales AND review counts
    mask = (
        df["wiki_sales_millions"].notna()
        & (df["wiki_sales_millions"] > 0)
        & (df["total_reviews"] >= MIN_REVIEWS_FOR_ESTIMATE)
        & df["Genre"].notna()
    )
    cal = df[mask].copy()
    print(f"[estimate] Calibration set: {len(cal):,} games with Wikipedia sales + reviews")

    if len(cal) < MIN_CALIBRATION_SAMPLES:
        print(f"[estimate] WARNING: Too few calibration samples ({len(cal)}), using default multiplier")
        return {}, {"calibration_size": len(cal), "method": "default_only"}

    # Compute empirical multiplier per game
    cal["empirical_multiplier"] = (
        cal["wiki_sales_millions"] * 1_000_000 / cal["total_reviews"]
    )

    # Global stats (before genre split)
    global_median = cal["empirical_multiplier"].median()
    global_mean = cal["empirical_multiplier"].mean()
    global_p10 = cal["empirical_multiplier"].quantile(0.1)
    global_p90 = cal["empirical_multiplier"].quantile(0.9)

    print(f"[estimate] Global multiplier — median: {global_median:.1f}, mean: {global_mean:.1f}")
    print(f"[estimate] Range (P10-P90): {global_p10:.1f} - {global_p90:.1f}")

    # Genre-specific multipliers (trimmed median)
    genre_multipliers: dict[str, float] = {}
    genre_stats: dict[str, dict] = {}

    for genre, group in cal.groupby("Genre"):
        if len(group) < MIN_CALIBRATION_SAMPLES:
            continue

        # Trim outliers at 10th/90th percentile
        p10 = group["empirical_multiplier"].quantile(0.1)
        p90 = group["empirical_multiplier"].quantile(0.9)
        trimmed = group["empirical_multiplier"].clip(lower=p10, upper=p90)

        multiplier = trimmed.median()
        genre_multipliers[str(genre)] = round(float(multiplier), 1)
        genre_stats[str(genre)] = {
            "samples": len(group),
            "multiplier": round(float(multiplier), 1),
            "median_raw": round(float(group["empirical_multiplier"].median()), 1),
            "p10": round(float(p10), 1),
            "p90": round(float(p90), 1),
        }

    print(f"[estimate] Calibrated {len(genre_multipliers)} genre-specific multipliers:")
    for genre, mult in sorted(genre_multipliers.items()):
        n = genre_stats[genre]["samples"]
        print(f"  {genre}: {mult}x ({n} samples)")

    calibration_report = {
        "calibration_size": len(cal),
        "method": "genre_trimmed_median",
        "global_median_multiplier": round(float(global_median), 1),
        "global_mean_multiplier": round(float(global_mean), 1),
        "default_fallback": DEFAULT_MULTIPLIER,
        "genre_stats": genre_stats,
    }

    return genre_multipliers, calibration_report


def apply_estimates(
    df: pd.DataFrame,
    genre_multipliers: dict[str, float],
) -> pd.DataFrame:
    """Apply review-multiplier estimates to all games with review data."""
    df = df.copy()

    # Get the multiplier for each game's genre
    df["review_multiplier"] = df["Genre"].map(genre_multipliers)

    # Fill missing genres with global median of calibrated multipliers
    if genre_multipliers:
        global_fallback = float(np.median(list(genre_multipliers.values())))
    else:
        global_fallback = DEFAULT_MULTIPLIER
    df["review_multiplier"] = df["review_multiplier"].fillna(global_fallback)

    # Compute estimate: total_reviews * multiplier / 1,000,000 (to get millions)
    mask = df["total_reviews"] >= MIN_REVIEWS_FOR_ESTIMATE
    df["review_estimated_sales"] = np.where(
        mask,
        df["total_reviews"] * df["review_multiplier"] / 1_000_000,
        np.nan,
    )

    estimated_count = df["review_estimated_sales"].notna().sum()
    print(f"[estimate] Generated estimates for {estimated_count:,} games")

    if estimated_count > 0:
        valid = df["review_estimated_sales"].dropna()
        print(f"[estimate] Estimated sales — median: {valid.median():.3f}M, "
              f"mean: {valid.mean():.3f}M, max: {valid.max():.1f}M")

    return df


def cross_validate(df: pd.DataFrame) -> dict:
    """Compare review-multiplier estimates against known VGChartz sales."""
    mask = (
        (df["Global_Sales"] > 0)
        & df["review_estimated_sales"].notna()
    )
    overlap = df[mask].copy()

    if len(overlap) < 10:
        print(f"[estimate] Cross-validation: too few overlapping games ({len(overlap)})")
        return {"overlap_count": len(overlap)}

    overlap["error"] = overlap["review_estimated_sales"] - overlap["Global_Sales"]
    overlap["abs_error"] = overlap["error"].abs()
    overlap["pct_error"] = (
        overlap["abs_error"] / overlap["Global_Sales"] * 100
    )

    mae = overlap["abs_error"].mean()
    median_ae = overlap["abs_error"].median()
    median_pct = overlap["pct_error"].median()
    correlation = overlap[["review_estimated_sales", "Global_Sales"]].corr().iloc[0, 1]

    print(f"\n[estimate] Cross-validation ({len(overlap):,} games with both VGChartz + estimate):")
    print(f"  MAE: {mae:.3f}M")
    print(f"  Median AE: {median_ae:.3f}M")
    print(f"  Median % error: {median_pct:.1f}%")
    print(f"  Correlation: {correlation:.3f}")

    # Spot-check: top 10 games by VGChartz sales
    top = overlap.nlargest(10, "Global_Sales")[
        ["Name", "Genre", "Global_Sales", "review_estimated_sales", "total_reviews"]
    ]
    print("\n[estimate] Top 10 games — VGChartz vs estimate:")
    for _, row in top.iterrows():
        name = str(row["Name"])[:35]
        print(
            f"  {name:<35} VGC={row['Global_Sales']:.2f}M  "
            f"Est={row['review_estimated_sales']:.2f}M  "
            f"Reviews={int(row['total_reviews']):,}"
        )

    return {
        "overlap_count": len(overlap),
        "mae_millions": round(float(mae), 4),
        "median_ae_millions": round(float(median_ae), 4),
        "median_pct_error": round(float(median_pct), 1),
        "correlation": round(float(correlation), 4),
    }


def estimate_sales(force: bool = False) -> Path:
    """Run the full sales estimation pipeline.

    Parameters
    ----------
    force:
        Overwrite output even if it exists.

    Returns
    -------
    Path to the output CSV.
    """
    REPORTS_DIR.mkdir(exist_ok=True)

    if OUTPUT_PATH.exists() and not force:
        print(f"[estimate] Already exists: {OUTPUT_PATH} (use --force)")
        return OUTPUT_PATH

    if not V3_PATH.exists():
        raise FileNotFoundError(
            f"Merged v3 dataset not found at {V3_PATH}\n"
            "Run the data collection pipeline first."
        )

    # Load v3 merged dataset
    df = pd.read_csv(V3_PATH)
    print(f"[estimate] Loaded {len(df):,} rows from {V3_PATH.name}")

    # Step 1: Enrich with authoritative review counts
    df = _load_and_enrich_reviews(df)

    has_reviews = (df["total_reviews"] >= MIN_REVIEWS_FOR_ESTIMATE).sum()
    print(f"[estimate] {has_reviews:,} games have >= {MIN_REVIEWS_FOR_ESTIMATE} reviews")

    # Step 2: Calibrate genre multipliers
    genre_multipliers, calibration_report = calibrate_multipliers(df)

    # Step 3: Apply estimates
    df = apply_estimates(df, genre_multipliers)

    # Step 4: Cross-validate
    cv_report = cross_validate(df)

    # Step 5: Save output — only columns needed for merge
    output_cols = [
        "Name", "Platform",
        "total_reviews", "review_multiplier", "review_estimated_sales",
    ]
    # Add appid if available for alternative merging
    for col in ["steam_appid", "steam_store_appid"]:
        if col in df.columns:
            output_cols.append(col)

    df_out = df[output_cols].dropna(subset=["review_estimated_sales"])
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[estimate] Saved {len(df_out):,} estimates → {OUTPUT_PATH}")

    # Step 6: Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "input_rows": len(df),
        "games_with_reviews": int(has_reviews),
        "estimates_generated": len(df_out),
        "calibration": calibration_report,
        "cross_validation": cv_report,
        "genre_multipliers": genre_multipliers,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[estimate] Report → {REPORT_PATH}")

    return OUTPUT_PATH


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Estimate sales via review-multiplier method")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    estimate_sales(force=args.force)
