"""Build a clean, quality-filtered dataset from the merged v3 data.

Applies quality tiers, filters out zero-sales rows, and produces a
training-ready dataset with documented provenance.

Two target columns are built:
- Global_Sales: VGChartz physical sales only (unchanged)
- estimated_total_sales: composite (Wikipedia > VGChartz > review-multiplier)

Input:  data/Ventes_jeux_video_v3.csv  (raw merge, ~64K rows)
        data/raw/estimated_sales.csv   (review-multiplier estimates, optional)
Output: data/Ventes_jeux_video_clean.csv  (quality-filtered)
        reports/data_quality_report.json  (audit trail)
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

INPUT_PATH = DATA_DIR / "Ventes_jeux_video_v3.csv"
ESTIMATES_PATH = RAW_DIR / "estimated_sales.csv"
OUTPUT_PATH = DATA_DIR / "Ventes_jeux_video_clean.csv"
REPORT_PATH = REPORTS_DIR / "data_quality_report.json"


def assign_quality_tier(row: pd.Series) -> str:
    """Assign a data quality tier based on available sales evidence.

    Tiers (best to worst):
        tier_1_verified   — Wikipedia official + VGChartz (highest confidence)
        tier_2_physical   — VGChartz >= 0.1M
        tier_3_estimated  — Review-multiplier calibrated estimate (medium confidence)
        tier_4_marginal   — VGChartz > 0 but < 0.1M
        tier_5_digital_proxy — SteamSpy owners > 100K, no calibrated estimate
        tier_6_no_sales   — No sales evidence at all
    """
    has_vgchartz = row.get("Global_Sales", 0) > 0
    has_wiki = pd.notna(row.get("wiki_sales_millions"))
    has_review_est = pd.notna(row.get("review_estimated_sales"))
    steam_owners = row.get("steam_owners_midpoint", 0)
    has_steam = pd.notna(steam_owners) and steam_owners > 0

    if has_wiki and has_vgchartz:
        return "tier_1_verified"
    if has_vgchartz and row["Global_Sales"] >= 0.1:
        return "tier_2_physical"
    if has_review_est:
        return "tier_3_estimated"
    if has_vgchartz and row["Global_Sales"] > 0:
        return "tier_4_marginal"
    if has_steam and steam_owners > 100_000:
        return "tier_5_digital_proxy"
    return "tier_6_no_sales"


def compute_sales_estimate(row: pd.Series) -> tuple[float, str, str]:
    """Compute best available sales estimate with provenance and confidence.

    Returns (estimate_in_millions, provenance_string, confidence).
    """
    wiki = row.get("wiki_sales_millions")
    if pd.notna(wiki) and wiki > 0:
        return float(wiki), "wikipedia_verified", "high"

    vgchartz = row.get("Global_Sales", 0)
    if vgchartz >= 0.1:
        return float(vgchartz), "vgchartz_physical", "high"

    review_est = row.get("review_estimated_sales")
    if pd.notna(review_est) and review_est > 0:
        return float(review_est), "review_multiplier", "medium"

    if vgchartz > 0:
        return float(vgchartz), "vgchartz_marginal", "medium"

    return 0.0, "no_sales_data", "none"


def _merge_review_estimates(df: pd.DataFrame) -> pd.DataFrame:
    """Merge review-multiplier sales estimates into the dataset."""
    if not ESTIMATES_PATH.exists():
        print("[clean] No estimated_sales.csv found — skipping review estimates")
        df["review_estimated_sales"] = np.nan
        return df

    estimates = pd.read_csv(ESTIMATES_PATH)
    print(f"[clean] Loaded {len(estimates):,} review-multiplier estimates")

    # Merge via Name + Platform (same key as v3 merge)
    est_cols = ["Name", "Platform", "review_estimated_sales"]
    est_available = [c for c in est_cols if c in estimates.columns]
    if "review_estimated_sales" not in est_available:
        print("[clean] WARNING: estimated_sales.csv missing review_estimated_sales column")
        df["review_estimated_sales"] = np.nan
        return df

    df = df.merge(
        estimates[est_available],
        on=["Name", "Platform"],
        how="left",
        suffixes=("", "_est"),
    )

    matched = df["review_estimated_sales"].notna().sum()
    print(f"[clean] Matched {matched:,} games with review-multiplier estimates")

    return df


def build_clean_dataset(
    min_tier: str = "tier_4_marginal",
    force: bool = False,
) -> Path:
    """Build a quality-filtered dataset with two target columns.

    Builds both Global_Sales (VGChartz physical) and estimated_total_sales
    (composite: Wikipedia > VGChartz > review-multiplier).

    Parameters
    ----------
    min_tier:
        Minimum quality tier to include. Default keeps tiers 1-4
        (VGChartz sales + review-multiplier estimates).
    force:
        Overwrite output even if it exists.
    """
    REPORTS_DIR.mkdir(exist_ok=True)

    if OUTPUT_PATH.exists() and not force:
        print(f"[clean] Already exists: {OUTPUT_PATH} (use --force)")
        return OUTPUT_PATH

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Merged v3 dataset not found at {INPUT_PATH}")

    # Load
    df = pd.read_csv(INPUT_PATH)
    original_count = len(df)
    print(f"[clean] Loaded {original_count:,} rows, {len(df.columns)} columns")

    # ── Step 1: Merge review-multiplier estimates ──
    df = _merge_review_estimates(df)

    # ── Step 2: Assign quality tiers ──
    df["quality_tier"] = df.apply(assign_quality_tier, axis=1)
    tier_counts = df["quality_tier"].value_counts().sort_index()
    print("\n[clean] Quality tier distribution:")
    for tier, count in tier_counts.items():
        print(f"  {tier}: {count:,} ({count/len(df)*100:.1f}%)")

    # ── Step 3: Build sales estimates + provenance + confidence ──
    results = df.apply(compute_sales_estimate, axis=1, result_type="expand")
    df["estimated_total_sales"] = results[0]
    df["sales_provenance"] = results[1]
    df["sales_confidence"] = results[2]

    provenance_counts = df["sales_provenance"].value_counts()
    print("\n[clean] Sales provenance:")
    for prov, count in provenance_counts.items():
        print(f"  {prov}: {count:,}")

    # ── Step 4: Add binary flags ──
    df["has_verified_sales"] = (
        df["wiki_sales_millions"].notna() & (df["wiki_sales_millions"] > 0)
    ).astype(int) if "wiki_sales_millions" in df.columns else 0

    # ── Step 5: Basic validation filters ──
    before_filter = len(df)
    df = df[df["Year"].notna()]
    df = df[df["Year"].between(1980, 2026)]
    df = df[df["Publisher"].notna()]
    df = df[df["Genre"].notna()]
    print(f"\n[clean] Validation filters: {before_filter:,} → {len(df):,} rows")

    # ── Step 6: Quality filter ──
    tier_order = [
        "tier_1_verified", "tier_2_physical", "tier_3_estimated",
        "tier_4_marginal", "tier_5_digital_proxy", "tier_6_no_sales",
    ]
    min_idx = tier_order.index(min_tier)
    allowed_tiers = tier_order[:min_idx + 1]

    df_clean = df[df["quality_tier"].isin(allowed_tiers)].copy()
    print(f"[clean] Quality filter (>= {min_tier}): {len(df):,} → {len(df_clean):,} rows")

    # ── Step 7: Feature coverage report ──
    coverage = {}
    coverage_cols = {
        "SteamSpy": "steam_owners_midpoint",
        "RAWG": "rawg_rating",
        "IGDB": "igdb_total_rating",
        "HLTB": "hltb_main",
        "OpenCritic": "oc_top_critic_score",
        "Wikipedia": "wiki_sales_millions",
        "Steam Store": "steam_store_price_usd",
        "Review Estimates": "review_estimated_sales",
        "meta_score": "meta_score",
    }
    print("\n[clean] Feature coverage in filtered dataset:")
    for name, col in coverage_cols.items():
        if col in df_clean.columns:
            valid = df_clean[col].notna().sum()
            if col in ("steam_owners_midpoint", "hltb_main", "oc_top_critic_score"):
                valid = (pd.to_numeric(df_clean[col], errors="coerce") > 0).sum()
            pct = valid / len(df_clean) * 100 if len(df_clean) > 0 else 0
            coverage[name] = {"count": int(valid), "pct": round(pct, 1)}
            print(f"  {name}: {valid:,} ({pct:.1f}%)")

    # ── Step 8: Sales distribution — both targets ──
    gs = df_clean["Global_Sales"]
    print("\n[clean] Global_Sales (VGChartz physical):")
    print(f"  Non-zero: {(gs > 0).sum():,} / {len(df_clean):,}")
    if (gs > 0).sum() > 0:
        gs_nz = gs[gs > 0]
        print(f"  Mean: {gs_nz.mean():.3f}M | Median: {gs_nz.median():.3f}M | Max: {gs_nz.max():.1f}M")

    ets = df_clean["estimated_total_sales"]
    print("\n[clean] estimated_total_sales (composite):")
    print(f"  Non-zero: {(ets > 0).sum():,} / {len(df_clean):,}")
    if (ets > 0).sum() > 0:
        ets_nz = ets[ets > 0]
        print(f"  Mean: {ets_nz.mean():.3f}M | Median: {ets_nz.median():.3f}M | Max: {ets_nz.max():.1f}M")

    confidence_counts = df_clean["sales_confidence"].value_counts()
    print("\n[clean] Sales confidence distribution:")
    for conf, count in confidence_counts.items():
        print(f"  {conf}: {count:,}")

    # ── Step 9: Save ──
    df_clean = df_clean.sort_values("estimated_total_sales", ascending=False).reset_index(drop=True)
    df_clean["Rank"] = range(1, len(df_clean) + 1)

    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[clean] Saved {len(df_clean):,} rows → {OUTPUT_PATH}")

    # ── Step 10: Quality report ──
    report = {
        "timestamp": datetime.now().isoformat(),
        "input_file": str(INPUT_PATH.name),
        "input_rows": original_count,
        "output_rows": len(df_clean),
        "min_quality_tier": min_tier,
        "tier_distribution": {
            tier: int(tier_counts.get(tier, 0)) for tier in tier_order
        },
        "filtered_tier_distribution": df_clean["quality_tier"].value_counts().to_dict(),
        "provenance_distribution": df_clean["sales_provenance"].value_counts().to_dict(),
        "confidence_distribution": confidence_counts.to_dict(),
        "global_sales_stats": {
            "non_zero_count": int((gs > 0).sum()),
            "mean": round(float(gs[gs > 0].mean()), 4) if (gs > 0).sum() > 0 else 0,
            "median": round(float(gs[gs > 0].median()), 4) if (gs > 0).sum() > 0 else 0,
        },
        "estimated_total_sales_stats": {
            "non_zero_count": int((ets > 0).sum()),
            "mean": round(float(ets[ets > 0].mean()), 4) if (ets > 0).sum() > 0 else 0,
            "median": round(float(ets[ets > 0].median()), 4) if (ets > 0).sum() > 0 else 0,
        },
        "feature_coverage": coverage,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[clean] Quality report → {REPORT_PATH}")

    return OUTPUT_PATH


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build clean dataset from v3 merge")
    parser.add_argument(
        "--min-tier", default="tier_4_marginal",
        choices=[
            "tier_1_verified", "tier_2_physical", "tier_3_estimated",
            "tier_4_marginal", "tier_5_digital_proxy", "tier_6_no_sales",
        ],
        help="Minimum quality tier to include (default: tier_4_marginal)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    build_clean_dataset(min_tier=args.min_tier, force=args.force)
