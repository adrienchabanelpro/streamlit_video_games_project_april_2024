"""Step 1: Download VGChartz 2024 dataset (64K rows) from Kaggle.

Uses kagglehub to download the dataset and maps columns to the app schema.
Supports manual CSV fallback when Kaggle auth is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = RAW_DIR / "vgchartz_2024.csv"

KAGGLE_DATASET = "asaniczka/video-game-sales-2024"

# Maps Kaggle columns → app schema columns
COLUMN_MAP: dict[str, str] = {
    "title": "Name",
    "console": "Platform",
    "genre": "Genre",
    "publisher": "Publisher",
    "developer": "developer",
    "na_sales": "NA_Sales",
    "jp_sales": "JP_Sales",
    "pal_sales": "EU_Sales",
    "other_sales": "Other_Sales",
    "total_sales": "Global_Sales",
    "critic_score": "meta_score",
    "release_date": "release_date",
}

# Normalize console names from Kaggle to short platform codes
PLATFORM_MAP: dict[str, str] = {
    "playstation 5": "PS5",
    "playstation 4": "PS4",
    "playstation 3": "PS3",
    "playstation 2": "PS2",
    "playstation": "PS",
    "playstation vita": "PSV",
    "playstation portable": "PSP",
    "xbox series x": "XSX",
    "xbox one": "XOne",
    "xbox 360": "X360",
    "xbox": "XB",
    "nintendo switch": "NS",
    "wii u": "WiiU",
    "wii": "Wii",
    "nintendo 3ds": "3DS",
    "nintendo ds": "DS",
    "game boy advance": "GBA",
    "gamecube": "GC",
    "nintendo 64": "N64",
    "super nintendo": "SNES",
    "nintendo entertainment system": "NES",
    "sega genesis": "GEN",
    "sega dreamcast": "DC",
    "sega saturn": "SAT",
    "pc": "PC",
    "stadia": "Stadia",
}


def normalize_platform(name: str) -> str:
    """Map verbose platform name to short code."""
    return PLATFORM_MAP.get(name.strip().lower(), name.strip())


def extract_year(date_str: str) -> float | None:
    """Extract year from a release_date string (various formats)."""
    if pd.isna(date_str) or not str(date_str).strip():
        return None
    try:
        dt = pd.to_datetime(str(date_str), errors="coerce")
        if pd.notna(dt):
            return float(dt.year)
    except Exception:
        pass
    # Try extracting a 4-digit year directly
    import re

    match = re.search(r"((?:19|20)\d{2})", str(date_str))
    if match:
        return float(match.group(1))
    return None


def download_kaggle(force: bool = False) -> Path:
    """Download the VGChartz 2024 dataset from Kaggle.

    Parameters
    ----------
    force:
        Re-download even if the output file already exists.

    Returns
    -------
    Path
        Path to the downloaded/mapped CSV.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not force:
        print(f"[kaggle] Already exists: {OUTPUT_PATH} (use --force to re-download)")
        return OUTPUT_PATH

    print("[kaggle] Downloading dataset from Kaggle...")
    try:
        import kagglehub

        dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)
        dataset_path = Path(dataset_path)
    except Exception as exc:
        print(f"\n[kaggle] ERROR: Kaggle download failed: {exc}")
        print(
            "\n  To set up Kaggle API authentication:\n"
            "  1. Go to https://www.kaggle.com/settings\n"
            "  2. Scroll to 'API' → click 'Create New API Token'\n"
            "  3. Run: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ "
            "&& chmod 600 ~/.kaggle/kaggle.json\n"
            "\n  Or manually download the CSV from:\n"
            "  https://www.kaggle.com/datasets/asaniczka/video-game-sales-2024\n"
            f"  and place it at: {OUTPUT_PATH}\n"
        )
        raise

    # Find the CSV in the downloaded directory
    csv_files = list(dataset_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in downloaded dataset: {dataset_path}")

    source_csv = csv_files[0]
    print(f"[kaggle] Found: {source_csv.name}")

    # Map columns to app schema
    df = _map_kaggle_to_schema(source_csv)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[kaggle] Saved {len(df):,} rows → {OUTPUT_PATH}")
    return OUTPUT_PATH


def load_manual_csv(csv_path: Path, force: bool = False) -> Path:
    """Fallback: load a manually-downloaded Kaggle CSV and map it.

    Parameters
    ----------
    csv_path:
        Path to the manually downloaded CSV file.
    force:
        Overwrite output even if it exists.

    Returns
    -------
    Path
        Path to the mapped CSV.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not force:
        print(f"[kaggle] Already exists: {OUTPUT_PATH}")
        return OUTPUT_PATH

    if not csv_path.exists():
        raise FileNotFoundError(f"Manual CSV not found: {csv_path}")

    df = _map_kaggle_to_schema(csv_path)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[kaggle] Saved {len(df):,} rows → {OUTPUT_PATH}")
    return OUTPUT_PATH


def _map_kaggle_to_schema(csv_path: Path) -> pd.DataFrame:
    """Read a Kaggle CSV and map it to the app's column schema."""
    df = pd.read_csv(csv_path)
    print(f"[kaggle] Read {len(df):,} rows, {len(df.columns)} columns")

    # Rename columns
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Normalize platform names
    if "Platform" in df.columns:
        df["Platform"] = df["Platform"].astype(str).map(normalize_platform)

    # Extract Year from release_date
    if "release_date" in df.columns:
        df["Year"] = df["release_date"].apply(extract_year)

    # Ensure sales columns are numeric
    sales_cols = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
    for col in sales_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Ensure meta_score is numeric
    if "meta_score" in df.columns:
        df["meta_score"] = pd.to_numeric(df["meta_score"], errors="coerce")

    # Add placeholder for user_review (not in Kaggle dataset)
    if "user_review" not in df.columns:
        df["user_review"] = float("nan")

    # Compute Rank by Global_Sales descending
    df = df.sort_values("Global_Sales", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download VGChartz 2024 from Kaggle")
    parser.add_argument("--force", action="store_true", help="Re-download even if exists")
    parser.add_argument("--manual-csv", type=Path, default=None, help="Path to manually downloaded CSV")
    args = parser.parse_args()

    if args.manual_csv:
        load_manual_csv(args.manual_csv, force=args.force)
    else:
        download_kaggle(force=args.force)
