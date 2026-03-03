"""Collect verified game sales data from Wikipedia bestseller lists.

Scrapes structured tables from Wikipedia pages listing best-selling games:
- Best-selling video games of all time
- Best-selling by platform (PS4, Switch, Xbox One, etc.)

These are verified official figures — highest reliability tier.
Output: data/raw/wikipedia_sales.csv
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = RAW_DIR / "wikipedia_sales.csv"

# Wikipedia API endpoint for parsing HTML tables
WIKI_API = "https://en.wikipedia.org/w/api.php"

# Pages to scrape for sales data
WIKI_PAGES = [
    "List_of_best-selling_video_games",
    "List_of_best-selling_Nintendo_Switch_video_games",
    "List_of_best-selling_PlayStation_4_video_games",
    "List_of_best-selling_PlayStation_5_video_games",
    "List_of_best-selling_Xbox_One_video_games",
    "List_of_best-selling_Wii_video_games",
    "List_of_best-selling_Wii_U_video_games",
    "List_of_best-selling_Game_Boy_video_games",
    "List_of_best-selling_Nintendo_DS_video_games",
    "List_of_best-selling_Nintendo_3DS_video_games",
    "List_of_best-selling_PlayStation_3_video_games",
    "List_of_best-selling_PlayStation_2_video_games",
    "List_of_best-selling_Xbox_360_video_games",
]

RATE_LIMIT_SECONDS = 1.0


def _fetch_wiki_tables(page_title: str) -> list[pd.DataFrame]:
    """Fetch all HTML tables from a Wikipedia page via the API."""
    params = {
        "action": "parse",
        "page": page_title,
        "prop": "wikitext",
        "format": "json",
    }
    try:
        resp = requests.get(WIKI_API, params=params, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[wikipedia] ERROR fetching {page_title}: {exc}")
        return []

    # Use the simpler HTML table parsing approach
    params_html = {
        "action": "parse",
        "page": page_title,
        "prop": "text",
        "format": "json",
    }
    try:
        resp = requests.get(WIKI_API, params=params_html, timeout=30)
        resp.raise_for_status()
        html = resp.json().get("parse", {}).get("text", {}).get("*", "")
    except Exception as exc:
        print(f"[wikipedia] ERROR parsing {page_title}: {exc}")
        return []

    try:
        tables = pd.read_html(html, flavor="lxml")
    except Exception:
        try:
            tables = pd.read_html(html)
        except Exception as exc:
            print(f"[wikipedia] No tables found in {page_title}: {exc}")
            return []

    return tables


def _extract_sales_number(value: str) -> float | None:
    """Extract a numeric sales figure from a Wikipedia cell.

    Handles formats like: '30 million', '30,000,000', '30M', '30.5 million'.
    Returns sales in millions.
    """
    if not isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    value = value.strip().replace(",", "").replace("[", "").replace("]", "")
    # Remove reference markers like [1], [a], etc.
    value = re.sub(r"\[.*?\]", "", value)

    # Match "X million" or "X Million"
    match = re.search(r"([\d.]+)\s*million", value, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # Match plain number (assume it's already in millions if > 100, else raw units)
    match = re.search(r"([\d.]+)", value)
    if match:
        num = float(match.group(1))
        if num > 1_000_000:
            return num / 1_000_000
        return num

    return None


def _infer_platform_from_page(page_title: str) -> str | None:
    """Infer the platform from the Wikipedia page title."""
    platform_map = {
        "Nintendo_Switch": "NS",
        "PlayStation_4": "PS4",
        "PlayStation_5": "PS5",
        "Xbox_One": "XOne",
        "Wii_video": "Wii",
        "Wii_U": "WiiU",
        "Game_Boy": "GB",
        "Nintendo_DS": "DS",
        "Nintendo_3DS": "3DS",
        "PlayStation_3": "PS3",
        "PlayStation_2": "PS2",
        "Xbox_360": "X360",
    }
    for key, platform in platform_map.items():
        if key in page_title:
            return platform
    return None  # Multi-platform (all-time list)


def _find_sales_table(tables: list[pd.DataFrame], page_title: str) -> pd.DataFrame | None:
    """Find the table most likely containing sales data."""
    for df in tables:
        cols_lower = [str(c).lower() for c in df.columns]
        # Look for tables with sales-related columns
        has_sales = any(
            keyword in col
            for col in cols_lower
            for keyword in ["copies", "sales", "units", "sold"]
        )
        has_game = any(
            keyword in col
            for col in cols_lower
            for keyword in ["game", "title", "name"]
        )
        if has_sales and has_game and len(df) >= 5:
            return df

    # Fallback: return the largest table
    if tables:
        largest = max(tables, key=len)
        if len(largest) >= 5:
            return largest

    return None


def _process_table(
    df: pd.DataFrame, page_title: str
) -> list[dict]:
    """Extract game sales records from a Wikipedia table."""
    records = []
    platform = _infer_platform_from_page(page_title)

    cols_lower = {str(c).lower(): c for c in df.columns}

    # Find the game name column
    name_col = None
    for keyword in ["game", "title", "name"]:
        for col_lower, col_orig in cols_lower.items():
            if keyword in col_lower:
                name_col = col_orig
                break
        if name_col:
            break

    # Find the sales column
    sales_col = None
    for keyword in ["copies sold", "sales", "units sold", "copies"]:
        for col_lower, col_orig in cols_lower.items():
            if keyword in col_lower:
                sales_col = col_orig
                break
        if sales_col:
            break

    if not name_col or not sales_col:
        return records

    # Find optional columns
    publisher_col = None
    developer_col = None
    date_col = None
    platform_col = None

    for col_lower, col_orig in cols_lower.items():
        if "publisher" in col_lower:
            publisher_col = col_orig
        elif "developer" in col_lower:
            developer_col = col_orig
        elif "release" in col_lower or "date" in col_lower or "year" in col_lower:
            date_col = col_orig
        elif "platform" in col_lower:
            platform_col = col_orig

    for _, row in df.iterrows():
        name = str(row.get(name_col, "")).strip()
        if not name or name.lower() in ("nan", "", "total"):
            continue

        sales = _extract_sales_number(str(row.get(sales_col, "")))
        if sales is None or sales <= 0:
            continue

        record = {
            "wiki_name": name,
            "wiki_sales_millions": sales,
            "wiki_platform": platform or (str(row.get(platform_col, "")) if platform_col else ""),
            "wiki_publisher": str(row.get(publisher_col, "")) if publisher_col else "",
            "wiki_developer": str(row.get(developer_col, "")) if developer_col else "",
            "wiki_release_date": str(row.get(date_col, "")) if date_col else "",
            "wiki_source_page": page_title,
            "wiki_sales_type": "verified_official",
        }
        records.append(record)

    return records


def collect_wikipedia(force: bool = False) -> Path:
    """Collect verified game sales from Wikipedia bestseller lists.

    Parameters
    ----------
    force:
        Re-collect even if output CSV exists.

    Returns
    -------
    Path to the output CSV.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not force:
        print(f"[wikipedia] Already exists: {OUTPUT_PATH} (use --force to re-collect)")
        return OUTPUT_PATH

    all_records: list[dict] = []

    for i, page_title in enumerate(WIKI_PAGES, 1):
        print(f"[wikipedia] [{i}/{len(WIKI_PAGES)}] Fetching {page_title}...")

        tables = _fetch_wiki_tables(page_title)
        if not tables:
            print(f"[wikipedia] No tables found in {page_title}")
            continue

        sales_table = _find_sales_table(tables, page_title)
        if sales_table is None:
            print(f"[wikipedia] No sales table found in {page_title}")
            continue

        records = _process_table(sales_table, page_title)
        all_records.extend(records)
        print(f"[wikipedia]   Found {len(records)} games")

        time.sleep(RATE_LIMIT_SECONDS)

    if not all_records:
        print("[wikipedia] WARNING: No sales records collected")
        return OUTPUT_PATH

    df = pd.DataFrame(all_records)
    # Deduplicate: keep the entry with highest sales for each game name
    df = df.sort_values("wiki_sales_millions", ascending=False)
    df = df.drop_duplicates(subset=["wiki_name"], keep="first")
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"[wikipedia] Saved {len(df):,} unique games → {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect Wikipedia game sales data")
    parser.add_argument("--force", action="store_true", help="Re-collect even if exists")
    args = parser.parse_args()

    collect_wikipedia(force=args.force)
