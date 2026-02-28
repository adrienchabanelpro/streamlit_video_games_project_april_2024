"""Step 2: Collect SteamSpy data (~50K+ Steam games).

Uses the steamspypi package to page through all games. Each page is saved as
a JSON file for resumability. After all pages are collected, consolidates into
a single CSV with parsed numeric fields.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
STEAMSPY_DIR = RAW_DIR / "steamspy"
PROGRESS_FILE = STEAMSPY_DIR / "_progress.json"
OUTPUT_PATH = RAW_DIR / "steamspy_all.csv"

GAMES_PER_PAGE = 1000
RATE_LIMIT_SECONDS = 62  # steamspypi enforces ~1 req/min for "all" endpoint


def parse_owners(owners_str: str) -> float | None:
    """Parse SteamSpy owners string into numeric midpoint.

    Examples
    --------
    >>> parse_owners("1,000,000 .. 2,000,000")
    1500000.0
    >>> parse_owners("0 .. 20,000")
    10000.0
    """
    if not owners_str or not isinstance(owners_str, str):
        return None
    parts = owners_str.split("..")
    if len(parts) != 2:
        return None
    try:
        lo = int(parts[0].strip().replace(",", ""))
        hi = int(parts[1].strip().replace(",", ""))
        return (lo + hi) / 2.0
    except (ValueError, TypeError):
        return None


def compute_review_pct(positive: int | float, negative: int | float) -> float | None:
    """Compute review percentage from positive/negative counts."""
    try:
        pos = int(positive)
        neg = int(negative)
    except (ValueError, TypeError):
        return None
    total = pos + neg
    if total == 0:
        return None
    return round(pos / total * 100, 1)


def _load_progress() -> dict:
    """Load the progress manifest."""
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed_pages": [], "total_pages": None}


def _save_progress(progress: dict) -> None:
    """Save the progress manifest."""
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def collect_steamspy(num_pages: int = 50, force: bool = False) -> Path:
    """Collect SteamSpy data page by page (resumable).

    Parameters
    ----------
    num_pages:
        Number of pages to collect (1000 games/page).
    force:
        Re-collect even if pages already exist.

    Returns
    -------
    Path
        Path to the consolidated CSV.
    """
    import steamspypi

    STEAMSPY_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not force:
        print(f"[steamspy] Already exists: {OUTPUT_PATH} (use --force to re-collect)")
        return OUTPUT_PATH

    progress = _load_progress()
    if force:
        progress = {"completed_pages": [], "total_pages": num_pages}

    progress["total_pages"] = num_pages
    completed = set(progress["completed_pages"])

    start_time = time.time()
    for page_num in range(num_pages):
        page_file = STEAMSPY_DIR / f"page_{page_num:03d}.json"

        if page_num in completed and page_file.exists() and not force:
            print(f"[steamspy] Skipping page {page_num} (already collected)")
            continue

        elapsed = time.time() - start_time
        remaining_pages = num_pages - page_num
        if page_num > 0:
            avg_per_page = elapsed / page_num
            est_remaining = avg_per_page * remaining_pages
            est_min = int(est_remaining / 60)
        else:
            est_min = remaining_pages  # rough estimate: ~1 min/page

        games_so_far = page_num * GAMES_PER_PAGE
        print(
            f"[steamspy] [page {page_num + 1}/{num_pages}] "
            f"{games_so_far:,} games (~{est_min} min remaining)"
        )

        try:
            data = steamspypi.download({"request": "all", "page": str(page_num)})
        except Exception as exc:
            print(f"[steamspy] ERROR on page {page_num}: {exc}")
            print("[steamspy] Progress saved. Re-run to resume.")
            _save_progress(progress)
            raise

        if not data:
            print(f"[steamspy] Page {page_num} returned empty — assuming end of data.")
            break

        page_file.write_text(json.dumps(data, indent=2))
        progress["completed_pages"].append(page_num)
        _save_progress(progress)

        # Rate limiting (skip on last page)
        if page_num < num_pages - 1:
            print(f"[steamspy] Rate limiting: waiting {RATE_LIMIT_SECONDS}s...")
            time.sleep(RATE_LIMIT_SECONDS)

    # Consolidate all pages into a single CSV
    print("[steamspy] Consolidating pages into CSV...")
    output = _consolidate_pages()
    print(f"[steamspy] Saved {len(output):,} games → {OUTPUT_PATH}")
    return OUTPUT_PATH


def _consolidate_pages() -> pd.DataFrame:
    """Merge all page JSON files into a single DataFrame and save as CSV."""
    all_games: dict[str, dict] = {}

    page_files = sorted(STEAMSPY_DIR.glob("page_*.json"))
    for page_file in page_files:
        data = json.loads(page_file.read_text())
        for appid, game_data in data.items():
            game_data["appid"] = int(appid)
            all_games[appid] = game_data

    if not all_games:
        raise RuntimeError("No SteamSpy data found to consolidate")

    df = pd.DataFrame.from_dict(all_games, orient="index")

    # Parse owners into numeric midpoint
    if "owners" in df.columns:
        df["owners_midpoint"] = df["owners"].apply(parse_owners)

    # Compute review percentage
    if "positive" in df.columns and "negative" in df.columns:
        df["review_pct"] = df.apply(
            lambda r: compute_review_pct(r["positive"], r["negative"]), axis=1
        )

    # Convert price from cents to dollars
    for price_col in ["price", "initialprice"]:
        if price_col in df.columns:
            df[price_col] = pd.to_numeric(df[price_col], errors="coerce") / 100.0

    df.to_csv(OUTPUT_PATH, index=False)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect SteamSpy data")
    parser.add_argument("--pages", type=int, default=50, help="Number of pages (default: 50)")
    parser.add_argument("--force", action="store_true", help="Re-collect even if exists")
    args = parser.parse_args()

    collect_steamspy(num_pages=args.pages, force=args.force)
