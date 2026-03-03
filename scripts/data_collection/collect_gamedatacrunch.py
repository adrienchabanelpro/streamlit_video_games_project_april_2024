"""Collect Steam analytics data from Gamedatacrunch.

Gamedatacrunch provides free Steam game analytics including:
- Revenue estimates
- Concurrent player counts
- Regional pricing
- Review breakdowns

Output: data/raw/gamedatacrunch.csv
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
GDC_DIR = RAW_DIR / "gamedatacrunch"
PROGRESS_FILE = GDC_DIR / "_progress.json"
OUTPUT_PATH = RAW_DIR / "gamedatacrunch.csv"

BASE_URL = "https://www.gamedatacrunch.com/api"
RATE_LIMIT_SECONDS = 2.0  # Be conservative with rate limiting


def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"last_page": 0, "total_fetched": 0}


def _save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def _fetch_games_page(page: int = 1) -> list[dict]:
    """Fetch a page of games from Gamedatacrunch."""
    try:
        resp = requests.get(
            f"{BASE_URL}/games",
            params={"page": page, "sort": "revenue", "order": "desc"},
            headers={
                "Accept": "application/json",
                "User-Agent": "VideoGameSalesPredictor/1.0 (research project)",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else data.get("games", data.get("results", []))
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            print("[gamedatacrunch] Rate limited, waiting 30s...")
            time.sleep(30)
            return _fetch_games_page(page)
        print(f"[gamedatacrunch] HTTP error on page {page}: {e}")
        return []
    except Exception as exc:
        print(f"[gamedatacrunch] ERROR on page {page}: {exc}")
        return []


def _fetch_game_details(app_id: int) -> dict | None:
    """Fetch detailed analytics for a single game."""
    try:
        resp = requests.get(
            f"{BASE_URL}/game/{app_id}",
            headers={
                "Accept": "application/json",
                "User-Agent": "VideoGameSalesPredictor/1.0 (research project)",
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _parse_game(game: dict) -> dict:
    """Parse a game entry into a flat record."""
    return {
        "gdc_appid": game.get("appid") or game.get("id") or game.get("steam_appid"),
        "gdc_name": game.get("name", ""),
        "gdc_revenue_estimate": game.get("revenue") or game.get("estimated_revenue"),
        "gdc_owners_estimate": game.get("owners") or game.get("estimated_owners"),
        "gdc_ccu_max": game.get("ccu") or game.get("peak_ccu"),
        "gdc_price_usd": game.get("price") or game.get("price_usd"),
        "gdc_review_score": game.get("review_score") or game.get("score"),
        "gdc_review_count": game.get("review_count") or game.get("reviews"),
        "gdc_release_date": game.get("release_date", ""),
        "gdc_developer": game.get("developer", ""),
        "gdc_publisher": game.get("publisher", ""),
        "gdc_tags": "|".join(game.get("tags", []) if isinstance(game.get("tags"), list) else []),
        "gdc_genres": "|".join(
            game.get("genres", []) if isinstance(game.get("genres"), list) else []
        ),
    }


def collect_gamedatacrunch(
    max_pages: int = 100,
    force: bool = False,
) -> Path:
    """Collect game analytics from Gamedatacrunch (resumable).

    Parameters
    ----------
    max_pages:
        Maximum number of pages to collect.
    force:
        Re-collect even if output CSV exists.

    Returns
    -------
    Path to the output CSV.
    """
    GDC_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not force:
        print(f"[gamedatacrunch] Already exists: {OUTPUT_PATH} (use --force to re-collect)")
        return OUTPUT_PATH

    progress = _load_progress() if not force else {"last_page": 0, "total_fetched": 0}
    all_records: list[dict] = []

    start_page = progress["last_page"] + 1 if progress["last_page"] > 0 else 1
    start_time = time.time()

    for page_num in range(start_page, max_pages + 1):
        games = _fetch_games_page(page_num)
        if not games:
            print(f"[gamedatacrunch] No more games at page {page_num}")
            break

        page_records = [_parse_game(g) for g in games if g.get("name")]
        all_records.extend(page_records)

        progress["last_page"] = page_num
        progress["total_fetched"] = len(all_records)
        _save_progress(progress)

        elapsed = time.time() - start_time
        pages_done = page_num - start_page + 1
        if pages_done > 0:
            avg = elapsed / pages_done
            remaining = avg * (max_pages - page_num)
            est_min = int(remaining / 60)
        else:
            est_min = max_pages - page_num

        print(
            f"[gamedatacrunch] [page {page_num}/{max_pages}] "
            f"{len(all_records):,} games (~{est_min}m remaining)"
        )

        # Save batch to disk periodically
        if page_num % 10 == 0:
            batch_file = GDC_DIR / "records_batch.json"
            batch_file.write_text(json.dumps(all_records, indent=2))

        time.sleep(RATE_LIMIT_SECONDS)

    if not all_records:
        print("[gamedatacrunch] WARNING: No game records collected")
        return OUTPUT_PATH

    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=["gdc_appid"])
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"[gamedatacrunch] Saved {len(df):,} games → {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect Gamedatacrunch analytics data")
    parser.add_argument("--max-pages", type=int, default=100, help="Max pages (default: 100)")
    parser.add_argument("--force", action="store_true", help="Re-collect even if exists")
    args = parser.parse_args()

    collect_gamedatacrunch(max_pages=args.max_pages, force=args.force)
