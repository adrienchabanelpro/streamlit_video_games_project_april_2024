"""Collect game data from the Steam Store API (free, no key needed).

Fetches: name, price, review scores, tags, DLC count, categories,
release date, developer, publisher, and more.

Uses the Steam Web API + Steam Store API.
Output: data/raw/steam_store.csv
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
STEAM_DIR = RAW_DIR / "steam_store"
PROGRESS_FILE = STEAM_DIR / "_progress.json"
OUTPUT_PATH = RAW_DIR / "steam_store.csv"

# Steam API endpoints
APP_LIST_URL = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
APP_DETAILS_URL = "https://store.steampowered.com/api/appdetails"
REVIEW_URL = "https://store.steampowered.com/appreviews/{appid}"

RATE_LIMIT_SECONDS = 1.5  # Steam is strict about rate limiting
BATCH_SIZE = 100  # Save progress every N apps


def _get_all_app_ids() -> list[dict]:
    """Fetch the complete list of Steam app IDs."""
    try:
        resp = requests.get(APP_LIST_URL, timeout=30)
        resp.raise_for_status()
        apps = resp.json().get("applist", {}).get("apps", [])
        # Filter to only apps with names (removes test entries)
        return [a for a in apps if a.get("name", "").strip()]
    except Exception as exc:
        print(f"[steam_store] ERROR fetching app list: {exc}")
        return []


def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"processed_ids": [], "last_index": 0}


def _save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def _fetch_app_details(app_id: int) -> dict | None:
    """Fetch details for a single Steam app."""
    try:
        resp = requests.get(
            APP_DETAILS_URL,
            params={"appids": app_id, "cc": "us", "l": "en"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        app_data = data.get(str(app_id), {})
        if not app_data.get("success"):
            return None

        details = app_data.get("data", {})
        if details.get("type") != "game":
            return None

        # Extract price
        price_overview = details.get("price_overview", {})
        price_usd = price_overview.get("final", 0) / 100 if price_overview else 0

        # Extract review data
        recommendations = details.get("recommendations", {})

        # Extract categories and genres
        categories = [c.get("description", "") for c in details.get("categories", [])]
        genres = [g.get("description", "") for g in details.get("genres", [])]

        return {
            "steam_appid": app_id,
            "steam_store_name": details.get("name", ""),
            "steam_store_type": details.get("type", ""),
            "steam_store_developer": "|".join(details.get("developers", [])),
            "steam_store_publisher": "|".join(details.get("publishers", [])),
            "steam_store_price_usd": price_usd,
            "steam_store_is_free": details.get("is_free", False),
            "steam_store_release_date": details.get("release_date", {}).get("date", ""),
            "steam_store_coming_soon": details.get("release_date", {}).get("coming_soon", False),
            "steam_store_recommendations": recommendations.get("total", 0),
            "steam_store_categories": "|".join(categories),
            "steam_store_genres": "|".join(genres),
            "steam_store_dlc_count": len(details.get("dlc", [])),
            "steam_store_metacritic": details.get("metacritic", {}).get("score"),
            "steam_store_platforms_win": details.get("platforms", {}).get("windows", False),
            "steam_store_platforms_mac": details.get("platforms", {}).get("mac", False),
            "steam_store_platforms_linux": details.get("platforms", {}).get("linux", False),
        }
    except requests.exceptions.HTTPError:
        return None
    except Exception:
        return None


def collect_steam_store(
    max_games: int = 5000,
    force: bool = False,
) -> Path:
    """Collect game details from Steam Store API (resumable).

    Parameters
    ----------
    max_games:
        Maximum number of games to collect details for.
    force:
        Re-collect even if output CSV exists.

    Returns
    -------
    Path to the output CSV.
    """
    STEAM_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not force:
        print(f"[steam_store] Already exists: {OUTPUT_PATH} (use --force to re-collect)")
        return OUTPUT_PATH

    print("[steam_store] Fetching Steam app list...")
    all_apps = _get_all_app_ids()
    if not all_apps:
        print("[steam_store] ERROR: Could not fetch app list")
        return OUTPUT_PATH

    print(f"[steam_store] Found {len(all_apps):,} apps total")

    progress = _load_progress() if not force else {"processed_ids": [], "last_index": 0}
    processed_set = set(progress["processed_ids"])

    records: list[dict] = []
    # Load existing records if resuming
    batch_file = STEAM_DIR / "records_batch.json"
    if batch_file.exists() and not force:
        existing = json.loads(batch_file.read_text())
        records.extend(existing)
        print(f"[steam_store] Resuming with {len(records)} existing records")

    start_idx = progress["last_index"]
    collected = 0
    start_time = time.time()

    for i in range(start_idx, len(all_apps)):
        if collected >= max_games:
            break

        app = all_apps[i]
        app_id = app["appid"]

        if app_id in processed_set:
            continue

        collected += 1
        if collected % 50 == 0:
            elapsed = time.time() - start_time
            rate = collected / elapsed if elapsed > 0 else 0
            remaining = (max_games - collected) / rate if rate > 0 else 0
            print(
                f"[steam_store] [{collected}/{max_games}] "
                f"Processing {app['name'][:40]}... "
                f"(~{int(remaining / 60)}m remaining)"
            )

        details = _fetch_app_details(app_id)
        if details:
            records.append(details)

        processed_set.add(app_id)
        progress["processed_ids"] = list(processed_set)
        progress["last_index"] = i

        # Save progress periodically
        if collected % BATCH_SIZE == 0:
            _save_progress(progress)
            batch_file.write_text(json.dumps(records, indent=2))

        time.sleep(RATE_LIMIT_SECONDS)

    # Final save
    _save_progress(progress)

    if not records:
        print("[steam_store] WARNING: No game records collected")
        return OUTPUT_PATH

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["steam_appid"])
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"[steam_store] Saved {len(df):,} games → {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect Steam Store game data")
    parser.add_argument("--max-games", type=int, default=5000, help="Max games (default: 5000)")
    parser.add_argument("--force", action="store_true", help="Re-collect even if exists")
    args = parser.parse_args()

    collect_steam_store(max_games=args.max_games, force=args.force)
