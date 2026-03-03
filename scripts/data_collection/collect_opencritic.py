"""Collect review scores from the OpenCritic API.

OpenCritic aggregates reviews from 100+ outlets, providing a more diverse
review landscape than Metacritic alone.

Free tier: no API key needed for basic game search and scores.
Output: data/raw/opencritic.csv
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OPENCRITIC_DIR = RAW_DIR / "opencritic"
PROGRESS_FILE = OPENCRITIC_DIR / "_progress.json"
OUTPUT_PATH = RAW_DIR / "opencritic.csv"

BASE_URL = "https://api.opencritic.com/api"
RATE_LIMIT_SECONDS = 1.0


def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"last_offset": 0, "total_fetched": 0}


def _save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def _fetch_game_list(skip: int = 0, sort: str = "score") -> list[dict]:
    """Fetch a page of games from OpenCritic."""
    try:
        resp = requests.get(
            f"{BASE_URL}/game",
            params={"skip": skip, "sort": sort, "order": "desc"},
            headers={"Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"[opencritic] ERROR at offset {skip}: {exc}")
        return []


def _fetch_game_details(game_id: int) -> dict | None:
    """Fetch detailed info for a single game."""
    try:
        resp = requests.get(
            f"{BASE_URL}/game/{game_id}",
            headers={"Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _parse_game(game: dict, details: dict | None = None) -> dict:
    """Parse a game entry into a flat record."""
    record = {
        "oc_id": game.get("id"),
        "oc_name": game.get("name", ""),
        "oc_top_critic_score": game.get("topCriticScore"),
        "oc_percent_recommended": game.get("percentRecommended"),
        "oc_num_reviews": game.get("numReviews", 0),
        "oc_num_top_critic_reviews": game.get("numTopCriticReviews", 0),
        "oc_tier": game.get("tier", ""),
        "oc_first_release_date": game.get("firstReleaseDate", ""),
    }

    if details:
        record["oc_description"] = (details.get("description") or "")[:200]
        platforms = details.get("Platforms", [])
        record["oc_platforms"] = "|".join(
            p.get("name", "") for p in platforms if isinstance(p, dict)
        )
        genres = details.get("Genres", [])
        record["oc_genres"] = "|".join(
            g.get("name", "") for g in genres if isinstance(g, dict)
        )
        companies = details.get("Companies", [])
        publishers = [
            c.get("name", "") for c in companies
            if isinstance(c, dict) and c.get("type") == "PUBLISHER"
        ]
        developers = [
            c.get("name", "") for c in companies
            if isinstance(c, dict) and c.get("type") == "DEVELOPER"
        ]
        record["oc_publisher"] = "|".join(publishers)
        record["oc_developer"] = "|".join(developers)

    return record


def collect_opencritic(
    max_games: int = 5000,
    fetch_details: bool = False,
    force: bool = False,
) -> Path:
    """Collect game review data from OpenCritic API (resumable).

    Parameters
    ----------
    max_games:
        Maximum number of games to collect.
    fetch_details:
        If True, fetch detailed info for each game (slower, more data).
    force:
        Re-collect even if output CSV exists.

    Returns
    -------
    Path to the output CSV.
    """
    OPENCRITIC_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not force:
        print(f"[opencritic] Already exists: {OUTPUT_PATH} (use --force to re-collect)")
        return OUTPUT_PATH

    progress = _load_progress() if not force else {"last_offset": 0, "total_fetched": 0}
    all_records: list[dict] = []

    offset = progress["last_offset"]
    page_size = 20  # OpenCritic default
    start_time = time.time()

    while len(all_records) < max_games:
        games = _fetch_game_list(skip=offset)
        if not games:
            print(f"[opencritic] No more games at offset {offset}")
            break

        for game in games:
            if len(all_records) >= max_games:
                break

            details = None
            if fetch_details:
                game_id = game.get("id")
                if game_id:
                    details = _fetch_game_details(game_id)
                    time.sleep(RATE_LIMIT_SECONDS)

            record = _parse_game(game, details)
            all_records.append(record)

        offset += page_size
        progress["last_offset"] = offset
        progress["total_fetched"] = len(all_records)
        _save_progress(progress)

        elapsed = time.time() - start_time
        rate = len(all_records) / elapsed if elapsed > 0 else 0
        remaining = (max_games - len(all_records)) / rate if rate > 0 else 0

        print(
            f"[opencritic] [{len(all_records)}/{max_games}] "
            f"Collected {len(all_records):,} games "
            f"(~{int(remaining / 60)}m remaining)"
        )

        time.sleep(RATE_LIMIT_SECONDS)

    if not all_records:
        print("[opencritic] WARNING: No game records collected")
        return OUTPUT_PATH

    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=["oc_id"])
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"[opencritic] Saved {len(df):,} games → {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect OpenCritic game data")
    parser.add_argument("--max-games", type=int, default=5000, help="Max games (default: 5000)")
    parser.add_argument("--details", action="store_true", help="Fetch detailed info per game")
    parser.add_argument("--force", action="store_true", help="Re-collect even if exists")
    args = parser.parse_args()

    collect_opencritic(max_games=args.max_games, fetch_details=args.details, force=args.force)
