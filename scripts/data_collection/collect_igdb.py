"""Collect game metadata from the IGDB API (700K+ games).

Fetches: themes, game_modes, player_perspectives, franchises,
game_type (category), hype_count, total_rating, involved_companies.

Resumable via progress JSON. Rate-limited to 4 requests/second.
Raw batches saved to data/raw/igdb/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from scripts.data_collection.api_config import get_igdb_access_token, get_igdb_credentials

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
IGDB_DIR = RAW_DIR / "igdb"
PROGRESS_FILE = IGDB_DIR / "_progress.json"
OUTPUT_PATH = RAW_DIR / "igdb_all.csv"

BASE_URL = "https://api.igdb.com/v4/games"
BATCH_SIZE = 500  # max per request
RATE_LIMIT_SECONDS = 0.25  # 4 req/s


def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"offset": 0, "total_fetched": 0}


def _save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def collect_igdb(max_games: int = 200_000, force: bool = False) -> Path:
    """Collect IGDB game data in batches (resumable).

    Parameters
    ----------
    max_games:
        Maximum number of games to collect.
    force:
        Re-collect even if output CSV exists.

    Returns
    -------
    Path to the consolidated CSV.
    """
    IGDB_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not force:
        print(f"[igdb] Already exists: {OUTPUT_PATH} (use --force to re-collect)")
        return OUTPUT_PATH

    client_id, _ = get_igdb_credentials()
    access_token = get_igdb_access_token()

    headers = {
        "Client-ID": client_id,
        "Authorization": f"Bearer {access_token}",
    }

    progress = _load_progress() if not force else {"offset": 0, "total_fetched": 0}
    offset = progress["offset"]
    start_time = time.time()

    batch_num = offset // BATCH_SIZE

    while offset < max_games:
        batch_file = IGDB_DIR / f"batch_{batch_num:04d}.json"

        if batch_file.exists() and not force:
            print(f"[igdb] Skipping batch {batch_num} (already collected)")
            offset += BATCH_SIZE
            batch_num += 1
            continue

        elapsed = time.time() - start_time
        games_remaining = max_games - offset
        if offset > progress["offset"] and elapsed > 0:
            rate = (offset - progress["offset"]) / elapsed
            est_min = int(games_remaining / max(rate, 1) / 60)
        else:
            est_min = games_remaining // BATCH_SIZE

        print(
            f"[igdb] [batch {batch_num}] offset={offset:,} "
            f"(~{est_min} min remaining)"
        )

        body = (
            f"fields name,slug,first_release_date,category,total_rating,"
            f"total_rating_count,hypes,follows,"
            f"themes.name,game_modes.name,player_perspectives.name,"
            f"franchises.name,involved_companies.company.name,"
            f"involved_companies.developer,involved_companies.publisher;"
            f"where category = (0,8,9,10,11);"
            f"sort id asc;"
            f"limit {BATCH_SIZE};"
            f"offset {offset};"
        )

        try:
            resp = requests.post(BASE_URL, headers=headers, data=body, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"[igdb] ERROR at offset {offset}: {exc}")
            _save_progress(progress)
            raise

        if not data:
            print(f"[igdb] Empty response at offset {offset} — assuming end of data.")
            break

        batch_file.write_text(json.dumps(data, indent=2))

        offset += len(data)
        batch_num += 1
        progress["offset"] = offset
        progress["total_fetched"] = offset
        _save_progress(progress)

        time.sleep(RATE_LIMIT_SECONDS)

    print("[igdb] Consolidating batches into CSV...")
    df = _consolidate_batches()
    print(f"[igdb] Saved {len(df):,} games → {OUTPUT_PATH}")
    return OUTPUT_PATH


def _parse_game(game: dict) -> dict:
    """Extract fields from a single IGDB game entry."""
    # Parse release date from Unix timestamp
    release_ts = game.get("first_release_date")
    release_date = None
    if release_ts:
        from datetime import datetime, timezone

        release_date = datetime.fromtimestamp(release_ts, tz=timezone.utc).strftime("%Y-%m-%d")

    # Parse involved companies
    developers = []
    publishers = []
    for ic in game.get("involved_companies") or []:
        company_name = (ic.get("company") or {}).get("name", "")
        if ic.get("developer"):
            developers.append(company_name)
        if ic.get("publisher"):
            publishers.append(company_name)

    return {
        "igdb_id": game.get("id"),
        "igdb_name": game.get("name"),
        "igdb_slug": game.get("slug"),
        "igdb_released": release_date,
        "igdb_category": game.get("category"),
        "igdb_total_rating": game.get("total_rating"),
        "igdb_rating_count": game.get("total_rating_count"),
        "igdb_hypes": game.get("hypes"),
        "igdb_follows": game.get("follows"),
        "igdb_themes": "|".join(t["name"] for t in (game.get("themes") or [])),
        "igdb_game_modes": "|".join(m["name"] for m in (game.get("game_modes") or [])),
        "igdb_perspectives": "|".join(
            p["name"] for p in (game.get("player_perspectives") or [])
        ),
        "igdb_franchises": "|".join(f["name"] for f in (game.get("franchises") or [])),
        "igdb_developers": "|".join(developers),
        "igdb_publishers": "|".join(publishers),
    }


def _consolidate_batches() -> pd.DataFrame:
    """Merge all batch JSON files into a single DataFrame."""
    all_games = []
    batch_files = sorted(IGDB_DIR.glob("batch_*.json"))

    for batch_file in batch_files:
        games = json.loads(batch_file.read_text())
        for game in games:
            all_games.append(_parse_game(game))

    if not all_games:
        raise RuntimeError("No IGDB data found to consolidate")

    df = pd.DataFrame(all_games)
    df = df.drop_duplicates(subset=["igdb_id"])
    df.to_csv(OUTPUT_PATH, index=False)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect IGDB game data")
    parser.add_argument(
        "--max-games", type=int, default=200_000, help="Max games (default: 200000)"
    )
    parser.add_argument("--force", action="store_true", help="Re-collect even if exists")
    args = parser.parse_args()

    collect_igdb(max_games=args.max_games, force=args.force)
