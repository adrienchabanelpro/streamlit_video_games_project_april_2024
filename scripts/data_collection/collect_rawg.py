"""Collect game metadata from the RAWG API (500K+ games).

Fetches: name, slug, metacritic score, playtime, ESRB rating,
genres, tags, platforms, developers, publishers, release date.

Resumable via progress JSON. Rate-limited to ~0.5s per request.
Raw pages saved to data/raw/rawg/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from scripts.data_collection.api_config import get_rawg_api_key

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAWG_DIR = RAW_DIR / "rawg"
PROGRESS_FILE = RAWG_DIR / "_progress.json"
OUTPUT_PATH = RAW_DIR / "rawg_all.csv"

BASE_URL = "https://api.rawg.io/api/games"
PAGE_SIZE = 40  # max allowed by RAWG
RATE_LIMIT_SECONDS = 0.5


def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"last_page": 0, "total_count": None}


def _save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def collect_rawg(max_pages: int = 500, force: bool = False) -> Path:
    """Collect RAWG game data page by page (resumable).

    Parameters
    ----------
    max_pages:
        Maximum number of pages to collect (40 games/page).
    force:
        Re-collect even if output CSV exists.

    Returns
    -------
    Path to the consolidated CSV.
    """
    RAWG_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not force:
        print(f"[rawg] Already exists: {OUTPUT_PATH} (use --force to re-collect)")
        return OUTPUT_PATH

    api_key = get_rawg_api_key()
    progress = _load_progress() if not force else {"last_page": 0, "total_count": None}

    start_page = progress["last_page"] + 1 if progress["last_page"] > 0 else 1
    start_time = time.time()

    for page_num in range(start_page, max_pages + 1):
        page_file = RAWG_DIR / f"page_{page_num:04d}.json"

        if page_file.exists() and not force:
            print(f"[rawg] Skipping page {page_num} (already collected)")
            progress["last_page"] = page_num
            continue

        elapsed = time.time() - start_time
        pages_done = page_num - start_page
        if pages_done > 0:
            avg = elapsed / pages_done
            remaining = avg * (max_pages - page_num)
            est_min = int(remaining / 60)
        else:
            est_min = max_pages - page_num

        games_so_far = page_num * PAGE_SIZE
        print(
            f"[rawg] [page {page_num}/{max_pages}] "
            f"~{games_so_far:,} games (~{est_min} min remaining)"
        )

        try:
            resp = requests.get(
                BASE_URL,
                params={"key": api_key, "page": page_num, "page_size": PAGE_SIZE},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"[rawg] ERROR on page {page_num}: {exc}")
            _save_progress(progress)
            raise

        if not data.get("results"):
            print(f"[rawg] Page {page_num} returned no results — assuming end of data.")
            break

        progress["total_count"] = data.get("count")
        page_file.write_text(json.dumps(data["results"], indent=2))

        progress["last_page"] = page_num
        _save_progress(progress)

        time.sleep(RATE_LIMIT_SECONDS)

    print("[rawg] Consolidating pages into CSV...")
    df = _consolidate_pages()
    print(f"[rawg] Saved {len(df):,} games → {OUTPUT_PATH}")
    return OUTPUT_PATH


def _parse_game(game: dict) -> dict:
    """Extract fields from a single RAWG game entry."""
    return {
        "rawg_id": game.get("id"),
        "rawg_name": game.get("name"),
        "rawg_slug": game.get("slug"),
        "rawg_released": game.get("released"),
        "rawg_metacritic": game.get("metacritic"),
        "rawg_rating": game.get("rating"),
        "rawg_ratings_count": game.get("ratings_count"),
        "rawg_playtime": game.get("playtime"),
        "rawg_esrb": (game.get("esrb_rating") or {}).get("name"),
        "rawg_genres": "|".join(g["name"] for g in (game.get("genres") or [])),
        "rawg_platforms": "|".join(
            (p.get("platform") or {}).get("name", "")
            for p in (game.get("platforms") or [])
        ),
        "rawg_tags_top5": "|".join(
            t["name"] for t in (game.get("tags") or [])[:5]
        ),
        "rawg_developers": "|".join(
            d["name"] for d in (game.get("developers") or [])
        ),
        "rawg_publishers": "|".join(
            p["name"] for p in (game.get("publishers") or [])
        ),
    }


def _consolidate_pages() -> pd.DataFrame:
    """Merge all page JSON files into a single DataFrame."""
    all_games = []
    page_files = sorted(RAWG_DIR.glob("page_*.json"))

    for page_file in page_files:
        games = json.loads(page_file.read_text())
        for game in games:
            all_games.append(_parse_game(game))

    if not all_games:
        raise RuntimeError("No RAWG data found to consolidate")

    df = pd.DataFrame(all_games)
    df = df.drop_duplicates(subset=["rawg_id"])
    df.to_csv(OUTPUT_PATH, index=False)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect RAWG game data")
    parser.add_argument("--pages", type=int, default=500, help="Max pages (default: 500)")
    parser.add_argument("--force", action="store_true", help="Re-collect even if exists")
    args = parser.parse_args()

    collect_rawg(max_pages=args.pages, force=args.force)
