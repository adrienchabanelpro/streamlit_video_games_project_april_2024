"""Collect authoritative review counts from the Steam Reviews API.

Fetches total_positive, total_negative, total_reviews, and review_score
for each Steam app. No API key needed.

Uses SteamSpy data as the source of app IDs, sorted by owner count
(most popular first) so partial runs still capture high-value games.

Output: data/raw/steam_reviews_summary.csv
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REVIEWS_DIR = RAW_DIR / "steam_reviews"
PROGRESS_FILE = REVIEWS_DIR / "_progress.json"
OUTPUT_PATH = RAW_DIR / "steam_reviews_summary.csv"

STEAMSPY_PATH = RAW_DIR / "steamspy_all.csv"

REVIEW_API_URL = "https://store.steampowered.com/appreviews/{appid}"
RATE_LIMIT_SECONDS = 1.5
BATCH_SIZE = 100


def _load_appids() -> list[int]:
    """Load Steam app IDs from SteamSpy data, sorted by owner count descending."""
    if not STEAMSPY_PATH.exists():
        raise FileNotFoundError(
            f"SteamSpy data not found at {STEAMSPY_PATH}\n"
            "Run the SteamSpy collection step first."
        )

    df = pd.read_csv(STEAMSPY_PATH)
    appid_col = "appid" if "appid" in df.columns else "steam_appid"
    owners_col = "owners_midpoint" if "owners_midpoint" in df.columns else None

    df = df.dropna(subset=[appid_col])
    df[appid_col] = df[appid_col].astype(int)

    if owners_col and owners_col in df.columns:
        df = df.sort_values(owners_col, ascending=False)

    return df[appid_col].tolist()


def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"processed_ids": [], "last_index": 0}


def _save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def _fetch_review_summary(app_id: int) -> dict | None:
    """Fetch review summary for a single Steam app."""
    try:
        resp = requests.get(
            REVIEW_API_URL.format(appid=app_id),
            params={
                "json": "1",
                "language": "all",
                "num_per_page": "0",
                "purchase_type": "all",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("success") != 1:
            return None

        summary = data.get("query_summary", {})
        total_positive = summary.get("total_positive", 0)
        total_negative = summary.get("total_negative", 0)
        total_reviews = total_positive + total_negative
        review_score = summary.get("review_score_desc", "")
        review_score_pct = (
            round(total_positive / total_reviews * 100, 1)
            if total_reviews > 0
            else 0
        )

        return {
            "appid": app_id,
            "review_total_positive": total_positive,
            "review_total_negative": total_negative,
            "review_total": total_reviews,
            "review_score_desc": review_score,
            "review_score_pct": review_score_pct,
        }
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 429:
            print(f"[steam_reviews] Rate limited on appid {app_id}, backing off...")
            time.sleep(10)
        return None
    except Exception:
        return None


def collect_steam_reviews(
    max_games: int = 15_000,
    force: bool = False,
) -> Path:
    """Collect Steam review summaries (resumable).

    Parameters
    ----------
    max_games:
        Maximum number of apps to query.
    force:
        Re-collect even if output CSV exists.

    Returns
    -------
    Path to the output CSV.
    """
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not force:
        print(f"[steam_reviews] Already exists: {OUTPUT_PATH} (use --force)")
        return OUTPUT_PATH

    print("[steam_reviews] Loading app IDs from SteamSpy data...")
    all_appids = _load_appids()
    print(f"[steam_reviews] Found {len(all_appids):,} apps (sorted by popularity)")

    progress = _load_progress() if not force else {"processed_ids": [], "last_index": 0}
    processed_set = set(progress["processed_ids"])

    records: list[dict] = []
    batch_file = REVIEWS_DIR / "records_batch.json"
    if batch_file.exists() and not force:
        existing = json.loads(batch_file.read_text())
        records.extend(existing)
        print(f"[steam_reviews] Resuming with {len(records)} existing records")

    start_idx = progress["last_index"]
    collected = 0
    start_time = time.time()

    for i in range(start_idx, len(all_appids)):
        if collected >= max_games:
            break

        app_id = all_appids[i]
        if app_id in processed_set:
            continue

        collected += 1
        if collected % 50 == 0:
            elapsed = time.time() - start_time
            rate = collected / elapsed if elapsed > 0 else 0
            remaining = (max_games - collected) / rate if rate > 0 else 0
            print(
                f"[steam_reviews] [{collected}/{max_games}] "
                f"appid={app_id} ({len(records)} with reviews, "
                f"~{int(remaining / 60)}m remaining)"
            )

        summary = _fetch_review_summary(app_id)
        if summary and summary["review_total"] > 0:
            records.append(summary)

        processed_set.add(app_id)
        progress["processed_ids"] = list(processed_set)
        progress["last_index"] = i

        if collected % BATCH_SIZE == 0:
            _save_progress(progress)
            batch_file.write_text(json.dumps(records, indent=2))

        time.sleep(RATE_LIMIT_SECONDS)

    # Final save
    _save_progress(progress)

    if not records:
        print("[steam_reviews] WARNING: No review records collected")
        return OUTPUT_PATH

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["appid"])
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"[steam_reviews] Saved {len(df):,} games → {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect Steam review summaries")
    parser.add_argument(
        "--max-games", type=int, default=15_000, help="Max games (default: 15000)"
    )
    parser.add_argument("--force", action="store_true", help="Re-collect even if exists")
    args = parser.parse_args()

    collect_steam_reviews(max_games=args.max_games, force=args.force)
