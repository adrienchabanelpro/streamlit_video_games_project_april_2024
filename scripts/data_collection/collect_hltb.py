"""Collect game completion times from HowLongToBeat.

Fetches: main_story, main+extras, completionist, all_styles hours.
Searches by game name from the VGChartz dataset.

Resumable via progress JSON. Rate-limited to 1 request per 2 seconds.
Results cached in data/raw/hltb/results.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
HLTB_DIR = RAW_DIR / "hltb"
RESULTS_FILE = HLTB_DIR / "results.json"
PROGRESS_FILE = HLTB_DIR / "_progress.json"
OUTPUT_PATH = RAW_DIR / "hltb_all.csv"

RATE_LIMIT_SECONDS = 2.0  # be conservative — no official API


def _load_results() -> dict[str, dict]:
    """Load cached HLTB results."""
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {}


def _save_results(results: dict[str, dict]) -> None:
    """Save HLTB results cache."""
    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))


def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"last_index": -1, "total_searched": 0, "total_found": 0}


def _save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def _search_hltb(game_name: str) -> dict | None:
    """Search HLTB for a game and return completion times."""
    try:
        from howlongtobeatpy import HowLongToBeat
    except ImportError:
        raise ImportError(
            "howlongtobeatpy not installed. Run: pip install howlongtobeatpy"
        )

    results = HowLongToBeat().search(game_name)

    if not results:
        return None

    # Take the best match (first result, highest similarity)
    best = results[0]

    return {
        "hltb_name": best.game_name,
        "hltb_main": best.main_story,
        "hltb_main_extra": best.main_extra,
        "hltb_completionist": best.completionist,
        "hltb_all_styles": best.all_styles,
        "hltb_similarity": best.similarity,
    }


def collect_hltb(
    max_games: int = 10_000,
    min_sales: float = 0.1,
    force: bool = False,
) -> Path:
    """Collect HLTB data for top-selling games (resumable).

    Parameters
    ----------
    max_games:
        Maximum number of games to search for.
    min_sales:
        Minimum Global_Sales (millions) to include a game.
    force:
        Re-collect even if output CSV exists.

    Returns
    -------
    Path to the output CSV.
    """
    HLTB_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not force:
        print(f"[hltb] Already exists: {OUTPUT_PATH} (use --force to re-collect)")
        return OUTPUT_PATH

    # Load game names from the main dataset
    dataset_path = PROJECT_ROOT / "data" / "Ventes_jeux_video_final.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Run the Kaggle + SteamSpy pipeline first."
        )

    df = pd.read_csv(dataset_path)
    # Get unique game names sorted by sales (highest first)
    df = df.dropna(subset=["Name", "Global_Sales"])
    df = df[df["Global_Sales"] >= min_sales]

    # Deduplicate by name (same game on multiple platforms)
    game_names = (
        df.sort_values("Global_Sales", ascending=False)
        .drop_duplicates(subset=["Name"])["Name"]
        .head(max_games)
        .tolist()
    )

    print(f"[hltb] Searching for {len(game_names):,} unique games (min_sales={min_sales}M)")

    results = _load_results()
    progress = _load_progress() if not force else {"last_index": -1, "total_searched": 0, "total_found": 0}
    start_idx = progress["last_index"] + 1
    start_time = time.time()

    for i, name in enumerate(game_names[start_idx:], start=start_idx):
        # Skip if already cached
        normalized = name.strip().lower()
        if normalized in results and not force:
            progress["last_index"] = i
            continue

        elapsed = time.time() - start_time
        searched = i - start_idx + 1
        if searched > 1 and elapsed > 0:
            rate = searched / elapsed
            remaining = (len(game_names) - i) / max(rate, 0.01)
            est_min = int(remaining / 60)
        else:
            est_min = (len(game_names) - i) * 2 // 60

        found = progress["total_found"]
        print(
            f"[hltb] [{i + 1}/{len(game_names)}] "
            f"Searching: {name[:50]} ({found} found, ~{est_min} min remaining)"
        )

        try:
            result = _search_hltb(name)
        except Exception as exc:
            print(f"[hltb] ERROR searching '{name}': {exc}")
            _save_results(results)
            _save_progress(progress)
            time.sleep(RATE_LIMIT_SECONDS * 2)
            continue

        if result is not None:
            results[normalized] = result
            progress["total_found"] += 1

        progress["last_index"] = i
        progress["total_searched"] += 1

        # Save periodically
        if (i + 1) % 50 == 0:
            _save_results(results)
            _save_progress(progress)

        time.sleep(RATE_LIMIT_SECONDS)

    # Final save
    _save_results(results)
    _save_progress(progress)

    # Consolidate to CSV
    print("[hltb] Consolidating results to CSV...")
    df_hltb = _consolidate_results(results)
    print(f"[hltb] Saved {len(df_hltb):,} games → {OUTPUT_PATH}")
    return OUTPUT_PATH


def _consolidate_results(results: dict[str, dict]) -> pd.DataFrame:
    """Convert cached results dict to DataFrame and save."""
    rows = []
    for query_name, data in results.items():
        row = {"query_name": query_name, **data}
        rows.append(row)

    if not rows:
        raise RuntimeError("No HLTB data found to consolidate")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect HowLongToBeat data")
    parser.add_argument(
        "--max-games", type=int, default=10_000, help="Max games to search (default: 10000)"
    )
    parser.add_argument(
        "--min-sales", type=float, default=0.1, help="Min Global_Sales in millions (default: 0.1)"
    )
    parser.add_argument("--force", action="store_true", help="Re-collect even if exists")
    args = parser.parse_args()

    collect_hltb(max_games=args.max_games, min_sales=args.min_sales, force=args.force)
