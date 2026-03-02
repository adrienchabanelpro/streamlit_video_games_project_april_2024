"""Orchestrator for the full data collection pipeline.

Usage
-----
    python scripts/data_collection/run_pipeline.py [options]

Sources: Kaggle VGChartz, SteamSpy, RAWG API, IGDB API, HowLongToBeat.
Final output: data/Ventes_jeux_video_v3.csv
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data collection pipeline: 5 sources → unified v3 dataset"
    )
    parser.add_argument("--skip-kaggle", action="store_true", help="Skip Kaggle download")
    parser.add_argument("--skip-steamspy", action="store_true", help="Skip SteamSpy collection")
    parser.add_argument("--skip-rawg", action="store_true", help="Skip RAWG API collection")
    parser.add_argument("--skip-igdb", action="store_true", help="Skip IGDB API collection")
    parser.add_argument("--skip-hltb", action="store_true", help="Skip HLTB collection")
    parser.add_argument("--skip-merge", action="store_true", help="Skip final merge step")
    parser.add_argument("--steamspy-pages", type=int, default=50, help="SteamSpy pages (default: 50)")
    parser.add_argument("--rawg-pages", type=int, default=500, help="RAWG pages (default: 500)")
    parser.add_argument("--igdb-max-games", type=int, default=200_000, help="IGDB max games (default: 200000)")
    parser.add_argument("--hltb-max-games", type=int, default=10_000, help="HLTB max games (default: 10000)")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    parser.add_argument("--match-threshold", type=int, default=85, help="Fuzzy match cutoff (default: 85)")
    args = parser.parse_args()

    start = time.time()
    total_steps = 6
    step = 0

    print("=" * 60)
    print("Data Collection Pipeline v3")
    print("Sources: Kaggle, SteamSpy, RAWG, IGDB, HLTB")
    print("=" * 60)

    # Step 1: Kaggle VGChartz
    step += 1
    if not args.skip_kaggle:
        print(f"\n--- Step {step}/{total_steps}: Kaggle VGChartz 64K ---")
        from scripts.data_collection.download_kaggle import download_kaggle

        download_kaggle(force=args.force)
    else:
        print(f"\n--- Step {step}/{total_steps}: Kaggle (skipped) ---")

    # Step 2: SteamSpy
    step += 1
    if not args.skip_steamspy:
        print(f"\n--- Step {step}/{total_steps}: SteamSpy ---")
        from scripts.data_collection.collect_steamspy import collect_steamspy

        collect_steamspy(num_pages=args.steamspy_pages, force=args.force)
    else:
        print(f"\n--- Step {step}/{total_steps}: SteamSpy (skipped) ---")

    # Step 3: RAWG API
    step += 1
    if not args.skip_rawg:
        print(f"\n--- Step {step}/{total_steps}: RAWG API ---")
        from scripts.data_collection.collect_rawg import collect_rawg

        collect_rawg(max_pages=args.rawg_pages, force=args.force)
    else:
        print(f"\n--- Step {step}/{total_steps}: RAWG (skipped) ---")

    # Step 4: IGDB API
    step += 1
    if not args.skip_igdb:
        print(f"\n--- Step {step}/{total_steps}: IGDB API ---")
        from scripts.data_collection.collect_igdb import collect_igdb

        collect_igdb(max_games=args.igdb_max_games, force=args.force)
    else:
        print(f"\n--- Step {step}/{total_steps}: IGDB (skipped) ---")

    # Step 5: HowLongToBeat
    step += 1
    if not args.skip_hltb:
        print(f"\n--- Step {step}/{total_steps}: HowLongToBeat ---")
        from scripts.data_collection.collect_hltb import collect_hltb

        collect_hltb(max_games=args.hltb_max_games, force=args.force)
    else:
        print(f"\n--- Step {step}/{total_steps}: HLTB (skipped) ---")

    # Step 6: Merge all sources
    step += 1
    if not args.skip_merge:
        print(f"\n--- Step {step}/{total_steps}: Merge All Sources ---")
        from scripts.data_collection.merge_all_sources import merge_all_sources

        merge_all_sources(match_threshold=args.match_threshold, force=args.force)
    else:
        print(f"\n--- Step {step}/{total_steps}: Merge (skipped) ---")

    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {minutes}m {seconds}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
