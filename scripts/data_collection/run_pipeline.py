"""Orchestrator for the full data collection pipeline.

Usage
-----
    python scripts/data_collection/run_pipeline.py [options]

Sources (10): Kaggle VGChartz, SteamSpy, Steam Reviews, RAWG API, IGDB API,
              HowLongToBeat, Wikipedia, Steam Store, OpenCritic, Gamedatacrunch.
Final output: data/Ventes_jeux_video_v3.csv
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data collection pipeline: 9 sources → unified v3 dataset"
    )
    parser.add_argument("--skip-kaggle", action="store_true", help="Skip Kaggle download")
    parser.add_argument("--skip-steamspy", action="store_true", help="Skip SteamSpy collection")
    parser.add_argument("--skip-steam-reviews", action="store_true", help="Skip Steam Reviews API")
    parser.add_argument("--skip-rawg", action="store_true", help="Skip RAWG API collection")
    parser.add_argument("--skip-igdb", action="store_true", help="Skip IGDB API collection")
    parser.add_argument("--skip-hltb", action="store_true", help="Skip HLTB collection")
    parser.add_argument("--skip-wikipedia", action="store_true", help="Skip Wikipedia scraping")
    parser.add_argument("--skip-steam-store", action="store_true", help="Skip Steam Store API")
    parser.add_argument("--skip-opencritic", action="store_true", help="Skip OpenCritic API")
    parser.add_argument("--skip-gamedatacrunch", action="store_true", help="Skip Gamedatacrunch")
    parser.add_argument("--skip-merge", action="store_true", help="Skip final merge step")
    parser.add_argument("--steamspy-pages", type=int, default=50, help="SteamSpy pages (default: 50)")
    parser.add_argument("--rawg-pages", type=int, default=500, help="RAWG pages (default: 500)")
    parser.add_argument("--igdb-max-games", type=int, default=200_000, help="IGDB max games (default: 200000)")
    parser.add_argument("--hltb-max-games", type=int, default=10_000, help="HLTB max games (default: 10000)")
    parser.add_argument("--steam-reviews-max", type=int, default=15_000, help="Steam Reviews max games (default: 15000)")
    parser.add_argument("--steam-store-max", type=int, default=20_000, help="Steam Store max games (default: 20000)")
    parser.add_argument("--opencritic-max", type=int, default=5000, help="OpenCritic max games (default: 5000)")
    parser.add_argument("--gdc-max-pages", type=int, default=100, help="Gamedatacrunch max pages (default: 100)")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    parser.add_argument("--match-threshold", type=int, default=85, help="Fuzzy match cutoff (default: 85)")
    args = parser.parse_args()

    start = time.time()
    total_steps = 11
    step = 0

    print("=" * 60)
    print("Data Collection Pipeline v3")
    print("Sources: Kaggle, SteamSpy, Steam Reviews, RAWG, IGDB, HLTB,")
    print("         Wikipedia, Steam Store, OpenCritic, Gamedatacrunch")
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

    # Step 3: Steam Reviews API
    step += 1
    if not args.skip_steam_reviews:
        print(f"\n--- Step {step}/{total_steps}: Steam Reviews API ---")
        from scripts.data_collection.collect_steam_reviews import collect_steam_reviews

        collect_steam_reviews(max_games=args.steam_reviews_max, force=args.force)
    else:
        print(f"\n--- Step {step}/{total_steps}: Steam Reviews (skipped) ---")

    # Step 4: RAWG API
    step += 1
    if not args.skip_rawg:
        print(f"\n--- Step {step}/{total_steps}: RAWG API ---")
        from scripts.data_collection.collect_rawg import collect_rawg

        collect_rawg(max_pages=args.rawg_pages, force=args.force)
    else:
        print(f"\n--- Step {step}/{total_steps}: RAWG (skipped) ---")

    # Step 5: IGDB API
    step += 1
    if not args.skip_igdb:
        print(f"\n--- Step {step}/{total_steps}: IGDB API ---")
        from scripts.data_collection.collect_igdb import collect_igdb

        collect_igdb(max_games=args.igdb_max_games, force=args.force)
    else:
        print(f"\n--- Step {step}/{total_steps}: IGDB (skipped) ---")

    # Step 6: HowLongToBeat
    step += 1
    if not args.skip_hltb:
        print(f"\n--- Step {step}/{total_steps}: HowLongToBeat ---")
        from scripts.data_collection.collect_hltb import collect_hltb

        collect_hltb(max_games=args.hltb_max_games, force=args.force)
    else:
        print(f"\n--- Step {step}/{total_steps}: HLTB (skipped) ---")

    # Step 7: Wikipedia (verified official sales figures)
    step += 1
    if not args.skip_wikipedia:
        print(f"\n--- Step {step}/{total_steps}: Wikipedia Sales ---")
        from scripts.data_collection.collect_wikipedia import collect_wikipedia

        collect_wikipedia(force=args.force)
    else:
        print(f"\n--- Step {step}/{total_steps}: Wikipedia (skipped) ---")

    # Step 8: Steam Store API
    step += 1
    if not args.skip_steam_store:
        print(f"\n--- Step {step}/{total_steps}: Steam Store API ---")
        from scripts.data_collection.collect_steam_store import collect_steam_store

        collect_steam_store(max_games=args.steam_store_max, force=args.force)
    else:
        print(f"\n--- Step {step}/{total_steps}: Steam Store (skipped) ---")

    # Step 9: OpenCritic
    step += 1
    if not args.skip_opencritic:
        print(f"\n--- Step {step}/{total_steps}: OpenCritic ---")
        from scripts.data_collection.collect_opencritic import collect_opencritic

        collect_opencritic(max_games=args.opencritic_max, force=args.force)
    else:
        print(f"\n--- Step {step}/{total_steps}: OpenCritic (skipped) ---")

    # Step 10: Gamedatacrunch
    step += 1
    if not args.skip_gamedatacrunch:
        print(f"\n--- Step {step}/{total_steps}: Gamedatacrunch ---")
        from scripts.data_collection.collect_gamedatacrunch import (
            collect_gamedatacrunch,
        )

        collect_gamedatacrunch(max_pages=args.gdc_max_pages, force=args.force)
    else:
        print(f"\n--- Step {step}/{total_steps}: Gamedatacrunch (skipped) ---")

    # Step 11: Merge all sources
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
