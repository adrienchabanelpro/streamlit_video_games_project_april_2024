"""Orchestrator for the full data collection pipeline.

Usage
-----
    python scripts/data_collection/run_pipeline.py [options]

    --skip-kaggle       Skip Kaggle download
    --skip-steamspy     Skip SteamSpy collection
    --skip-merge        Skip merge step
    --steamspy-pages N  Number of SteamSpy pages (default: 50)
    --force             Re-download/re-collect even if files exist
    --match-threshold N Fuzzy match cutoff (default: 85)
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data collection pipeline: Kaggle VGChartz 64K + SteamSpy enrichment"
    )
    parser.add_argument("--skip-kaggle", action="store_true", help="Skip Kaggle download")
    parser.add_argument("--skip-steamspy", action="store_true", help="Skip SteamSpy collection")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge step")
    parser.add_argument("--steamspy-pages", type=int, default=50, help="Number of SteamSpy pages (default: 50)")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    parser.add_argument("--match-threshold", type=int, default=85, help="Fuzzy match cutoff (default: 85)")
    args = parser.parse_args()

    start = time.time()
    print("=" * 60)
    print("Data Collection Pipeline")
    print("=" * 60)

    # Step 1: Kaggle
    if not args.skip_kaggle:
        print("\n--- Step 1/3: Kaggle VGChartz 64K ---")
        from scripts.data_collection.download_kaggle import download_kaggle

        download_kaggle(force=args.force)
    else:
        print("\n--- Step 1/3: Kaggle (skipped) ---")

    # Step 2: SteamSpy
    if not args.skip_steamspy:
        print("\n--- Step 2/3: SteamSpy ---")
        from scripts.data_collection.collect_steamspy import collect_steamspy

        collect_steamspy(num_pages=args.steamspy_pages, force=args.force)
    else:
        print("\n--- Step 2/3: SteamSpy (skipped) ---")

    # Step 3: Merge
    if not args.skip_merge:
        print("\n--- Step 3/3: Merge ---")
        from scripts.data_collection.merge_datasets import merge_datasets

        merge_datasets(match_threshold=args.match_threshold, force=args.force)
    else:
        print("\n--- Step 3/3: Merge (skipped) ---")

    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {minutes}m {seconds}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
