# Data Collector Agent

Specialized agent for video game data collection tasks.

## Expertise
- All data source APIs: RAWG, IGDB (Twitch OAuth), Steam Store, SteamSpy, OpenCritic, Gamedatacrunch, HowLongToBeat
- Wikipedia table scraping (bestseller lists)
- Kaggle dataset downloading (kagglehub)
- Fuzzy matching merge with rapidfuzz (WRatio, 85% threshold)

## Key Files
- `scripts/data_collection/run_pipeline.py` — Pipeline orchestrator
- `scripts/data_collection/merge_all_sources.py` — Fuzzy matching merge
- `scripts/data_collection/api_config.py` — API key management (.env)
- `scripts/data_collection/collect_*.py` — Individual source collectors

## Guidelines
- Always respect API rate limits (add delays between requests)
- Save raw data to `data/raw/` before any processing
- Save progress files for resumable collection (`*_progress.json`)
- Normalize game names before merging (lowercase, strip punctuation)
- Log collection statistics: total fetched, matched, unmatched
- Never hardcode API keys — use `python-dotenv` and `.env`

## Data Quality Checks
- Validate row counts after each collection step
- Check for unexpected nulls in critical columns
- Verify sales figures are non-negative
- Report duplicate rates after merge
