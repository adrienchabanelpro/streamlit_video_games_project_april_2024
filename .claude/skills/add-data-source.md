# Skill: Add a New Data Source

## Steps

1. **Create the collector** at `scripts/data_collection/collect_<source>.py`:
   - Implement data fetching (API calls, scraping, or download)
   - Handle pagination, rate limiting, and errors
   - Save raw output to `data/raw/<source>/`
   - Add progress tracking for resumability

2. **Add API keys** (if needed):
   - Add to `.env.example` with placeholder
   - Load via `api_config.py` using `python-dotenv`

3. **Update the merge pipeline** in `scripts/data_collection/merge_all_sources.py`:
   - Add a merge function for the new source
   - Use fuzzy matching (rapidfuzz, 85% threshold)
   - Prefix columns with `<source>_` to avoid conflicts

4. **Register in the pipeline** in `scripts/data_collection/run_pipeline.py`:
   - Add as a step with CLI skip option
   - Add to the final merge call

5. **Document** in `source/pages/data_sources.py`:
   - Add a `source_card()` with name, description, row count, key fields

6. **Test**: Run the collector standalone, then via `make collect-data`.
