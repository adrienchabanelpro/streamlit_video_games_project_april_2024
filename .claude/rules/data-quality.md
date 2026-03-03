# Data Quality Rules

## Source Reliability Tiers
1. **Tier 1 (Official):** Publisher/developer press releases, NPD/GfK reports, Wikipedia verified figures
2. **Tier 2 (API):** Steam Store API, RAWG, IGDB, OpenCritic (structured, versioned)
3. **Tier 3 (Estimates):** SteamSpy, Gamedatacrunch, VGChartz (crowd-sourced or estimated)
4. **Tier 4 (Scraped):** Web scraping, manual collection

When sources conflict, prefer higher-tier data. Document the tier for each source in the merge pipeline.

## Required Fields
Every game record must have:
- `Name` (string, non-null)
- `Platform` (string, normalized)
- `Year` (int, 1970–current)
- `Global_Sales` (float, >= 0) OR a valid sales proxy

## Deduplication
- Normalize game names: lowercase, strip punctuation, remove edition suffixes
- Fuzzy match threshold: 85% (rapidfuzz WRatio)
- Exact match first, then fuzzy match
- When duplicates found, keep the record with the most complete data

## Physical vs Digital Sales
- Always track sales type: `physical`, `digital`, `combined`, `estimated`
- Never mix physical-only and digital-only figures without labeling
- VGChartz = physical only, SteamSpy = digital estimates, Wikipedia = often combined

## Validation
- Use Pandera schemas for all loaded datasets
- Validate value ranges: sales >= 0, scores in expected range, year reasonable
- Log warnings for suspicious values (e.g., sales > 100M for non-AAA titles)
