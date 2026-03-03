# Project Structure Rules

## Current Layout
All app code lives in `source/`. Data in `data/`. Models in `models/` and `reports/`.

## When Adding New Pages
1. Create `source/pages/<page_name>.py` with a `<page_name>_page()` function
2. Add a `st.Page(_lazy("pages.<page_name>", "<page_name>_page"), ...)` entry in `source/main.py`'s `st.navigation()` list

## When Adding New Models
1. Save trained models to `models/` (sklearn: `.pkl` or `.joblib`, LightGBM: `.txt`)
2. Save transformers/encoders alongside in `models/`
3. Document in root `CLAUDE.md` under ML Models & Artifacts table

## When Adding New Datasets
1. Place in `data/`
2. Document schema and row count in root `CLAUDE.md`
3. Add data loading with `@st.cache_data`

## When Adding New Data Sources
1. Create `scripts/data_collection/collect_<source_name>.py`
2. Add the source to `scripts/data_collection/run_pipeline.py`
3. Update `scripts/data_collection/merge_all_sources.py` with merge logic
4. Document the source in `source/pages/data_sources.py`

## Assets
- Images → `images/`
- Keep image sizes reasonable (<500KB preferred, compress large ones)
