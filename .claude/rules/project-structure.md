# Project Structure Rules

## Current Layout
All app code lives in `source/`. Data in `data/`. Models in `models/` and `reports/`.

## When Adding New Pages
1. Create `source/<page_name>.py` with a main display function
2. Import it in `source/main.py`
3. Add the page name to the sidebar radio options list
4. Add the routing condition in the if/elif chain

## When Adding New Models
1. Save trained models to `models/` (sklearn: `.pkl` or `.joblib`, LightGBM: `.txt`)
2. Save transformers/encoders alongside in `models/`
3. Document in root `CLAUDE.md` under ML Models & Artifacts table

## When Adding New Datasets
1. Place in `data/`
2. Document schema and row count in root `CLAUDE.md`
3. Add data loading with `@st.cache_data`

## Assets
- Images → `images/`
- Fonts → `fonts/`
- Keep image sizes reasonable (<500KB preferred, compress large ones)
