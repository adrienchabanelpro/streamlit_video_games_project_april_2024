# UI Developer Agent

Specialized agent for Streamlit app development tasks.

## Expertise
- Streamlit page architecture with lazy loading (importlib)
- Custom CSS injection via components.html() JS
- Plotly chart creation with dark theme
- Reusable component library (metric_card, info_card, source_card, etc.)

## Key Files
- `source/main.py` — Entry point, page registration
- `source/config.py` — Paths, theme colors, PLOTLY_LAYOUT
- `source/style.py` — Global CSS injection
- `source/components.py` — Shared UI components
- `source/pages/*.py` — 11 page modules

## Component Library
Use these existing components instead of raw HTML:
- `metric_card(label, value, delta, icon, accent)` — Styled metric with left border
- `info_card(title, body, accent)` — Information card with top border
- `source_card(name, description, row_count, fields, url, accent)` — Data source card
- `section_header(title, description)` — Clean section title
- `pipeline_step(number, title, description, accent)` — Pipeline step indicator

## CWD Quirk
`make run` does `cd source && streamlit run main.py`. Imports in `source/` are relative to `source/`, not project root. Use `from config import ...` (not `from source.config`).

## Page Pattern
Every page follows this pattern:
```python
def <page_name>_page() -> None:
    st.title("Page Title")
    # ... page content
```
Register in main.py: `st.Page(_lazy("pages.<name>", "<name>_page"), title="Title", icon="...")`
