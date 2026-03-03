# Skill: Add a New Streamlit Page

## Steps

1. **Create the page module** at `source/pages/<page_name>.py`:
   ```python
   import streamlit as st
   from config import ACCENT, PLOTLY_LAYOUT  # as needed

   def <page_name>_page() -> None:
       """Render the <page description> page."""
       st.title("Page Title")
       # ... page content
   ```

2. **Register in main.py** — add to the `st.navigation()` list:
   ```python
   st.Page(_lazy("pages.<page_name>", "<page_name>_page"), title="Title", icon="..."),
   ```

3. **Use shared components** from `source/components.py`:
   - `metric_card()` for key metrics
   - `info_card()` for explanatory content
   - `section_header()` for section titles

4. **Follow conventions**:
   - All UI text in English
   - Use Plotly with `**PLOTLY_LAYOUT` for charts
   - Use `@st.cache_data` for data loading
   - Use `@st.cache_resource` for model loading

5. **Test**: Run `make run` and navigate to the new page.
