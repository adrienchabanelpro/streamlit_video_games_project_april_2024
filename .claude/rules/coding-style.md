# Coding Style Rules

## Language
- UI text: French
- Code (variables, functions, comments): English preferred for new code
- Docstrings: English

## Python
- Python 3.11+ features allowed
- Use type hints on all new/modified functions
- Use f-strings, not .format() or %
- Snake_case for functions and variables, PascalCase for classes

## Streamlit
- Always use `@st.cache_data` for data loading functions
- Always use `@st.cache_resource` for model/transformer loading
- Use `st.spinner()` for operations >1s
- Prefer Plotly over Matplotlib for new charts (interactivity)
- Use `st.columns()` for side-by-side layouts

## File Paths
- Use `pathlib.Path` for all file paths
- Define paths relative to project root via a config constant
- Never hardcode absolute paths

## Error Handling
- Wrap model loading in try/except with `st.error()` fallback
- Validate user uploads before processing
- Use `st.warning()` for recoverable issues

## Git
- Commit messages in English, imperative mood
- Keep commits atomic (one logical change per commit)
