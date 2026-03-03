# Visualization Rules

## Plotly Standards
- Always use `config.PLOTLY_LAYOUT` as the base layout (dark theme, Inter font)
- Pass layout via `**PLOTLY_LAYOUT` in `fig.update_layout()`
- Use `st.plotly_chart(fig, use_container_width=True)` for responsive sizing

## Color Palette (from config.py)
- `ACCENT` (#3B82F6 blue) — primary highlights, active elements
- `SECONDARY` (#8B5CF6 purple) — secondary elements, comparisons
- `SUCCESS` (#10B981 green) — positive values, improvements
- `WARNING` (#F59E0B amber) — caution, attention needed
- `DANGER` (#EF4444 red) — negative values, errors
- `TEXT_COLOR` (#F1F5F9) — primary text
- `TEXT_MUTED` (#94A3B8) — secondary/caption text

## Chart Guidelines
- Every chart must have a clear title and axis labels
- Use hover tooltips for detailed information
- Prefer bar charts for comparisons, line charts for trends, scatter for correlations
- Limit legends to 8 items max; use "Other" for the rest
- Use consistent number formatting (e.g., "1.2M" for millions)

## Interactivity
- Add filters (st.multiselect, st.slider) above charts when datasets are large
- Show record count after filtering (e.g., "1,234 games selected out of 64,000")
- Use st.tabs() to group related charts and reduce page scrolling
