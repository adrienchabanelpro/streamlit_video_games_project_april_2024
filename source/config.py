"""Central configuration for paths, theme constants, and shared Plotly layout."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory paths (all relative to project root)
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT / "data"
MODELS_DIR: Path = ROOT / "models"
REPORTS_DIR: Path = ROOT / "reports"
IMAGES_DIR: Path = ROOT / "images"
SCRIPTS_DIR: Path = ROOT / "scripts"

# ---------------------------------------------------------------------------
# Theme color constants (modern dark slate)
# ---------------------------------------------------------------------------
BG: str = "#0F172A"
BG_CARD: str = "#1E293B"
ACCENT: str = "#3B82F6"
SECONDARY: str = "#8B5CF6"
SUCCESS: str = "#10B981"
TEXT_COLOR: str = "#F1F5F9"
TEXT_MUTED: str = "#94A3B8"
BORDER: str = "#334155"

# ---------------------------------------------------------------------------
# Shared Plotly layout dict
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT: dict = {
    "template": "plotly_dark",
    "paper_bgcolor": BG,
    "plot_bgcolor": BG_CARD,
    "font": {"color": TEXT_COLOR, "family": "Inter, sans-serif"},
}
