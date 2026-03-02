"""Tests for the config module (source/config.py)."""

from pathlib import Path

from config import (
    ACCENT,
    BG,
    BG_CARD,
    BORDER,
    DATA_DIR,
    FONTS_DIR,
    IMAGES_DIR,
    MODELS_DIR,
    PLOTLY_LAYOUT,
    REPORTS_DIR,
    ROOT,
    SECONDARY,
    SUCCESS,
    TEXT_COLOR,
    TEXT_MUTED,
)


class TestPaths:
    def test_root_is_project_root(self):
        assert ROOT.is_dir()
        assert (ROOT / "source").is_dir()
        assert (ROOT / "data").is_dir()

    def test_data_dir_exists(self):
        assert DATA_DIR.is_dir()

    def test_models_dir_exists(self):
        assert MODELS_DIR.is_dir()

    def test_reports_dir_exists(self):
        assert REPORTS_DIR.is_dir()

    def test_images_dir_exists(self):
        assert IMAGES_DIR.is_dir()

    def test_fonts_dir_exists(self):
        assert FONTS_DIR.is_dir()

    def test_all_paths_are_pathlib(self):
        for p in [ROOT, DATA_DIR, MODELS_DIR, REPORTS_DIR, IMAGES_DIR, FONTS_DIR]:
            assert isinstance(p, Path)


class TestThemeConstants:
    def test_colors_are_hex_strings(self):
        for color in [BG, BG_CARD, ACCENT, SECONDARY, SUCCESS, TEXT_COLOR, TEXT_MUTED, BORDER]:
            assert isinstance(color, str)
            assert color.startswith("#")
            assert len(color) == 7  # #RRGGBB

    def test_plotly_layout_has_required_keys(self):
        assert "template" in PLOTLY_LAYOUT
        assert "paper_bgcolor" in PLOTLY_LAYOUT
        assert "plot_bgcolor" in PLOTLY_LAYOUT
        assert "font" in PLOTLY_LAYOUT

    def test_plotly_layout_template(self):
        assert PLOTLY_LAYOUT["template"] == "plotly_dark"

    def test_plotly_layout_font_color(self):
        assert PLOTLY_LAYOUT["font"]["color"] == TEXT_COLOR
