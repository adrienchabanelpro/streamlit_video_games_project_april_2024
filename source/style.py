"""Inject global CSS for the modern dark slate theme.

Uses JavaScript DOM injection via ``streamlit.components.v1.html()`` to bypass
Streamlit's markdown HTML sanitizer, which strips ``<style>`` tags in recent
versions.  The script inserts CSS and Google Fonts directly into the parent
document ``<head>`` — idempotent across Streamlit re-runs.
"""

import json

import streamlit.components.v1 as components

_GOOGLE_FONTS_URL = (
    "https://fonts.googleapis.com/css2"
    "?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
)

_CSS = """\
/* ---- Modern Dark Slate Theme ---- */

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1E293B;
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * {
    color: #F1F5F9;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    color: #F1F5F9 !important;
}
h1 { color: #3B82F6 !important; }

/* Body text */
.stMarkdown, .stText, p, li, span {
    color: #F1F5F9;
    font-family: 'Inter', sans-serif;
}

/* Links */
a {
    color: #3B82F6 !important;
    text-decoration: none;
}
a:hover {
    color: #60A5FA !important;
}

/* Buttons — blue accent */
.stButton > button {
    background-color: #3B82F6;
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 14px;
    padding: 8px 20px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background-color: #2563EB;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
}

/* Metrics — glassmorphism card */
[data-testid="stMetric"] {
    background-color: rgba(30, 41, 59, 0.8);
    backdrop-filter: blur(10px);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px;
}
[data-testid="stMetricValue"] {
    color: #3B82F6 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
}
[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Selectbox / number input */
.stSelectbox label, .stNumberInput label, .stSlider label {
    color: #F1F5F9 !important;
}

/* Dataframe styling */
.stDataFrame {
    border: 1px solid #334155;
    border-radius: 8px;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #94A3B8;
    font-family: 'Inter', sans-serif;
}
.stTabs [aria-selected="true"] {
    color: #3B82F6 !important;
    border-bottom-color: #3B82F6 !important;
}

/* Divider lines */
hr {
    border-color: #334155;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #3B82F6 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed #334155;
    border-radius: 12px;
    padding: 16px;
}

/* Warning / info / error boxes */
.stAlert {
    border-radius: 8px;
}

/* Navigation items */
[data-testid="stSidebarNav"] a {
    color: #F1F5F9 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebarNav"] a:hover {
    color: #3B82F6 !important;
}

/* ---- Expanders ---- */
details[data-testid="stExpander"] {
    background-color: rgba(30, 41, 59, 0.8);
    border: 1px solid #334155;
    border-radius: 12px;
}
details[data-testid="stExpander"] summary {
    color: #F1F5F9 !important;
    font-family: 'Inter', sans-serif !important;
}
details[data-testid="stExpander"] > div {
    background-color: #1E293B;
}

/* ---- Text inputs and text areas ---- */
.stTextInput input,
.stTextArea textarea {
    background-color: #1E293B !important;
    color: #F1F5F9 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
    outline: none !important;
}

/* ---- Sliders ---- */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: #3B82F6 !important;
    border: 2px solid #3B82F6 !important;
}
.stSlider [data-baseweb="slider"] > div > div:first-child {
    background-color: #334155 !important;
}
.stSlider [data-baseweb="slider"] > div > div:nth-child(2) {
    background: linear-gradient(90deg, #3B82F6, #8B5CF6) !important;
}

/* ---- Radio buttons and checkboxes ---- */
.stRadio label,
.stCheckbox label {
    color: #F1F5F9 !important;
    transition: color 0.2s;
}
.stRadio label:hover,
.stCheckbox label:hover {
    color: #3B82F6 !important;
}
.stRadio [data-baseweb="radio"] div:first-child {
    border-color: #3B82F6 !important;
}
.stRadio [data-baseweb="radio"] div:first-child::after {
    background-color: #3B82F6 !important;
}
.stCheckbox [data-baseweb="checkbox"] span {
    border-color: #3B82F6 !important;
}
.stCheckbox [data-baseweb="checkbox"] span[aria-checked="true"] {
    background-color: #3B82F6 !important;
    border-color: #3B82F6 !important;
}

/* ---- Progress bars ---- */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #3B82F6, #8B5CF6) !important;
    border-radius: 4px;
}
.stProgress > div > div {
    background-color: #1E293B !important;
    border-radius: 4px;
}

/* ---- Custom scrollbar ---- */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #0F172A;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #475569;
}

/* ---- Mobile-responsive layout ---- */
@media screen and (max-width: 768px) {
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1rem !important; }
    h4, h5, h6 { font-size: 0.9rem !important; }

    [data-testid="stSidebar"] {
        min-width: 180px !important;
        max-width: 220px !important;
    }

    .stPlotlyChart,
    [data-testid="stPlotlyChart"],
    .stDataFrame {
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: auto;
    }

    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 8px !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        width: 100% !important;
        flex: 1 1 100% !important;
    }

    .block-container {
        padding: 0.5rem 0.8rem !important;
    }
    [data-testid="stMetric"] {
        padding: 10px !important;
    }
    .stButton > button {
        font-size: 13px !important;
        padding: 6px 14px !important;
    }
}
"""


def apply_style() -> None:
    """Inject global CSS and Google Fonts into the parent document head.

    Uses JavaScript DOM injection via a zero-height ``components.html()``
    iframe.  This bypasses Streamlit's markdown sanitizer which strips
    ``<style>`` tags in recent versions.  The injection is idempotent —
    a unique ``id`` prevents duplicates across Streamlit re-runs.
    """
    css_json = json.dumps(_CSS)
    font_url_json = json.dumps(_GOOGLE_FONTS_URL)

    components.html(
        f"""
        <script>
        (function() {{
            var doc = window.parent.document;

            // Inject Google Fonts (once)
            if (!doc.getElementById('modern-google-fonts')) {{
                var link = doc.createElement('link');
                link.id = 'modern-google-fonts';
                link.rel = 'stylesheet';
                link.href = {font_url_json};
                doc.head.appendChild(link);
            }}

            // Inject CSS (once)
            if (!doc.getElementById('modern-theme-css')) {{
                var style = doc.createElement('style');
                style.id = 'modern-theme-css';
                style.textContent = {css_json};
                doc.head.appendChild(style);
            }}
        }})();
        </script>
        """,
        height=0,
    )
