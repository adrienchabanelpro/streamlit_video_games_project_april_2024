"""Main entry point: multi-page Streamlit app with retro neon theme."""

import importlib

import streamlit as st
from config import IMAGES_DIR
from style import apply_style

st.set_page_config(page_title="Prediction Jeux Video", page_icon="🎮", layout="wide")

# Apply global style (persists across all pages)
apply_style()

# Sidebar branding
navigation_gif_path = IMAGES_DIR / "chun-li-walking-animation.gif"
if navigation_gif_path.exists():
    st.sidebar.image(str(navigation_gif_path), width=200)
else:
    st.sidebar.write(
        f"Erreur : l'image {navigation_gif_path.name} est introuvable. Verifiez le dossier images/."
    )


# ---------------------------------------------------------------------------
# Lazy page loader — modules are only imported when the user visits a page.
# This keeps startup memory under ~200 MB (critical for Streamlit Cloud 1 GB).
# ---------------------------------------------------------------------------
def _lazy(module: str, func: str):
    """Return a callable that lazy-imports and runs a page function."""

    def page():
        mod = importlib.import_module(module)
        getattr(mod, func)()

    page.__name__ = func
    page.__qualname__ = func
    return page


# Native multi-page navigation (Streamlit 1.36+)
pg = st.navigation(
    [
        st.Page(_lazy("pages.presentation", "presentation_page"), title="Presentation", icon="🎯"),
        st.Page(_lazy("pages.methodologie", "methodologie_page"), title="Methodologie", icon="📋"),
        st.Page(_lazy("pages.dataviz", "dataviz_page"), title="DataViz", icon="📊"),
        st.Page(_lazy("pages.feature_engineering", "feature_engineering_page"), title="Feature Engineering", icon="⚙️"),
        st.Page(_lazy("pages.modelisation", "modelisation_page"), title="Modelisation", icon="🧠"),
        st.Page(_lazy("pages.prediction", "prediction_page"), title="Prediction", icon="🔮"),
        st.Page(_lazy("pages.what_if", "what_if_page"), title="What-If", icon="🔬"),
        st.Page(_lazy("pages.recommendation", "recommendation_page"), title="Recommandations", icon="💡"),
        st.Page(_lazy("pages.comparison", "comparison_page"), title="Comparaison", icon="⚖️"),
        st.Page(_lazy("pages.trends", "trends_page"), title="Tendances", icon="📈"),
        st.Page(_lazy("pages.perception", "perception_page"), title="Perception", icon="💬"),
        st.Page(_lazy("pages.perspectives", "perspectives_page"), title="Perspectives", icon="🔭"),
    ]
)

pg.run()
