"""Main entry point: multi-page Streamlit app for video game sales prediction."""

import importlib

import streamlit as st
from style import apply_style

st.set_page_config(page_title="Prediction Jeux Video", page_icon="🎮", layout="wide")

# Apply global style
apply_style()


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
        st.Page(_lazy("pages.home", "home_page"), title="Accueil", icon="🏠"),
        st.Page(_lazy("pages.data_sources", "data_sources_page"), title="Sources de Donnees", icon="🗄️"),
        st.Page(_lazy("pages.dataviz", "dataviz_page"), title="Analyse Exploratoire", icon="📊"),
        st.Page(_lazy("pages.feature_engineering", "feature_engineering_page"), title="Feature Engineering", icon="⚙️"),
        st.Page(_lazy("pages.model_training", "model_training_page"), title="Entrainement", icon="🧠"),
        st.Page(_lazy("pages.prediction", "prediction_page"), title="Predictions", icon="🔮"),
        st.Page(_lazy("pages.interpretability", "interpretability_page"), title="Interpretabilite", icon="🔍"),
        st.Page(_lazy("pages.what_if", "what_if_page"), title="What-If", icon="🔬"),
        st.Page(_lazy("pages.market_insights", "market_insights_page"), title="Tendances", icon="📈"),
        st.Page(_lazy("pages.perception", "perception_page"), title="Sentiment NLP", icon="💬"),
        st.Page(_lazy("pages.about", "about_page"), title="A Propos", icon="ℹ️"),
    ]
)

pg.run()
