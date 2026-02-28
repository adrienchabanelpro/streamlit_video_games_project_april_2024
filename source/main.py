"""Main entry point: multi-page Streamlit app with retro neon theme."""

import importlib
import os
import random
import subprocess
from pathlib import Path

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


def _has_display() -> bool:
    """Check if a graphical display is available (False on cloud)."""
    return bool(
        os.environ.get("DISPLAY")
        or os.environ.get("WAYLAND_DISPLAY")
        or os.name == "nt"  # Windows
        or ("darwin" in os.uname().sysname.lower())
    )  # macOS


def jeu_surprise_page() -> None:
    """Jeu Surprise page -- launches a random pygame game."""
    st.title("THE GAME!!")
    st.write("Es-tu pret a donner le meilleur de toi meme?")

    if not _has_display():
        st.warning(
            "Les mini-jeux Pygame necessitent un affichage graphique local. "
            "Ils ne sont pas disponibles dans le deploiement cloud. "
            "Clonez le projet et lancez-le localement pour y jouer!"
        )
        return

    if st.button("Clique ICI"):
        game_choice = random.choices(
            ["games/casse_brique.py", "games/snake.py", "games/space_invaders.py"],
            weights=[1, 1, 1],
            k=1,
        )[0]
        game_path = str(Path(__file__).parent / game_choice)
        subprocess.Popen(["python", game_path])
        game_name = game_choice.split("/")[1].split(".")[0]
        st.write(f"Le jeu {game_name} se lance dans une nouvelle fenetre.")


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
        st.Page(jeu_surprise_page, title="Jeu Surprise", icon="🎮"),
        st.Page(_lazy("pages.pong_streamlit", "pong_page"), title="Pong", icon="🏓"),
        st.Page(_lazy("pages.leaderboard", "leaderboard_page"), title="Classement", icon="🏆"),
    ]
)

pg.run()
