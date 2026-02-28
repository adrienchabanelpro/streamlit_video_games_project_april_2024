# main.py
import os
import random
import subprocess
from pathlib import Path

import streamlit as st
from comparison import comparison_page
from config import IMAGES_DIR
from dataviz import dataviz_page
from feature_engineering import feature_engineering_page
from leaderboard import leaderboard_page
from methodologie import methodologie_page
from modelisation import modelisation_page
from perception import perception_page
from perspectives import perspectives_page
from pong_streamlit import pong_page
from prediction import prediction_page
from presentation import presentation_page
from recommendation import recommendation_page
from style import apply_style
from trends import trends_page
from what_if import what_if_page

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
            ["casse_brique.py", "snake.py", "space_invaders.py"],
            weights=[1, 1, 1],
            k=1,
        )[0]
        game_path = str(Path(__file__).parent / game_choice)
        subprocess.Popen(["python", game_path])
        st.write(f"Le jeu {game_choice.split('.')[0]} se lance dans une nouvelle fenetre.")


# Native multi-page navigation (Streamlit 1.36+)
pg = st.navigation(
    [
        st.Page(presentation_page, title="Presentation", icon="🎯"),
        st.Page(methodologie_page, title="Methodologie", icon="📋"),
        st.Page(dataviz_page, title="DataViz", icon="📊"),
        st.Page(feature_engineering_page, title="Feature Engineering", icon="⚙️"),
        st.Page(modelisation_page, title="Modelisation", icon="🧠"),
        st.Page(prediction_page, title="Prediction", icon="🔮"),
        st.Page(what_if_page, title="What-If", icon="🔬"),
        st.Page(recommendation_page, title="Recommandations", icon="💡"),
        st.Page(comparison_page, title="Comparaison", icon="⚖️"),
        st.Page(trends_page, title="Tendances", icon="📈"),
        st.Page(perception_page, title="Perception", icon="💬"),
        st.Page(perspectives_page, title="Perspectives", icon="🔭"),
        st.Page(jeu_surprise_page, title="Jeu Surprise", icon="🎮"),
        st.Page(pong_page, title="Pong", icon="🏓"),
        st.Page(leaderboard_page, title="Classement", icon="🏆"),
    ]
)

pg.run()
