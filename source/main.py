# main.py
import streamlit as st
import os
import subprocess
import random
from style import apply_style
from presentation import presentation_et_objectif
from methodologie import methodologie
from dataviz import dataviz
from feature_engineering import feature_engineering
from modelisation import modelisation
from perspectives import perspectives
from perception import perception
from prediction import prediction_page
from what_if import what_if_page

st.set_page_config(page_title="Prediction Jeux Video", page_icon="🎮", layout="wide")

# Apply global style (persists across all pages)
apply_style()
# Sidebar branding
navigation_gif_path = os.path.join(
    os.path.dirname(__file__), '..', 'images', 'chun-li-walking-animation.gif'
)
if os.path.exists(navigation_gif_path):
    st.sidebar.image(navigation_gif_path, width=200)
else:
    st.sidebar.write(
        f"Erreur : l'image {os.path.basename(navigation_gif_path)} "
        "est introuvable. Verifiez le dossier images/."
    )


def _has_display() -> bool:
    """Check if a graphical display is available (False on cloud)."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
                or os.name == "nt"  # Windows
                or ("darwin" in os.uname().sysname.lower()))  # macOS


def jeu_surprise():
    """Jeu Surprise page — launches a random pygame game."""
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
            ['casse_brique.py', 'snake.py'], weights=[1, 1], k=1
        )[0]
        game_path = os.path.join(os.path.dirname(__file__), game_choice)
        subprocess.Popen(['python', game_path])
        st.write(
            f"Le jeu {game_choice.split('.')[0]} se lance dans une nouvelle fenetre."
        )


# Native multi-page navigation (Streamlit 1.36+)
pg = st.navigation([
    st.Page(presentation_et_objectif, title="Presentation", icon="🎯"),
    st.Page(methodologie, title="Methodologie", icon="📋"),
    st.Page(dataviz, title="DataViz", icon="📊"),
    st.Page(feature_engineering, title="Feature Engineering", icon="⚙️"),
    st.Page(modelisation, title="Modelisation", icon="🧠"),
    st.Page(prediction_page, title="Prediction", icon="🔮"),
    st.Page(what_if_page, title="What-If", icon="🔬"),
    st.Page(perception, title="Perception", icon="💬"),
    st.Page(perspectives, title="Perspectives", icon="🔭"),
    st.Page(jeu_surprise, title="Jeu Surprise", icon="🎮"),
])

pg.run()
