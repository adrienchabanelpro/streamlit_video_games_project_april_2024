"""Leaderboard -- Displays high scores from all mini-games in a styled neon table."""

import json
from pathlib import Path

import streamlit as st
from config import DATA_DIR

_HIGHSCORES_FILE: Path = DATA_DIR / "highscores.json"

# Human-readable French names for each game key
_GAME_LABELS: dict[str, str] = {
    "snake": "Snake",
    "casse_brique": "Casse-Briques",
    "space_invaders": "Space Invaders",
}


def _load_highscores() -> dict[str, int]:
    """Load high scores from the shared JSON file."""
    if _HIGHSCORES_FILE.exists():
        try:
            with open(_HIGHSCORES_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"snake": 0, "casse_brique": 0, "space_invaders": 0}


def leaderboard_page() -> None:
    """Render the leaderboard page with a neon-styled score table."""
    st.title("Classement")
    st.markdown(
        "Voici les meilleurs scores enregistres pour chaque mini-jeu Pygame. "
        "Les scores sont sauvegardes localement dans `data/highscores.json`."
    )

    scores = _load_highscores()

    # Build rows sorted by score descending
    rows: list[dict[str, str | int]] = []
    for key, label in _GAME_LABELS.items():
        rows.append({"Jeu": label, "Meilleur Score": scores.get(key, 0)})
    # Include any extra game keys not in _GAME_LABELS
    for key, value in scores.items():
        if key not in _GAME_LABELS:
            rows.append({"Jeu": key.replace("_", " ").title(), "Meilleur Score": value})

    rows.sort(key=lambda r: r["Meilleur Score"], reverse=True)

    # Styled HTML table with neon theme
    table_html = """
    <style>
    .lb-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Press Start 2P', monospace;
        font-size: 14px;
    }
    .lb-table th {
        background-color: #1A1A2E;
        color: #00FFCC;
        padding: 14px 20px;
        text-align: left;
        border-bottom: 2px solid #00FFCC;
        text-shadow: 0 0 8px rgba(0, 255, 204, 0.5);
    }
    .lb-table td {
        background-color: #0D0D0D;
        color: #E0E0E0;
        padding: 12px 20px;
        border-bottom: 1px solid #333366;
    }
    .lb-table tr:hover td {
        background-color: #1A1A2E;
        color: #FF6EC7;
        text-shadow: 0 0 6px rgba(255, 110, 199, 0.4);
    }
    .lb-rank {
        color: #FFFF00;
        font-weight: bold;
    }
    .lb-score {
        color: #00FFCC;
        font-weight: bold;
    }
    </style>
    <table class="lb-table">
        <thead>
            <tr>
                <th>#</th>
                <th>Jeu</th>
                <th>Meilleur Score</th>
            </tr>
        </thead>
        <tbody>
    """
    for i, row in enumerate(rows, start=1):
        medal = ""
        if i == 1:
            medal = " 🥇"
        elif i == 2:
            medal = " 🥈"
        elif i == 3:
            medal = " 🥉"
        table_html += f"""
            <tr>
                <td class="lb-rank">{i}{medal}</td>
                <td>{row["Jeu"]}</td>
                <td class="lb-score">{row["Meilleur Score"]}</td>
            </tr>
        """
    table_html += """
        </tbody>
    </table>
    """

    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("---")
    st.info(
        "Les mini-jeux Pygame (Snake, Casse-Briques, Space Invaders) necessitent "
        "un affichage graphique local. Le jeu Pong fonctionne directement dans le navigateur."
    )
