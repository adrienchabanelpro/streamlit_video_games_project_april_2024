"""Recommendation engine: find similar games using cosine similarity."""

import numpy as np
import pandas as pd
import streamlit as st
from config import DATA_DIR
from data_validation import validate_dataframe
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


@st.cache_data
def _load_games_data() -> pd.DataFrame:
    """Load and prepare games dataset for similarity computation."""
    df = pd.read_csv(DATA_DIR / "Ventes_jeux_video_final.csv")
    # Advisory validation — warn but continue
    is_valid, errors = validate_dataframe(df)
    if not is_valid:
        st.warning(
            f"Validation des donnees : {len(errors)} probleme(s) detecte(s). "
            "Les recommandations peuvent etre affectees."
        )
    df = df.dropna(subset=["Name", "Genre", "Platform", "Publisher", "Year"])
    df["Year"] = df["Year"].astype(int)
    df["meta_score"] = df["meta_score"].fillna(df["meta_score"].median())
    df["user_review"] = df["user_review"].fillna(df["user_review"].median())
    return df.reset_index(drop=True)


@st.cache_data
def _build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build normalized feature matrix for cosine similarity.

    Features: one-hot Genre + one-hot Platform + scaled numericals.
    """
    genre_dummies = pd.get_dummies(df["Genre"], prefix="genre")
    platform_dummies = pd.get_dummies(df["Platform"], prefix="platform")

    numericals = df[["Year", "meta_score", "user_review", "Global_Sales"]].copy()
    numericals = numericals.fillna(numericals.median())
    scaler = StandardScaler()
    numericals_scaled = pd.DataFrame(
        scaler.fit_transform(numericals),
        columns=numericals.columns,
        index=numericals.index,
    )

    feature_df = pd.concat([numericals_scaled, genre_dummies, platform_dummies], axis=1)
    feature_names = list(feature_df.columns)
    return feature_df.values, feature_names


def _find_similar(
    df: pd.DataFrame,
    feature_matrix: np.ndarray,
    game_idx: int,
    n: int = 10,
) -> pd.DataFrame:
    """Find the n most similar games to the one at game_idx."""
    query = feature_matrix[game_idx].reshape(1, -1)
    similarities = cosine_similarity(query, feature_matrix)[0]

    # Exclude the game itself
    similarities[game_idx] = -1

    top_indices = np.argsort(similarities)[::-1][:n]
    result = df.iloc[top_indices][
        [
            "Name",
            "Platform",
            "Genre",
            "Publisher",
            "Year",
            "Global_Sales",
            "meta_score",
            "user_review",
        ]
    ].copy()
    result["Similarite"] = [f"{similarities[i]:.3f}" for i in top_indices]
    return result.reset_index(drop=True)


def recommendation_page():
    """Recommendation engine page: find games similar to a selected game."""
    st.title("Moteur de recommandation")
    st.write(
        "Selectionnez un jeu et decouvrez les titres les plus similaires "
        "en termes de genre, plateforme, scores et ventes."
    )

    with st.spinner("Chargement des donnees..."):
        df = _load_games_data()
        feature_matrix, _ = _build_feature_matrix(df)

    # --- Game selection ---
    st.subheader("Selectionnez un jeu")

    # Filter helpers
    col1, col2 = st.columns(2)
    with col1:
        genre_filter = st.selectbox(
            "Filtrer par genre (optionnel)",
            ["Tous"] + sorted(df["Genre"].unique().tolist()),
        )
    with col2:
        platform_filter = st.selectbox(
            "Filtrer par plateforme (optionnel)",
            ["Tous"] + sorted(df["Platform"].unique().tolist()),
        )

    filtered = df.copy()
    if genre_filter != "Tous":
        filtered = filtered[filtered["Genre"] == genre_filter]
    if platform_filter != "Tous":
        filtered = filtered[filtered["Platform"] == platform_filter]

    game_options = filtered["Name"].unique().tolist()
    if not game_options:
        st.warning("Aucun jeu ne correspond aux filtres selectionnes.")
        return

    selected_game = st.selectbox(
        "Choisir un jeu",
        sorted(game_options),
    )

    # Find the game index (first occurrence if multiple platforms)
    matches = df[df["Name"] == selected_game]
    if len(matches) > 1:
        platform_choice = st.selectbox(
            "Ce jeu existe sur plusieurs plateformes :",
            matches["Platform"].tolist(),
        )
        game_idx = matches[matches["Platform"] == platform_choice].index[0]
    else:
        game_idx = matches.index[0]

    # Show selected game info
    game_info = df.iloc[game_idx]
    st.markdown("---")
    st.subheader(f"{game_info['Name']} ({game_info['Platform']})")

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Genre", game_info["Genre"])
    with col_b:
        st.metric("Annee", int(game_info["Year"]))
    with col_c:
        st.metric("Metacritic", f"{game_info['meta_score']:.0f}")
    with col_d:
        st.metric("Ventes", f"{game_info['Global_Sales']:.2f}M")

    # --- Find similar games ---
    n_results = st.slider("Nombre de recommandations", 5, 20, 10)

    similar = _find_similar(df, feature_matrix, game_idx, n=n_results)

    st.subheader("Jeux similaires")
    st.dataframe(
        similar.rename(
            columns={
                "Name": "Jeu",
                "Platform": "Plateforme",
                "Publisher": "Editeur",
                "Year": "Annee",
                "Global_Sales": "Ventes (M)",
                "meta_score": "Metacritic",
                "user_review": "Score utilisateur",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
