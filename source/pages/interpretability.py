"""Interpretability page: SHAP analysis, feature importance, individual explanations."""

import json

import pandas as pd
import streamlit as st
from components import info_card, section_header
from config import DATA_DIR, REPORTS_DIR


@st.cache_data
def _load_training_log() -> dict | None:
    for name in ["training_log_v3.json", "training_log.json"]:
        path = REPORTS_DIR / name
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


@st.cache_data
def _load_dataset() -> pd.DataFrame:
    for name in ["Ventes_jeux_video_v3.csv", "Ventes_jeux_video_final.csv"]:
        path = DATA_DIR / name
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def interpretability_page() -> None:
    """Render the model interpretability page."""
    st.title("Interpretabilite des Modeles")
    st.caption("Comprendre pourquoi le modele fait ses predictions : SHAP, importance des variables")

    log = _load_training_log()

    # SHAP section
    section_header("SHAP (SHapley Additive exPlanations)")

    info_card(
        "Qu'est-ce que SHAP ?",
        """
        SHAP attribue a chaque variable une <b>contribution marginale</b> a la prediction.
        Contrairement a l'importance classique (basee sur le gain), SHAP fournit des
        explications <b>consistantes et locales</b> : on peut expliquer chaque prediction
        individuellement, pas seulement le modele global.
        """,
    )

    shap_bar = REPORTS_DIR / "shap_bar_v3.png"
    if not shap_bar.exists():
        shap_bar = REPORTS_DIR / "shap_bar.png"
    shap_summary = REPORTS_DIR / "shap_summary_v3.png"
    if not shap_summary.exists():
        shap_summary = REPORTS_DIR / "shap_summary.png"

    if shap_bar.exists() or shap_summary.exists():
        tab1, tab2 = st.tabs(["Importance globale (bar)", "Distribution (beeswarm)"])
        with tab1:
            if shap_bar.exists():
                st.image(str(shap_bar), use_container_width=True)
                st.caption(
                    "Chaque barre represente la contribution moyenne absolue "
                    "de la variable aux predictions. Plus la barre est longue, "
                    "plus la variable est importante."
                )
        with tab2:
            if shap_summary.exists():
                st.image(str(shap_summary), use_container_width=True)
                st.caption(
                    "Chaque point represente un jeu. La couleur indique la valeur "
                    "de la variable (rouge = haute, bleu = basse). La position "
                    "horizontale montre l'impact sur la prediction."
                )
    else:
        st.info("Les plots SHAP seront generes apres l'entrainement (`make train`).")

    st.divider()

    # Feature descriptions
    section_header("Description des variables", "Que mesure chaque feature et pourquoi elle est utile")

    if log:
        features = log.get("features", [])
        if features:
            feature_desc = _get_feature_descriptions()
            rows = []
            for f in features:
                rows.append({
                    "Variable": f,
                    "Description": feature_desc.get(f, "—"),
                    "Categorie": _categorize_feature(f),
                })
            df_features = pd.DataFrame(rows)

            # Group by category
            for cat in df_features["Categorie"].unique():
                with st.expander(f"**{cat}** ({len(df_features[df_features['Categorie'] == cat])} variables)"):
                    subset = df_features[df_features["Categorie"] == cat][["Variable", "Description"]]
                    st.dataframe(subset, use_container_width=True, hide_index=True)

    st.divider()

    # Methodology explanation
    section_header("Methodologie d'interpretabilite")

    info_card(
        "Approche multi-niveaux",
        """
        <b>1. SHAP TreeExplainer</b> — Calcul exact des valeurs SHAP pour les modeles
        a base d'arbres (LightGBM, XGBoost, CatBoost). Complexite O(TLD²).<br><br>
        <b>2. Importance par permutation</b> — Mesure la degradation de performance quand
        on melange aleatoirement une variable. Independant du modele.<br><br>
        <b>3. Analyse des residus</b> — Verification que les erreurs sont aleatoires et
        non systematiques (pas de biais structurel).
        """,
    )


def _get_feature_descriptions() -> dict[str, str]:
    """Return human-readable descriptions for known features."""
    return {
        "Year": "Annee de sortie du jeu",
        "meta_score": "Score Metacritic (critique professionnelle)",
        "user_review": "Score des utilisateurs",
        "Global_Sales_mean_genre": "Ventes moyennes des jeux du meme genre (donnees d'entrainement)",
        "Global_Sales_mean_platform": "Ventes moyennes des jeux de la meme plateforme",
        "Year_Global_Sales_mean_genre": "Interaction : Annee × ventes moyennes du genre",
        "Year_Global_Sales_mean_platform": "Interaction : Annee × ventes moyennes de la plateforme",
        "Cumulative_Sales_Genre": "Ventes cumulees du genre jusqu'a l'annee de sortie",
        "Cumulative_Sales_Platform": "Ventes cumulees de la plateforme jusqu'a l'annee de sortie",
        "Publisher_encoded": "Editeur encode par target encoding (ventes moyennes de l'editeur)",
        "publisher_avg_sales_prior": "Ventes moyennes des jeux precedents de l'editeur",
        "publisher_game_count_prior": "Nombre de jeux precedents de l'editeur",
        "publisher_hit_rate": "Pourcentage de hits de l'editeur (ventes > mediane)",
        "developer_avg_sales_prior": "Ventes moyennes des jeux precedents du developpeur",
        "competition_density": "Nombre de jeux sortis la meme annee",
        "genre_market_share": "Part de marche du genre l'annee de sortie",
        "review_count_total": "Nombre total d'avis Steam (positifs + negatifs)",
        "review_ratio": "Ratio d'avis positifs sur Steam",
        "playtime_avg": "Temps de jeu moyen sur Steam (minutes)",
        "concurrent_users": "Pic de joueurs simultanes sur Steam",
        "rawg_playtime": "Temps de jeu moyen estime (RAWG)",
        "rawg_ratings_count": "Nombre de notes communautaires RAWG",
        "rawg_rating": "Note communautaire RAWG (0-5)",
        "rawg_metacritic": "Score Metacritic via RAWG (0-100)",
        "hltb_main": "Temps pour finir l'histoire principale (heures)",
        "hltb_main_extra": "Temps histoire + extras (heures)",
        "hltb_completionist": "Temps pour le 100% (heures)",
        "hltb_depth_ratio": "Ratio completionniste / histoire (profondeur)",
        "release_month": "Mois de sortie (1-12)",
        "release_quarter": "Trimestre de sortie (1-4)",
        "is_holiday_release": "Sortie pendant la periode des fetes (oct-dec)",
        "cross_platform_count": "Nombre de plateformes supportees",
        "is_multi_platform": "Jeu multi-plateforme (oui/non)",
        "esrb_encoded": "Classification d'age ESRB (ordinal)",
        "has_franchise": "Appartient a une franchise connue",
        "is_remake": "Est un remake",
        "is_remaster": "Est un remaster",
        "igdb_total_rating": "Note globale IGDB (0-100)",
        "igdb_hypes": "Score de hype pre-sortie (IGDB)",
        "igdb_follows": "Nombre de followers du jeu (IGDB)",
        "steam_price": "Prix actuel sur Steam",
        "steam_initialprice": "Prix de lancement sur Steam",
        "steam_review_pct": "Pourcentage d'avis positifs Steam",
    }


def _categorize_feature(name: str) -> str:
    """Categorize a feature by its type."""
    if name in ("Year", "release_month", "release_quarter", "is_holiday_release"):
        return "Temporel"
    if "publisher" in name or "developer" in name or "Publisher" in name:
        return "Track record editeur/dev"
    if name.startswith(("Global_Sales_mean", "Year_Global_Sales", "Cumulative")):
        return "Historique genre/plateforme"
    if name.startswith(("review_", "playtime", "concurrent", "steam_review_pct")):
        return "Engagement Steam"
    if name.startswith("rawg_"):
        return "Metadonnees RAWG"
    if name.startswith("hltb_"):
        return "Temps de completion (HLTB)"
    if name.startswith(("cross_platform", "is_multi")):
        return "Multi-plateforme"
    if name.startswith(("esrb", "has_franchise", "is_remake", "is_remaster")):
        return "Caracteristiques du jeu"
    if name.startswith("igdb_"):
        return "IGDB"
    if name.startswith("steam_") and "price" in name:
        return "Prix"
    if name in ("meta_score", "user_review"):
        return "Scores critiques"
    return "Autre"
