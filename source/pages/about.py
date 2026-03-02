"""About page: methodology, tech stack, limitations, author info."""

import streamlit as st
from components import info_card, section_header


def about_page() -> None:
    """Render the About / Methodology page."""
    st.title("A Propos du Projet")
    st.caption("Methodologie, outils, limitations et perspectives")

    # Methodology
    section_header("Methodologie")
    st.write(
        """
        Ce projet suit une approche **CRISP-DM** (Cross-Industry Standard Process
        for Data Mining) en 6 phases :

        1. **Comprehension du probleme** — Predire les ventes mondiales de jeux video
           a partir de metadonnees (genre, plateforme, editeur, scores, etc.)
        2. **Collecte de donnees** — 5 sources : VGChartz (ventes physiques),
           SteamSpy (digital PC), RAWG API (metadonnees), IGDB (themes/franchises),
           HowLongToBeat (temps de completion)
        3. **Preparation** — Fusion par fuzzy matching, nettoyage, imputation,
           winsorisation des outliers
        4. **Feature engineering** — 30+ variables ingenierees : track record editeur,
           contexte de marche, engagement, temporalite, franchises
        5. **Modelisation** — 7 modeles avec tuning Optuna, stacking ensemble avec
           meta-learner Ridge
        6. **Evaluation** — R², RMSE, MAE, MAPE, analyse des residus, SHAP
        """
    )

    st.divider()

    # Tech stack
    section_header("Stack technique")

    c1, c2 = st.columns(2)
    with c1:
        info_card(
            "Donnees & ML",
            """
            <ul style="margin:0;padding-left:16px">
                <li><b>Python 3.11+</b></li>
                <li><b>pandas</b> — manipulation de donnees</li>
                <li><b>scikit-learn</b> — preprocessing, metriques, RF, HGB, ElasticNet</li>
                <li><b>LightGBM</b> — gradient boosting</li>
                <li><b>XGBoost</b> — gradient boosting</li>
                <li><b>CatBoost</b> — gradient boosting</li>
                <li><b>Optuna</b> — tuning bayesien des hyperparametres</li>
                <li><b>SHAP</b> — interpretabilite des modeles</li>
                <li><b>category_encoders</b> — target encoding</li>
                <li><b>rapidfuzz</b> — fuzzy matching pour la fusion</li>
            </ul>
            """,
        )

    with c2:
        info_card(
            "Application & NLP",
            """
            <ul style="margin:0;padding-left:16px">
                <li><b>Streamlit</b> — framework web interactif</li>
                <li><b>Plotly</b> — visualisations interactives</li>
                <li><b>Transformers</b> — DistilBERT pour l'analyse de sentiments</li>
                <li><b>pytest</b> — tests unitaires</li>
                <li><b>ruff</b> — linting et formatage</li>
                <li><b>GitHub Actions</b> — CI/CD</li>
                <li><b>Docker</b> — conteneurisation</li>
                <li><b>MLflow</b> — tracking des experiences</li>
            </ul>
            """,
            accent="#8B5CF6",
        )

    st.divider()

    # Decisions
    section_header("Decisions techniques cles")

    decisions = [
        ("Pourquoi le stacking plutot que le moyennage ?",
         "Le stacking apprend les poids optimaux via un meta-learner Ridge, "
         "tandis que le moyennage suppose que tous les modeles sont egalement bons. "
         "Le stacking attribue plus de poids aux modeles les plus performants sur "
         "les donnees out-of-fold."),
        ("Pourquoi le target encoding ?",
         "Avec ~600 editeurs uniques, le one-hot encoding creerait 600 colonnes "
         "creuses. Le target encoding remplace chaque editeur par la moyenne de "
         "ses ventes, passant de 576 a 1 colonne tout en capturant l'information."),
        ("Pourquoi la transformation log ?",
         "Les ventes de jeux video suivent une distribution tres asymetrique "
         "(quelques hits massifs, beaucoup de petites ventes). La transformation "
         "log1p() normalise cette distribution et ameliore les predictions des "
         "modeles sur toute la gamme."),
        ("Pourquoi le split temporel ?",
         "Un split aleatoire laisserait des jeux de 2020 dans le train et de 2010 "
         "dans le test — le modele 'tricherait' en voyant l'avenir. Le split "
         "temporel (train <= 2015, test > 2015) simule un usage reel."),
    ]

    for q, a in decisions:
        with st.expander(q):
            st.write(a)

    st.divider()

    # Limitations
    section_header("Limitations connues")
    st.write(
        """
        - **Donnees physiques uniquement** pour la variable cible (VGChartz).
          Les ventes digitales (Steam, PS Store, etc.) ne sont pas incluses dans
          Global_Sales.
        - **Completude des donnees** — Pas tous les jeux sont presents dans toutes
          les sources. Le fuzzy matching peut introduire de faux positifs.
        - **R² modere** — La prediction des ventes de jeux video est un probleme
          intrinsequement difficile (forte variance, facteurs non observables comme
          le marketing et le bouche-a-oreille).
        - **Biais temporel** — Le modele est entraine sur des donnees historiques.
          Les tendances du marche evoluent (montee du free-to-play, cloud gaming).
        """
    )

    st.divider()

    # Future work
    section_header("Perspectives")
    st.write(
        """
        - Integration de donnees de ventes digitales (Epic Games Store, PS Store)
        - Deep learning tabulaire (TabNet, FT-Transformer)
        - Series temporelles pour les tendances (Prophet, ARIMA)
        - Analyse de sentiments des reviews comme feature predictive
        - LLM pour le querying en langage naturel des donnees
        """
    )
