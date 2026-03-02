import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from config import IMAGES_DIR, PLOTLY_LAYOUT, REPORTS_DIR
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def modelisation_page() -> None:
    """Render the LightGBM model presentation and interactive demo page."""
    import lightgbm as lgb

    # Titre de la présentation
    st.title("🚀 Présentation du Modèle LightGBM")

    # Introduction
    st.header("Introduction 🌟")
    st.write("""
    Avant de sélectionner le modèle ci-dessous, nous avons utilisé un LazyRegressor qui a généré 29 modèles et leurs scores respectifs. Le LightGBM ayant obtenu les meilleurs résultats, c'est celui que nous avons choisi pour ce projet.
    LightGBM est un framework de boosting de gradient développé par Microsoft. Il est conçu pour être extrêmement efficace, rapide et performant.
    """)

    # Fonctionnement de LightGBM
    st.header("Fonctionnement de LightGBM 🛠️")

    st.write("""
    Le LightGBM Regressor fonctionne en combinant plusieurs techniques avancées pour optimiser l'entraînement et la précision des modèles de régression :
    Gradient Boosting : Combine plusieurs modèles faibles séquentiellement pour créer un modèle puissant.

    Exclusive Feature Bundling (EFB) : Réduit la dimensionnalité en combinant des variables non chevauchantes.

    Gradient-based One-Side Sampling (GOSS) : Améliore l'efficacité de l'entraînement en sélectionnant intelligemment les échantillons de données.
    Croissance verticale des arbres : Ajoute des niveaux de profondeur aux arbres de décision pour améliorer les prédictions.
    """)

    # Schéma de fonctionnement de LightGBM
    chemin_image = IMAGES_DIR / "A_stylized_diagram_illustrating_the_workflow_of_Li.jpg"

    st.subheader("Schéma de Fonctionnement 🔍")
    if chemin_image.exists():
        st.image(
            str(chemin_image),
            caption="Schéma de Fonctionnement de LightGBM",
            use_container_width=True,
        )
    else:
        st.write(
            f"Erreur : l'image {chemin_image.name} est introuvable. Vérifiez le dossier images/."
        )

    # Avantages de LightGBM
    st.header("Avantages de LightGBM 💡")

    st.write("""
    - **Vitesse et Efficacité** : LightGBM est conçu pour être très rapide et efficace.
    - **Précision** : Grâce à ses techniques avancées de boosting, il offre une grande précision.
    - **Scalabilité** : Il peut gérer des ensembles de données volumineux avec de nombreuses fonctionnalités.
    - **Support des Valeurs Manquantes** : LightGBM gère nativement les valeurs manquantes.
    """)

    # Schéma des avantages
    st.subheader("Avantages en un coup d'œil 📊")

    adv_df = pd.DataFrame(
        {
            "Avantage": ["Vitesse", "Precision", "Scalabilite", "Support des Valeurs Manquantes"],
            "Importance (%)": [90, 85, 95, 80],
        }
    )
    fig_adv = px.bar(
        adv_df,
        x="Importance (%)",
        y="Avantage",
        orientation="h",
        color="Importance (%)",
        color_continuous_scale="Teal",
        title="Avantages de LightGBM",
    )
    fig_adv.update_layout(
        **PLOTLY_LAYOUT,
        showlegend=False,
    )
    st.plotly_chart(fig_adv, use_container_width=True)

    # Interactivité avec l'utilisateur
    st.header("Essayez par vous-même ! 🎮")

    # Slider interactif pour ajuster les paramètres (exemple)
    max_depth = st.slider("Choisissez la profondeur maximale de l'arbre", 1, 20, 6)
    learning_rate = st.slider("Choisissez le taux d'apprentissage", 0.01, 0.5, 0.1)

    # Chargement des données
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with st.spinner("Entrainement du modele..."):
        # Création et entraînement du modèle avec les paramètres ajustés
        model = lgb.LGBMRegressor(max_depth=max_depth, learning_rate=learning_rate)
        model.fit(X_train, y_train)

        # Prédiction et évaluation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

    # Affichage des résultats
    st.write(
        f"Erreur Quadratique Moyenne (MSE) avec profondeur {max_depth} et taux d'apprentissage {learning_rate}: {mse:.2f}"
    )

    # Comparaison des valeurs prédites et réelles
    fig_scatter = go.Figure()
    fig_scatter.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred,
            mode="markers",
            marker=dict(color="#00FFCC", opacity=0.4, size=5),
            name="Predictions",
        )
    )
    fig_scatter.add_trace(
        go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode="lines",
            line=dict(color="#FF6EC7", dash="dash", width=2),
            name="Ideal",
        )
    )
    fig_scatter.update_layout(
        title="Comparaison des valeurs reelles et predites",
        xaxis_title="Valeurs reelles",
        yaxis_title="Valeurs predites",
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Conclusion
    st.header("Conclusion 🎯")
    st.write("""
            LightGBM est un outil puissant pour les taches de regression et de classification,
            particulierement adapte aux grands ensembles de donnees.

            **Modele v1 (16K donnees, fuite de donnees) :**

            LGBMRegressor R² : Moyenne = 0.3107 (avant feature engineering)
            LGBMRegressor R² : Moyenne = 0.9880 (apres feature engineering — score gonfle par la fuite de donnees)

            *Note : le modele v1 utilisait les ventes regionales (NA, EU, JP, Other) comme features,
            ce qui constituait une fuite de donnees car elles composent directement Global_Sales.
            Les scores v1 ne sont donc pas comparables aux scores v2.*
""")

    # ------------------------------------------------------------------
    # V2 Model: Optuna + SHAP results
    # ------------------------------------------------------------------
    st.markdown("---")
    st.header("Modele v2 : Optimisation Optuna + SHAP")

    training_log_path = REPORTS_DIR / "training_log.json"
    shap_summary_path = REPORTS_DIR / "shap_summary.png"
    shap_bar_path = REPORTS_DIR / "shap_bar.png"

    if training_log_path.exists():
        with open(training_log_path) as f:
            training_log = json.load(f)

        raw_metrics = training_log["metrics"]
        best_params = training_log["best_params"]

        # Handle both old format (flat metrics) and new format (per-model metrics)
        if "ensemble" in raw_metrics:
            # New ensemble format
            metrics_lgb = raw_metrics["lightgbm"]
            metrics_xgb = raw_metrics["xgboost"]
            metrics_cb = raw_metrics["catboost"]
            metrics_ens = raw_metrics["ensemble"]
            has_ensemble = True
        else:
            # Old single-model format
            metrics_lgb = raw_metrics
            has_ensemble = False

        # --- Ensemble comparison ---
        st.subheader("Comparaison des modeles")
        st.write("""
        Le modele v2 corrige la fuite de donnees, remplace le one-hot encoding
        de Publisher (567 colonnes) par du target encoding (1 colonne), et
        optimise les hyperparametres avec Optuna. Trois modeles sont entraines
        (LightGBM, XGBoost, CatBoost) et combines en ensemble (moyenne).
        """)

        if has_ensemble:
            comparison_data = {
                "Modele": ["v1 (original)", "LightGBM", "XGBoost", "CatBoost", "Ensemble"],
                "R2": [
                    "0.9880*",
                    f"{metrics_lgb['r2']:.4f}",
                    f"{metrics_xgb['r2']:.4f}",
                    f"{metrics_cb['r2']:.4f}",
                    f"{metrics_ens['r2']:.4f}",
                ],
                "RMSE": [
                    "0.0265*",
                    f"{metrics_lgb['rmse']:.4f}",
                    f"{metrics_xgb['rmse']:.4f}",
                    f"{metrics_cb['rmse']:.4f}",
                    f"{metrics_ens['rmse']:.4f}",
                ],
                "MAE": [
                    "0.0132*",
                    f"{metrics_lgb['mae']:.4f}",
                    f"{metrics_xgb['mae']:.4f}",
                    f"{metrics_cb['mae']:.4f}",
                    f"{metrics_ens['mae']:.4f}",
                ],
            }
            st.dataframe(
                pd.DataFrame(comparison_data),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "*v1 : 16K donnees, 576 features, fuite de donnees — "
                "v2 : 64K donnees, 10 features, split temporel, pas de fuite"
            )
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Modele v1 (original)**")
                st.metric("R2", "0.9880")
                st.caption("16K donnees, 576 features, fuite de donnees")
            with col2:
                st.markdown("**Modele v2 (Optuna)**")
                st.metric("R2", f"{metrics_lgb['r2']:.4f}")
                st.caption("64K donnees, 10 features, split temporel")

        # Baseline comparison
        baseline_r2 = metrics_lgb.get("baseline_r2", "N/A")
        baseline_rmse = metrics_lgb.get("baseline_rmse", "N/A")
        st.write(f"**Baseline (predicteur moyen) — R2 : {baseline_r2}, RMSE : {baseline_rmse}**")

        # --- Best hyperparameters ---
        st.subheader("Meilleurs hyperparametres (Optuna)")

        if isinstance(best_params.get("lightgbm"), dict):
            # New format: per-model params
            tab_lgb, tab_xgb, tab_cb = st.tabs(["LightGBM", "XGBoost", "CatBoost"])
            for tab, name in [(tab_lgb, "lightgbm"), (tab_xgb, "xgboost"), (tab_cb, "catboost")]:
                with tab:
                    p = best_params[name]
                    st.dataframe(
                        pd.DataFrame(list(p.items()), columns=["Parametre", "Valeur"]),
                        use_container_width=True,
                        hide_index=True,
                    )
        else:
            # Old format: flat params
            params_display = {k: v for k, v in best_params.items() if k != "split_year"}
            params_display["split_year"] = best_params.get("split_year", "N/A")
            st.dataframe(
                pd.DataFrame(list(params_display.items()), columns=["Parametre", "Valeur"]),
                use_container_width=True,
                hide_index=True,
            )

        # --- SHAP plots ---
        st.subheader("Importance des features (SHAP)")
        st.write("""
        Les graphiques SHAP montrent quelles variables influencent le plus les
        predictions du modele. Contrairement a l'importance classique basee sur
        les splits, SHAP attribue a chaque feature une contribution precise et
        interpretable pour chaque prediction.
        """)

        if shap_bar_path.exists():
            st.image(
                str(shap_bar_path),
                caption="Importance moyenne des features (|SHAP|)",
                use_container_width=True,
            )
        else:
            st.warning("Le graphique SHAP (bar) est introuvable.")

        if shap_summary_path.exists():
            st.image(
                str(shap_summary_path),
                caption="Distribution SHAP par feature (beeswarm)",
                use_container_width=True,
            )
        else:
            st.warning("Le graphique SHAP (summary) est introuvable.")

    else:
        st.info(
            "Le modele v2 n'a pas encore ete entraine. "
            "Lancez `python scripts/train_model.py` pour generer les resultats."
        )

    # Ajout d'un GIF fun lié aux jeux vidéo
    st.markdown(
        """
    <div style="display: flex; justify-content: center;">
        <iframe src="https://giphy.com/embed/mHv5sLKI1b1I8r4wmp" width="680" height="370" frameBorder="0" class="giphy-embed" allowFullScreen></iframe>
    </div>
    """,
        unsafe_allow_html=True,
    )
