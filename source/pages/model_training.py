"""Model Training & Evaluation page: model comparison, stacking architecture, metrics."""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from components import info_card, metric_card, section_header
from config import ACCENT, PLOTLY_LAYOUT, REPORTS_DIR, SECONDARY


@st.cache_data
def _load_training_log() -> dict | None:
    """Load the training log (v3 first, v2 fallback)."""
    for name in ["training_log_v3.json", "training_log.json"]:
        path = REPORTS_DIR / name
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def model_training_page() -> None:
    """Render the Model Training & Evaluation page."""
    st.title("Entrainement & Evaluation des Modeles")
    st.caption("Comparaison des modeles, architecture du stacking ensemble, et metriques detaillees")

    log = _load_training_log()
    if log is None:
        st.warning("Aucun log d'entrainement trouve. Lancez `make train` d'abord.")
        return

    metrics = log.get("metrics", {})
    features = log.get("features", [])

    # Key metrics
    ensemble_key = "stacking_ensemble" if "stacking_ensemble" in metrics else "ensemble"
    ensemble_m = metrics.get(ensemble_key, {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("R² (Ensemble)", f"{ensemble_m.get('r2', 0):.4f}", icon="🎯")
    with c2:
        metric_card("RMSE", f"{ensemble_m.get('rmse', 0):.4f}", icon="📉")
    with c3:
        metric_card("MAE", f"{ensemble_m.get('mae', 0):.4f}", icon="📊")
    with c4:
        metric_card("Features", len(features), icon="⚙️")

    st.divider()

    # Model comparison table
    section_header("Comparaison des modeles", "Performance sur le jeu de test (donnees post-split)")

    model_rows = []
    for name, m in metrics.items():
        if isinstance(m, dict) and "r2" in m:
            model_rows.append({
                "Modele": name.replace("_", " ").title(),
                "R²": round(m.get("r2", 0), 4),
                "RMSE": round(m.get("rmse", 0), 4),
                "MAE": round(m.get("mae", 0), 4),
                "MAPE": round(m.get("mape", 0), 4) if "mape" in m else None,
            })

    if model_rows:
        df_models = pd.DataFrame(model_rows).sort_values("R²", ascending=False)
        st.dataframe(df_models, use_container_width=True, hide_index=True)

        # Bar chart comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_models["Modele"],
            y=df_models["R²"],
            marker_color=[ACCENT if "Ensemble" in n or "Stacking" in n else SECONDARY
                         for n in df_models["Modele"]],
            text=df_models["R²"].apply(lambda x: f"{x:.4f}"),
            textposition="outside",
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="R² par modele",
            yaxis_title="R²",
            showlegend=False,
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Stacking architecture
    section_header("Architecture du Stacking Ensemble")

    if "stacking_meta_weights" in log:
        weights = log["stacking_meta_weights"]
        base_names = [name.replace("_", " ").title() for name in list(log.get("best_params", {}).keys())
                     if name != "elastic_net"][:len(weights)]

        info_card(
            "Stacking a 2 niveaux",
            f"""
            <b>Niveau 0 (base models)</b> : {', '.join(base_names)}<br>
            <b>Niveau 1 (meta-learner)</b> : Ridge Regression (alpha={log.get('stacking_meta_alpha', '?'):.4f})<br>
            <br>
            Les predictions out-of-fold (5-fold CV) des modeles de base servent
            d'entrees au meta-learner, evitant le surapprentissage du simple moyennage.
            """
        )

        # Weights bar chart
        if len(base_names) == len(weights):
            fig_w = go.Figure(go.Bar(
                x=base_names,
                y=weights,
                marker_color=ACCENT,
                text=[f"{w:.3f}" for w in weights],
                textposition="outside",
            ))
            fig_w.update_layout(
                **PLOTLY_LAYOUT,
                title="Poids du meta-learner par modele",
                yaxis_title="Coefficient Ridge",
                height=350,
            )
            st.plotly_chart(fig_w, use_container_width=True)

        # Simple avg vs stacking comparison
        if "simple_avg_r2" in ensemble_m:
            c1, c2 = st.columns(2)
            with c1:
                metric_card("Stacking R²", f"{ensemble_m['r2']:.4f}", icon="🏆", accent=ACCENT)
            with c2:
                metric_card("Simple Average R²", f"{ensemble_m['simple_avg_r2']:.4f}", icon="📊", accent=SECONDARY)
    else:
        info_card(
            "Ensemble par moyennage",
            "Les predictions des 3 modeles (LightGBM, XGBoost, CatBoost) sont moyennees. "
            "Le pipeline v3 utilise un stacking ensemble avec meta-learner Ridge.",
        )

    st.divider()

    # Hyperparameters
    section_header("Hyperparametres optimaux", "Trouves par Optuna (tuning bayesien)")

    best_params = log.get("best_params", {})
    if best_params:
        tabs = st.tabs(list(best_params.keys()))
        for tab, (name, params) in zip(tabs, best_params.items()):
            with tab:
                st.json(params)

    st.divider()

    # SHAP feature importance
    section_header("Importance des variables (SHAP)")

    shap_bar = REPORTS_DIR / "shap_bar_v3.png"
    if not shap_bar.exists():
        shap_bar = REPORTS_DIR / "shap_bar.png"
    shap_summary = REPORTS_DIR / "shap_summary_v3.png"
    if not shap_summary.exists():
        shap_summary = REPORTS_DIR / "shap_summary.png"

    if shap_bar.exists() or shap_summary.exists():
        tab1, tab2 = st.tabs(["Importance moyenne", "Beeswarm"])
        with tab1:
            if shap_bar.exists():
                st.image(str(shap_bar), use_container_width=True)
            else:
                st.info("Plot non disponible")
        with tab2:
            if shap_summary.exists():
                st.image(str(shap_summary), use_container_width=True)
            else:
                st.info("Plot non disponible")
    else:
        st.info("Les plots SHAP seront generes apres l'entrainement du modele.")

    # Training metadata
    with st.expander("Metadata d'entrainement"):
        st.write(f"**Date** : {log.get('timestamp', 'N/A')}")
        st.write(f"**Split year** : {log.get('split_year', 'N/A')}")
        st.write(f"**Log transform** : {log.get('log_transform', 'N/A')}")
        st.write(f"**Random state** : {log.get('random_state', 'N/A')}")
        st.write(f"**Features** ({len(features)}) :")
        st.code(", ".join(features))
