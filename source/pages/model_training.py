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
    st.title("Model Training & Evaluation")
    st.caption("Model comparison, stacking ensemble architecture, and detailed metrics")

    log = _load_training_log()
    if log is None:
        st.warning("No training log found. Run `make train` first.")
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
    section_header("Model Comparison", "Performance on the test set (post-split data)")

    model_rows = []
    for name, m in metrics.items():
        if isinstance(m, dict) and "r2" in m:
            model_rows.append({
                "Model": name.replace("_", " ").title(),
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
            x=df_models["Model"],
            y=df_models["R²"],
            marker_color=[ACCENT if "Ensemble" in n or "Stacking" in n else SECONDARY
                         for n in df_models["Model"]],
            text=df_models["R²"].apply(lambda x: f"{x:.4f}"),
            textposition="outside",
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="R² by Model",
            yaxis_title="R²",
            showlegend=False,
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Stacking architecture
    section_header("Stacking Ensemble Architecture")

    if "stacking_meta_weights" in log:
        weights = log["stacking_meta_weights"]
        base_names = [name.replace("_", " ").title() for name in list(log.get("best_params", {}).keys())
                     if name != "elastic_net"][:len(weights)]

        info_card(
            "2-Level Stacking",
            f"""
            <b>Level 0 (base models)</b>: {', '.join(base_names)}<br>
            <b>Level 1 (meta-learner)</b>: Ridge Regression (alpha={log.get('stacking_meta_alpha', '?'):.4f})<br>
            <br>
            Out-of-fold predictions (5-fold CV) from base models serve as inputs
            to the meta-learner, avoiding the overfitting of simple averaging.
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
                title="Meta-Learner Weights by Model",
                yaxis_title="Ridge Coefficient",
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
            "Averaging Ensemble",
            "Predictions from 3 models (LightGBM, XGBoost, CatBoost) are averaged. "
            "The v3 pipeline uses a stacking ensemble with a Ridge meta-learner.",
        )

    st.divider()

    # Hyperparameters
    section_header("Best Hyperparameters", "Found by Optuna (Bayesian tuning)")

    best_params = log.get("best_params", {})
    if best_params:
        tabs = st.tabs(list(best_params.keys()))
        for tab, (name, params) in zip(tabs, best_params.items()):
            with tab:
                st.json(params)

    st.divider()

    # SHAP feature importance
    section_header("Feature Importance (SHAP)")

    shap_bar = REPORTS_DIR / "shap_bar_v3.png"
    if not shap_bar.exists():
        shap_bar = REPORTS_DIR / "shap_bar.png"
    shap_summary = REPORTS_DIR / "shap_summary_v3.png"
    if not shap_summary.exists():
        shap_summary = REPORTS_DIR / "shap_summary.png"

    if shap_bar.exists() or shap_summary.exists():
        tab1, tab2 = st.tabs(["Mean Importance", "Beeswarm"])
        with tab1:
            if shap_bar.exists():
                st.image(str(shap_bar), use_container_width=True)
            else:
                st.info("Plot not available")
        with tab2:
            if shap_summary.exists():
                st.image(str(shap_summary), use_container_width=True)
            else:
                st.info("Plot not available")
    else:
        st.info("SHAP plots will be generated after model training.")

    # Training metadata
    with st.expander("Training Metadata"):
        st.write(f"**Date**: {log.get('timestamp', 'N/A')}")
        st.write(f"**Split year**: {log.get('split_year', 'N/A')}")
        st.write(f"**Log transform**: {log.get('log_transform', 'N/A')}")
        st.write(f"**Random state**: {log.get('random_state', 'N/A')}")
        st.write(f"**Features** ({len(features)}):")
        st.code(", ".join(features))
