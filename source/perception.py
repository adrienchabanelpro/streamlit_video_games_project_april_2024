import base64
import os
from io import BytesIO

import plotly.graph_objects as go
import streamlit as st
from analyse_avis_utilisateurs import predict_user_reviews
from PIL import Image


def perception():
    st.title("Perception")
    st.write(
        "Decouvrez ce que vos utilisateurs pensent de votre produit en "
        "analysant leurs avis. Telechargez un fichier CSV et laissez notre "
        "modele DistilBERT predire la perception de vos clients."
    )
    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
    st.write(
        "IMPORTANT: Le fichier CSV doit avoir une colonne nommee 'user_review' en langue anglaise."
    )

    if uploaded_file is not None:
        if st.button("Lancer la prediction"):
            with st.spinner("Analyse des avis avec DistilBERT..."):
                data, positive_percentage, negative_percentage = predict_user_reviews(uploaded_file)

            if data is not None and positive_percentage is not None:
                # --- Summary metrics ---
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avis positifs", f"{positive_percentage:.1f}%")
                with col2:
                    st.metric("Avis negatifs", f"{negative_percentage:.1f}%")
                with col3:
                    avg_confidence = data["confidence"].mean() * 100
                    st.metric("Confiance moyenne", f"{avg_confidence:.1f}%")

                # --- Gauge chart with Street Fighter background ---
                _display_gauge(positive_percentage)

                # --- Per-review details ---
                st.subheader("Detail des predictions")
                display_df = data[["user_review", "sentiment", "confidence"]].copy()
                display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.1%}")
                display_df.columns = ["Avis", "Sentiment", "Confiance"]
                st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Disclaimer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        "Modele : DistilBERT (distilbert-base-uncased-finetuned-sst-2-english). "
        "Ce modele est en version beta et peut faire des erreurs. "
        "Envisagez de verifier les informations importantes.</p>",
        unsafe_allow_html=True,
    )


def _display_gauge(positive_percentage: float) -> None:
    """Render the Plotly gauge with Street Fighter background."""

    def _pil_to_base64(img_path: str) -> str:
        img = Image.open(img_path)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    def _get_color(value: float, alpha: float = 0.5) -> str:
        if value <= 25:
            red, green, blue = 255, int((value / 25) * 165), 0
        elif value <= 50:
            red = 255
            green = 165 + int(((value - 25) / 25) * (255 - 165))
            blue = 0
        elif value <= 75:
            red = 255 - int(((value - 50) / 25) * 200)
            green = 240 - int((value / 100) * (255 - 128))
            blue = 0
        else:
            red, green = 0, 255 - int(((value - 55) / 25) * (255 - 128))
            blue = 0
        return f"rgba({red},{green},{blue},{alpha})"

    image_path = os.path.join(os.path.dirname(__file__), "..", "images", "street_fighter2.png")
    if not os.path.exists(image_path):
        st.warning("Image street_fighter2.png introuvable.")
        return

    encoded_image = _pil_to_base64(image_path)

    steps = [{"range": [i, i + 1], "color": _get_color(i, alpha=0.7)} for i in range(101)]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickfont": {"size": 25, "family": "Arial", "color": "white"},
                },
                "bar": {"color": "rgba(0,0,0,0)"},
                "steps": steps,
                "threshold": {
                    "line": {"color": "#00FFCC", "width": 5},
                    "thickness": 0.75,
                    "value": positive_percentage,
                },
            },
        )
    )

    fig.update_layout(
        title={
            "text": "Pourcentage d'avis positifs",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"color": "#00FFCC"},
        },
        paper_bgcolor="#0D0D0D",
        font=dict(family="Arial", size=18, color="#E0E0E0"),
        margin=dict(t=50, b=0, l=0, r=0),
        images=[
            dict(
                source=encoded_image,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                sizex=1,
                sizey=1,
                xanchor="center",
                yanchor="middle",
                layer="below",
            )
        ],
    )

    st.plotly_chart(fig)
