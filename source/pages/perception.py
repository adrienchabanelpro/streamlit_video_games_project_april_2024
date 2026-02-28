"""User perception analysis page with DistilBERT sentiment, word clouds, and aspect analysis."""

import base64
from io import BytesIO

import plotly.graph_objects as go
import streamlit as st
from analyse_avis_utilisateurs import analyze_aspects, predict_user_reviews
from config import BG, CYAN, IMAGES_DIR, PINK, PLOTLY_LAYOUT, TEXT_COLOR
from PIL import Image


def perception_page() -> None:
    """Render the user perception / sentiment analysis page."""
    st.title("Perception")
    st.write(
        "Analysez le sentiment de vos utilisateurs avec nos modeles de NLP. "
        "Telechargez un fichier CSV et laissez notre IA predire la perception "
        "de vos clients."
    )

    # --- Mode selector ---
    mode = st.radio(
        "Mode d'analyse",
        ["Binaire (Positif/Negatif)", "5 etoiles (1-5)", "Analyse par aspect"],
        horizontal=True,
        key="sentiment_mode",
    )

    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
    st.caption(
        "Le fichier CSV doit avoir une colonne nommee 'user_review'. "
        "Le modele 5 etoiles supporte le francais, anglais, allemand, "
        "espagnol, italien et neerlandais."
    )

    if uploaded_file is None:
        return

    if not st.button("Lancer l'analyse"):
        return

    if mode == "Analyse par aspect":
        _run_aspect_analysis(uploaded_file)
    elif mode == "5 etoiles (1-5)":
        _run_star_analysis(uploaded_file)
    else:
        _run_binary_analysis(uploaded_file)

    # Disclaimer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        "Modeles : DistilBERT (binaire) et BERT multilingual (5 etoiles). "
        "Ces modeles sont en version beta et peuvent faire des erreurs. "
        "Envisagez de verifier les informations importantes.</p>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# Binary sentiment analysis
# ------------------------------------------------------------------


def _run_binary_analysis(uploaded_file: object) -> None:
    """Run binary sentiment and display results with word clouds."""
    with st.spinner("Analyse binaire avec DistilBERT..."):
        data, positive_pct, negative_pct = predict_user_reviews(uploaded_file, granularity="binary")

    if data is None or positive_pct is None:
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avis positifs", f"{positive_pct:.1f}%")
    with col2:
        st.metric("Avis negatifs", f"{negative_pct:.1f}%")
    with col3:
        avg_conf = data["confidence"].mean() * 100
        st.metric("Confiance moyenne", f"{avg_conf:.1f}%")

    # Gauge chart
    _display_gauge(positive_pct)

    # Word clouds
    _display_word_clouds(data)

    # Per-review details
    st.subheader("Detail des predictions")
    display_df = data[["user_review", "sentiment", "confidence"]].copy()
    display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.1%}")
    display_df.columns = ["Avis", "Sentiment", "Confiance"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------
# 5-star sentiment analysis
# ------------------------------------------------------------------


def _run_star_analysis(uploaded_file: object) -> None:
    """Run 5-star sentiment analysis and display results."""
    with st.spinner("Analyse 5 etoiles avec BERT multilingual..."):
        data, avg_stars, _ = predict_user_reviews(uploaded_file, granularity="5-star")

    if data is None or avg_stars is None:
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Note moyenne", f"{avg_stars:.1f} / 5")
    with col2:
        avg_conf = data["confidence"].mean() * 100
        st.metric("Confiance moyenne", f"{avg_conf:.1f}%")
    with col3:
        st.metric("Nombre d'avis", str(len(data)))

    # Star distribution chart
    star_counts = data["stars"].value_counts().sort_index()
    colors = ["#FF4444", "#FF8800", "#FFCC00", "#88CC00", "#00CC66"]

    fig = go.Figure()
    for star_val in range(1, 6):
        count = star_counts.get(star_val, 0)
        fig.add_trace(
            go.Bar(
                x=[f"{star_val} etoile{'s' if star_val > 1 else ''}"],
                y=[count],
                name=f"{star_val} etoile{'s' if star_val > 1 else ''}",
                marker_color=colors[star_val - 1],
                text=[count],
                textposition="outside",
            )
        )

    fig.update_layout(
        title="Distribution des notes",
        xaxis_title="Note",
        yaxis_title="Nombre d'avis",
        showlegend=False,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-review details
    st.subheader("Detail des predictions")
    display_df = data[["user_review", "stars", "sentiment", "confidence"]].copy()
    display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.1%}")
    display_df.columns = ["Avis", "Etoiles", "Sentiment", "Confiance"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------
# Aspect-based sentiment analysis
# ------------------------------------------------------------------


def _run_aspect_analysis(uploaded_file: object) -> None:
    """Run aspect-based sentiment analysis and display results."""
    import pandas as pd

    try:
        uploaded_file.seek(0)
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        return

    if "user_review" not in data.columns:
        st.warning("Le fichier CSV doit contenir une colonne 'user_review'.")
        return

    data = data.dropna(subset=["user_review"]).reset_index(drop=True)
    reviews = data["user_review"].astype(str).tolist()

    if not reviews:
        st.warning("Aucun avis valide trouve.")
        return

    with st.spinner("Analyse par aspect avec DistilBERT..."):
        aspect_results = analyze_aspects(reviews)

    # Build chart data
    aspects = []
    pos_counts = []
    neg_counts = []
    totals = []
    for aspect, counts in aspect_results.items():
        aspects.append(aspect.capitalize())
        pos_counts.append(counts["positive"])
        neg_counts.append(counts["negative"])
        totals.append(counts["total"])

    if sum(totals) == 0:
        st.info("Aucun aspect de jeu detecte dans les avis.")
        return

    # Grouped bar chart
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Positif",
            x=aspects,
            y=pos_counts,
            marker_color=CYAN,
            text=pos_counts,
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Negatif",
            x=aspects,
            y=neg_counts,
            marker_color=PINK,
            text=neg_counts,
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Sentiment par aspect du jeu",
        xaxis_title="Aspect",
        yaxis_title="Nombre d'avis",
        barmode="group",
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.subheader("Resume par aspect")
    summary_rows = []
    for aspect, counts in aspect_results.items():
        total = counts["total"]
        if total > 0:
            pos_pct = counts["positive"] / total * 100
            summary_rows.append(
                {
                    "Aspect": aspect.capitalize(),
                    "Mentions": total,
                    "Positif (%)": f"{pos_pct:.0f}%",
                    "Negatif (%)": f"{100 - pos_pct:.0f}%",
                }
            )
    if summary_rows:
        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True,
        )


# ------------------------------------------------------------------
# Word clouds
# ------------------------------------------------------------------


def _display_word_clouds(data: object) -> None:
    """Generate and display word clouds for positive and negative reviews."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        st.info("Installez 'wordcloud' (pip install wordcloud) pour afficher les nuages de mots.")
        return

    pos_reviews = data[data["predictions"] == 1]["user_review"].str.cat(sep=" ")
    neg_reviews = data[data["predictions"] == 0]["user_review"].str.cat(sep=" ")

    if not pos_reviews.strip() and not neg_reviews.strip():
        return

    st.subheader("Nuages de mots")
    col1, col2 = st.columns(2)

    with col1:
        if pos_reviews.strip():
            wc = WordCloud(
                width=600,
                height=400,
                background_color="#0D0D0D",
                colormap="cool",
                max_words=100,
            ).generate(pos_reviews)
            fig = go.Figure()
            fig.add_layout_image(
                dict(
                    source=Image.fromarray(wc.to_array()),
                    x=0,
                    y=1,
                    xref="paper",
                    yref="paper",
                    sizex=1,
                    sizey=1,
                    xanchor="left",
                    yanchor="top",
                    layer="above",
                )
            )
            fig.update_layout(
                title="Avis positifs",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                paper_bgcolor=BG,
                margin=dict(t=40, b=0, l=0, r=0),
                font=dict(color=TEXT_COLOR),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun avis positif pour generer un nuage de mots.")

    with col2:
        if neg_reviews.strip():
            wc = WordCloud(
                width=600,
                height=400,
                background_color="#0D0D0D",
                colormap="spring",
                max_words=100,
            ).generate(neg_reviews)
            fig = go.Figure()
            fig.add_layout_image(
                dict(
                    source=Image.fromarray(wc.to_array()),
                    x=0,
                    y=1,
                    xref="paper",
                    yref="paper",
                    sizex=1,
                    sizey=1,
                    xanchor="left",
                    yanchor="top",
                    layer="above",
                )
            )
            fig.update_layout(
                title="Avis negatifs",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                paper_bgcolor=BG,
                margin=dict(t=40, b=0, l=0, r=0),
                font=dict(color=TEXT_COLOR),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun avis negatif pour generer un nuage de mots.")


# ------------------------------------------------------------------
# Gauge chart
# ------------------------------------------------------------------


def _display_gauge(positive_percentage: float) -> None:
    """Render the Plotly gauge with Street Fighter background."""

    def _pil_to_base64(img_path: str) -> str:
        """Convert an image file to a base64-encoded data URI."""
        img = Image.open(img_path)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    def _get_color(value: float, alpha: float = 0.5) -> str:
        """Return an RGBA color string for the given gauge *value*."""
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

    image_path = IMAGES_DIR / "street_fighter2.png"
    if not image_path.exists():
        st.warning("Image street_fighter2.png introuvable.")
        return

    encoded_image = _pil_to_base64(str(image_path))

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
                    "line": {"color": CYAN, "width": 5},
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
            "font": {"color": CYAN},
        },
        paper_bgcolor=BG,
        font=dict(family="Arial", size=18, color=TEXT_COLOR),
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
