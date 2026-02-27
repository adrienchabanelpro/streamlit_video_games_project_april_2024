import os

import pandas as pd
import streamlit as st
from transformers import pipeline

_BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


@st.cache_resource
def _load_sentiment_pipeline():
    """Load the DistilBERT sentiment analysis pipeline (cached)."""
    return pipeline(
        "sentiment-analysis",
        model=_MODEL_NAME,
        truncation=True,
        max_length=512,
    )


def predict_user_reviews(
    uploaded_file,
) -> tuple[pd.DataFrame | None, float | None, float | None]:
    """Predict sentiment for each review in uploaded CSV.

    Returns (dataframe_with_predictions, positive_pct, negative_pct).
    """
    classifier = _load_sentiment_pipeline()

    if uploaded_file is None:
        return None, None, None

    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier CSV : {e}")
        return None, None, None

    if "user_review" not in data.columns:
        st.warning("Le fichier CSV doit contenir une colonne 'user_review'.")
        return None, None, None

    try:
        # Drop empty / NaN reviews
        data = data.dropna(subset=["user_review"]).reset_index(drop=True)
        reviews = data["user_review"].astype(str).tolist()

        if not reviews:
            st.warning("Aucun avis valide trouve dans le fichier.")
            return None, None, None

        # Batch prediction with DistilBERT
        results = classifier(reviews, batch_size=32)

        data["sentiment"] = [r["label"] for r in results]
        data["confidence"] = [r["score"] for r in results]
        data["predictions"] = [1 if r["label"] == "POSITIVE" else 0 for r in results]

        positive_percentage = (data["predictions"] == 1).mean() * 100
        negative_percentage = (data["predictions"] == 0).mean() * 100

        return data, positive_percentage, negative_percentage

    except Exception as e:
        st.error(f"Erreur lors de l'analyse des avis : {e}")
        return None, None, None
