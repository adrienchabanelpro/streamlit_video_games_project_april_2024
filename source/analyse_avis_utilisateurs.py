"""NLP sentiment analysis with DistilBERT, 5-star ratings, and multilingual support."""

import pandas as pd
import streamlit as st
from transformers import pipeline

_BINARY_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
_STAR_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"

# Gaming aspects and their associated keywords
GAMING_ASPECTS: dict[str, list[str]] = {
    "gameplay": [
        "gameplay",
        "controls",
        "mechanics",
        "combat",
        "movement",
        "physics",
        "difficulty",
        "challenge",
        "fun",
        "boring",
        "addictive",
        "repetitive",
    ],
    "graphics": [
        "graphics",
        "visuals",
        "art",
        "design",
        "animation",
        "textures",
        "resolution",
        "framerate",
        "fps",
        "beautiful",
        "ugly",
        "stunning",
    ],
    "story": [
        "story",
        "narrative",
        "plot",
        "characters",
        "dialogue",
        "writing",
        "campaign",
        "ending",
        "lore",
        "protagonist",
        "villain",
        "quest",
    ],
    "value": [
        "price",
        "value",
        "worth",
        "money",
        "expensive",
        "cheap",
        "content",
        "hours",
        "length",
        "dlc",
        "microtransaction",
        "free",
        "subscription",
    ],
    "performance": [
        "performance",
        "lag",
        "crash",
        "bug",
        "glitch",
        "loading",
        "optimize",
        "stuttering",
        "freeze",
        "patch",
        "update",
        "stable",
        "broken",
    ],
    "multiplayer": [
        "multiplayer",
        "online",
        "coop",
        "co-op",
        "pvp",
        "matchmaking",
        "server",
        "connection",
        "friends",
        "team",
        "competitive",
        "lobby",
    ],
}


@st.cache_resource
def _load_binary_pipeline() -> object:
    """Load the DistilBERT binary sentiment pipeline (cached)."""
    return pipeline(
        "sentiment-analysis",
        model=_BINARY_MODEL,
        truncation=True,
        max_length=512,
    )


@st.cache_resource
def _load_star_pipeline() -> object:
    """Load the 5-star multilingual sentiment pipeline (cached)."""
    return pipeline(
        "sentiment-analysis",
        model=_STAR_MODEL,
        truncation=True,
        max_length=512,
    )


def predict_user_reviews(
    uploaded_file: object,
    granularity: str = "binary",
) -> tuple[pd.DataFrame | None, float | None, float | None]:
    """Predict sentiment for each review in uploaded CSV.

    Args:
        uploaded_file: A Streamlit ``UploadedFile`` (CSV) with a
            ``user_review`` column.
        granularity: ``"binary"`` for positive/negative or ``"5-star"``
            for 1-5 star ratings.

    Returns:
        Tuple of (dataframe_with_predictions, positive_pct, negative_pct).
        For 5-star mode, positive_pct is the average star rating and
        negative_pct is None.
    """
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
        data = data.dropna(subset=["user_review"]).reset_index(drop=True)
        reviews = data["user_review"].astype(str).tolist()

        if not reviews:
            st.warning("Aucun avis valide trouve dans le fichier.")
            return None, None, None

        if granularity == "5-star":
            return _predict_star(data, reviews)
        return _predict_binary(data, reviews)

    except Exception as e:
        st.error(f"Erreur lors de l'analyse des avis : {e}")
        return None, None, None


def _predict_binary(data: pd.DataFrame, reviews: list[str]) -> tuple[pd.DataFrame, float, float]:
    """Run binary (positive/negative) sentiment analysis."""
    classifier = _load_binary_pipeline()
    results = classifier(reviews, batch_size=32)

    data["sentiment"] = [r["label"] for r in results]
    data["confidence"] = [r["score"] for r in results]
    data["predictions"] = [1 if r["label"] == "POSITIVE" else 0 for r in results]

    positive_pct = (data["predictions"] == 1).mean() * 100
    negative_pct = (data["predictions"] == 0).mean() * 100
    return data, positive_pct, negative_pct


def _predict_star(data: pd.DataFrame, reviews: list[str]) -> tuple[pd.DataFrame, float, None]:
    """Run 5-star sentiment analysis (multilingual)."""
    classifier = _load_star_pipeline()
    results = classifier(reviews, batch_size=32)

    # Label format: "1 star", "2 stars", ..., "5 stars"
    data["stars"] = [int(r["label"][0]) for r in results]
    data["confidence"] = [r["score"] for r in results]
    data["sentiment"] = data["stars"].map(
        {1: "Tres negatif", 2: "Negatif", 3: "Neutre", 4: "Positif", 5: "Tres positif"}
    )

    avg_stars = data["stars"].mean()
    return data, avg_stars, None


def analyze_aspects(reviews: list[str]) -> dict[str, dict[str, int]]:
    """Perform aspect-based sentiment analysis on reviews.

    For each gaming aspect, finds reviews containing relevant keywords,
    then classifies them as positive or negative.

    Returns:
        Dict mapping aspect name to {"positive": count, "negative": count}.
    """
    classifier = _load_binary_pipeline()
    aspect_results: dict[str, dict[str, int]] = {}

    for aspect, keywords in GAMING_ASPECTS.items():
        # Find reviews mentioning this aspect
        matching_reviews = [r for r in reviews if any(kw in r.lower() for kw in keywords)]

        if not matching_reviews:
            aspect_results[aspect] = {"positive": 0, "negative": 0, "total": 0}
            continue

        results = classifier(matching_reviews, batch_size=32)
        pos = sum(1 for r in results if r["label"] == "POSITIVE")
        neg = len(results) - pos
        aspect_results[aspect] = {
            "positive": pos,
            "negative": neg,
            "total": len(results),
        }

    return aspect_results
