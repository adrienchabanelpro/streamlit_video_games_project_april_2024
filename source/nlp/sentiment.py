"""NLP sentiment analysis with DistilBERT, 5-star ratings, and multilingual support.

Re-exports from ``analyse_avis_utilisateurs`` for clean package imports.
The original module is kept for backwards compatibility with tests.
"""

from analyse_avis_utilisateurs import (
    GAMING_ASPECTS,
    analyze_aspects,
    predict_user_reviews,
)

__all__ = ["GAMING_ASPECTS", "analyze_aspects", "predict_user_reviews"]
