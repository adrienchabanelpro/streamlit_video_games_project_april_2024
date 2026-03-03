import streamlit as st


def feature_engineering_page() -> None:
    """Render the feature engineering and pre-processing page."""

    st.title("Feature Engineering & Pre-processing")

    st.header("Initial Dataset")
    st.markdown("""
    **Size:** ~64,000 rows and 30 columns (VGChartz 2024 + SteamSpy).
    **Columns excluded from prediction:** 'Name', 'Rank', 'NA_Sales', 'EU_Sales',
    'JP_Sales', 'Other_Sales' (data leakage), 'img', 'developer', 'release_date',
    'last_update', and all `steam_*` columns (not used as features).
    """)

    st.header("Data Cleaning")
    st.markdown("""
    - Removal of duplicates.
    - Column format corrections ('Year' converted to datetime, 'user_review' to numeric).
    - Platform column alignment to facilitate merges.
    - Removal of rows with missing values ('Publisher' and 'Year').
    - Imputation of missing 'user_review' and 'meta_score' with median values by platform and genre.
    """)

    st.header("Transformation")
    st.markdown("""
    - **v1 (legacy)**: OneHotEncoder on Publisher — 576 sparse columns.
    - **v2 (current)**: Target Encoding on Publisher — a single column (`Publisher_encoded`),
      much more efficient for high-cardinality variables.
    - Normalization of all numerical columns with StandardScaler.
    """)

    st.header("Normalization & Encoding")
    st.markdown("""
    **Objective:** Prepare data for modeling by normalizing numerical columns
    and encoding categorical variables.
    **Result (v2):** After transformation, the training dataset contains ~60,000+
    rows and only **10 features** (thanks to target encoding instead of one-hot).
    """)

    st.header("Principal Component Analysis (PCA)")
    st.markdown("""
    **Issue Identified:**
    - The "Publisher" variable had a large number of unique values, complicating PCA.
    - Over 30 principal components were needed to explain ~90% of the variance, making dimensionality reduction ineffective.
    - Difficulty in visualizing and interpreting the results.

    **Conclusion:**
    PCA did not yield the expected results, so we opted for direct use of dimensionality reduction techniques within our ML model (LightGBM Regressor) via EFB (exclusive feature bundling).
    """)

    st.header("Feature Engineering")
    st.markdown("""
    **Creative Process:**
    - Creation of new variables to improve model performance.
    - Testing multiple iterations to find the best features.

    **New Variables Created:**
    - Global_Sales_Mean_Platform and Global_Sales_Mean_genre: Average global sales by genre and platform.
    - Year_Global_Sales_mean_platform and Year_Global_Sales_mean_genre: Interactions between release year and average global sales.
    - Cumulative_Sales_Platform and Cumulative_Sales_Genre: Popularity indicators based on historical sales.
    """)

    st.subheader("Inconclusive Hypotheses")
    st.markdown("""
    The following variables did not improve model performance:
    - Genre_Count
    - Publisher_Count
    - Platform_Count
    - Publisher_Popularity_Sales
    - Age
    - Decade
    - Score_Interaction
    """)

    st.header("Final Dataset (v2)")
    st.markdown("""
    **Size:** ~60,000+ rows and **10 features** for prediction.
    **Features used:**
    - `Year`
    - `meta_score`, `user_review`
    - `Publisher_encoded` (target encoding — 1 column instead of 576)
    - `Global_Sales_mean_genre`, `Global_Sales_mean_platform`
    - `Year_Global_Sales_mean_genre`, `Year_Global_Sales_mean_platform`
    - `Cumulative_Sales_Genre`, `Cumulative_Sales_Platform`

    **Target:** `Global_Sales` (with log1p transformation).
    **Temporal split:** training on games before the split year,
    testing on games after (no data leakage).
    """)
