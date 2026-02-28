"""Lottie animation URL registry for the retro arcade Streamlit app.

Each key maps a semantic animation name to a public Lottie JSON URL hosted on
LottieFiles.  These URLs are used with ``streamlit-lottie`` to render
lightweight, themeable animations without bundling assets locally.
"""

LOTTIE_URLS: dict[str, str] = {
    # Retro / gaming loading spinner
    "loading_game": ("https://lottie.host/0198b445-7545-4e30-9578-4cecdc498611/bqj3sgF6RQ.json"),
    # Celebration / confetti burst
    "celebration": ("https://lottie.host/4d1ad81e-a1b2-437c-9a21-57a3efbfc0c6/RyaANRkiGj.json"),
    # Data analysis / chart animation
    "analysis": ("https://lottie.host/d5b0d5f9-1a52-4a85-85de-c3b1c6d3faed/9WnKIgCzKI.json"),
}
