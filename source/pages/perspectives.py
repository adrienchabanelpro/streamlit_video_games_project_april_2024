import requests
import streamlit as st
from config import IMAGES_DIR
from lottie_urls import LOTTIE_URLS
from PIL import Image
from streamlit_lottie import st_lottie


def _load_lottie_url(url: str) -> dict | None:
    """Fetch a Lottie animation JSON from *url*.

    Returns the parsed JSON dict on success, or ``None`` if the request fails.
    """
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    return None


def perspectives_page() -> None:
    """Render the perspectives and future improvements page."""
    # Titre de la page
    st.title("Perspectives")

    # Chemin du GIF local
    gif_path = IMAGES_DIR / "ryu-ken.gif"
    # Afficher le GIF sous le titre
    if gif_path.exists():
        st.image(str(gif_path), use_container_width=True)
    else:
        st.write("Erreur : le GIF de perspectives est introuvable.")

    # Section 12.1 - Pistes d'amélioration pour le modèle
    st.subheader("Pistes d'amélioration pour le modèle")
    st.write("""
    Afin d'améliorer la pertinence et la précision de notre modèle, plusieurs pistes ont été envisagées :
    """)
    improvements = [
        "Enrichissement du Dataset : Pour obtenir une vision plus fiable et actualisée du marché des jeux vidéo, il serait bénéfique d'incorporer des données supplémentaires, notamment des ventes digitales. Les données actuelles sont principalement basées sur les ventes physiques, ce qui ne reflète pas entièrement les tendances actuelles du marché, dominé par les achats numériques. En intégrant ces informations, nous pourrions améliorer la représentativité et la précision de notre modèle.",
        "Intégration des Net Promoter Scores (NPS) : Nous avons récemment commencé à récupérer des informations sur les Net Promoter Scores (NPS) des jeux vidéo, qui mesurent la satisfaction et la fidélité des clients. L'ajout de cette métrique pourrait fournir des insights précieux sur la relation entre la satisfaction des utilisateurs et les ventes, permettant ainsi de modéliser de manière plus précise les facteurs influençant la performance des jeux.",
        "Analyse des tendances du marché : Une analyse plus approfondie des tendances actuelles du marché, telles que la popularité croissante des jeux mobiles et des plateformes de streaming, pourrait fournir des variables supplémentaires pertinentes pour nos modèles. Cela permettrait d'anticiper les évolutions du marché et de mieux prédire les ventes futures.",
        "Collaboration avec des experts du domaine : Travailler en collaboration avec des experts de l'industrie des jeux vidéo pourrait nous aider à identifier des variables clés et des tendances émergentes. Leur expertise pourrait également orienter l'interprétation de nos résultats et suggérer des améliorations pratiques pour nos modèles.",
    ]

    for i, improvement in enumerate(improvements):
        st.write(f"**{i + 1}.** {improvement}")

    st.write("""
    En intégrant ces pistes d'amélioration, nous visons à renforcer la robustesse et la précision de notre modèle, offrant ainsi des insights plus pertinents et fiables pour l'analyse du marché des jeux vidéo.
    """)

    # Section 12.2 - Contributions à la connaissance scientifique
    st.subheader("Contributions à la connaissance scientifique")
    st.write("""
    Notre projet vise à apporter des contributions significatives à la connaissance scientifique dans le domaine de l'analyse des données de l'industrie des jeux vidéo. Pour partager nos découvertes et nos méthodes avec la communauté scientifique et les professionnels du secteur, nous prévoyons de publier notre travail sur GitHub.
    """)

    # Ajout d'une zone interactive pour l'engagement
    st.subheader("Participez à l'amélioration du modèle")
    st.write(
        "Nous serions ravis de recevoir vos suggestions et idées pour améliorer notre modèle. Partagez vos commentaires ci-dessous :"
    )

    # Formulaire de commentaires
    with st.form("feedback_form"):
        name = st.text_input("Votre nom")
        st.text_area("Vos suggestions")
        submitted = st.form_submit_button("Envoyer")

        if submitted:
            st.write(f"Merci pour vos suggestions, {name}!")

    # Ajout d'un jeu interactif simple (exemple : quiz)
    st.subheader("Quiz sur le Marché des Jeux Vidéo")
    st.write("Testez vos connaissances sur le marché des jeux vidéo avec ce petit quiz !")

    quiz_questions = {
        "Quelle est la plateforme de jeux vidéo la plus vendue de tous les temps ?": [
            "PlayStation 2",
            "Nintendo Switch",
            "Xbox 360",
        ],
        "Quel jeu a généré le plus de revenus en 2020 ?": [
            "Fortnite",
            "Call of Duty: Modern Warfare",
            "League of Legends",
        ],
        "Quelle entreprise développe la série de jeux 'The Legend of Zelda' ?": [
            "Nintendo",
            "Sony",
            "Microsoft",
        ],
    }

    quiz_answers = {
        "Quelle est la plateforme de jeux vidéo la plus vendue de tous les temps ?": "PlayStation 2",
        "Quel jeu a généré le plus de revenus en 2020 ?": "Call of Duty: Modern Warfare",
        "Quelle entreprise développe la série de jeux 'The Legend of Zelda' ?": "Nintendo",
    }

    score = 0
    for question, options in quiz_questions.items():
        st.write(question)
        answer = st.radio("", options, key=question)
        if answer == quiz_answers[question]:
            score += 1

    if st.button("Soumettre le quiz"):
        st.write(f"Votre score : {score} / {len(quiz_questions)}")
        if score == len(quiz_questions):
            st.balloons()
            st.write("Félicitations ! Vous avez tout juste.")

            # Lottie celebration animation for perfect score
            celebration_anim = _load_lottie_url(LOTTIE_URLS["celebration"])
            if celebration_anim is not None:
                st_lottie(celebration_anim, height=250, key="quiz_celebration")

            youwin_path = IMAGES_DIR / "youwin.png"
            if youwin_path.exists():
                image = Image.open(youwin_path)
                st.image(image, caption="You Win!")
            else:
                st.warning("Image youwin.png introuvable.")
        else:
            gameover_path = IMAGES_DIR / "game_over.png"
            if gameover_path.exists():
                image = Image.open(gameover_path)
                st.image(image, caption="Game Over")
            else:
                st.warning("Image game_over.png introuvable.")
            st.write("Réessayez pour améliorer votre score.")

    # Fin de la page
    st.write("Merci de votre participation et de vos précieux commentaires !")
