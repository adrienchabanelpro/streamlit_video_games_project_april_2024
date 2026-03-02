import streamlit as st


def perspectives_page() -> None:
    """Render the perspectives and future improvements page."""
    st.title("Perspectives")

    # Section 12.1 - Pistes d'amelioration pour le modele
    st.subheader("Pistes d'amelioration pour le modele")
    st.write("""
    Afin d'ameliorer la pertinence et la precision de notre modele, plusieurs pistes ont ete envisagees :
    """)
    improvements = [
        "Enrichissement du Dataset : Pour obtenir une vision plus fiable et actualisee du marche des jeux video, il serait benefique d'incorporer des donnees supplementaires, notamment des ventes digitales. Les donnees actuelles sont principalement basees sur les ventes physiques, ce qui ne reflete pas entierement les tendances actuelles du marche, domine par les achats numeriques. En integrant ces informations, nous pourrions ameliorer la representativite et la precision de notre modele.",
        "Integration des Net Promoter Scores (NPS) : Nous avons recemment commence a recuperer des informations sur les Net Promoter Scores (NPS) des jeux video, qui mesurent la satisfaction et la fidelite des clients. L'ajout de cette metrique pourrait fournir des insights precieux sur la relation entre la satisfaction des utilisateurs et les ventes, permettant ainsi de modeliser de maniere plus precise les facteurs influencant la performance des jeux.",
        "Analyse des tendances du marche : Une analyse plus approfondie des tendances actuelles du marche, telles que la popularite croissante des jeux mobiles et des plateformes de streaming, pourrait fournir des variables supplementaires pertinentes pour nos modeles. Cela permettrait d'anticiper les evolutions du marche et de mieux predire les ventes futures.",
        "Collaboration avec des experts du domaine : Travailler en collaboration avec des experts de l'industrie des jeux video pourrait nous aider a identifier des variables cles et des tendances emergentes. Leur expertise pourrait egalement orienter l'interpretation de nos resultats et suggerer des ameliorations pratiques pour nos modeles.",
    ]

    for i, improvement in enumerate(improvements):
        st.write(f"**{i + 1}.** {improvement}")

    st.write("""
    En integrant ces pistes d'amelioration, nous visons a renforcer la robustesse et la precision de notre modele, offrant ainsi des insights plus pertinents et fiables pour l'analyse du marche des jeux video.
    """)

    # Section 12.2 - Contributions a la connaissance scientifique
    st.subheader("Contributions a la connaissance scientifique")
    st.write("""
    Notre projet vise a apporter des contributions significatives a la connaissance scientifique dans le domaine de l'analyse des donnees de l'industrie des jeux video. Pour partager nos decouvertes et nos methodes avec la communaute scientifique et les professionnels du secteur, nous prevoyons de publier notre travail sur GitHub.
    """)

    # Zone interactive pour l'engagement
    st.subheader("Participez a l'amelioration du modele")
    st.write(
        "Nous serions ravis de recevoir vos suggestions et idees pour ameliorer notre modele. Partagez vos commentaires ci-dessous :"
    )

    with st.form("feedback_form"):
        name = st.text_input("Votre nom")
        st.text_area("Vos suggestions")
        submitted = st.form_submit_button("Envoyer")

        if submitted:
            st.write(f"Merci pour vos suggestions, {name}!")

    # Quiz
    st.subheader("Quiz sur le Marche des Jeux Video")
    st.write("Testez vos connaissances sur le marche des jeux video avec ce petit quiz !")

    quiz_questions = {
        "Quelle est la plateforme de jeux video la plus vendue de tous les temps ?": [
            "PlayStation 2",
            "Nintendo Switch",
            "Xbox 360",
        ],
        "Quel jeu a genere le plus de revenus en 2020 ?": [
            "Fortnite",
            "Call of Duty: Modern Warfare",
            "League of Legends",
        ],
        "Quelle entreprise developpe la serie de jeux 'The Legend of Zelda' ?": [
            "Nintendo",
            "Sony",
            "Microsoft",
        ],
    }

    quiz_answers = {
        "Quelle est la plateforme de jeux video la plus vendue de tous les temps ?": "PlayStation 2",
        "Quel jeu a genere le plus de revenus en 2020 ?": "Call of Duty: Modern Warfare",
        "Quelle entreprise developpe la serie de jeux 'The Legend of Zelda' ?": "Nintendo",
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
            st.write("Felicitations ! Vous avez tout juste.")
        else:
            st.write("Reessayez pour ameliorer votre score.")

    st.write("Merci de votre participation et de vos precieux commentaires !")
