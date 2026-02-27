import streamlit as st
import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import subprocess
import random
from style import apply_style
from presentation import presentation_et_objectif
from methodologie import methodologie
from dataviz import dataviz
from  feature_engineering import feature_engineering
from modelisation import modelisation
from perspectives import perspectives
from prediction import prediction_page
from analyse_avis_utilisateurs import predict_user_reviews
import plotly.graph_objects as go
import base64
from io import BytesIO
from PIL import Image




def  perception():    
    st.title("Perception")
    st.write("Découvrez ce que vos utilisateurs pensent de votre produit en analysant leurs avis. Téléchargez un fichier CSV et laissez notre modèle prédire la perception de vos clients.")
    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
    st.write("IMPORTANT: Le fichier CSV doit avoir une colonne unique nommer 'user_review'en langue anglaise.")
    
    if uploaded_file is not None:
        if st.button("Lancer la prédiction"):
            data, positive_percentage, negative_percentage = predict_user_reviews(uploaded_file)
        
            if data is not None and positive_percentage is not None:
                st.write(f"Pourcentage de réponses positives (1) : {positive_percentage:.2f}%")
                st.write(f"Pourcentage de réponses négatives (0) : {negative_percentage:.2f}%")
        

            # Fonction pour lire et encoder l'image en base64
            def pil_to_base64(img_path):
                img = Image.open(img_path)
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode()
                return f"data:image/png;base64,{img_str}"
            # Afficher le DataFrame avec les prédictions
            #st.write(data)
                # Fonction pour générer une couleur en dégradé du rouge au vert
            # Fonction pour générer une couleur en dégradé du rouge au vert avec transparence
            def get_color(value, alpha=0.5):
                if value <= 25:
                    # Rouge à orange
                    red = 255
                    green = int((value / 25) * 165)
                    blue = 0
                elif value <= 50:
                    # Orange à jaune
                    red = 255
                    green = 165 + int(((value - 25) / 25) * (255 - 165))
                    blue = 0
                elif value <= 75:
                    # Jaune à vert
                    red = 255 - int(((value - 50) / 25) * 200)
                    green = 240 - int((value / 100) * (255 - 128))
                    blue = 0
                else:
                # Vert clair à vert foncé
                    red = 0
                    green = 255 - int(((value - 55) / 25) * (255 - 128))
                    blue = 0
                return f'rgba({red},{green},{blue},{alpha})'

                # Chemin de votre image locale
            image_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'street_fighter2.png')
            encoded_image = pil_to_base64(image_path)

            # Créer les étapes de dégradé
            steps = []
            for i in range(0, 101, 1):  # Plus la granularité est fine, plus le dégradé sera lisse
                steps.append({'range': [i, i+1], 'color': get_color(i, alpha=0.7)})  

            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                gauge = {
                    'axis': {'range': [0, 100], 'tickfont': {'size': 25, 'family': 'Arial', 'color': 'white'}},
                    'bar': {'color': 'rgba(0,0,0,0)'},  # Couleur transparente
                    'steps': steps,
                    'threshold': {
                        'line': {'color': "black", 'width': 5},
                        'thickness': 0.75,
                        'value': positive_percentage}}))

            # Mettre à jour la mise en page pour inclure une image de fond
            fig.update_layout(
                title={
                            'text': "Pourcentage d'avis positifs",  # Titre du graphique
                            'y':0.95,  # Positionnement du titre
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top',
                            'font': {'color': 'black'} 
                            }, # Couleur du titre

                paper_bgcolor='white',  # Arrière-plan noir pour le thème Street Fighter
                font=dict(family="Arial", size=18, color="white"),
                margin=dict(t=50, b=0, l=0, r=0),
                images=[dict(
                    source=encoded_image,  # Utilisation de l'image encodée en base64
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,  # Position au centre
                    sizex=1, sizey=1,  # Prend toute la taille de la figure
                    xanchor="center", yanchor="middle",
                    layer="below")],  # Image en arrière-plan
            )

            st.plotly_chart(fig)
    # Ajouter le disclaimer en bas de la page
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: gray;'>Ce modèle est en version bêta et peut faire des erreurs. Envisagez de vérifier les informations importantes.</p>", unsafe_allow_html=True)              