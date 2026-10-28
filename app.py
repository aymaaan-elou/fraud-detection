import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Détection de Fraude Bancaire", page_icon="🛡️")
st.title("🛡️ Détection de Fraude Bancaire")
st.markdown("Simulez une transaction pour obtenir une prédiction de fraude avec explication SHAP.")

# Charger le modèle
@st.cache_resource
def load_model():
    return joblib.load('model_fraud.pkl')

model = load_model()

# Formulaire de saisie
st.sidebar.header("Paramètres de la transaction")

# Champs pour les features principales (V1-V28 sont anonymisées, on les simplifie)
st.sidebar.subheader("Montant et temps")
amount = st.sidebar.number_input("Montant (Amount)", min_value=0.0, max_value=10000.0, value=100.0)
time_val = st.sidebar.number_input("Temps (Time)", min_value=0.0, max_value=200000.0, value=50000.0)

st.sidebar.subheader("Features PCA (V1-V28)")
st.sidebar.caption("Les valeurs sont normalisées autour de 0. Ajustez quelques sliders pour simuler une transaction.")

# Générer les sliders pour V1-V10 (les plus importantes généralement)
v_values = {}
for i in range(1, 11):
    v_values[f'V{i}'] = st.sidebar.slider(f'V{i}', -5.0, 5.0, 0.0, step=0.1)

# Bouton de prédiction
if st.button("Analyser la transaction"):
    # Construire le DataFrame d'entrée avec les 30 features
    # Pour les V11-V28 non modifiés, on met 0 (valeur moyenne)
    input_dict = {'Time': time_val, 'Amount': amount}
    for i in range(1, 29):
        input_dict[f'V{i}'] = v_values.get(f'V{i}', 0.0)
    
    input_df = pd.DataFrame([input_dict])
    
    # Prédiction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100
    
    # Affichage
    st.subheader("Résultat de l'analyse")
    if prediction == 1:
        st.error(f"⚠️ **Transaction suspecte** — Probabilité de fraude : {probability:.1f}%")
    else:
        st.success(f"✅ **Transaction normale** — Probabilité de fraude : {probability:.1f}%")
    
    st.info("💡 Ce modèle utilise XGBoost avec SMOTE et des explications SHAP pour la transparence.")