# 🛡️ Détection de Fraude Bancaire avec IA Explicable (XAI)

## 📌 Contexte du Projet
Ce projet a été réalisé dans le cadre de mon exploration du domaine du **Big Data et de l'Intelligence Artificielle**. L'objectif est de construire un système capable de détecter les transactions bancaires frauduleuses, tout en fournissant des explications claires sur les décisions du modèle (un enjeu crucial pour le secteur bancaire).

## ⚠️ Le Défi : Le Déséquilibre des Données
Le jeu de données utilisé (Credit Card Fraud Detection) contient **284 807 transactions**, dont seulement **492 fraudes (0,172%)**. Ce déséquilibre extrême est un défi classique en Machine Learning. 

## 🛠️ Solutions Techniques Apportées
*   **Gestion du déséquilibre :** Utilisation de la technique **SMOTE** (Synthetic Minority Over-sampling Technique) pour générer des exemples synthétiques de fraudes.
*   **Modélisation :** Entraînement d'un classifieur **XGBoost**, reconnu pour sa performance sur les données tabulaires déséquilibrées.
*   **Optimisation :** Ajustement du seuil de décision (threshold) basé sur le score F1 pour maximiser la détection des fraudes sans trop de faux positifs.
*   **Explicabilité (XAI) :** Utilisation de **SHAP** (SHapley Additive exPlanations) pour comprendre et visualiser les variables qui ont le plus influencé la prédiction du modèle.

## 🚀 Tester l'application
L'application interactive est déployée sur Streamlit Cloud. Elle permet de simuler une transaction et d'obtenir une prédiction en temps réel.
👉 **[Insérer ici le lien de ton application Streamlit]**

## 📂 Structure du Dépôt
*   `app.py` : Code de l'application web Streamlit.
*   `model_fraud.pkl` : Modèle XGBoost entraîné et sauvegardé.
*   `requirements.txt` : Dépendances Python nécessaires au déploiement.
*   `shap_summary.png` : Graphique généré par SHAP montrant l'importance des variables.

## 💻 Technologies Utilisées
*   **Langage :** Python
*   **Data Science & ML :** Pandas, Scikit-learn, XGBoost, Imbalanced-learn (SMOTE)
*   **Explicabilité :** SHAP
*   **Interface Web :** Streamlit
*   **Déploiement :** Streamlit Community Cloud
