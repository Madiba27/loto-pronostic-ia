# IA Loto / EuroMillions

Cette application utilise le Machine Learning (LSTM) pour générer des prédictions de numéros pour le Loto et l'EuroMillions.

## Fonctionnalités

- Scraping automatique des derniers tirages
- Analyse statistique
- Modèle LSTM pour suggestions probabilistes
- Interface interactive Streamlit

## Lancer l'application en local

```bash
pip install -r requirements.txt
streamlit run loto_ai_app.py
```

## Déploiement Streamlit Cloud

1. Crée un dépôt GitHub avec ces fichiers
2. Va sur https://streamlit.io/cloud
3. Clique sur "New App" et sélectionne ton dépôt + `loto_ai_app.py`
