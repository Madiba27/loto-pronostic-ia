import pandas as pd
import numpy as np
from collections import Counter
import random
import requests
from bs4 import BeautifulSoup
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import streamlit as st

@st.cache_data(show_spinner=False)
def recuperer_tirages_loto(n=50):
    url = "https://tirage-gagnant.com/resultats-loto/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    blocs = soup.find_all("div", class_="results")[:n]
    tirages = []
    for bloc in blocs:
        boules = bloc.find_all("span", class_="boule")
        nums = [int(b.text.strip()) for b in boules if b.text.strip().isdigit()]
        if len(nums) >= 5:
            tirages.append(nums[:5])
    return tirages

try:
    loto_tirages = recuperer_tirages_loto()
except:
    st.error("Erreur de chargement des tirages. Utilisation de valeurs par défaut.")
    loto_tirages = [[7, 30, 37, 40, 45], [5, 12, 27, 30, 44]] * 25

X = [[1 if i in tirage else 0 for i in range(1, 50)] for tirage in loto_tirages[:-1]]
Y = [1 if i in loto_tirages[-1] else 0 for i in range(1, 50)]
X = np.array(X)
Y = np.array(Y)
X_seq = X.reshape((X.shape[0], 1, X.shape[1]))

model = Sequential()
model.add(LSTM(64, input_shape=(1, 49), activation='relu'))
model.add(Dense(49, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X_seq, np.tile(Y, (X.shape[0], 1)), epochs=50, verbose=0)

prediction = model.predict(X_seq[-1:])[0]
indices_probables = np.argsort(prediction)[-10:] + 1

st.title("🔮 Pronostics Loto & EuroMillions avec IA")
st.subheader("Loto - Tirage IA")

if st.button("🎰 Générer une grille Loto IA"):
    grille_lstm = sorted(random.sample(list(indices_probables), 5))
    num_chance = random.randint(1, 10)
    st.success(f"Numéros : {grille_lstm}  |  Chance : {num_chance}")

st.subheader("EuroMillions - Simulation classique")
frequents_euro = [5, 15, 29, 38, 47]

def generer_grille_euromillions(frequents, total=5):
    grille = set()
    while len(grille) < total:
        if random.random() < 0.7:
            grille.add(random.choice(frequents))
        else:
            grille.add(random.randint(1, 50))
    return sorted(grille)

def generer_etoiles():
    etoiles = set()
    while len(etoiles) < 2:
        etoiles.add(random.randint(1, 12))
    return sorted(etoiles)

if st.button("⭐ Générer grille EuroMillions"):
    grille_euro = generer_grille_euromillions(frequents_euro)
    etoiles = generer_etoiles()
    st.success(f"Numéros : {grille_euro}  |  Étoiles : {etoiles}")
