import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import tensorflow as tf

# ✅ Chargement du modèle avec gestion des exceptions
model_path = "model_augmented_keras.h5"
scaler_path = "scaler_augmented.pkl"

try:
    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ Le fichier 'model_augmented_keras.h5' est introuvable.")
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"❌ Erreur lors du chargement du modèle : {e}")
    st.stop()

try:
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("❌ Le fichier 'scaler_augmented.pkl' est introuvable.")
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
except Exception as e:
    st.error(f"❌ Erreur lors du chargement du scaler : {e}")
    st.stop()

# ✅ Interface utilisateur
st.title("🔍 Détection de Malware")
st.sidebar.header("🛠️ Entrez les caractéristiques")
st.sidebar.info("Remplissez les champs suivants pour prédire si le fichier est légitime ou malveillant.")

def user_input():
    AddressOfEntryPoint = st.sidebar.number_input('AddressOfEntryPoint', min_value=0, max_value=1000000, value=10407)
    MajorLinkerVersion = st.sidebar.number_input('MajorLinkerVersion', min_value=0, max_value=255, value=9)
    MajorImageVersion = st.sidebar.number_input('MajorImageVersion', min_value=0, max_value=1000, value=6)
    MajorOperatingSystemVersion = st.sidebar.number_input('MajorOperatingSystemVersion', min_value=0, max_value=1000, value=6)
    DllCharacteristics = st.sidebar.number_input('DllCharacteristics', min_value=0, max_value=65535, value=33088)
    SizeOfStackReserve = st.sidebar.number_input('SizeOfStackReserve', min_value=0, max_value=33554432, value=262144)
    NumberOfSections = st.sidebar.number_input('NumberOfSections', min_value=1, max_value=40, value=4)
    ResourceSize = st.sidebar.number_input('ResourceSize', min_value=0, max_value=4294967295, value=952)
    return pd.DataFrame([{
        "AddressOfEntryPoint": AddressOfEntryPoint,
        "MajorLinkerVersion": MajorLinkerVersion,
        "MajorImageVersion": MajorImageVersion,
        "MajorOperatingSystemVersion": MajorOperatingSystemVersion,
        "DllCharacteristics": DllCharacteristics,
        "SizeOfStackReserve": SizeOfStackReserve,
        "NumberOfSections": NumberOfSections,
        "ResourceSize": ResourceSize
    }])

# ✅ Données d'entrée utilisateur
input_data = user_input()

# ✅ Vérification des colonnes avant normalisation
columns_to_normalize = ['AddressOfEntryPoint', 'ResourceSize', 'SizeOfStackReserve']
for col in columns_to_normalize:
    if col not in input_data.columns:
        st.error(f"⚠️ La colonne '{col}' est manquante dans les données saisies.")
        st.stop()

# ✅ Normalisation avec le scaler pré-entrainé
try:
    input_data[columns_to_normalize] = scaler.transform(input_data[columns_to_normalize])
except Exception as e:
    st.error(f"⚠️ Erreur lors de la normalisation : {e}")
    st.stop()

# ✅ Prédiction
if st.button("🔮 Prédire"):
    try:
        prediction = model.predict(input_data)
        if isinstance(prediction, (np.ndarray, list)):
            prediction = prediction[0]
        if prediction >= 0.5:
            st.success("✅ **Le fichier est LÉGITIME**")
        else:
            st.error("❌ **Le fichier est MALVEILLANT**")
        st.write("### 📝 **Données saisies :**")
        st.dataframe(input_data)
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")
        st.stop()

st.write("🔗 **Fin de l'analyse. Merci d'avoir utilisé l'application !**")
