import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import pickle
import os

# Fixer les graines pour la reproductibilité
np.random.seed(42)
tf.random.set_seed(42)

# Charger les données
file_path = "DatasetmalwareExtrait.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")

data = pd.read_csv(file_path)

# Vérification des colonnes nécessaires
required_columns = ['AddressOfEntryPoint', 'ResourceSize', 'SizeOfStackReserve', 'legitimate']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Les colonnes suivantes sont manquantes : {missing_columns}")

# Gestion des valeurs manquantes
if data.isnull().sum().any():
    print("Des valeurs manquantes ont été détectées. Elles seront supprimées.")
    data = data.dropna()

# Normalisation des colonnes
columns_to_normalize = ['AddressOfEntryPoint', 'ResourceSize', 'SizeOfStackReserve']
scaler = StandardScaler()
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

# Séparation des données en X (features) et y (cible)
X = data.drop('legitimate', axis=1)
y = data['legitimate']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Construction du modèle Keras
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Évaluation sur les données de test
y_pred = (model.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision sur les données de test : {accuracy:.2f}")
print("Rapport de classification :")
print(classification_report(y_test, y_pred))

# Sauvegarder le modèle et le scaler
model.save("model_keras.h5")
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("✅ Modèle et scaler sauvegardés avec succès.")
