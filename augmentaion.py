from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import pickle
import os
import numpy as np

# Fixer les graines pour la reproductibilité
np.random.seed(42)
tf.random.set_seed(42)

# Charger les données
file_path = "DatasetmalwareExtrait.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")

data = pd.read_csv(file_path)

# Vérification des colonnes nécessaires
required_columns = ['legitimate']
if 'legitimate' not in data.columns:
    raise ValueError("La colonne 'legitimate' est manquante dans le dataset.")

# Séparation des features et de la cible
X = data.drop('legitimate', axis=1)
y = data['legitimate']

# Application de SMOTE pour équilibrer les classes
print("⚙️ Application de SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Afficher la distribution après SMOTE
print("🔄 Répartition après SMOTE :")
print(pd.Series(y_resampled).value_counts())

# Normalisation des données après SMOTE
columns_to_normalize = ['AddressOfEntryPoint', 'ResourceSize', 'SizeOfStackReserve']
scaler = StandardScaler()
try:
    X_resampled[columns_to_normalize] = scaler.fit_transform(X_resampled[columns_to_normalize])
except KeyError as e:
    raise ValueError(f"Erreur lors de la normalisation : {e}")

# Division des données équilibrées en train/test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

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
print(f"✅ Précision sur les données de test : {accuracy:.2f}")
print("✅ Rapport de classification :")
print(classification_report(y_test, y_pred))

# Sauvegarder le modèle et le scaler
model.save("model_augmented_keras.h5")
with open("scaler_augmented.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

# Sauvegarder le dataset augmenté
augmented_path = "dataset_augmented.csv"
pd.concat([X_resampled, pd.Series(y_resampled, name='legitimate')], axis=1).to_csv(augmented_path, index=False)
print(f"✅ Dataset augmenté sauvegardé sous '{augmented_path}'.")

print("🎯 Modèle ré-entraîné et sauvegardé avec succès après application de SMOTE.")
