import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.models import load_model
import pickle
import os
import numpy as np

# Fixer les graines pour la reproductibilité
np.random.seed(42)

# Charger le modèle
model_path = "model_keras.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Le fichier {model_path} est introuvable.")
model = load_model(model_path)

# Charger le scaler
scaler_path = "scaler.pkl"
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Le fichier {scaler_path} est introuvable.")
with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

# Charger les données originales
file_path = "DatasetmalwareExtrait.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")
data = pd.read_csv(file_path)
X = data.drop('legitimate', axis=1)
y = data['legitimate']

# Normalisation
columns_to_normalize = ['AddressOfEntryPoint', 'ResourceSize', 'SizeOfStackReserve']
X[columns_to_normalize] = scaler.transform(X[columns_to_normalize])

# Évaluation sur les données originales
y_pred_original = (model.predict(X) > 0.5).astype(int)
accuracy_original = accuracy_score(y, y_pred_original)
print(f"🔹 Précision sur les données originales : {accuracy_original:.2f}")
print("🔹 Rapport de classification :")
print(classification_report(y, y_pred_original))
print("🔹 Matrice de confusion :")
print(confusion_matrix(y, y_pred_original))

# Charger et évaluer les jeux adversariaux
adv_files = {
    "FGSM": "dataset_adversarial_fgsm.csv",
    
}

for key, path in adv_files.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le fichier {path} est introuvable.")
    adv_data = pd.read_csv(path)
    if 'legitimate' not in adv_data.columns:
        raise ValueError(f"Le fichier {path} doit contenir une colonne 'legitimate'.")
    X_adv = adv_data.drop('legitimate', axis=1)
    y_adv = adv_data['legitimate']
    X_adv[columns_to_normalize] = scaler.transform(X_adv[columns_to_normalize])
    
    y_pred_adv = (model.predict(X_adv) > 0.5).astype(int)
    accuracy_adv = accuracy_score(y_adv, y_pred_adv)
    print(f"\n⚔️ Précision sur les données adversariales ({key}) : {accuracy_adv:.2f}")
    print(f"⚔️ Rapport de classification ({key}):")
    print(classification_report(y_adv, y_pred_adv))
    print(f"⚔️ Matrice de confusion ({key}):")
    print(confusion_matrix(y_adv, y_pred_adv))

print("🎯 Évaluation de la robustesse terminée avec succès.")
