from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, DeepFool
from art.estimators.classification import KerasClassifier
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle
import os
import numpy as np

import tensorflow as tf  
tf.compat.v1.disable_eager_execution()  # Désactive l'exécution immédiate

# Fixer les graines pour la reproductibilité
np.random.seed(42)

# Charger le modèle
model_path = "model_keras.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Le fichier {model_path} est introuvable.")
model = load_model(model_path)
classifier = KerasClassifier(model=model, clip_values=(0, 1))

# Charger le scaler
scaler_path = "scaler.pkl"
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Le fichier {scaler_path} est introuvable.")
with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

# Charger les données
file_path = "DatasetmalwareExtrait.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")
data = pd.read_csv(file_path)
X = data.drop('legitimate', axis=1)
y = data['legitimate']

# Normalisation des données
columns_to_normalize = ['AddressOfEntryPoint', 'ResourceSize', 'SizeOfStackReserve']
X[columns_to_normalize] = scaler.transform(X[columns_to_normalize])

# Génération des exemples adversariaux
print("⚔️ Génération des exemples adversariaux avec FGSM...")
fgsm_attack = FastGradientMethod(estimator=classifier, eps=0.1)
X_adv_fgsm = fgsm_attack.generate(X=np.array(X))
fgsm_df = pd.DataFrame(X_adv_fgsm, columns=X.columns)
fgsm_df['legitimate'] = y.values
fgsm_df.to_csv("dataset_adversarial_fgsm.csv", index=False)
print("✅ Dataset FGSM sauvegardé.")

print("⚔️ Génération des exemples adversariaux avec PGD...")
pgd_attack = ProjectedGradientDescent(estimator=classifier, eps=0.1)
X_adv_pgd = pgd_attack.generate(X=np.array(X))
pgd_df = pd.DataFrame(X_adv_pgd, columns=X.columns)
pgd_df['legitimate'] = y.values
pgd_df.to_csv("dataset_adversarial_pgd.csv", index=False)
print("✅ Dataset PGD sauvegardé.")

print("⚔️ Génération des exemples adversariaux avec DeepFool...")
deepfool_attack = DeepFool(estimator=classifier)
X_adv_deepfool = deepfool_attack.generate(X=np.array(X))
deepfool_df = pd.DataFrame(X_adv_deepfool, columns=X.columns)
deepfool_df['legitimate'] = y.values
deepfool_df.to_csv("dataset_adversarial_deepfool.csv", index=False)
print("✅ Dataset DeepFool sauvegardé.")

print("🎯 Datasets adversariaux générés et sauvegardés avec succès : FGSM, PGD, DeepFool.")
