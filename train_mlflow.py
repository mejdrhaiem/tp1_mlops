
# ===== 1. Importer les bibliothèques =====

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ===== 2. Charger les données (une seule fois) =====

df = pd.read_csv("churn_data.csv")  # Le tableau de 500 clients

y = df["churn"]  # Réponse à prédire : 1 = part, 0 = reste


# ===== 3. Fonction : préparer les données pour une liste de colonnes =====

def preparer(colonnes):
    # Transformer le texte 'abonnement' en colonnes 0/1 si présent
    X = pd.get_dummies(
        df[colonnes],
        drop_first=True
    )

    # 80 % apprentissage / 20 % test
    # Même découpage à chaque fois grâce à random_state=42
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


# ===== 4. Fonction : lancer UN essai et l'enregistrer dans MLflow =====

def lancer_essai(nom, n_estimators, max_depth, colonnes):

    X_train, X_test, y_train, y_test = preparer(colonnes)

    # Ouvrir un run au nom de l'essai
    with mlflow.start_run(run_name=nom):

        # Créer le modèle
        model = RandomForestClassifier(
            n_estimators=n_estimators,  # Nombre d'arbres
            max_depth=max_depth,        # Profondeur maximale des arbres
            random_state=42             # Résultats reproductibles
        )

        # Entraînement
        model.fit(X_train, y_train)

        # Prédictions sur les données de test
        pred = model.predict(X_test)

        # --- Paramètres : les réglages de cet essai ---

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features", ",".join(colonnes))

        # --- Métriques : les résultats de cet essai ---

        mlflow.log_metric(
            "accuracy",
            accuracy_score(y_test, pred)
        )

        # zero_division=0 évite un avertissement lorsque
        # le modèle ne prédit aucun départ
        mlflow.log_metric(
            "precision",
            precision_score(y_test, pred, zero_division=0)
        )

        mlflow.log_metric(
            "recall",
            recall_score(y_test, pred, zero_division=0)
        )

        mlflow.log_metric(
            "f1",
            f1_score(y_test, pred, zero_division=0)
        )

        # --- Artifact : le modèle sauvegardé dans ce run ---

        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle"
        )

        # Afficher les résultats dans le terminal
        print(
            nom,
            "-> accuracy :",
            round(accuracy_score(y_test, pred), 3),
            "| f1 :",
            round(f1_score(y_test, pred, zero_division=0), 3)
        )


# ===== 5. Programme principal =====

# Utiliser la même expérience que dans l'étape 1
mlflow.set_experiment("churn-prediction")

TOUT = [
    "age",
    "salaire",
    "abonnement",
    "anciennete",
    "consommation"
]

# Essai 1 : 100 arbres, profondeur 5, toutes les colonnes
lancer_essai("essai-1", 100, 5, TOUT)

# Essai 2 : modèle plus simple (50 arbres, profondeur 3)
lancer_essai("essai-2", 50, 3, TOUT)

# Essai 3 : sans la colonne salaire
lancer_essai(
    "essai-3",
    100,
    5,
    ["age", "abonnement", "anciennete", "consommation"]
)
