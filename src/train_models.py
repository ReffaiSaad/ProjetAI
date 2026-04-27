"""
train_models.py
---------------
Entraînement et évaluation des modèles de classification :
- SVM
- Régression Logistique
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


# ─────────────────────────────────────────────
# PREPARATION DATA
# ─────────────────────────────────────────────
def prepare_data(X, y, test_size=0.2):
    """
    Split + normalisation
    """

    # Conversion labels texte → 0 / 1
    y = np.array([1 if label == "positive" else 0 for label in y])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    # Normalisation (IMPORTANT pour SVM & Logistic)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
def train_svm(X_train, y_train):
    """
    SVM rapide et optimisé pour gros dataset
    """
    model = LinearSVC()
    model.fit(X_train, y_train)
    return model


def train_logistic(X_train, y_train):
    """
    Régression logistique
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, name="Model"):
    """
    Affiche performance
    """

    y_pred = model.predict(X_test)

    print("\n" + "="*60)
    print(f"📊 Résultats : {name}")
    print("="*60)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {acc:.4f}")

    print("\nClassification Report :")
    print(classification_report(y_test, y_pred))


# ─────────────────────────────────────────────
# MAIN FUNCTION (IMPORTANT POUR run_pipeline.py)
# ─────────────────────────────────────────────
def run_all_models(X_skipgram, X_cbow, y_labels):

    # =========================
    # SVM Skip-gram
    # =========================
    print("\n🚀 SVM + Skip-gram")
    X_train, X_test, y_train, y_test = prepare_data(X_skipgram, y_labels)
    model = train_svm(X_train, y_train)
    evaluate_model(model, X_test, y_test, "SVM Skip-gram")

    # =========================
    # SVM CBOW
    # =========================
    print("\n🚀 SVM + CBOW")
    X_train, X_test, y_train, y_test = prepare_data(X_cbow, y_labels)
    model = train_svm(X_train, y_train)
    evaluate_model(model, X_test, y_test, "SVM CBOW")

    # =========================
    # Logistic Skip-gram
    # =========================
    print("\n🚀 Logistic Regression + Skip-gram")
    X_train, X_test, y_train, y_test = prepare_data(X_skipgram, y_labels)
    model = train_logistic(X_train, y_train)
    evaluate_model(model, X_test, y_test, "Logistic Skip-gram")

    # =========================
    # Logistic CBOW
    # =========================
    print("\n🚀 Logistic Regression + CBOW")
    X_train, X_test, y_train, y_test = prepare_data(X_cbow, y_labels)
    model = train_logistic(X_train, y_train)
    evaluate_model(model, X_test, y_test, "Logistic CBOW")
