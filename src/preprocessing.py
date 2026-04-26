"""
preprocessing.py
----------------
Nettoyage et lemmatisation des reviews IMDB avec spaCy.
Les résultats sont sauvegardés en .pkl pour éviter de relancer
le prétraitement à chaque test.
"""

import re
import pickle
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as stop_en


# Charger le modèle spaCy une seule fois (désactiver les pipes inutiles pour la vitesse)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def preprocess_fast(texts: list[str], batch_size: int = 1000) -> list[list[str]]:
    """
    Pipeline de prétraitement rapide :
      1. Nettoyage regex (HTML, URLs, chiffres, ponctuation)
      2. Lemmatisation + suppression des stop words via spaCy en batch

    Args:
        texts      (list[str]): Reviews brutes
        batch_size (int)      : Taille des batches spaCy (défaut 1000)

    Returns:
        list[list[str]]: Liste de listes de tokens lemmatisés
    """
    # ── Étape 1 : nettoyage rapide (sans spaCy) ──────────────────────────────
    cleaned = []
    for text in texts:
        text = text.lower()
        text = re.sub(r'<.*?>',      ' ', text)   # balises HTML
        text = re.sub(r'http\S+|www\S+', ' ', text)  # URLs
        text = re.sub(r'\d+',        ' ', text)   # chiffres
        text = re.sub(r'[^\w\s]',    ' ', text)   # ponctuation
        text = re.sub(r'\s+',        ' ', text).strip()
        cleaned.append(text)

    # ── Étape 2 : lemmatisation spaCy en batch (très rapide) ─────────────────
    results = []
    for doc in nlp.pipe(cleaned, batch_size=batch_size):
        tokens = [
            token.lemma_
            for token in doc
            if token.text not in stop_en and not token.is_space
        ]
        results.append(tokens)

    return results


def save_cleaned(cleaned_texts: list[list[str]], path: str = "data/cleaned_texts.pkl") -> None:
    """
    Sauvegarde les textes nettoyés dans un fichier pickle.

    Args:
        cleaned_texts (list[list[str]]): Textes prétraités
        path          (str)            : Chemin de sauvegarde
    """
    with open(path, "wb") as f:
        pickle.dump(cleaned_texts, f)
    print(f"✅ Cleaned texts sauvegardés dans : {path}")


def load_cleaned(path: str = "data/cleaned_texts.pkl") -> list[list[str]]:
    """
    Charge les textes nettoyés depuis un fichier pickle.

    Args:
        path (str): Chemin vers le fichier .pkl

    Returns:
        list[list[str]]: Textes prétraités
    """
    with open(path, "rb") as f:
        cleaned_texts = pickle.load(f)
    print(f"✅ Cleaned texts chargés depuis : {path}  ({len(cleaned_texts)} reviews)")
    return cleaned_texts
