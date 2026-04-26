"""
vectorization.py
----------------
Entraînement Word2Vec (Skip-gram & CBOW), vectorisation des reviews
et sauvegarde / chargement des modèles.
"""

import numpy as np
from gensim.models import Word2Vec


# ── Entraînement ──────────────────────────────────────────────────────────────

def train_word2vec_skipgram(cleaned_texts: list[list[str]]) -> Word2Vec:
    """
    Entraîne un modèle Word2Vec en mode Skip-gram (sg=1).

    Args:
        cleaned_texts (list[list[str]]): Corpus tokenisé

    Returns:
        Word2Vec: Modèle entraîné
    """
    model = Word2Vec(
        sentences=cleaned_texts,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        sg=1          # Skip-gram
    )
    return model


def train_word2vec_cbow(cleaned_texts: list[list[str]]) -> Word2Vec:
    """
    Entraîne un modèle Word2Vec en mode CBOW (sg=0).

    Args:
        cleaned_texts (list[list[str]]): Corpus tokenisé

    Returns:
        Word2Vec: Modèle entraîné
    """
    model = Word2Vec(
        sentences=cleaned_texts,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        sg=0          # CBOW
    )
    return model


# ── Vectorisation ─────────────────────────────────────────────────────────────

def review_to_vector(tokens: list[str], model: Word2Vec) -> np.ndarray:
    """
    Convertit une liste de tokens en un vecteur (moyenne des embeddings).

    Args:
        tokens (list[str]): Tokens d'une review
        model  (Word2Vec) : Modèle Word2Vec entraîné

    Returns:
        np.ndarray: Vecteur de dimension (vector_size,)
    """
    vectors = [model.wv[word] for word in tokens if word in model.wv]

    if len(vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


def vectorize_texts(cleaned_texts: list[list[str]], model: Word2Vec) -> np.ndarray:
    """
    Vectorise l'ensemble du corpus.

    Args:
        cleaned_texts (list[list[str]]): Corpus tokenisé
        model         (Word2Vec)        : Modèle Word2Vec entraîné

    Returns:
        np.ndarray: Matrice de forme (n_samples, vector_size)
    """
    return np.array([
        review_to_vector(tokens, model)
        for tokens in cleaned_texts
    ])


# ── Sauvegarde / Chargement ───────────────────────────────────────────────────

def save_w2v(model: Word2Vec, path: str) -> None:
    """
    Sauvegarde un modèle Word2Vec sur disque.

    Args:
        model (Word2Vec): Modèle à sauvegarder
        path  (str)     : Chemin de sauvegarde (ex: 'models/w2v.model')
    """
    model.save(path)
    print(f"✅ Modèle Word2Vec sauvegardé dans : {path}")


def load_w2v(path: str) -> Word2Vec:
    """
    Charge un modèle Word2Vec depuis le disque.

    Args:
        path (str): Chemin vers le fichier .model

    Returns:
        Word2Vec: Modèle chargé
    """
    model = Word2Vec.load(path)
    print(f"✅ Modèle Word2Vec chargé depuis : {path}")
    return model
