"""
run_pipeline.py
---------------
Script principal — PARTIE 1.
  1. Chargement des données
  2. Prétraitement (ou chargement du cache .pkl)
  3. Chargement des modèles Word2Vec sauvegardés
  4. Vectorisation

Usage :
    python run_pipeline.py
"""

import os
import numpy as np

from src.data_loader   import load_data
from src.preprocessing import preprocess_fast, save_cleaned, load_cleaned
from src.vectorization import vectorize_texts, load_w2v

# ─── Chemins ──────────────────────────────────────────────────────────────────
DATASET_PATH      = "data/IMDB Dataset.csv"
CLEANED_PATH      = "data/cleaned_texts.pkl"
W2V_SKIPGRAM_PATH = "models/w2v.model"
W2V_CBOW_PATH     = "models/w2v_cbow.model"
# ──────────────────────────────────────────────────────────────────────────────


def main():

    # ── 1. Chargement des données ─────────────────────────────────────────────
    print("\n[1/3] Chargement du dataset...")
    texts, labels = load_data(DATASET_PATH)

    # ── 2. Prétraitement (cache) ──────────────────────────────────────────────
    if os.path.exists(CLEANED_PATH):
        print(f"\n[2/3] Cache trouvé → chargement de {CLEANED_PATH}")
        cleaned_texts = load_cleaned(CLEANED_PATH)
    else:
        print("\n[2/3] Prétraitement en cours (peut prendre quelques minutes)...")
        cleaned_texts = preprocess_fast(texts)
        save_cleaned(cleaned_texts, CLEANED_PATH)

    print(f"       Total reviews    : {len(cleaned_texts)}")

    # ── Aperçu avant / après prétraitement ───────────────────────────────────
    print("\n" + "─"*60)
    print("  APERÇU PRÉTRAITEMENT — review #0")
    print("─"*60)
    print("  AVANT :")
    print(f"    {texts[0][:300]}...")
    print("\n  APRÈS (tokens) :")
    print(f"    {cleaned_texts[0][:20]}")
    print(f"    ({len(cleaned_texts[0])} tokens)")
    print("─"*60)

    # ── 3. Chargement des modèles Word2Vec ────────────────────────────────────
    print("\n[3/3] Chargement des modèles Word2Vec...")
    w2v_skipgram = load_w2v(W2V_SKIPGRAM_PATH)
    w2v_cbow     = load_w2v(W2V_CBOW_PATH)

    # ── Vectorisation ─────────────────────────────────────────────────────────
    print("\nVectorisation en cours...")
    X_skipgram = vectorize_texts(cleaned_texts, w2v_skipgram)
    X_cbow     = vectorize_texts(cleaned_texts, w2v_cbow)
    y          = np.array(labels)

    # ── Aperçu vectorisation ──────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("  APERÇU VECTORISATION")
    print("─"*60)
    print(f"  Nombre de reviews    : {X_skipgram.shape[0]}")
    print(f"  Taille du vecteur    : {X_skipgram.shape[1]} dimensions")
    print()
    for i in range(3):
        print(f"  Review #{i} — label : {labels[i]}")
        print(f"    Skip-gram  : [{', '.join(f'{v:.4f}' for v in X_skipgram[i][:6])} ...]")
        print(f"    CBOW       : [{', '.join(f'{v:.4f}' for v in X_cbow[i][:6])} ...]")
        print()
    print("─"*60)

    print("\n✅ Pipeline PARTIE 1 terminé.")
    print("   Données prêtes pour la PARTIE 2 :")
    print(f"     - X_skipgram : {X_skipgram.shape}")
    print(f"     - X_cbow     : {X_cbow.shape}")
    print(f"     - y          : {y.shape}\n")

    return X_skipgram, X_cbow, y, labels


if __name__ == "__main__":
    main()