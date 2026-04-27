"""
run_pipeline.py
---------------
Pipeline complet IMDB Sentiment Analysis :

1. Chargement des données
2. Prétraitement (cache)
3. Chargement Word2Vec
4. Vectorisation
5. Classification (SVM + Logistic Regression)
"""

import os
import numpy as np

from src.data_loader import load_data
from src.preprocessing import preprocess_fast, save_cleaned, load_cleaned
from src.vectorization import vectorize_texts, load_w2v
from src.train_models import run_all_models


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATASET_PATH = "data/IMDB Dataset.csv"
CLEANED_PATH = "data/cleaned_texts.pkl"

W2V_SKIPGRAM_PATH = "models/w2v.model"
W2V_CBOW_PATH     = "models/w2v_cbow.model"


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def main():

    print("\n" + "="*70)
    print("🎬 IMDB SENTIMENT ANALYSIS PIPELINE")
    print("="*70)

    # ── 1. LOAD DATA ───────────────────────────
    print("\n[1/5] Loading dataset...")
    texts, labels = load_data(DATASET_PATH)

    # ── 2. PREPROCESSING ───────────────────────
    if os.path.exists(CLEANED_PATH):
        print("\n[2/5] Loading cached preprocessing...")
        cleaned_texts = load_cleaned(CLEANED_PATH)
    else:
        print("\n[2/5] Preprocessing texts...")
        cleaned_texts = preprocess_fast(texts)
        save_cleaned(cleaned_texts, CLEANED_PATH)

    print(f"Total reviews: {len(cleaned_texts)}")

    # ── SAMPLE ────────────────────────────────
    print("\n" + "-"*60)
    print("Sample review:")
    print("-"*60)
    print("RAW   :", texts[0][:200])
    print("CLEAN :", cleaned_texts[0][:15])
    print("-"*60)

    # ── 3. LOAD WORD2VEC ───────────────────────
    print("\n[3/5] Loading Word2Vec models...")
    w2v_skipgram = load_w2v(W2V_SKIPGRAM_PATH)
    w2v_cbow     = load_w2v(W2V_CBOW_PATH)

    # ── 4. VECTORISATION ───────────────────────
    print("\n[4/5] Vectorizing texts...")

    X_skipgram = vectorize_texts(cleaned_texts, w2v_skipgram)
    X_cbow     = vectorize_texts(cleaned_texts, w2v_cbow)

    y = np.array(labels)

    print(f"\nShape Skip-gram : {X_skipgram.shape}")
    print(f"Shape CBOW      : {X_cbow.shape}")

    # ── 5. CLASSIFICATION ─────────────────────
    print("\n[5/5] Training models...")

    run_all_models(X_skipgram, X_cbow, labels)

    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
