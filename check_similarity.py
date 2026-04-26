"""
check_similarity.py
-------------------
Vérifie la qualité des modèles Word2Vec en calculant
la similarité cosinus entre des mots liés au sentiment IMDB.

Usage :
    python check_similarity.py
"""

from gensim.models import Word2Vec

W2V_SKIPGRAM_PATH = "models/w2v.model"
W2V_CBOW_PATH     = "models/w2v_cbow.model"

# Paires de mots à tester
# Format : (mot1, mot2, relation_attendue)
PAIRES = [
    # Mots positifs — doivent être très proches
    ("great",     "excellent",  "positif ↔ positif"),
    ("amazing",   "wonderful",  "positif ↔ positif"),
    ("love",      "enjoy",      "positif ↔ positif"),

    # Mots négatifs — doivent être très proches
    ("terrible",  "awful",      "négatif ↔ négatif"),
    ("bad",       "horrible",   "négatif ↔ négatif"),
    ("boring",    "dull",       "négatif ↔ négatif"),

    # Positif ↔ Négatif — doivent être éloignés
    ("great",     "terrible",   "positif ↔ négatif"),
    ("love",      "hate",       "positif ↔ négatif"),
    ("amazing",   "awful",      "positif ↔ négatif"),

    # Mots liés au cinéma
    ("actor",     "actress",    "cinéma ↔ cinéma"),
    ("film",      "movie",      "cinéma ↔ cinéma"),
    ("director",  "producer",   "cinéma ↔ cinéma"),
]


def check_model(model, model_name: str):
    print(f"\n{'='*58}")
    print(f"  {model_name}")
    print(f"{'='*58}")
    print(f"  {'Mot 1':<12} {'Mot 2':<12} {'Similarité':>10}   Relation")
    print(f"  {'-'*54}")

    for mot1, mot2, relation in PAIRES:
        if mot1 in model.wv and mot2 in model.wv:
            score = model.wv.similarity(mot1, mot2)
            barre = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"  {mot1:<12} {mot2:<12} {score:>8.4f}   {barre}  {relation}")
        else:
            manquant = mot1 if mot1 not in model.wv else mot2
            print(f"  {mot1:<12} {mot2:<12} {'N/A':>8}   ⚠ '{manquant}' absent du vocabulaire")

    print()
    print("  Top 5 mots les plus proches de 'good' :")
    if "good" in model.wv:
        for mot, score in model.wv.most_similar("good", topn=5):
            print(f"    {mot:<15} {score:.4f}")

    print()
    print("  Top 5 mots les plus proches de 'bad' :")
    if "bad" in model.wv:
        for mot, score in model.wv.most_similar("bad", topn=5):
            print(f"    {mot:<15} {score:.4f}")


def main():
    print("\nChargement des modèles...")
    w2v_skipgram = Word2Vec.load(W2V_SKIPGRAM_PATH)
    w2v_cbow     = Word2Vec.load(W2V_CBOW_PATH)
    print("Modèles chargés.")

    check_model(w2v_skipgram, "Skip-gram")
    check_model(w2v_cbow,     "CBOW")

    # Comparaison rapide
    print(f"\n{'='*58}")
    print("  COMPARAISON RAPIDE Skip-gram vs CBOW")
    print(f"{'='*58}")
    print(f"  {'Paire':<26} {'Skip-gram':>10} {'CBOW':>10}")
    print(f"  {'-'*50}")
    for mot1, mot2, _ in PAIRES:
        if mot1 in w2v_skipgram.wv and mot2 in w2v_skipgram.wv:
            sg = w2v_skipgram.wv.similarity(mot1, mot2)
            cb = w2v_cbow.wv.similarity(mot1, mot2)
            paire = f"{mot1} / {mot2}"
            print(f"  {paire:<26} {sg:>10.4f} {cb:>10.4f}")
    print()


if __name__ == "__main__":
    main()
