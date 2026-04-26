# IMDB Sentiment Analysis — NLP Project

Analyse de sentiment sur le dataset IMDB (50 000 reviews) — Partie 1 : Prétraitement & Vectorisation.

---

## Objectif du projet

Prédire si une review de film est **positive** ou **négative**.  
Cette partie couvre la préparation des données et la vectorisation avec Word2Vec.  
La classification (SVM, Régression Logistique) et l'API sont à compléter.

---

## État d'avancement

| Partie | Tâche | Statut |
|--------|-------|--------|
| Partie 1 | Chargement & exploration des données | ✅ Terminé |
| Partie 1 | Pipeline de prétraitement | ✅ Terminé |
| Partie 1 | Vectorisation Word2Vec Skip-gram | ✅ Terminé |
| Partie 1 | Vectorisation Word2Vec CBOW | ✅ Terminé |
| **Partie 2** | **Classification SVM + Régression Logistique** | ⏳ À faire |
| **Partie 3** | **API FastAPI + Frontend** | ⏳ À faire |

---

## Structure du projet

```
imdb_sentiment/
│
├── data/
│   ├── IMDB Dataset.csv          ← dataset original (voir ci-dessous)
│   └── cleaned_texts.pkl         ← reviews prétraitées et sauvegardées
│
├── models/
│   ├── w2v.model                 ← modèle Word2Vec Skip-gram (déjà entraîné)
│   └── w2v_cbow.model            ← modèle Word2Vec CBOW (déjà entraîné)
│
├── src/
│   ├── data_loader.py            ← chargement du CSV
│   ├── preprocessing.py          ← nettoyage + lemmatisation
│   ├── vectorization.py          ← vectorisation avec Word2Vec
│   └── train_models.py           ← ⏳ PARTIE 2 : à compléter
│
├── run_pipeline.py               ← script principal Partie 1
├── check_similarity.py           ← vérification qualité des modèles
├── requirements.txt
└── README.md
```

---

## Dataset

**IMDB Movie Reviews Dataset** — 50 000 reviews, équilibré 25 000 positif / 25 000 négatif.

Téléchargement : [Kaggle — IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

Placer le fichier dans `data/IMDB Dataset.csv`.

> Les fichiers `cleaned_texts.pkl`, `w2v.model` et `w2v_cbow.model` sont **déjà fournis** — inutile de relancer le prétraitement ni l'entraînement Word2Vec.

---

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Pipeline Partie 1

```bash
python run_pipeline.py
```

---

## Prétraitement

Le prétraitement est fait dans `src/preprocessing.py` via la fonction `preprocess_fast()`.

Chaque review passe par les étapes suivantes :

1. Mise en minuscules
2. Suppression des balises HTML (`<br />`, etc.)
3. Suppression des URLs
4. Suppression des chiffres
5. Suppression de la ponctuation
6. Lemmatisation avec **spaCy** (`en_core_web_sm`)
7. Suppression des stop words anglais

**Exemple :**

```
AVANT :
  "One of the other reviewers has mentioned that after watching just 1 Oz episode
   you'll be hooked. They are right, as this is exactly what happened with me..."

APRÈS (tokens) :
  ['reviewer', 'mention', 'watch', 'oz', 'episode', 'hook',
   'right', 'exactly', 'happen', 'strike', 'brutality', ...]
```

Le résultat est sauvegardé dans `data/cleaned_texts.pkl` pour éviter de relancer le prétraitement à chaque exécution.

---

## Vectorisation — Word2Vec

La vectorisation est faite dans `src/vectorization.py`.

Deux modèles Word2Vec ont été entraînés sur les 50 000 reviews prétraitées :

### Skip-gram (`models/w2v.model`)

Skip-gram prend **un mot central** et apprend à prédire les mots qui l'entourent.  
Il est plus lent à entraîner mais capture mieux les mots rares et les nuances de sens.

```python
Word2Vec(sentences=cleaned_texts, vector_size=100, window=5, min_count=2, sg=1)
```

### CBOW (`models/w2v_cbow.model`)

CBOW prend **les mots du contexte** et apprend à prédire le mot central manquant.  
Il est plus rapide à entraîner et fonctionne mieux sur les mots fréquents.

```python
Word2Vec(sentences=cleaned_texts, vector_size=100, window=5, min_count=2, sg=0)
```

### Paramètres communs

| Paramètre | Valeur | Signification |
|-----------|--------|---------------|
| `vector_size` | 100 | chaque mot → vecteur de 100 dimensions |
| `window` | 5 | 5 mots de contexte de chaque côté |
| `min_count` | 2 | ignorer les mots apparaissant moins de 2 fois |

### De la review au vecteur

Chaque review est convertie en un seul vecteur de 100 dimensions en faisant la **moyenne** des vecteurs de tous ses tokens :

```
"great film"  →  (vecteur("great") + vecteur("film")) / 2  →  [0.12, -0.34, ..., 0.08]
                                                                   100 valeurs
```

**Résultat final :**

```
X_skipgram : (50000, 100)   ← 50 000 reviews × 100 dimensions
X_cbow     : (50000, 100)   ← 50 000 reviews × 100 dimensions
```

---

## Vérification qualité des modèles

```bash
python check_similarity.py
```

Ce script calcule la similarité cosinus entre des paires de mots pour vérifier que les modèles ont bien appris le sens des mots.

**Résultats obtenus :**

| Paire | Skip-gram | CBOW |
|-------|-----------|------|
| great / excellent | 0.8503 | 0.7531 |
| terrible / awful | 0.8886 | 0.8511 |
| boring / dull | 0.8913 | 0.8583 |
| film / movie | 0.8908 | 0.8333 |
| great / terrible | 0.5608 | 0.3632 |
| love / hate | 0.6940 | 0.5031 |

Les mots de même polarité ont des similarités élevées (> 0.75), ce qui confirme que les deux modèles sont bien entraînés. Skip-gram donne de meilleurs résultats que CBOW sur toutes les paires.

---

## Ce qui reste à faire

### Partie 2 — Classification (`src/train_models.py`)

Utiliser les vecteurs produits pour entraîner et comparer :
- **SVM** avec les vecteurs Skip-gram
- **SVM** avec les vecteurs CBOW
- **Régression Logistique** avec les vecteurs Skip-gram
- **Régression Logistique** avec les vecteurs CBOW

Les vecteurs `X_skipgram`, `X_cbow` et les labels `y` sont retournés par `run_pipeline.py`.

### Partie 3 — API & Frontend

- Exposer le meilleur modèle via **FastAPI**
- Construire un frontend pour soumettre une review et recevoir la prédiction

---

## Dépendances

```
pandas
numpy
spacy
gensim
scikit-learn
```