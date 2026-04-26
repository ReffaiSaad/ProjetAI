"""
data_loader.py
--------------
Chargement du dataset IMDB depuis un fichier CSV.
"""

import pandas as pd


def load_data(filepath: str):
    """
    Charge le dataset IMDB et retourne les textes et les labels.

    Args:
        filepath (str): Chemin vers le fichier CSV (ex: 'data/IMDB Dataset.csv')

    Returns:
        texts  (list[str]): Liste des reviews brutes
        labels (list[str]): Liste des sentiments ('positive' / 'negative')
    """
    df = pd.read_csv(filepath)

    print(df.head())                          # aperçu
    print(df['sentiment'].value_counts())     # distribution
    print(df['review'].str.len().describe())  # longueur des reviews

    texts  = df['review'].tolist()
    labels = df['sentiment'].tolist()

    return texts, labels
