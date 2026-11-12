"""Chargement des données : Hugging Face (tweet_eval) ou CSV local."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_data():
    """Retourne (train, val, test) : DataFrames avec colonnes `text` et `label`.

    - Si data/reviews.csv existe (colonnes text,label), il est utilisé et découpé 70/15/15.
    - Sinon, le dataset Hugging Face `tweet_eval/sentiment` est téléchargé.
    """
    if config.LOCAL_CSV.exists():
        df = pd.read_csv(config.LOCAL_CSV)[["text", "label"]].dropna()
        train, tmp = train_test_split(
            df, test_size=0.30, stratify=df["label"], random_state=config.SEED
        )
        val, test = train_test_split(
            tmp, test_size=0.50, stratify=tmp["label"], random_state=config.SEED
        )
    else:
        from datasets import load_dataset

        try:
            ds = load_dataset(*config.HF_DATASET)
        except Exception as e:  # nom du dépôt changé / réseau : on tente l'ancien nom
            print(f"Échec avec {config.HF_DATASET[0]} ({type(e).__name__}), essai de secours...")
            try:
                ds = load_dataset(*config.HF_DATASET_FALLBACK)
            except Exception:
                raise RuntimeError(
                    "Impossible de télécharger le dataset Hugging Face. Vérifie la connexion "
                    "Internet de Colab, ou utilise l'option « Mon fichier CSV » "
                    "(colonnes text,label)."
                ) from e
        train, val, test = (
            ds[s].to_pandas()[["text", "label"]] for s in ("train", "validation", "test")
        )
    return (d.reset_index(drop=True) for d in (train, val, test))
