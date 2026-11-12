"""Baseline classique : TF-IDF + Logistic Regression / SVM linéaire (la référence à battre)."""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src import config


def make_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAMS,
        sublinear_tf=True,
        min_df=2,
    )


def train_baselines(X_train, y_train):
    """Retourne un dict {nom: pipeline entraîné}."""
    models = {
        "TF-IDF + LogReg": Pipeline([
            ("tfidf", make_tfidf()),
            ("clf", LogisticRegression(max_iter=1000, C=2.0, random_state=config.SEED)),
        ]),
        "TF-IDF + LinearSVM": Pipeline([
            ("tfidf", make_tfidf()),
            ("clf", LinearSVC(C=0.5, random_state=config.SEED)),
        ]),
    }
    for name, pipe in models.items():
        print(f"Entraînement {name} ...")
        pipe.fit(X_train, y_train)
    return models


def save(pipe, name: str):
    joblib.dump(pipe, config.MODELS_DIR / f"{name}.joblib")
