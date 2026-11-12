"""Prétraitement NLP : nettoyage, stopwords, lemmatisation, tokenisation, padding."""
import re

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src import config


def _ensure_nltk():
    for pkg, path in [("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk()

# On garde les négations : elles portent le sentiment ("not good" != "good")
_NEGATIONS = {"no", "nor", "not", "don", "didn", "doesn", "isn", "wasn", "aren",
              "weren", "won", "wouldn", "couldn", "shouldn", "hasn", "haven", "hadn"}
STOPWORDS = set(stopwords.words("english")) - _NEGATIONS
_lemmatizer = WordNetLemmatizer()

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """minuscules → suppression URL/mentions/ponctuation → stopwords → lemmatisation."""
    text = text.lower()
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = text.replace("#", " ")
    text = _NON_ALPHA.sub(" ", text)
    tokens = _SPACES.sub(" ", text).strip().split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if lemmatize:
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def clean_series(series):
    return series.astype(str).map(clean_text)


def fit_tokenizer(train_texts) -> Tokenizer:
    tok = Tokenizer(num_words=config.VOCAB_SIZE, oov_token="<OOV>")
    tok.fit_on_texts(train_texts)
    return tok


def to_padded(tok: Tokenizer, texts) -> np.ndarray:
    seqs = tok.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=config.MAX_LEN, padding="post", truncating="post")
