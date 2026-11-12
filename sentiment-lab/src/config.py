"""Configuration centrale du projet."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"

for d in (DATA_DIR, MODELS_DIR, FIG_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
LABELS = ["négatif", "neutre", "positif"]  # 0, 1, 2 (convention tweet_eval)
NUM_CLASSES = len(LABELS)

# Dataset Hugging Face (3 classes). Remplaçable par un CSV local (colonnes: text,label)
HF_DATASET = ("cardiffnlp/tweet_eval", "sentiment")  # ancien nom court "tweet_eval" refusé par huggingface_hub récent
HF_DATASET_FALLBACK = ("tweet_eval", "sentiment")
LOCAL_CSV = DATA_DIR / "reviews.csv"

# Séquences / vocabulaire
MAX_LEN = 50
VOCAB_SIZE = 20_000
EMB_DIM = 128

# TF-IDF
TFIDF_MAX_FEATURES = 20_000
TFIDF_NGRAMS = (1, 2)

# Entraînement NN
EPOCHS = 15
BATCH_SIZE = 64
