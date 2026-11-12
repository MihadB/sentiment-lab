"""Interface Streamlit : saisir un texte → prédiction en direct (CNN vs baseline).

Lancer :  streamlit run app.py   (après avoir exécuté train_all.py)
"""
import joblib
import pandas as pd
import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from src import config
from src.preprocess import clean_text, to_padded

st.set_page_config(page_title="Analyse de sentiments", page_icon="💬")


@st.cache_resource
def load_models():
    cnn = keras.models.load_model(config.MODELS_DIR / "cnn.keras")
    tok = tokenizer_from_json((config.MODELS_DIR / "tokenizer.json").read_text(encoding="utf-8"))
    base = joblib.load(config.MODELS_DIR / "TF-IDF_LogReg.joblib")
    return cnn, tok, base


st.title("💬 Analyse de sentiments multi-modèles")
st.caption("Compare un CNN 1D (Kim) et une baseline TF-IDF + Régression logistique.")

try:
    cnn, tok, base = load_models()
except Exception:
    st.error("Modèles introuvables : lance d'abord `python train_all.py`.")
    st.stop()

text = st.text_area("Entre un texte (anglais) :", "I absolutely love this product, it works great!")
EMOJI = {"négatif": "😡", "neutre": "😐", "positif": "😊"}

if st.button("Analyser") and text.strip():
    clean = clean_text(text)
    p_cnn = cnn.predict(to_padded(tok, [clean]), verbose=0)[0]
    p_base = base.predict_proba([clean])[0]

    col1, col2 = st.columns(2)
    for col, name, p in [(col1, "CNN 1D (Kim)", p_cnn), (col2, "TF-IDF + LogReg", p_base)]:
        label = config.LABELS[int(p.argmax())]
        col.metric(name, f"{EMOJI[label]} {label}", f"{p.max():.1%} de confiance")
    st.bar_chart(pd.DataFrame({"CNN": p_cnn, "Baseline": p_base}, index=config.LABELS))
