"""Pipeline complet : EDA → prétraitement → baselines → MLP → CNN → comparaison.

Usage :  python train_all.py [--sample 5000]   (--sample pour un test rapide)
"""
import argparse
import json

import joblib
import numpy as np
import tensorflow as tf

from src import config
from src.baseline import make_tfidf, save, train_baselines
from src.cnn import train_cnn
from src.data import load_data
from src.eda import run_eda
from src.evaluate import (evaluate, plot_confusion_matrices, plot_learning_curves,
                          save_comparison)
from src.mlp import predict_proba, train_mlp
from src.preprocess import clean_series, fit_tokenizer, to_padded


def main(sample=None, epochs=None):
    if epochs:
        config.EPOCHS = epochs
    np.random.seed(config.SEED)
    tf.random.set_seed(config.SEED)

    train, val, test = load_data()
    if sample:
        train, val, test = (d.sample(min(sample, len(d)), random_state=config.SEED)
                            .reset_index(drop=True) for d in (train, val, test))
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    # 1) Prétraitement
    for d in (train, val, test):
        d["clean"] = clean_series(d["text"])

    # 2) EDA
    run_eda(train, train["clean"])

    y_tr, y_va, y_te = (d["label"].values for d in (train, val, test))
    results, preds, histories = [], {}, {}

    # 3) Baselines (TF-IDF + LogReg / SVM)
    baselines = train_baselines(train["clean"], y_tr)
    for name, pipe in baselines.items():
        y_pred = pipe.predict(test["clean"])
        results.append(evaluate(name, y_te, y_pred))
        preds[name] = y_pred
        save(pipe, name.replace(" ", "_").replace("+", "").replace("__", "_"))

    # 4) MLP sur TF-IDF
    tfidf = make_tfidf().fit(train["clean"])
    Xtr, Xva, Xte = (tfidf.transform(d["clean"]) for d in (train, val, test))
    mlp, h_mlp = train_mlp(Xtr, y_tr, Xva, y_va)
    y_pred = predict_proba(mlp, Xte).argmax(1)
    results.append(evaluate("MLP (TF-IDF)", y_te, y_pred))
    preds["MLP (TF-IDF)"], histories["MLP"] = y_pred, h_mlp
    mlp.save(config.MODELS_DIR / "mlp.keras")
    joblib.dump(tfidf, config.MODELS_DIR / "tfidf_mlp.joblib")

    # 5) CNN (Kim)
    tok = fit_tokenizer(train["clean"])
    Ptr, Pva, Pte = (to_padded(tok, d["clean"]) for d in (train, val, test))
    cnn, h_cnn = train_cnn(Ptr, y_tr, Pva, y_va)
    y_pred = cnn.predict(Pte, verbose=0).argmax(1)
    results.append(evaluate("CNN 1D (Kim)", y_te, y_pred))
    preds["CNN 1D (Kim)"], histories["CNN"] = y_pred, h_cnn
    cnn.save(config.MODELS_DIR / "cnn.keras")
    (config.MODELS_DIR / "tokenizer.json").write_text(tok.to_json(), encoding="utf-8")

    # 6) Comparaison
    save_comparison(results)
    plot_confusion_matrices(preds, y_te)
    plot_learning_curves(histories)
    (config.REPORT_DIR / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("\nTerminé. Résultats dans reports/ — modèles dans models/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None,
                    help="Sous-échantillonne chaque split (test rapide)")
    ap.add_argument("--epochs", type=int, default=None, help="Nombre max d'époques")
    args = ap.parse_args()
    main(args.sample, args.epochs)
