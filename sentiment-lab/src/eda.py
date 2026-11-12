"""Analyse exploratoire (EDA) : classes, longueurs, mots fréquents."""
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config

sns.set_theme(style="whitegrid")


def run_eda(train: pd.DataFrame, clean_texts: pd.Series) -> None:
    df = train.copy()
    df["sentiment"] = df["label"].map(dict(enumerate(config.LABELS)))
    df["n_words"] = df["text"].str.split().str.len()
    df["n_chars"] = df["text"].str.len()

    # 1) Distribution des classes
    fig, ax = plt.subplots(figsize=(5, 4))
    order = config.LABELS
    sns.countplot(data=df, x="sentiment", order=order, ax=ax)
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    ax.set_title("Distribution des classes (train)")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "eda_class_distribution.png", dpi=150)
    plt.close(fig)

    # 2) Longueur des textes
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["n_words"], bins=40, ax=axes[0])
    axes[0].set_title("Nombre de mots par texte")
    sns.boxplot(data=df, x="sentiment", y="n_words", order=order, ax=axes[1])
    axes[1].set_title("Longueur par classe")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "eda_text_length.png", dpi=150)
    plt.close(fig)

    # 3) Mots les plus fréquents par classe (texte nettoyé)
    fig, axes = plt.subplots(1, config.NUM_CLASSES, figsize=(15, 5))
    for i, name in enumerate(order):
        words = Counter(" ".join(clean_texts[df["label"] == i]).split()).most_common(15)
        w, c = zip(*words)
        sns.barplot(x=list(c), y=list(w), ax=axes[i])
        axes[i].set_title(f"Top 15 mots — {name}")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "eda_top_words.png", dpi=150)
    plt.close(fig)

    stats = df.groupby("sentiment")[["n_words", "n_chars"]].describe().round(1)
    stats.to_csv(config.REPORT_DIR / "eda_stats.csv")
    print("EDA terminée → reports/figures/")
