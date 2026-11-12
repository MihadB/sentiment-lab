"""Évaluation comparative : métriques, matrices de confusion, courbes d'apprentissage."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from src import config

sns.set_theme(style="whitegrid")


def evaluate(name, y_true, y_pred) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                  zero_division=0)
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, target_names=config.LABELS, digits=3))
    return {"modèle": name, "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": p, "recall_macro": r, "f1_macro": f1}


def plot_confusion_matrices(preds: dict, y_true):
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    axes = [axes] if n == 1 else axes
    for ax, (name, y_pred) in zip(axes, preds.items()):
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=config.LABELS, yticklabels=config.LABELS)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_learning_curves(histories: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in histories.items():
        ep = list(range(1, len(h.history["loss"]) + 1))
        axes[0].plot(ep, h.history["loss"], label=f"{name} train")
        axes[0].plot(ep, h.history["val_loss"], "--", label=f"{name} val")
        axes[1].plot(ep, h.history["accuracy"], label=f"{name} train")
        axes[1].plot(ep, h.history["val_accuracy"], "--", label=f"{name} val")
    axes[0].set(title="Loss", xlabel="Époque")
    axes[1].set(title="Accuracy", xlabel="Époque")
    for a in axes:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        a.legend()
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "learning_curves.png", dpi=150)
    plt.close(fig)


def save_comparison(rows: list):
    df = pd.DataFrame(rows).set_index("modèle").round(4)
    df.to_csv(config.REPORT_DIR / "results.csv")
    df.to_markdown(config.REPORT_DIR / "results.md")
    print("\n", df)

    ax = df[["accuracy", "precision_macro", "recall_macro", "f1_macro"]].plot.bar(
        figsize=(10, 5), rot=15)
    ax.set_ylim(df.values.min() - 0.05, min(1.0, df.values.max() + 0.03))
    ax.set_title("Comparaison des modèles (jeu de test)")
    ax.set_xlabel("")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)
    plt.savefig(config.FIG_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
