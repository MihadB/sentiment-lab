"""Interface Gradio « Sentiment Lab » — pensée pour Google Colab (fonctionne aussi en local).

Onglets : 1) Données & EDA  2) Entraînement en direct  3) Résultats  4) Démo live  5) À propos
Lancer :  python ui.py        ou, dans Colab :  from ui import build; build().queue().launch(share=True)
"""
import matplotlib

matplotlib.use("Agg")

import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402

SRC_HF = "🤗 Dataset tweet_eval (anglais, 3 classes)"
SRC_CSV = "📄 Mon fichier CSV"
EMOJI = {"négatif": "😡", "neutre": "😐", "positif": "😊"}
COLOR = {"négatif": "#ef4444", "neutre": "#64748b", "positif": "#22c55e"}

# --------------------------------------------------------------------------- style
THEME = gr.themes.Soft(
    primary_hue="indigo", secondary_hue="violet", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
    radius_size="lg",
)

CSS = """
.gradio-container { max-width: 1180px !important; margin: auto; }
.hero { background: linear-gradient(120deg,#4f46e5 0%,#7c3aed 55%,#db2777 100%);
        border-radius: 20px; padding: 30px 34px; color: #fff; margin-bottom: 14px;
        box-shadow: 0 10px 30px rgba(79,70,229,.35); }
.hero h1 { margin: 0 0 6px 0; font-size: 2.1rem; font-weight: 800; color:#fff; }
.hero p  { margin: 0 0 14px 0; opacity: .92; font-size: 1.02rem; color:#fff; }
.pill { display:inline-block; background: rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.35);
        padding: 4px 12px; border-radius: 999px; margin: 0 6px 6px 0; font-size:.82rem; color:#fff; }
.step-title { font-size: 1.05rem; font-weight: 700; margin: 4px 0 2px 0; }
.stepper { display:flex; gap:8px; flex-wrap:wrap; margin: 6px 0 2px 0; }
.chip { padding:6px 14px; border-radius:999px; font-size:.85rem; font-weight:600;
        background: var(--neutral-200); color: var(--neutral-600); }
.chip.done { background:#dcfce7; color:#166534; }
.chip.now  { background:#e0e7ff; color:#3730a3; animation: pulse 1.2s infinite; }
@keyframes pulse { 50% { opacity:.55; } }
.banner { border-radius:14px; padding:14px 18px; font-size:1.02rem; border:1px solid var(--border-color-primary); }
.banner.ok   { background:#ecfdf5; border-color:#86efac; color:#14532d; }
.banner.warn { background:#fffbeb; border-color:#fcd34d; color:#78350f; }
.banner.err  { background:#fef2f2; border-color:#fca5a5; color:#7f1d1d; }
.verdict { border-radius:18px; padding:20px 24px; color:#fff; text-align:center; }
.verdict .big { font-size:2.2rem; font-weight:800; }
.verdict .sub { opacity:.92; margin-top:4px; }
.banner.ok,   .banner.ok *   { color:#14532d !important; }
.banner.warn, .banner.warn * { color:#78350f !important; }
.banner.err,  .banner.err *  { color:#7f1d1d !important; }
.verdict, .verdict * { color: #fff !important; }
.hero, .hero * { color: #fff !important; }
.tabs > .tab-nav button { font-weight:600; }
button.primary { box-shadow: 0 4px 14px rgba(79,70,229,.35); }
footer { visibility: hidden; }
"""


# --------------------------------------------------------------------------- helpers
def _banner(text, kind="ok"):
    return f'<div class="banner {kind}">{text}</div>'


def _img(name):
    p = config.FIG_DIR / f"{name}.png"
    return str(p) if p.exists() else None


def _prepare_source(source, csv_path):
    """Applique le choix de source : copie le CSV utilisateur ou supprime l'ancien."""
    if source == SRC_CSV:
        if not csv_path:
            raise gr.Error("Importe d'abord un fichier CSV (colonnes `text` et `label`).")
        df = pd.read_csv(csv_path)
        if not {"text", "label"} <= set(df.columns):
            raise gr.Error("Le CSV doit contenir les colonnes `text` et `label`.")
        if not set(df["label"].unique()) <= {0, 1, 2}:
            raise gr.Error("`label` doit valoir 0 (négatif), 1 (neutre) ou 2 (positif).")
        df.to_csv(config.LOCAL_CSV, index=False)
    elif config.LOCAL_CSV.exists():
        config.LOCAL_CSV.unlink()


# --------------------------------------------------------------------------- 1) EDA
def explore(source, csv_path, sample):
    _prepare_source(source, csv_path)
    from src.data import load_data
    from src.eda import run_eda
    from src.preprocess import clean_series

    train, _, _ = load_data()
    if sample:
        train = train.sample(min(int(sample), len(train)), random_state=config.SEED)
    train = train.reset_index(drop=True)
    train["clean"] = clean_series(train["text"])
    run_eda(train, train["clean"])

    df = train.copy()
    df["classe"] = df["label"].map(dict(enumerate(config.LABELS)))
    df["n_mots"] = df["text"].str.split().str.len()
    summary = df.groupby("classe").agg(textes=("text", "size"),
                                       mots_moyens=("n_mots", "mean")).round(1).reset_index()
    summary["part_%"] = (summary["textes"] / summary["textes"].sum() * 100).round(1)
    examples = df[["classe", "text"]].sample(min(8, len(df)), random_state=1)
    top = Counter(" ".join(train["clean"]).split()).most_common(1)
    msg = _banner(f"✅ <b>{len(train):,}</b> textes analysés — mot le plus fréquent : "
                  f"<b>{top[0][0] if top else '—'}</b>. Les graphiques sont ci-dessous.")
    return (_img("eda_class_distribution"), _img("eda_text_length"), _img("eda_top_words"),
            summary, examples, msg)


# --------------------------------------------------------------------------- 2) Training
STEPS = ["Données", "EDA", "Baselines", "MLP", "CNN", "Rapport"]
MARKERS = [("train=", 1), ("EDA terminée", 2), ("=== TF-IDF + LinearSVM", 3),
           ("=== MLP", 4), ("=== CNN", 5), ("Terminé", 6)]
_CACHE = {}


def _stepper(done, failed=False):
    chips = []
    for i, name in enumerate(STEPS):
        cls = "done" if i < done else ("now" if i == done and not failed else "")
        chips.append(f'<span class="chip {cls}">{"✓ " if i < done else ""}{name}</span>')
    return f'<div class="stepper">{"".join(chips)}</div>'


def train(source, csv_path, sample, epochs):
    _prepare_source(source, csv_path)
    cmd = [sys.executable, "-u", "train_all.py", "--epochs", str(int(epochs))]
    if sample:
        cmd += ["--sample", str(int(sample))]
    env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, cwd=config.ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    log, done = [], 0
    yield "Démarrage…", _stepper(0)
    for line in proc.stdout:
        line = line.rstrip()
        if not line or line.startswith(("I0000", "W0000", "E0000", "WARNING:")):
            continue
        log.append(line)
        for marker, n in MARKERS:
            if marker in line:
                done = max(done, n)
        yield "\n".join(log[-250:]), _stepper(done)
    proc.wait()
    _CACHE.clear()
    if proc.returncode == 0:
        yield "\n".join(log[-250:]), _stepper(6) + _banner(
            "🎉 Entraînement terminé ! Va dans <b>Résultats</b> puis <b>Démo live</b>.")
    else:
        yield "\n".join(log[-250:]), _stepper(done, failed=True) + _banner(
            "❌ L'entraînement a échoué — regarde la fin du journal ci-dessus.", "err")


# --------------------------------------------------------------------------- 3) Results
def load_results():
    path = config.REPORT_DIR / "results.csv"
    if not path.exists():
        return (_banner("Aucun résultat pour l'instant — lance d'abord l'entraînement.", "warn"),
                None, None, None, None)
    df = pd.read_csv(path).round(4)
    best = df.loc[df["f1_macro"].idxmax()]
    base = df[df["modèle"].str.contains("LogReg")]["f1_macro"]
    gain = f" (<b>{(best['f1_macro'] - base.iloc[0]) * 100:+.1f} pts</b> de F1 vs baseline LogReg)" \
        if len(base) else ""
    banner = _banner(f"🏆 Meilleur modèle : <b>{best['modèle']}</b> — F1 macro "
                     f"<b>{best['f1_macro']:.3f}</b>, accuracy <b>{best['accuracy']:.3f}</b>{gain}")
    return (banner, df, _img("model_comparison"), _img("confusion_matrices"),
            _img("learning_curves"))


# --------------------------------------------------------------------------- 4) Live demo
def _models():
    if _CACHE:
        return _CACHE
    needed = ["cnn.keras", "tokenizer.json", "TF-IDF_LogReg.joblib", "mlp.keras",
              "tfidf_mlp.joblib"]
    if any(not (config.MODELS_DIR / f).exists() for f in needed):
        return None
    import joblib
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import tokenizer_from_json

    _CACHE.update(
        cnn=keras.models.load_model(config.MODELS_DIR / "cnn.keras"),
        mlp=keras.models.load_model(config.MODELS_DIR / "mlp.keras"),
        tok=tokenizer_from_json((config.MODELS_DIR / "tokenizer.json").read_text(encoding="utf-8")),
        tfidf=joblib.load(config.MODELS_DIR / "tfidf_mlp.joblib"),
        lr=joblib.load(config.MODELS_DIR / "TF-IDF_LogReg.joblib"),
    )
    return _CACHE


def predict(text):
    m = _models()
    if m is None:
        raise gr.Error("Aucun modèle entraîné : va d'abord dans l'onglet « Entraînement ».")
    if not text or not text.strip():
        raise gr.Error("Saisis un texte à analyser.")
    from src.preprocess import clean_text, to_padded

    clean = clean_text(text)
    probs = {
        "CNN 1D (Kim)": m["cnn"].predict(to_padded(m["tok"], [clean]), verbose=0)[0],
        "MLP (TF-IDF)": m["mlp"].predict(
            m["tfidf"].transform([clean]).toarray().astype("float32"), verbose=0)[0],
        "TF-IDF + LogReg": m["lr"].predict_proba([clean])[0],
    }
    votes = [config.LABELS[int(p.argmax())] for p in probs.values()]
    top, n = Counter(votes).most_common(1)[0]
    mean_conf = sum(p[config.LABELS.index(top)] for p in probs.values()) / len(probs)
    sub = ("Les 3 modèles sont d'accord" if n == 3
           else f"{n} modèles sur 3 — il y a un désaccord, le texte est ambigu")
    verdict = (f'<div class="verdict" style="background:{COLOR[top]}">'
               f'<div class="big">{EMOJI[top]} {top.capitalize()}</div>'
               f'<div class="sub">{sub} · confiance moyenne {mean_conf:.0%}</div></div>')
    out = [{l: float(x) for l, x in zip(config.LABELS, p)} for p in probs.values()]
    return (verdict, *out)


EXAMPLES = [
    ["I absolutely love this phone, the battery life is amazing!"],
    ["Worst customer service ever. I will never order here again."],
    ["The meeting is scheduled for Monday at 10am in room 4."],
    ["Not bad at all, but I expected a bit more for the price."],
    ["Can't believe how disappointing this was. Total waste of money."],
]

ABOUT = """
### 🎯 Objectif
Classer automatiquement des textes (tweets, avis…) en **négatif / neutre / positif** et
**comparer trois approches** avec le même protocole d'évaluation.

| Modèle | Entrée | Idée |
|---|---|---|
| **TF-IDF + LogReg / SVM** | n-grammes pondérés | Baseline rigoureuse à battre |
| **MLP** | vecteurs TF-IDF | Réseau dense : Dense → Dropout → Softmax |
| **CNN 1D (Kim 2014)** | embeddings appris | Conv1D k=3,4,5 → GlobalMaxPooling → Dense |

### 🧪 Rigueur méthodologique
- Négations conservées dans les stopwords (« not good » ≠ « good »)
- Vocabulaire et TF-IDF ajustés **uniquement sur le train** (pas de fuite de données)
- Early stopping sur la loss de validation, évaluation finale sur le **test**
- Métriques **macro** (précision, rappel, F1) + matrices de confusion normalisées

### 🛠️ Stack
Python · TensorFlow/Keras · scikit-learn · NLTK · pandas · matplotlib/seaborn · Gradio
"""


# --------------------------------------------------------------------------- app
def build() -> gr.Blocks:
    with gr.Blocks(theme=THEME, css=CSS, title="Sentiment Lab") as demo:
        gr.HTML(
            '<div class="hero"><h1>💬 Sentiment Lab</h1>'
            "<p>Analyse de sentiments multi-modèles : de la baseline TF-IDF au CNN 1D, "
            "comparés sur les mêmes données.</p>"
            '<span class="pill">TF-IDF + LogReg</span><span class="pill">MLP</span>'
            '<span class="pill">CNN 1D · Kim</span><span class="pill">NLP · Keras</span></div>'
        )

        with gr.Accordion("⚙️ Paramètres (communs aux onglets Données et Entraînement)", open=True):
            with gr.Row():
                source = gr.Radio([SRC_HF, SRC_CSV], value=SRC_HF, label="Source des données")
                csv_file = gr.File(label="CSV (colonnes `text`, `label` : 0/1/2)",
                                   file_types=[".csv"], type="filepath")
            with gr.Row():
                sample = gr.Slider(0, 50000, value=5000, step=500,
                                   label="Nombre de textes par split (0 = tout le dataset)")
                epochs = gr.Slider(2, 30, value=10, step=1, label="Époques max (early stopping actif)")

        with gr.Tabs():
            # ---- 1) Données & EDA
            with gr.Tab("📊 1 · Données & EDA"):
                eda_btn = gr.Button("Analyser les données", variant="primary")
                eda_msg = gr.HTML()
                with gr.Row():
                    eda_img1 = gr.Image(label="Distribution des classes", interactive=False,
                                        type="filepath")
                    eda_img2 = gr.Image(label="Longueur des textes", interactive=False,
                                        type="filepath")
                eda_img3 = gr.Image(label="Mots les plus fréquents par classe",
                                    interactive=False, type="filepath")
                with gr.Row():
                    eda_summary = gr.Dataframe(label="Résumé par classe", interactive=False)
                    eda_examples = gr.Dataframe(label="Exemples de textes", interactive=False,
                                                wrap=True)
                eda_btn.click(explore, [source, csv_file, sample],
                              [eda_img1, eda_img2, eda_img3, eda_summary, eda_examples, eda_msg])

            # ---- 2) Entraînement
            with gr.Tab("🏋️ 2 · Entraînement"):
                gr.Markdown("Entraîne les **4 modèles** (LogReg, SVM, MLP, CNN) d'un seul clic. "
                            "Astuce : commence avec 5 000 textes pour un test rapide.")
                train_btn = gr.Button("🚀 Lancer l'entraînement", variant="primary")
                stepper = gr.HTML(_stepper(0))
                logs = gr.Textbox(label="Journal d'entraînement", lines=16, max_lines=16,
                                  interactive=False, autoscroll=True)
                train_btn.click(train, [source, csv_file, sample, epochs], [logs, stepper])

            # ---- 3) Résultats
            with gr.Tab("📈 3 · Résultats"):
                res_btn = gr.Button("Afficher / rafraîchir les résultats", variant="primary")
                res_banner = gr.HTML()
                res_table = gr.Dataframe(label="Métriques sur le jeu de test", interactive=False)
                res_bar = gr.Image(label="Comparaison des modèles", interactive=False,
                                   type="filepath")
                with gr.Row():
                    res_cm = gr.Image(label="Matrices de confusion", interactive=False,
                                      type="filepath")
                    res_lc = gr.Image(label="Courbes d'apprentissage", interactive=False,
                                      type="filepath")
                res_btn.click(load_results, None, [res_banner, res_table, res_bar, res_cm, res_lc])

            # ---- 4) Démo live
            with gr.Tab("🔮 4 · Démo live"):
                gr.Markdown("Tape un texte **en anglais** (modèles entraînés sur tweet_eval) "
                            "et compare la réponse des 3 modèles.")
                txt = gr.Textbox(label="Ton texte", lines=3,
                                 placeholder="This product is fantastic, I love it!")
                go = gr.Button("Analyser le sentiment", variant="primary")
                verdict = gr.HTML()
                with gr.Row():
                    l_cnn = gr.Label(label="CNN 1D (Kim)", num_top_classes=3)
                    l_mlp = gr.Label(label="MLP (TF-IDF)", num_top_classes=3)
                    l_lr = gr.Label(label="TF-IDF + LogReg", num_top_classes=3)
                gr.Examples(EXAMPLES, inputs=txt, label="Exemples")
                outs = [verdict, l_cnn, l_mlp, l_lr]
                go.click(predict, txt, outs)
                txt.submit(predict, txt, outs)

            # ---- 5) À propos
            with gr.Tab("ℹ️ À propos"):
                gr.Markdown(ABOUT)
    return demo


if __name__ == "__main__":
    build().queue().launch()
