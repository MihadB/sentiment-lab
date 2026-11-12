"""MLP sur vecteurs TF-IDF : Dense → Dropout → Dense → Dropout → Softmax."""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from src import config


class SparseBatches(keras.utils.Sequence):
    """Densifie la matrice creuse TF-IDF batch par batch (évite d'exploser la RAM)."""

    def __init__(self, X, y=None, batch_size=config.BATCH_SIZE, shuffle=False):
        super().__init__()
        self.X, self.y, self.bs, self.shuffle = X, y, batch_size, shuffle
        self.idx = np.arange(X.shape[0])
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.idx) / self.bs))

    def __getitem__(self, i):
        b = self.idx[i * self.bs:(i + 1) * self.bs]
        xb = self.X[b].toarray().astype("float32")
        return (xb, self.y[b]) if self.y is not None else xb

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)


def build_mlp(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(config.NUM_CLASSES, activation="softmax"),
    ], name="mlp_tfidf")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_mlp(Xtr, ytr, Xva, yva):
    model = build_mlp(Xtr.shape[1])
    cb = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True,
                                        monitor="val_loss")]
    history = model.fit(
        SparseBatches(Xtr, ytr, shuffle=True),
        validation_data=SparseBatches(Xva, yva),
        epochs=config.EPOCHS, callbacks=cb, verbose=2,
    )
    return model, history


def predict_proba(model, X):
    return model.predict(SparseBatches(X), verbose=0)
