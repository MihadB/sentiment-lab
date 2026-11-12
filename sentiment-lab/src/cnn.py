"""CNN 1D pour texte, inspiré de Kim (2014) « Convolutional Neural Networks for Sentence Classification ».

Embedding → [Conv1D(k=3,4,5) → GlobalMaxPooling] en parallèle → Concat → Dropout → Dense → Softmax
"""
from tensorflow import keras
from tensorflow.keras import layers

from src import config


def build_kim_cnn(vocab_size=config.VOCAB_SIZE, emb_dim=config.EMB_DIM,
                  max_len=config.MAX_LEN, kernel_sizes=(3, 4, 5), n_filters=100,
                  embedding_matrix=None) -> keras.Model:
    inp = layers.Input(shape=(max_len,), name="tokens")
    if embedding_matrix is not None:  # ex. GloVe / Word2Vec pré-entraînés
        emb = layers.Embedding(vocab_size, embedding_matrix.shape[1],
                               weights=[embedding_matrix], trainable=True)(inp)
    else:
        emb = layers.Embedding(vocab_size, emb_dim)(inp)
    emb = layers.SpatialDropout1D(0.2)(emb)

    pooled = []
    for k in kernel_sizes:  # chaque noyau capte des n-grammes de taille k
        c = layers.Conv1D(n_filters, k, activation="relu")(emb)
        pooled.append(layers.GlobalMaxPooling1D()(c))
    x = layers.Concatenate()(pooled) if len(pooled) > 1 else pooled[0]
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(config.NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inp, out, name="kim_cnn")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_cnn(Xtr, ytr, Xva, yva):
    model = build_kim_cnn()
    cb = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True,
                                        monitor="val_loss")]
    history = model.fit(Xtr, ytr, validation_data=(Xva, yva),
                        epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                        callbacks=cb, verbose=2)
    return model, history
