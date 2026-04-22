"""Bidirectional LSTM classifier for micro-Doppler pollinator signals."""

import logging
from typing import Any, Dict, Optional

import numpy as np

from pollinator_classifier.config import (
    BATCH_SIZE,
    DROPOUT_RATE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    LSTM_UNITS_1,
    LSTM_UNITS_2,
    NUM_CLASSES,
    RANDOM_STATE,
)
from pollinator_classifier.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class BiLSTMModel(BaseModel):
    """Two-layer bidirectional LSTM.

    Architecture
    ------------
    Bidirectional(LSTM(64, return_sequences=True)) → Dropout(0.3)
    → Bidirectional(LSTM(128)) → Dropout(0.3) → Dense(5, softmax)
    """

    def __init__(self) -> None:
        import tensorflow as tf

        tf.random.set_seed(RANDOM_STATE)
        self._model: Optional[Any] = None
        self.history: Dict[str, Any] = {}

    def _build(self, input_shape: tuple) -> None:
        import tensorflow as tf

        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(LSTM_UNITS_1, return_sequences=True)
        )(inputs)
        x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_UNITS_2))(x)
        x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
        outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

        self._model = tf.keras.Model(inputs, outputs, name="bilstm")
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        logger.debug("BiLSTM model built: %s", input_shape)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weight: Optional[Dict[int, float]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        import tensorflow as tf

        self._build(X_train.shape[1:])
        y_train_cat = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
        y_val_cat = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=0,
            )
        ]

        hist = self._model.fit(
            X_train,
            y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=0,
        )
        self.history = hist.history
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self._model.predict(X, verbose=0)
        return np.argmax(probs, axis=1)

    def save(self, path: str) -> None:
        self._model.save(path)
        logger.info("BiLSTM model saved → %s", path)

    def load(self, path: str) -> None:
        import tensorflow as tf

        self._model = tf.keras.models.load_model(path)
        logger.info("BiLSTM model loaded ← %s", path)
