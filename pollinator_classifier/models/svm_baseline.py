"""SVM baseline with handcrafted FFT features."""

import logging
import pickle
from typing import Any, Dict, Optional

import numpy as np
from scipy.fft import rfft
from sklearn.svm import SVC

from pollinator_classifier.config import (
    RANDOM_STATE,
    SVM_KERNEL,
    SVM_N_FFT_PEAKS,
)
from pollinator_classifier.models.base_model import BaseModel

logger = logging.getLogger(__name__)


def extract_features(X: np.ndarray) -> np.ndarray:
    """Extract handcrafted FFT features from windowed radar signals.

    Per-window feature vector (24 values total):
        - Top-K FFT magnitudes of the amplitude channel  (K values)
        - Top-K FFT magnitudes of the phase channel      (K values)
        - RMS and std of amplitude                       (2 values)
        - RMS and std of phase                           (2 values)

    Parameters
    ----------
    X : np.ndarray, shape (N, T, 2)
        Windowed signals, channels [amplitude, phase].

    Returns
    -------
    np.ndarray, shape (N, 2*K + 4)
        Feature matrix; K = SVM_N_FFT_PEAKS (default 10).
    """
    amp = X[:, :, 0]    # (N, T)
    phase = X[:, :, 1]  # (N, T)

    fft_amp = np.abs(rfft(amp, axis=1))    # (N, T//2 + 1)
    fft_phase = np.abs(rfft(phase, axis=1))

    # Top-K peak magnitudes, sorted descending
    top_amp = np.sort(fft_amp, axis=1)[:, -SVM_N_FFT_PEAKS:][:, ::-1]
    top_phase = np.sort(fft_phase, axis=1)[:, -SVM_N_FFT_PEAKS:][:, ::-1]

    rms_amp = np.sqrt(np.mean(amp ** 2, axis=1, keepdims=True))
    std_amp = np.std(amp, axis=1, keepdims=True)
    rms_phase = np.sqrt(np.mean(phase ** 2, axis=1, keepdims=True))
    std_phase = np.std(phase, axis=1, keepdims=True)

    return np.hstack([top_amp, top_phase, rms_amp, std_amp, rms_phase, std_phase])


class SVMModel(BaseModel):
    """SVM classifier with RBF kernel trained on handcrafted FFT features.

    Feature dimensionality: 2 * SVM_N_FFT_PEAKS + 4  (default: 24).
    """

    def __init__(self) -> None:
        self._clf: Optional[SVC] = None
        self.history: Dict[str, Any] = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weight: Optional[Dict[int, float]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self._clf = SVC(
            kernel=SVM_KERNEL,
            probability=True,
            random_state=RANDOM_STATE,
            class_weight=class_weight,
        )

        F_train = extract_features(X_train)
        self._clf.fit(F_train, y_train)

        F_val = extract_features(X_val)
        val_acc = float(self._clf.score(F_val, y_val))
        self.history = {"val_accuracy": [val_acc]}
        logger.info("SVM validation accuracy: %.4f", val_acc)
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        F = extract_features(X)
        return self._clf.predict(F)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._clf, f)
        logger.info("SVM model saved → %s", path)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self._clf = pickle.load(f)
        logger.info("SVM model loaded ← %s", path)
