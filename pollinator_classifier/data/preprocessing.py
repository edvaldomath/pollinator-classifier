"""Feature scaling — scaler must be fitted ONLY on the training fold."""

import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from pollinator_classifier.data.windowing import WindowRecord

logger = logging.getLogger(__name__)


def windows_to_arrays(
    windows: List[WindowRecord],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a list of WindowRecords into stacked numpy arrays.

    Parameters
    ----------
    windows : list[WindowRecord]
        Windowed segments (all from one split: train or test).

    Returns
    -------
    X : np.ndarray, shape (N, window_size, 2)
        Signal windows — channels [amplitude, phase].
    y : np.ndarray, shape (N,), dtype int32
        Integer class labels.
    subjects : np.ndarray, shape (N,), dtype int32
        Subject ID for each window (needed for LOSO verification).
    """
    from pollinator_classifier.config import LABEL_TO_IDX

    X = np.stack([w.window for w in windows], axis=0).astype(np.float32)
    y = np.array([LABEL_TO_IDX[w.label] for w in windows], dtype=np.int32)
    subjects = np.array([w.subject_id for w in windows], dtype=np.int32)
    return X, y, subjects


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a per-channel StandardScaler on training windows only.

    The scaler operates on the (N*T, C) view so that mean and variance
    are computed across all training time steps, not per-window.

    Parameters
    ----------
    X_train : np.ndarray, shape (N, T, C)
        Training windows. Must NOT include any test-fold data.

    Returns
    -------
    StandardScaler
        Fitted scaler. Apply to test data with :func:`apply_scaler`.
    """
    N, T, C = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, C))
    logger.debug(
        "Scaler fitted on %d training windows — mean=%s  std=%s",
        N,
        scaler.mean_.round(4),
        scaler.scale_.round(4),
    )
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Apply a pre-fitted scaler to an array of windows.

    Parameters
    ----------
    X : np.ndarray, shape (N, T, C)
        Windows to transform (train or test).
    scaler : StandardScaler
        Scaler already fitted on the training fold only.

    Returns
    -------
    np.ndarray, shape (N, T, C)
        Scaled windows, same shape as input.
    """
    N, T, C = X.shape
    return scaler.transform(X.reshape(-1, C)).reshape(N, T, C).astype(np.float32)


def save_scaler(scaler: StandardScaler, path: str | Path) -> None:
    """Persist a fitted scaler to disk."""
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    logger.debug("Scaler saved to %s", path)


def load_scaler(path: str | Path) -> StandardScaler:
    """Load a scaler previously saved with :func:`save_scaler`."""
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    logger.debug("Scaler loaded from %s", path)
    return scaler
