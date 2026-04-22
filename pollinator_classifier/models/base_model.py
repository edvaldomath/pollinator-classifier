"""Abstract base class — uniform fit/predict/evaluate interface for all models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseModel(ABC):
    """Uniform interface required by every classifier in this package.

    All models accept X of shape (N, T, C):
        N = number of windows
        T = time steps (window_size)
        C = channels (2: amplitude, phase)
    """

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weight: Optional[Dict[int, float]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Train the model.

        Parameters
        ----------
        X_train : np.ndarray, shape (N, T, C)
        y_train : np.ndarray, shape (N,)  — integer class labels
        X_val   : np.ndarray, shape (M, T, C)  — validation / LOSO test fold
        y_val   : np.ndarray, shape (M,)
        class_weight : dict, optional
            Per-class weights for imbalanced datasets.

        Returns
        -------
        dict
            Training history (keys: loss, val_loss, accuracy, val_accuracy, …).
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices.

        Parameters
        ----------
        X : np.ndarray, shape (N, T, C)

        Returns
        -------
        np.ndarray, shape (N,), dtype int
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the model to *path*."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a previously saved model from *path*."""
