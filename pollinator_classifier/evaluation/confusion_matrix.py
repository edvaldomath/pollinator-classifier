"""Confusion matrix heatmap generation."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from pollinator_classifier.config import SPECIES_LABELS

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    output_path: Optional[Path] = None,
    normalize: bool = True,
) -> plt.Figure:
    """Generate a confusion matrix heatmap.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth integer labels.
    y_pred : np.ndarray
        Predicted integer labels.
    title : str
        Plot title.
    output_path : Path, optional
        If provided, the figure is saved at this path (PNG, 150 dpi).
    normalize : bool
        Normalize rows to recall values (0–1) if True; use raw counts otherwise.

    Returns
    -------
    plt.Figure
    """
    class_indices = list(range(len(SPECIES_LABELS)))
    cm = confusion_matrix(y_true, y_pred, labels=class_indices)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums == 0, 0.0, cm.astype(float) / row_sums)
        fmt = ".2f"
        vmin, vmax = 0.0, 1.0
    else:
        cm_plot = cm
        fmt = "d"
        vmin, vmax = None, None

    short_labels = [s.replace("_", "\n") for s in SPECIES_LABELS]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        vmin=vmin,
        vmax=vmax,
        xticklabels=short_labels,
        yticklabels=short_labels,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title, fontsize=13, pad=14)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved → %s", output_path)

    return fig
