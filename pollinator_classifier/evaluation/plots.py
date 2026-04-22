"""Visualization utilities: learning curves, FFT spectrum, boxplots."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq

from pollinator_classifier.config import SAMPLE_RATE, SPECIES_LABELS

logger = logging.getLogger(__name__)


def plot_learning_curve(
    history: Dict[str, List[float]],
    title: str = "Learning Curve",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot training vs. validation loss and accuracy per epoch.

    Parameters
    ----------
    history : dict
        Keras history dict. Expected keys: loss, val_loss, accuracy, val_accuracy.
    title : str
        Figure suptitle.
    output_path : Path, optional
        Save path (PNG, 150 dpi).

    Returns
    -------
    plt.Figure
    """
    epochs = range(1, len(history.get("loss", [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if "loss" in history:
        axes[0].plot(epochs, history["loss"], label="train", linewidth=1.5)
        if "val_loss" in history:
            axes[0].plot(epochs, history["val_loss"], label="val", linewidth=1.5, linestyle="--")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Categorical cross-entropy")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

    if "accuracy" in history:
        axes[1].plot(epochs, history["accuracy"], label="train", linewidth=1.5)
        if "val_accuracy" in history:
            axes[1].plot(epochs, history["val_accuracy"], label="val", linewidth=1.5, linestyle="--")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Learning curve saved → %s", output_path)

    return fig


def plot_fft_spectrum(
    signal: np.ndarray,
    title: str = "FFT Spectrum",
    output_path: Optional[Path] = None,
    sample_rate: int = SAMPLE_RATE,
) -> plt.Figure:
    """Plot FFT magnitude spectrum of a raw radar signal.

    Parameters
    ----------
    signal : np.ndarray, shape (N,) or (N, 2)
        Signal to transform. If 2-column, uses the amplitude channel (col 0).
    title : str
        Plot title.
    output_path : Path, optional
        Save path.
    sample_rate : int
        Sampling rate in Hz (default: 1000).

    Returns
    -------
    plt.Figure
    """
    amp = signal[:, 0] if signal.ndim == 2 else signal
    n = len(amp)
    freqs = rfftfreq(n, d=1.0 / sample_rate)
    magnitudes = np.abs(rfft(amp))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, magnitudes, linewidth=0.8, color="steelblue")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("FFT spectrum saved → %s", output_path)

    return fig


def plot_f1_boxplot(
    results: Dict[str, List[float]],
    title: str = "Macro-F1 per Model (LOSO)",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Boxplot of per-fold macro-F1 scores for each model.

    Parameters
    ----------
    results : dict
        Keys are model names; values are lists of per-fold F1 scores.
    title : str
        Plot title.
    output_path : Path, optional
        Save path.

    Returns
    -------
    plt.Figure
    """
    model_names = list(results.keys())
    data = [results[m] for m in model_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=model_names, patch_artist=True, notch=False)

    colors = ["#4C9BE8", "#F4A460", "#5CB85C", "#D9534F"]
    for patch, color in zip(bp["boxes"], colors[: len(model_names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.9, color="red", linestyle="--", linewidth=0.8, label="H1 threshold (0.90)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("F1 boxplot saved → %s", output_path)

    return fig


def plot_per_species_f1(
    per_species_f1: Dict[str, Dict[str, float]],
    title: str = "Per-species F1 by Model",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Grouped bar chart: per-species F1 across all models.

    Parameters
    ----------
    per_species_f1 : dict
        Outer keys: model names; inner keys: species names; values: F1 scores.
    title : str
    output_path : Path, optional

    Returns
    -------
    plt.Figure
    """
    model_names = list(per_species_f1.keys())
    n_models = len(model_names)
    n_species = len(SPECIES_LABELS)
    x = np.arange(n_species)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#4C9BE8", "#F4A460", "#5CB85C", "#D9534F"]

    for i, model in enumerate(model_names):
        f1_vals = [
            per_species_f1[model].get(sp, {}).get("f1", 0.0) for sp in SPECIES_LABELS
        ]
        ax.bar(x + i * width, f1_vals, width, label=model, color=colors[i % 4], alpha=0.8)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([s.replace("_", "\n") for s in SPECIES_LABELS], fontsize=9)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Per-species F1 chart saved → %s", output_path)

    return fig
