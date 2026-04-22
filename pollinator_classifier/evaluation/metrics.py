"""Classification metrics — per-fold and aggregated."""

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from pollinator_classifier.config import IDX_TO_LABEL, SPECIES_LABELS


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute accuracy, macro-F1, precision, recall, and per-species breakdown.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        Ground truth integer class labels.
    y_pred : np.ndarray, shape (N,)
        Predicted integer class labels.

    Returns
    -------
    dict
        Keys: accuracy, macro_f1, macro_precision, macro_recall, per_species.
    """
    labels = list(range(len(SPECIES_LABELS)))

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)

    per_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_prec = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_rec = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    per_species = {
        IDX_TO_LABEL[i]: {
            "f1": float(per_f1[i]),
            "precision": float(per_prec[i]),
            "recall": float(per_rec[i]),
        }
        for i in labels
    }

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "per_species": per_species,
    }


def print_metrics(metrics: Dict[str, Any], title: str = "Results") -> None:
    """Pretty-print a metrics dict to stdout."""
    bar = "─" * 55
    print(f"\n┌{bar}┐")
    print(f"│  {title:<53}│")
    print(f"├{bar}┤")
    print(f"│  Accuracy        : {metrics['accuracy']:.4f}{' ' * 31}│")
    print(f"│  Macro F1        : {metrics['macro_f1']:.4f}{' ' * 31}│")
    print(f"│  Macro Precision : {metrics['macro_precision']:.4f}{' ' * 31}│")
    print(f"│  Macro Recall    : {metrics['macro_recall']:.4f}{' ' * 31}│")

    per = metrics.get("per_species", {})
    if per:
        print(f"├{bar}┤")
        print(f"│  {'Species':<25}  {'F1':>6}  {'P':>6}  {'R':>6}  │")
        print(f"│  {'─'*25}  {'─'*6}  {'─'*6}  {'─'*6}  │")
        for species, vals in per.items():
            print(
                f"│  {species:<25}  {vals['f1']:>6.4f}  "
                f"{vals['precision']:>6.4f}  {vals['recall']:>6.4f}  │"
            )

    print(f"└{bar}┘\n")
