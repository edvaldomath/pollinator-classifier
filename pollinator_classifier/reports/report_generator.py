"""Export evaluation results to CSV and JSON."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def save_comparison_table(
    model_results: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> pd.DataFrame:
    """Build and save a multi-model comparison table.

    Parameters
    ----------
    model_results : dict
        Keys: model names.  Values: aggregated result dicts as returned
        by :func:`~pollinator_classifier.training.trainer.train_loso`.
    output_dir : Path
        Directory for the output files (created if absent).

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by macro_f1_mean (descending).
    """
    rows = []
    for model_name, res in model_results.items():
        rows.append(
            {
                "model": model_name,
                "accuracy_mean": res.get("accuracy_mean"),
                "accuracy_std": res.get("accuracy_std"),
                "macro_f1_mean": res.get("macro_f1_mean"),
                "macro_f1_std": res.get("macro_f1_std"),
                "macro_precision_mean": res.get("macro_precision_mean"),
                "macro_precision_std": res.get("macro_precision_std"),
                "macro_recall_mean": res.get("macro_recall_mean"),
                "macro_recall_std": res.get("macro_recall_std"),
                "n_folds": res.get("n_folds"),
            }
        )

    df = (
        pd.DataFrame(rows)
        .sort_values("macro_f1_mean", ascending=False)
        .reset_index(drop=True)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "model_comparison.csv"
    json_path = output_dir / "model_comparison.json"

    df.to_csv(csv_path, index=False, float_format="%.4f")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=float)

    logger.info("Comparison table → %s and %s", csv_path, json_path)

    print("\n" + "=" * 70)
    print("  MODEL COMPARISON (sorted by macro-F1)")
    print("=" * 70)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 70 + "\n")

    return df


def load_fold_predictions(
    results_dir: Path,
    model_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load concatenated y_true / y_pred across all saved folds for a model.

    Parameters
    ----------
    results_dir : Path
        Directory containing ``{model_name}_fold_*_y_true.npy`` files.
    model_name : str
        Model identifier (lstm, bilstm, cnn1d, svm).

    Returns
    -------
    y_true_all : np.ndarray
    y_pred_all : np.ndarray

    Raises
    ------
    FileNotFoundError
        If no prediction files are found for the given model.
    """
    y_true_parts: List[np.ndarray] = []
    y_pred_parts: List[np.ndarray] = []

    for true_path in sorted(results_dir.glob(f"{model_name}_fold_*_y_true.npy")):
        tag = true_path.stem.split("_y_true")[0]   # e.g. "lstm_fold_0"
        pred_path = results_dir / f"{tag}_y_pred.npy"
        if pred_path.exists():
            y_true_parts.append(np.load(true_path))
            y_pred_parts.append(np.load(pred_path))
        else:
            logger.warning("Missing predictions for %s — skipping", tag)

    if not y_true_parts:
        raise FileNotFoundError(
            f"No prediction files found for model '{model_name}' in {results_dir}"
        )

    return np.concatenate(y_true_parts), np.concatenate(y_pred_parts)


def export_per_fold_csv(
    fold_results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Export per-fold metric rows to a flat CSV.

    Parameters
    ----------
    fold_results : list[dict]
        List of fold metric dicts (as stored in the result JSON 'folds' key).
    output_path : Path
        Destination CSV file.
    """
    rows = []
    for fold in fold_results:
        row = {
            "fold_id": fold.get("fold_id"),
            "test_subject_id": fold.get("test_subject_id"),
            "accuracy": fold.get("accuracy"),
            "macro_f1": fold.get("macro_f1"),
            "macro_precision": fold.get("macro_precision"),
            "macro_recall": fold.get("macro_recall"),
        }
        for species, vals in fold.get("per_species", {}).items():
            for metric, value in vals.items():
                row[f"{species}_{metric}"] = value
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False, float_format="%.4f")
    logger.info("Per-fold CSV exported → %s", output_path)
