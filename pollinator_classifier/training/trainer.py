"""LOSO training loop — orchestrates preprocessing, model training, and saving."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from pollinator_classifier.config import MODEL_NAMES
from pollinator_classifier.data.preprocessing import (
    apply_scaler,
    fit_scaler,
    save_scaler,
    windows_to_arrays,
)
from pollinator_classifier.models.base_model import BaseModel
from pollinator_classifier.training.cross_validation import (
    LOSOFold,
    compute_class_weights,
)

logger = logging.getLogger(__name__)


def _instantiate_model(model_name: str) -> BaseModel:
    """Return a fresh model instance by name."""
    if model_name == "lstm":
        from pollinator_classifier.models.lstm_model import LSTMModel

        return LSTMModel()
    if model_name == "bilstm":
        from pollinator_classifier.models.bilstm_model import BiLSTMModel

        return BiLSTMModel()
    if model_name == "cnn1d":
        from pollinator_classifier.models.cnn1d_model import CNN1DModel

        return CNN1DModel()
    if model_name == "svm":
        from pollinator_classifier.models.svm_baseline import SVMModel

        return SVMModel()
    raise ValueError(
        f"Unknown model '{model_name}'. Choose from: {MODEL_NAMES}"
    )


def train_loso(
    model_name: str,
    folds: List[LOSOFold],
    output_dir: Path,
    use_class_weight: bool = True,
) -> Dict[str, Any]:
    """Run full LOSO cross-validation for one model type.

    For each fold:
        1. Convert WindowRecords to arrays.
        2. Fit StandardScaler on training windows ONLY.
        3. Scale both train and test sets.
        4. Instantiate model, call fit(), call predict().
        5. Save model, scaler, and raw predictions.

    Parameters
    ----------
    model_name : str
        One of: lstm, bilstm, cnn1d, svm.
    folds : list[LOSOFold]
        LOSO fold definitions from :func:`loso_splits`.
    output_dir : Path
        Directory for model files, scalers, and result JSON.
    use_class_weight : bool
        Whether to compute and pass class weights to the model.

    Returns
    -------
    dict
        Aggregated results: per-fold metrics + mean ± std summary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_results: List[Dict[str, Any]] = []

    for fold in tqdm(folds, desc=f"LOSO [{model_name.upper()}]", unit="fold"):
        logger.info(
            "=== Fold %d / %d  |  test subject_id=%d ===",
            fold.fold_id,
            len(folds) - 1,
            fold.test_subject_id,
        )

        X_train_raw, y_train, _ = windows_to_arrays(fold.train_windows)
        X_test_raw, y_test, _ = windows_to_arrays(fold.test_windows)

        # ── Scaler: fit ONLY on training fold ──────────────────────────────
        scaler = fit_scaler(X_train_raw)
        X_train = apply_scaler(X_train_raw, scaler)
        X_test = apply_scaler(X_test_raw, scaler)

        class_weight: Optional[Dict[int, float]] = (
            compute_class_weights(y_train) if use_class_weight else None
        )

        model = _instantiate_model(model_name)
        history = model.fit(X_train, y_train, X_test, y_test, class_weight=class_weight)

        y_pred = model.predict(X_test)

        from pollinator_classifier.evaluation.metrics import compute_metrics

        metrics = compute_metrics(y_test, y_pred)
        metrics["fold_id"] = fold.fold_id
        metrics["test_subject_id"] = fold.test_subject_id
        metrics["history"] = {k: [float(v) for v in vals] for k, vals in history.items()}
        fold_results.append(metrics)

        # ── Persist model, scaler, and raw arrays ──────────────────────────
        ext = ".pkl" if model_name == "svm" else ".keras"
        model_path = output_dir / f"{model_name}_fold_{fold.fold_id}{ext}"
        scaler_path = output_dir / f"{model_name}_fold_{fold.fold_id}.scaler.pkl"

        model.save(str(model_path))
        save_scaler(scaler, scaler_path)

        np.save(output_dir / f"{model_name}_fold_{fold.fold_id}_y_true.npy", y_test)
        np.save(output_dir / f"{model_name}_fold_{fold.fold_id}_y_pred.npy", y_pred)

        logger.info(
            "Fold %d done — acc=%.4f  macro-F1=%.4f",
            fold.fold_id,
            metrics["accuracy"],
            metrics["macro_f1"],
        )

    summary = _aggregate_results(fold_results, model_name)

    out_json = output_dir / f"{model_name}_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    logger.info("Results saved → %s", out_json)

    return summary


def _aggregate_results(
    fold_results: List[Dict[str, Any]],
    model_name: str,
) -> Dict[str, Any]:
    """Compute mean ± std across folds for headline metrics."""
    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["macro_f1"] for r in fold_results]
    precs = [r["macro_precision"] for r in fold_results]
    recs = [r["macro_recall"] for r in fold_results]

    return {
        "model": model_name,
        "n_folds": len(fold_results),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "macro_precision_mean": float(np.mean(precs)),
        "macro_precision_std": float(np.std(precs)),
        "macro_recall_mean": float(np.mean(recs)),
        "macro_recall_std": float(np.std(recs)),
        "folds": fold_results,
    }
