"""Leave-One-Subject-Out (LOSO) cross-validation splits."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from pollinator_classifier.config import SPECIES_LABELS
from pollinator_classifier.data.windowing import WindowRecord

logger = logging.getLogger(__name__)


@dataclass
class LOSOFold:
    """Train / test split for one LOSO fold."""

    fold_id: int
    test_subject_id: int
    train_windows: List[WindowRecord] = field(repr=False)
    test_windows: List[WindowRecord] = field(repr=False)


def loso_splits(windows: List[WindowRecord]) -> List[LOSOFold]:
    """Generate LOSO folds — one per unique subject_id.

    For each fold, ALL windows whose source recording has
    ``subject_id == test_subject_id`` go to the test set; every other
    window goes to the training set.  This guarantees zero leakage
    between individuals across any fold.

    Parameters
    ----------
    windows : list[WindowRecord]
        Complete windowed dataset (all species, all subjects).

    Returns
    -------
    list[LOSOFold]
        One fold per unique subject_id, ordered by subject_id.
    """
    if not windows:
        raise ValueError("Cannot create LOSO splits from an empty window list.")

    unique_subjects = sorted({w.subject_id for w in windows})
    logger.info("LOSO: %d folds (subjects: %s)", len(unique_subjects), unique_subjects)

    folds: List[LOSOFold] = []
    for fold_idx, subject_id in enumerate(unique_subjects):
        train = [w for w in windows if w.subject_id != subject_id]
        test = [w for w in windows if w.subject_id == subject_id]

        _log_fold_distribution(fold_idx, subject_id, train, test)

        folds.append(
            LOSOFold(
                fold_id=fold_idx,
                test_subject_id=subject_id,
                train_windows=train,
                test_windows=test,
            )
        )

    return folds


def _log_fold_distribution(
    fold_idx: int,
    subject_id: int,
    train: List[WindowRecord],
    test: List[WindowRecord],
) -> None:
    train_counts: Dict[str, int] = {s: 0 for s in SPECIES_LABELS}
    test_counts: Dict[str, int] = {s: 0 for s in SPECIES_LABELS}
    for w in train:
        train_counts[w.label] += 1
    for w in test:
        test_counts[w.label] += 1

    logger.info(
        "Fold %d | test_subject_id=%d | train=%d windows | test=%d windows",
        fold_idx,
        subject_id,
        len(train),
        len(test),
    )
    for label in SPECIES_LABELS:
        if train_counts[label] or test_counts[label]:
            logger.debug(
                "  %-25s  train=%5d  test=%5d",
                label,
                train_counts[label],
                test_counts[label],
            )


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute inverse-frequency class weights to handle imbalanced folds.

    Parameters
    ----------
    y : np.ndarray, shape (N,)
        Integer class labels from the training fold.

    Returns
    -------
    dict[int, float]
        Mapping class index → weight.
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    weight_dict = dict(zip(classes.tolist(), weights.tolist()))
    logger.debug("Class weights: %s", weight_dict)
    return weight_dict
