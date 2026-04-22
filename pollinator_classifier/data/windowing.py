"""Sliding-window segmentation — preserves subject_id in every window."""

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

from pollinator_classifier.config import DEFAULT_OVERLAP, DEFAULT_WINDOW_SIZE
from pollinator_classifier.data.loader import SignalRecord

logger = logging.getLogger(__name__)


@dataclass
class WindowRecord:
    """A single windowed segment of a radar signal."""

    window: np.ndarray   # shape (window_size, 2)
    label: str
    subject_id: int      # CRITICAL: must be preserved for correct LOSO splits
    medicao: int
    window_idx: int
    filepath: str


def create_windows(
    records: List[SignalRecord],
    window_size: int = DEFAULT_WINDOW_SIZE,
    overlap: float = DEFAULT_OVERLAP,
) -> List[WindowRecord]:
    """Segment signal records into overlapping windows.

    Each window retains the subject_id of its source recording so that
    LOSO cross-validation can split correctly without contamination.

    Parameters
    ----------
    records : list[SignalRecord]
        Source signal records.
    window_size : int
        Number of samples per window (default 1000 = 1 s at 1 kHz).
    overlap : float
        Fractional overlap between consecutive windows, 0 ≤ overlap < 1
        (default 0.5 = 50 %).

    Returns
    -------
    list[WindowRecord]
        Windowed segments, each carrying subject_id from its source record.

    Raises
    ------
    ValueError
        If overlap is outside [0, 1) or window_size ≤ 0.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    step = max(1, int(window_size * (1.0 - overlap)))
    windows: List[WindowRecord] = []

    for record in records:
        n = len(record.signal)
        if n < window_size:
            logger.warning(
                "Signal '%s' has only %d samples (< window_size=%d) — skipped",
                record.filepath,
                n,
                window_size,
            )
            continue

        for idx, start in enumerate(range(0, n - window_size + 1, step)):
            windows.append(
                WindowRecord(
                    window=record.signal[start : start + window_size].copy(),
                    label=record.label,
                    subject_id=record.subject_id,
                    medicao=record.medicao,
                    window_idx=idx,
                    filepath=record.filepath,
                )
            )

    logger.info(
        "Created %d windows (size=%d, overlap=%.0f%%, step=%d) from %d records",
        len(windows),
        window_size,
        overlap * 100,
        step,
        len(records),
    )
    return windows
