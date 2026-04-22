"""CSV loading utilities for pollinator radar signals."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from pollinator_classifier.config import SIGNAL_COLUMNS, SPECIES_LABELS

logger = logging.getLogger(__name__)


@dataclass
class SignalRecord:
    """Container for a single radar signal recording."""

    signal: np.ndarray   # shape (N, 2): columns [amplitude, phase]
    label: str           # full species label, e.g. "apis_mellifera"
    subject_id: int
    medicao: int
    genus: str
    species: str
    filepath: str


def _parse_filename(path: Path) -> Optional[dict]:
    """Parse metadata from a filename: {genus}_{species}_{subject_id}_{medicao}.csv."""
    stem = path.stem
    for label in SPECIES_LABELS:
        prefix = label + "_"
        if stem.startswith(prefix):
            remainder = stem[len(prefix):]
            match = re.fullmatch(r"(\d+)_(\d+)", remainder)
            if match:
                genus, species_name = label.split("_", 1)
                return {
                    "label": label,
                    "genus": genus,
                    "species": species_name,
                    "subject_id": int(match.group(1)),
                    "medicao": int(match.group(2)),
                }
    logger.warning("Could not parse filename: %s", path.name)
    return None


def load_single(path: str | Path) -> SignalRecord:
    """Load a single CSV signal file.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.

    Returns
    -------
    SignalRecord
        Loaded signal with label and metadata.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are absent or the filename cannot be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Signal file not found: {path}")

    meta = _parse_filename(path)
    if meta is None:
        raise ValueError(
            f"Cannot parse metadata from '{path.name}'. "
            "Expected: {{genus}}_{{species}}_{{subject_id}}_{{medicao}}.csv"
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV '{path.name}': {exc}") from exc

    missing = [c for c in SIGNAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV '{path.name}' is missing columns: {missing}")

    signal = df[SIGNAL_COLUMNS].to_numpy(dtype=np.float64)
    logger.debug("Loaded %d samples from %s", len(signal), path.name)

    return SignalRecord(signal=signal, filepath=str(path), **meta)


def load_dataset(folder_path: str | Path) -> List[SignalRecord]:
    """Load all CSV files from a folder matching the naming convention.

    Parameters
    ----------
    folder_path : str or Path
        Directory containing signal CSV files.

    Returns
    -------
    list[SignalRecord]
        Successfully loaded records sorted by (label, subject_id, medicao).

    Raises
    ------
    FileNotFoundError
        If the folder does not exist.
    ValueError
        If no valid CSV files are found.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in: {folder}")

    records: List[SignalRecord] = []
    errors: int = 0

    for csv_path in csv_files:
        try:
            records.append(load_single(csv_path))
        except (ValueError, pd.errors.ParserError) as exc:
            logger.warning("Skipping %s: %s", csv_path.name, exc)
            errors += 1

    logger.info(
        "Loaded %d records from '%s' (%d skipped due to errors)",
        len(records),
        folder,
        errors,
    )

    if not records:
        raise ValueError(f"No valid signal files could be loaded from: {folder}")

    records.sort(key=lambda r: (r.label, r.subject_id, r.medicao))
    return records
