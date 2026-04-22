"""Automatic dataset inventory generation."""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from pollinator_classifier.data.loader import SignalRecord, load_dataset

logger = logging.getLogger(__name__)


def build_inventory(records: List[SignalRecord]) -> pd.DataFrame:
    """Build a DataFrame inventory from loaded signal records.

    Parameters
    ----------
    records : list[SignalRecord]
        Loaded signal records.

    Returns
    -------
    pd.DataFrame
        Columns: filepath, genus, species, label, subject_id, medicao, n_samples.
    """
    rows = [
        {
            "filepath": r.filepath,
            "genus": r.genus,
            "species": r.species,
            "label": r.label,
            "subject_id": r.subject_id,
            "medicao": r.medicao,
            "n_samples": len(r.signal),
        }
        for r in records
    ]
    return (
        pd.DataFrame(rows)
        .sort_values(["label", "subject_id", "medicao"])
        .reset_index(drop=True)
    )


def summarize_inventory(df: pd.DataFrame) -> None:
    """Print a concise inventory summary to stdout."""
    print("\n" + "=" * 70)
    print("  DATASET INVENTORY SUMMARY")
    print("=" * 70)
    print(f"  Total files  : {len(df)}")
    print(f"  Total species: {df['label'].nunique()}")
    print()

    summary = (
        df.groupby("label")
        .agg(
            files=("filepath", "count"),
            unique_subjects=("subject_id", "nunique"),
            total_measurements=("medicao", "count"),
            total_samples=("n_samples", "sum"),
        )
        .reset_index()
    )

    header = (
        f"  {'Species':<25} | {'Files':>5} | {'Subjects':>8} | "
        f"{'Measurements':>12} | {'Total samples':>13}"
    )
    print(header)
    print("  " + "-" * 68)

    for _, row in summary.iterrows():
        print(
            f"  {row['label']:<25} | {row['files']:>5} | "
            f"{row['unique_subjects']:>8} | {row['total_measurements']:>12} | "
            f"{row['total_samples']:>13,}"
        )

    print("=" * 70 + "\n")


def run_inventory(
    folder_path: str | Path,
    output_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Load dataset, build inventory, display summary, and export to CSV.

    Parameters
    ----------
    folder_path : str or Path
        Dataset directory containing CSV signal files.
    output_csv : str or Path, optional
        Destination CSV path. Defaults to ``<folder_path>/inventory.csv``.

    Returns
    -------
    pd.DataFrame
        Full inventory DataFrame.
    """
    folder = Path(folder_path)
    records = load_dataset(folder)
    df = build_inventory(records)
    summarize_inventory(df)

    out = Path(output_csv) if output_csv else folder / "inventory.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  Inventory saved to: {out}\n")

    return df
