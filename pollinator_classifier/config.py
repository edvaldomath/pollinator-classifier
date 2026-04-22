"""Global hyperparameters and constants for the pollinator classifier."""

from typing import Dict, List

# ── Dataset ──────────────────────────────────────────────────────────────────

SPECIES_LABELS: List[str] = [
    "apis_mellifera",
    "vespula_vulgaris",
    "bombus_terrestris",
    "bombus_lapidarius",
    "bombus_muscorum",
]

LABEL_TO_IDX: Dict[str, int] = {label: idx for idx, label in enumerate(SPECIES_LABELS)}
IDX_TO_LABEL: Dict[int, str] = {idx: label for label, idx in LABEL_TO_IDX.items()}
NUM_CLASSES: int = len(SPECIES_LABELS)

SAMPLE_RATE: int = 1_000        # Hz
EXPECTED_SAMPLES: int = 60_001
SIGNAL_COLUMNS: List[str] = ["amplitude", "phase"]

# ── Windowing ────────────────────────────────────────────────────────────────

DEFAULT_WINDOW_SIZE: int = 1_000   # samples = 1 s at 1 kHz
DEFAULT_OVERLAP: float = 0.5       # 50 %
WINDOW_SIZE_H4: int = 500          # 0.5 s window — hypothesis H4

# ── LSTM / Bi-LSTM ───────────────────────────────────────────────────────────

LSTM_UNITS_1: int = 64
LSTM_UNITS_2: int = 128
DROPOUT_RATE: float = 0.3

# ── CNN-1D ───────────────────────────────────────────────────────────────────

CNN_FILTERS_1: int = 64
CNN_FILTERS_2: int = 128
CNN_KERNEL_1: int = 7
CNN_KERNEL_2: int = 5
DENSE_UNITS: int = 64

# ── SVM ──────────────────────────────────────────────────────────────────────

SVM_N_FFT_PEAKS: int = 10
SVM_KERNEL: str = "rbf"

# ── Training ─────────────────────────────────────────────────────────────────

LEARNING_RATE: float = 0.001
BATCH_SIZE: int = 32
EPOCHS: int = 100
EARLY_STOPPING_PATIENCE: int = 10
RANDOM_STATE: int = 42

MODEL_NAMES: List[str] = ["lstm", "bilstm", "cnn1d", "svm"]
