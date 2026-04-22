# Pollinator Classifier — micro-Doppler LSTM Pipeline

CLI tool for classifying flying pollinators by their micro-Doppler radar signature using LSTM (and baseline) models. Developed for **Article 1** of the PPGMSB/UNEB scientific roadmap.

Target venues: *Remote Sensing* (MDPI) · *IEEE GRSL* · *Computers and Electronics in Agriculture*

---

## Installation

```bash
# 1. Clone / copy the project
cd "Polinizadores claude"

# 2. Create a virtual environment (Python 3.10+)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU note:** `tensorflow>=2.16` picks up CUDA automatically if a compatible GPU and drivers are present. CPU-only training works out of the box.

---

## Dataset structure

Place all signal CSV files inside a single flat folder (e.g. `./data/`).

### File naming convention

```
{genus}_{species}_{subject_id}_{medicao}.csv
```

| Field | Example | Notes |
|---|---|---|
| `genus` | `apis` | First part of the species name |
| `species` | `mellifera` | Second part |
| `subject_id` | `1` | Unique individual ID |
| `medicao` | `3` | Recording index for this individual |

**Supported species** (exactly as named):

| Label | Common name |
|---|---|
| `apis_mellifera` | Honeybee |
| `vespula_vulgaris` | Common wasp |
| `bombus_terrestris` | Buff-tailed bumblebee |
| `bombus_lapidarius` | Red-tailed bumblebee |
| `bombus_muscorum` | Moss carder bee |

### CSV format

Each CSV must have **exactly these two columns**:

| Column | Type | Description |
|---|---|---|
| `amplitude` | float64 | Radar return amplitude |
| `phase` | float64 | Radar return phase |

- **60 001 samples** per file (60.001 s at 1 kHz)
- No header row required beyond the column names

**Example directory layout:**

```
data/
├── apis_mellifera_1_1.csv
├── apis_mellifera_1_2.csv
├── apis_mellifera_2_1.csv
├── vespula_vulgaris_1_1.csv
├── bombus_terrestris_1_1.csv
└── ...
```

---

## CLI reference

All commands are run from the **project root** (`Polinizadores claude/`).

```
python -m pollinator_classifier.main [--verbose] COMMAND [OPTIONS]
```

### `inventory` — scan the dataset

```bash
python -m pollinator_classifier.main inventory \
    --dataset ./data/

# Save inventory CSV to a custom path
python -m pollinator_classifier.main inventory \
    --dataset ./data/ \
    --output ./reports/inventory.csv
```

Prints a per-species summary (files, unique subjects, total samples) and writes `inventory.csv`.

---

### `preprocess` — preview windowing (dry-run)

```bash
# Default: 1 s windows, 50 % overlap (Hypotheses H1-H3)
python -m pollinator_classifier.main preprocess \
    --dataset ./data/ \
    --window 1000 \
    --overlap 0.5

# 0.5 s windows to test Hypothesis H4
python -m pollinator_classifier.main preprocess \
    --dataset ./data/ \
    --window 500 \
    --overlap 0.5
```

No files are written. Prints window counts per species and confirms `subject_id` is preserved.

---

### `train` — LOSO cross-validation

```bash
# Train a single model
python -m pollinator_classifier.main train \
    --model lstm \
    --dataset ./data/ \
    --output ./results/

# Train all four models at once
python -m pollinator_classifier.main train \
    --model all \
    --dataset ./data/ \
    --output ./results/

# Test Hypothesis H4 (0.5 s windows)
python -m pollinator_classifier.main train \
    --model lstm \
    --dataset ./data/ \
    --output ./results_h4/ \
    --window 500

# Disable class weighting
python -m pollinator_classifier.main train \
    --model bilstm \
    --dataset ./data/ \
    --output ./results/ \
    --no-class-weight
```

**Available model names:** `lstm` · `bilstm` · `cnn1d` · `svm` · `all`

**Output per model** (inside `<output>/<model_name>/`):

| File | Description |
|---|---|
| `{model}_fold_N.keras` | Saved Keras model for fold N |
| `{model}_fold_N.pkl` | Saved SVM model for fold N |
| `{model}_fold_N.scaler.pkl` | StandardScaler fitted on fold N training data |
| `{model}_fold_N_y_true.npy` | Ground truth labels for fold N |
| `{model}_fold_N_y_pred.npy` | Predicted labels for fold N |
| `{model}_results.json` | Full metrics: per-fold + mean ± std |

When `--model all` is used, a `model_comparison.csv` and `model_comparison.json` are written to `<output>/`.

---

### `evaluate` — compare results and generate plots

```bash
# Text-only comparison
python -m pollinator_classifier.main evaluate \
    --results ./results/

# With plots (PNG, 150 dpi)
python -m pollinator_classifier.main evaluate \
    --results ./results/ \
    --plot
```

**Generated plots** (when `--plot`):

| File | Description |
|---|---|
| `<model>/confusion_matrix.png` | Normalised confusion matrix (all folds concatenated) |
| `<model>/learning_curve_fold0.png` | Loss & accuracy curves for fold 0 |
| `f1_boxplot.png` | Per-fold macro-F1 distribution per model |
| `per_species_f1.png` | Grouped bar chart — species F1 by model |

---

### `predict` — single-signal inference

```bash
python -m pollinator_classifier.main predict \
    --model ./results/lstm/lstm_fold_0.keras \
    --input sinal.csv

# SVM inference
python -m pollinator_classifier.main predict \
    --model ./results/svm/svm_fold_0.pkl \
    --input sinal.csv
```

The signal is segmented into windows, each window is classified, and a **majority vote** gives the final species prediction.

The scaler is loaded automatically from `{model_stem}.scaler.pkl` in the same directory as the model file. If not found, a warning is shown and the scaler is fitted on the input signal (acceptable for quick tests, not for rigorous evaluation).

---

## Model descriptions

### LSTM (unidirectional)
```
LSTM(64, return_sequences=True) → Dropout(0.3)
→ LSTM(128) → Dropout(0.3) → Dense(5, softmax)
```
Primary model for Hypothesis H1. Processes the raw 2-channel time series (amplitude + phase) end-to-end.

### Bi-LSTM
```
Bidirectional(LSTM(64, return_sequences=True)) → Dropout(0.3)
→ Bidirectional(LSTM(128)) → Dropout(0.3) → Dense(5, softmax)
```
Reads the sequence in both directions. Hypothesis H2 expects this to improve F1 for *Bombus* spp.

### CNN-1D
```
Conv1D(64, 7) → MaxPool(2) → Conv1D(128, 5) → MaxPool(2)
→ Flatten → Dense(64, ReLU) → Dense(5, softmax)
```
Convolutional baseline. Captures local temporal patterns efficiently.

### SVM + FFT features
Handcrafted feature vector per window (24 values):
- Top-10 FFT magnitudes — amplitude channel
- Top-10 FFT magnitudes — phase channel
- RMS + std — amplitude channel
- RMS + std — phase channel

Trained with `SVC(kernel='rbf', probability=True)`. Classical baseline for Hypothesis H3.

---

## Hypotheses this tool tests

| ID | Hypothesis | Key command |
|---|---|---|
| **H1** | LSTM + 1 s window achieves macro-F1 ≥ 0.90 in 5-species LOSO | `train --model lstm --window 1000` |
| **H2** | Bi-LSTM outperforms LSTM in F1 for *Bombus* spp. | `train --model all` then `evaluate` |
| **H3** | Raw-signal LSTM outperforms SVM + handcrafted FFT features | `train --model all` |
| **H4** | 0.5 s window achieves ≥ 85 % accuracy | `train --model lstm --window 500` |

---

## Scientific protocol — critical constraints

| Rule | Implementation |
|---|---|
| **Split always by `subject_id`** | `LOSOFold` partitions `WindowRecord` lists by source `subject_id`; never by window index |
| **Scaler fit only on training fold** | `fit_scaler(X_train)` is called inside the fold loop; test data only sees `apply_scaler()` |
| **`subject_id` preserved in every window** | `WindowRecord.subject_id` copied from `SignalRecord` during windowing |
| **Reproducibility** | `random_state=42` in all stochastic operations; `tf.random.set_seed(42)` per model |

---

## Global hyperparameters (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `SAMPLE_RATE` | 1000 Hz | Radar sampling rate |
| `DEFAULT_WINDOW_SIZE` | 1000 | Samples per window (1 s) |
| `DEFAULT_OVERLAP` | 0.5 | 50 % overlap |
| `LSTM_UNITS_1 / 2` | 64 / 128 | LSTM layer sizes |
| `DROPOUT_RATE` | 0.3 | Dropout probability |
| `LEARNING_RATE` | 0.001 | Adam optimizer LR |
| `BATCH_SIZE` | 32 | Mini-batch size |
| `EPOCHS` | 100 | Maximum epochs |
| `EARLY_STOPPING_PATIENCE` | 10 | Epochs without `val_loss` improvement before stopping |
| `SVM_N_FFT_PEAKS` | 10 | Number of FFT peaks per channel |
| `RANDOM_STATE` | 42 | Global seed |
