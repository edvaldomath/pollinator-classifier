"""Streamlit results dashboard for the pollinator micro-Doppler classifier."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── Make the package importable when running from project root ────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pollinator Classifier",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🐝 Pollinator Classifier")
st.sidebar.caption("micro-Doppler LSTM — PPGMSB/UNEB")

page = st.sidebar.radio(
    "Navigation",
    ["📂 Dataset Explorer", "🔍 Signal Inspector", "📊 LOSO Results", "📈 Model Comparison"],
)

DATASET_DIR = st.sidebar.text_input(
    "Dataset folder",
    value=r"C:\Users\Edvaldo\OneDrive\Desktop\Atividades metodos Computacionais\ATV polinizadores\dataset\Polliradar dataset",
)
RESULTS_DIR = st.sidebar.text_input("Results folder", value="./results")

st.sidebar.markdown("---")
st.sidebar.caption("Hypotheses")
st.sidebar.markdown(
    "- **H1** LSTM F1 ≥ 0.90 (1 s window)\n"
    "- **H2** Bi-LSTM > LSTM for *Bombus*\n"
    "- **H3** LSTM > SVM+FFT\n"
    "- **H4** 0.5 s window ≥ 85 % acc"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset…")
def load_records(folder: str):
    from pollinator_classifier.data.loader import load_dataset
    return load_dataset(folder)


@st.cache_data(show_spinner="Building inventory…")
def load_inventory(folder: str) -> pd.DataFrame:
    from pollinator_classifier.data.inventory import build_inventory
    records = load_records(folder)
    return build_inventory(records)


def load_results_json(results_dir: str, model_name: str) -> dict | None:
    p = Path(results_dir) / model_name / f"{model_name}_results.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


SPECIES_COLORS = {
    "apis_mellifera": "#F4A460",
    "vespula_vulgaris": "#DC143C",
    "bombus_terrestris": "#FFD700",
    "bombus_lapidarius": "#8B0000",
    "bombus_muscorum": "#228B22",
}

MODEL_COLORS = {
    "lstm": "#4C9BE8",
    "bilstm": "#F4A460",
    "cnn1d": "#5CB85C",
    "svm": "#D9534F",
}


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Dataset Explorer
# ─────────────────────────────────────────────────────────────────────────────
if page == "📂 Dataset Explorer":
    st.title("Dataset Explorer")

    if not Path(DATASET_DIR).is_dir():
        st.warning(f"Folder not found: `{DATASET_DIR}` — set the path in the sidebar.")
        st.stop()

    try:
        df = load_inventory(DATASET_DIR)
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        st.stop()

    # ── KPI row ───────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total files", len(df))
    col2.metric("Species", df["label"].nunique())
    col3.metric("Unique subjects", df["subject_id"].nunique())
    col4.metric("Total samples", f"{df['n_samples'].sum():,}")

    st.markdown("---")

    # ── Per-species summary ───────────────────────────────────────────────────
    st.subheader("Per-species summary")
    summary = (
        df.groupby("label")
        .agg(files=("filepath", "count"), subjects=("subject_id", "nunique"), samples=("n_samples", "sum"))
        .reset_index()
        .rename(columns={"label": "Species", "files": "Files", "subjects": "Unique subjects", "samples": "Total samples"})
    )
    st.dataframe(summary, use_container_width=True)

    # ── Bar charts ────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(5, 3))
        colors = [SPECIES_COLORS.get(s, "#aaa") for s in summary["Species"]]
        ax.bar(summary["Species"].str.replace("_", "\n"), summary["Files"], color=colors)
        ax.set_title("Files per species", fontsize=11)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", labelsize=7)
        plt.tight_layout()
        st.pyplot(fig)

    with col_b:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(summary["Species"].str.replace("_", "\n"), summary["Unique subjects"], color=colors)
        ax.set_title("Unique subjects per species", fontsize=11)
        ax.set_ylabel("Subjects")
        ax.tick_params(axis="x", labelsize=7)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    # ── Full inventory table ──────────────────────────────────────────────────
    with st.expander("Full inventory table"):
        st.dataframe(
            df.drop(columns=["filepath"]),
            use_container_width=True,
        )
        csv = df.to_csv(index=False).encode()
        st.download_button("Download inventory CSV", csv, "inventory.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Signal Inspector
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Signal Inspector":
    st.title("Signal Inspector")

    from pollinator_classifier.config import SAMPLE_RATE, SIGNAL_COLUMNS
    from scipy.fft import rfft, rfftfreq
    from scipy.signal import spectrogram as scipy_spectrogram

    # ── Source selection ──────────────────────────────────────────────────────
    source_mode = st.radio(
        "Signal source",
        ["📁 Select from dataset folder", "⬆ Upload CSV file"],
        horizontal=True,
    )

    signal: np.ndarray | None = None
    file_label = ""

    if source_mode == "📁 Select from dataset folder":
        if not Path(DATASET_DIR).is_dir():
            st.warning(f"Folder not found: `{DATASET_DIR}` — set the path in the sidebar.")
            st.stop()

        # ── Individual CSV selectbox (not just folder) ─────────────────────
        csv_files = sorted(Path(DATASET_DIR).glob("*.csv"))
        if not csv_files:
            st.error(f"No CSV files found in `{DATASET_DIR}`.")
            st.stop()

        csv_names = [f.name for f in csv_files]
        selected_name = st.selectbox("CSV file", csv_names)
        selected_path = Path(DATASET_DIR) / selected_name

        try:
            df_raw = pd.read_csv(selected_path)
        except Exception as exc:
            st.error(f"Failed to read `{selected_name}`: {exc}")
            st.stop()

        # ── Preview raw data ───────────────────────────────────────────────
        st.markdown("**Raw data preview** (first 5 rows):")
        st.write(df_raw.head())

        missing_cols = [c for c in SIGNAL_COLUMNS if c not in df_raw.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}. Found: {list(df_raw.columns)}")
            st.stop()

        signal = df_raw[SIGNAL_COLUMNS].to_numpy(dtype=np.float64)
        file_label = selected_name

    else:  # Upload mode
        uploaded = st.file_uploader("Upload a signal CSV", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV with `amplitude` and `phase` columns to begin.")
            st.stop()

        try:
            df_raw = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Failed to read uploaded file: {exc}")
            st.stop()

        # ── Preview raw data ───────────────────────────────────────────────
        st.markdown("**Raw data preview** (first 5 rows):")
        st.write(df_raw.head())

        missing_cols = [c for c in SIGNAL_COLUMNS if c not in df_raw.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}. Found: {list(df_raw.columns)}")
            st.stop()

        signal = df_raw[SIGNAL_COLUMNS].to_numpy(dtype=np.float64)
        file_label = uploaded.name

    # ── Common signal info ────────────────────────────────────────────────────
    n = len(signal)
    t = np.linspace(0, n / SAMPLE_RATE, n)
    st.markdown(f"**File:** `{file_label}` — {n:,} samples @ {SAMPLE_RATE} Hz")
    st.markdown("---")

    # ── Zoom slider ───────────────────────────────────────────────────────────
    zoom_s = st.slider("Zoom (seconds)", 0.1, min(5.0, n / SAMPLE_RATE), 1.0, 0.1)
    zoom_n = int(zoom_s * SAMPLE_RATE)

    # ── Time-domain plot ──────────────────────────────────────────────────────
    st.subheader("Time domain")
    fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    axes[0].plot(t[:zoom_n], signal[:zoom_n, 0], linewidth=0.7, color="#4C9BE8")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.3)
    axes[1].plot(t[:zoom_n], signal[:zoom_n, 1], linewidth=0.7, color="#F4A460")
    axes[1].set_ylabel("Phase")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(alpha=0.3)
    fig.suptitle(file_label, fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)

    # ── FFT plot ──────────────────────────────────────────────────────────────
    st.subheader("FFT spectrum")
    fft_window_opts = [x for x in [500, 1000, 2000, 5000] if x <= n] + ([n] if n not in [500, 1000, 2000, 5000] else [])
    default_idx = min(1, len(fft_window_opts) - 1)
    fft_window = st.selectbox("FFT window (samples)", fft_window_opts, index=default_idx)

    seg = signal[:fft_window, :]
    freqs = rfftfreq(fft_window, d=1.0 / SAMPLE_RATE)
    mag_amp = np.abs(rfft(seg[:, 0]))
    mag_phase = np.abs(rfft(seg[:, 1]))

    # ── FFT validation ────────────────────────────────────────────────────────
    fft_ok = True
    if len(mag_amp) == 0 or len(mag_phase) == 0:
        st.error("FFT returned an empty array. Check that the signal has enough samples.")
        fft_ok = False
    elif np.any(np.isnan(mag_amp)) or np.any(np.isnan(mag_phase)):
        nan_amp = int(np.isnan(mag_amp).sum())
        nan_phase = int(np.isnan(mag_phase).sum())
        st.error(
            f"FFT result contains NaN values — amplitude: {nan_amp} NaNs, "
            f"phase: {nan_phase} NaNs. The source signal may contain NaNs or Infs."
        )
        fft_ok = False

    if fft_ok:
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 3))
        axes2[0].plot(freqs, mag_amp, linewidth=0.8, color="#4C9BE8")
        axes2[0].set_title("Amplitude channel — FFT")
        axes2[0].set_xlabel("Frequency (Hz)")
        axes2[0].set_ylabel("Magnitude")
        axes2[0].grid(alpha=0.3)
        axes2[1].plot(freqs, mag_phase, linewidth=0.8, color="#F4A460")
        axes2[1].set_title("Phase channel — FFT")
        axes2[1].set_xlabel("Frequency (Hz)")
        axes2[1].grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)

    # ── Spectrogram ───────────────────────────────────────────────────────────
    st.subheader("Spectrogram (amplitude channel)")
    spec_len = min(10_000, n)
    f_spec, t_spec, Sxx = scipy_spectrogram(
        signal[:spec_len, 0], fs=SAMPLE_RATE, nperseg=256, noverlap=200
    )
    fig3, ax3 = plt.subplots(figsize=(12, 3))
    ax3.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-10), shading="gouraud", cmap="inferno")
    ax3.set_ylabel("Frequency (Hz)")
    ax3.set_xlabel("Time (s)")
    ax3.set_title("Spectrogram — amplitude channel")
    plt.tight_layout()
    st.pyplot(fig3)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: LOSO Results
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 LOSO Results":
    st.title("LOSO Results — Per-model detail")

    from pollinator_classifier.config import MODEL_NAMES, SPECIES_LABELS

    available = [m for m in MODEL_NAMES if load_results_json(RESULTS_DIR, m) is not None]
    if not available:
        st.warning(f"No result JSON files found in `{RESULTS_DIR}`. Run `train` first.")
        st.stop()

    selected_model = st.selectbox("Model", available)
    res = load_results_json(RESULTS_DIR, selected_model)

    # ── Headline metrics ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{res['accuracy_mean']:.4f}", f"±{res['accuracy_std']:.4f}")
    c2.metric("Macro F1", f"{res['macro_f1_mean']:.4f}", f"±{res['macro_f1_std']:.4f}")
    c3.metric("Macro Precision", f"{res['macro_precision_mean']:.4f}", f"±{res['macro_precision_std']:.4f}")
    c4.metric("Macro Recall", f"{res['macro_recall_mean']:.4f}", f"±{res['macro_recall_std']:.4f}")

    # Hypothesis H1 indicator
    if selected_model == "lstm":
        target = 0.90
        achieved = res["macro_f1_mean"] >= target
        icon = "✅" if achieved else "❌"
        st.info(f"{icon} **H1:** LSTM macro-F1 {'≥' if achieved else '<'} {target:.2f}  (observed {res['macro_f1_mean']:.4f})")

    st.markdown("---")

    # ── Per-fold table ────────────────────────────────────────────────────────
    st.subheader("Per-fold metrics")
    folds_data = res.get("folds", [])
    fold_rows = [
        {
            "Fold": f["fold_id"],
            "Test subject": f["test_subject_id"],
            "Accuracy": f"{f['accuracy']:.4f}",
            "Macro F1": f"{f['macro_f1']:.4f}",
            "Precision": f"{f['macro_precision']:.4f}",
            "Recall": f"{f['macro_recall']:.4f}",
        }
        for f in folds_data
    ]
    st.dataframe(pd.DataFrame(fold_rows), use_container_width=True)

    # ── Per-fold F1 bar chart ─────────────────────────────────────────────────
    if folds_data:
        fig, ax = plt.subplots(figsize=(10, 3))
        fold_ids = [f["fold_id"] for f in folds_data]
        f1_vals = [f["macro_f1"] for f in folds_data]
        bar_colors = ["#5CB85C" if v >= 0.90 else "#F4A460" if v >= 0.80 else "#D9534F" for v in f1_vals]
        ax.bar(fold_ids, f1_vals, color=bar_colors, edgecolor="white")
        ax.axhline(0.90, color="red", linestyle="--", linewidth=1, label="H1 target (0.90)")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Macro F1")
        ax.set_title(f"{selected_model.upper()} — Macro-F1 per LOSO fold")
        ax.set_xticks(fold_ids)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    # ── Per-species F1 heatmap across folds ──────────────────────────────────
    st.subheader("Per-species F1 across folds")
    if folds_data:
        matrix = np.zeros((len(folds_data), len(SPECIES_LABELS)))
        for i, fold in enumerate(folds_data):
            for j, sp in enumerate(SPECIES_LABELS):
                matrix[i, j] = fold.get("per_species", {}).get(sp, {}).get("f1", 0.0)

        import seaborn as sns

        fig2, ax2 = plt.subplots(figsize=(10, max(3, len(folds_data) * 0.5 + 1)))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            xticklabels=[s.replace("_", "\n") for s in SPECIES_LABELS],
            yticklabels=[f"Fold {f['fold_id']}" for f in folds_data],
            linewidths=0.5,
            ax=ax2,
        )
        ax2.set_title(f"{selected_model.upper()} — F1 per species × fold")
        plt.tight_layout()
        st.pyplot(fig2)

    # ── Learning curves ───────────────────────────────────────────────────────
    st.subheader("Learning curves")
    folds_with_history = [f for f in folds_data if "history" in f and "loss" in f.get("history", {})]
    if folds_with_history:
        fold_choice = st.selectbox("Select fold", [f["fold_id"] for f in folds_with_history])
        hist = next(f["history"] for f in folds_with_history if f["fold_id"] == fold_choice)
        epochs = range(1, len(hist["loss"]) + 1)

        fig3, axes3 = plt.subplots(1, 2, figsize=(11, 3.5))
        axes3[0].plot(epochs, hist["loss"], label="train", linewidth=1.5)
        if "val_loss" in hist:
            axes3[0].plot(epochs, hist["val_loss"], label="val", linewidth=1.5, linestyle="--")
        axes3[0].set_title("Loss")
        axes3[0].set_xlabel("Epoch")
        axes3[0].legend()
        axes3[0].grid(alpha=0.3)
        if "accuracy" in hist:
            axes3[1].plot(epochs, hist["accuracy"], label="train", linewidth=1.5)
            if "val_accuracy" in hist:
                axes3[1].plot(epochs, hist["val_accuracy"], label="val", linewidth=1.5, linestyle="--")
        axes3[1].set_title("Accuracy")
        axes3[1].set_xlabel("Epoch")
        axes3[1].set_ylim(0, 1.05)
        axes3[1].legend()
        axes3[1].grid(alpha=0.3)
        fig3.suptitle(f"{selected_model.upper()} — Fold {fold_choice}", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig3)
    else:
        st.info("No training history available (SVM baseline has no epoch-level history).")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.subheader("Confusion matrix (all folds concatenated)")
    results_path = Path(RESULTS_DIR) / selected_model
    true_files = sorted(results_path.glob(f"{selected_model}_fold_*_y_true.npy"))
    if true_files:
        y_true_all, y_pred_all = [], []
        for tf in true_files:
            tag = tf.stem.split("_y_true")[0]
            pf = results_path / f"{tag}_y_pred.npy"
            if pf.exists():
                y_true_all.append(np.load(str(tf)))
                y_pred_all.append(np.load(str(pf)))

        if y_true_all:
            from sklearn.metrics import confusion_matrix
            import seaborn as sns

            y_t = np.concatenate(y_true_all)
            y_p = np.concatenate(y_pred_all)
            cm = confusion_matrix(y_t, y_p, labels=list(range(len(SPECIES_LABELS))))
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.where(row_sums == 0, 0.0, cm / row_sums)

            short = [s.replace("_", "\n") for s in SPECIES_LABELS]
            fig4, ax4 = plt.subplots(figsize=(7, 5))
            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                        vmin=0, vmax=1, xticklabels=short, yticklabels=short,
                        linewidths=0.5, ax=ax4)
            ax4.set_title(f"{selected_model.upper()} — Confusion Matrix", fontsize=12)
            ax4.set_xlabel("Predicted")
            ax4.set_ylabel("True")
            plt.tight_layout()
            st.pyplot(fig4)
    else:
        st.info(f"No `.npy` prediction files found in `{results_path}`.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Model Comparison":
    st.title("Model Comparison")

    from pollinator_classifier.config import MODEL_NAMES, SPECIES_LABELS

    all_results = {
        m: load_results_json(RESULTS_DIR, m)
        for m in MODEL_NAMES
        if load_results_json(RESULTS_DIR, m) is not None
    }

    if not all_results:
        st.warning(f"No result files found in `{RESULTS_DIR}`. Run `train --model all` first.")
        st.stop()

    # ── Comparison table ──────────────────────────────────────────────────────
    st.subheader("Summary table")
    rows = []
    for m, r in all_results.items():
        rows.append({
            "Model": m.upper(),
            "Acc mean": f"{r['accuracy_mean']:.4f}",
            "Acc ±": f"{r['accuracy_std']:.4f}",
            "F1 mean": f"{r['macro_f1_mean']:.4f}",
            "F1 ±": f"{r['macro_f1_std']:.4f}",
            "Precision": f"{r['macro_precision_mean']:.4f}",
            "Recall": f"{r['macro_recall_mean']:.4f}",
            "Folds": r["n_folds"],
        })
    comp_df = pd.DataFrame(rows).sort_values("F1 mean", ascending=False)
    st.dataframe(comp_df, use_container_width=True)

    # Hypothesis annotations
    st.markdown("---")
    st.subheader("Hypothesis checks")
    hyp_cols = st.columns(len(all_results))
    for col, (m, r) in zip(hyp_cols, all_results.items()):
        col.markdown(f"**{m.upper()}**")
        if m == "lstm":
            ok = r["macro_f1_mean"] >= 0.90
            col.write(f"H1 ({'✅' if ok else '❌'}) F1={r['macro_f1_mean']:.4f}")
        if m == "bilstm" and "lstm" in all_results:
            ok = r["macro_f1_mean"] > all_results["lstm"]["macro_f1_mean"]
            col.write(f"H2 ({'✅' if ok else '❌'}) BiLSTM {'>' if ok else '≤'} LSTM")
        if m == "lstm" and "svm" in all_results:
            ok = r["macro_f1_mean"] > all_results["svm"]["macro_f1_mean"]
            col.write(f"H3 ({'✅' if ok else '❌'}) LSTM {'>' if ok else '≤'} SVM")

    st.markdown("---")

    # ── F1 boxplot ────────────────────────────────────────────────────────────
    st.subheader("Per-fold macro-F1 distribution")
    f1_data = {
        m: [f["macro_f1"] for f in r.get("folds", [])]
        for m, r in all_results.items()
        if r.get("folds")
    }
    if f1_data:
        fig, ax = plt.subplots(figsize=(8, 4))
        model_list = list(f1_data.keys())
        bp = ax.boxplot(
            [f1_data[m] for m in model_list],
            labels=[m.upper() for m in model_list],
            patch_artist=True,
            notch=False,
        )
        for patch, m in zip(bp["boxes"], model_list):
            patch.set_facecolor(MODEL_COLORS.get(m, "#aaa"))
            patch.set_alpha(0.8)
        ax.axhline(0.90, color="red", linestyle="--", linewidth=1, label="H1 threshold")
        ax.set_ylabel("Macro-F1")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_title("Macro-F1 per model — LOSO folds")
        plt.tight_layout()
        st.pyplot(fig)

    # ── Per-species F1 grouped bar chart ──────────────────────────────────────
    st.subheader("Per-species F1 by model")
    per_species: dict = {}
    for m, r in all_results.items():
        folds_data = r.get("folds", [])
        if not folds_data:
            continue
        sp_f1: dict = {}
        for fold in folds_data:
            for sp, vals in fold.get("per_species", {}).items():
                sp_f1.setdefault(sp, []).append(vals.get("f1", 0.0))
        per_species[m] = {sp: np.mean(v) for sp, v in sp_f1.items()}

    if per_species:
        model_names = list(per_species.keys())
        n_m = len(model_names)
        x = np.arange(len(SPECIES_LABELS))
        width = 0.8 / n_m

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        for i, m in enumerate(model_names):
            vals = [per_species[m].get(sp, 0.0) for sp in SPECIES_LABELS]
            ax2.bar(x + i * width, vals, width, label=m.upper(),
                    color=MODEL_COLORS.get(m, "#aaa"), alpha=0.85)

        ax2.set_xticks(x + width * (n_m - 1) / 2)
        ax2.set_xticklabels([s.replace("_", "\n") for s in SPECIES_LABELS], fontsize=9)
        ax2.set_ylabel("Mean F1 (over folds)")
        ax2.set_ylim(0, 1.05)
        ax2.set_title("Per-species F1 — model comparison")
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)

        # H2: highlight Bombus species
        bombus_indices = [i for i, s in enumerate(SPECIES_LABELS) if "bombus" in s]
        for bi in bombus_indices:
            ax2.axvspan(bi - 0.5, bi + 0.5, alpha=0.05, color="purple")
        ax2.text(
            np.mean(bombus_indices), 1.02, "← H2: Bombus spp. →",
            ha="center", va="bottom", fontsize=8, color="purple"
        )

        plt.tight_layout()
        st.pyplot(fig2)

    # ── Download comparison CSV ───────────────────────────────────────────────
    st.markdown("---")
    csv_path = Path(RESULTS_DIR) / "model_comparison.csv"
    if csv_path.exists():
        with open(csv_path, "rb") as f:
            st.download_button(
                "⬇ Download comparison CSV",
                f.read(),
                file_name="model_comparison.csv",
                mime="text/csv",
            )
