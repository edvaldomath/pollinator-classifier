#!/usr/bin/env python3
"""CLI entry point for the pollinator radar signal classifier.

Usage examples
--------------
    python -m pollinator_classifier.main inventory  --dataset ./data/
    python -m pollinator_classifier.main preprocess --dataset ./data/ --window 1000 --overlap 0.5
    python -m pollinator_classifier.main train      --model lstm  --dataset ./data/ --output ./results/
    python -m pollinator_classifier.main train      --model all   --dataset ./data/ --output ./results/
    python -m pollinator_classifier.main evaluate   --results ./results/ --plot
    python -m pollinator_classifier.main predict    --model ./results/lstm/lstm_fold_0.keras --input sinal.csv
"""

import argparse
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=level,
        datefmt="%H:%M:%S",
    )


# ── Subcommand handlers ───────────────────────────────────────────────────────


def cmd_inventory(args: argparse.Namespace) -> None:
    from pollinator_classifier.data.inventory import run_inventory

    run_inventory(args.dataset, output_csv=args.output)


def cmd_preprocess(args: argparse.Namespace) -> None:
    from collections import Counter

    from pollinator_classifier.data.loader import load_dataset
    from pollinator_classifier.data.windowing import create_windows

    records = load_dataset(args.dataset)
    windows = create_windows(records, window_size=args.window, overlap=args.overlap)

    print(f"\nWindowing preview")
    print(f"  Window size : {args.window} samples ({args.window / 1000:.2f} s at 1 kHz)")
    print(f"  Overlap     : {args.overlap:.0%}")
    print(f"  Total windows created: {len(windows)}")
    print(f"  subject_id preserved : YES\n")

    counts = Counter(w.label for w in windows)
    print(f"  {'Species':<25}  Windows")
    print(f"  {'─'*25}  ───────")
    for label in sorted(counts):
        print(f"  {label:<25}  {counts[label]:>7}")
    print()

    subjects = Counter(w.subject_id for w in windows)
    print(f"  Unique subject IDs in windows: {sorted(subjects.keys())}\n")


def cmd_train(args: argparse.Namespace) -> None:
    from pollinator_classifier.config import MODEL_NAMES
    from pollinator_classifier.data.loader import load_dataset
    from pollinator_classifier.data.windowing import create_windows
    from pollinator_classifier.evaluation.metrics import print_metrics
    from pollinator_classifier.training.cross_validation import loso_splits
    from pollinator_classifier.training.trainer import train_loso

    records = load_dataset(args.dataset)
    windows = create_windows(records, window_size=args.window, overlap=args.overlap)

    if not windows:
        print("ERROR: No windows created — check dataset and window size.", file=sys.stderr)
        sys.exit(1)

    folds = loso_splits(windows)
    output_dir = Path(args.output)
    use_class_weight = not args.no_class_weight

    models_to_train = MODEL_NAMES if args.model == "all" else [args.model]
    all_results: dict = {}

    for model_name in models_to_train:
        bar = "=" * 60
        print(f"\n{bar}")
        print(f"  Training: {model_name.upper()}")
        print(f"  Folds   : {len(folds)}  |  Windows: {len(windows)}")
        print(f"{bar}")

        result = train_loso(
            model_name,
            folds,
            output_dir / model_name,
            use_class_weight=use_class_weight,
        )
        all_results[model_name] = result

        print_metrics(
            {
                "accuracy": result["accuracy_mean"],
                "macro_f1": result["macro_f1_mean"],
                "macro_precision": result["macro_precision_mean"],
                "macro_recall": result["macro_recall_mean"],
                "per_species": {},
            },
            title=f"{model_name.upper()} — LOSO mean (±std F1 = {result['macro_f1_std']:.4f})",
        )

    if len(models_to_train) > 1:
        from pollinator_classifier.reports.report_generator import save_comparison_table

        save_comparison_table(all_results, output_dir)


def cmd_evaluate(args: argparse.Namespace) -> None:
    import json

    from pollinator_classifier.config import MODEL_NAMES
    from pollinator_classifier.evaluation.metrics import print_metrics
    from pollinator_classifier.reports.report_generator import (
        load_fold_predictions,
        save_comparison_table,
    )

    results_dir = Path(args.results)
    all_results: dict = {}

    for model_name in MODEL_NAMES:
        model_dir = results_dir / model_name
        json_path = model_dir / f"{model_name}_results.json"
        if not json_path.exists():
            continue

        with open(json_path, encoding="utf-8") as f:
            res = json.load(f)
        all_results[model_name] = res

        print_metrics(
            {
                "accuracy": res["accuracy_mean"],
                "macro_f1": res["macro_f1_mean"],
                "macro_precision": res["macro_precision_mean"],
                "macro_recall": res["macro_recall_mean"],
                "per_species": {},
            },
            title=f"{model_name.upper()} — mean over {res['n_folds']} folds",
        )

        if args.plot:
            try:
                y_true, y_pred = load_fold_predictions(model_dir, model_name)
            except FileNotFoundError as exc:
                print(f"  WARNING: {exc}")
                continue

            from pollinator_classifier.evaluation.confusion_matrix import plot_confusion_matrix

            plot_confusion_matrix(
                y_true,
                y_pred,
                title=f"{model_name.upper()} — Confusion Matrix (all folds)",
                output_path=model_dir / "confusion_matrix.png",
            )
            print(f"  Confusion matrix → {model_dir / 'confusion_matrix.png'}")

            folds_data = res.get("folds", [])
            if folds_data and "history" in folds_data[0]:
                from pollinator_classifier.evaluation.plots import plot_learning_curve

                plot_learning_curve(
                    folds_data[0]["history"],
                    title=f"{model_name.upper()} — Learning Curve (fold 0)",
                    output_path=model_dir / "learning_curve_fold0.png",
                )
                print(f"  Learning curve  → {model_dir / 'learning_curve_fold0.png'}")

        from pollinator_classifier.reports.report_generator import export_per_fold_csv

        export_per_fold_csv(
            res.get("folds", []),
            model_dir / f"{model_name}_per_fold.csv",
        )

    if not all_results:
        print("No result JSON files found. Run 'train' first.", file=sys.stderr)
        sys.exit(1)

    save_comparison_table(all_results, results_dir)

    if args.plot:
        f1_data = {
            m: [f["macro_f1"] for f in r.get("folds", [])]
            for m, r in all_results.items()
            if r.get("folds")
        }
        if f1_data:
            from pollinator_classifier.evaluation.plots import plot_f1_boxplot

            plot_f1_boxplot(f1_data, output_path=results_dir / "f1_boxplot.png")
            print(f"  F1 boxplot → {results_dir / 'f1_boxplot.png'}")

        # Per-species F1 chart using mean over folds
        per_species_data: dict = {}
        for m, r in all_results.items():
            folds_data = r.get("folds", [])
            if not folds_data:
                continue
            species_f1: dict = {}
            for fold in folds_data:
                for sp, vals in fold.get("per_species", {}).items():
                    species_f1.setdefault(sp, []).append(vals.get("f1", 0.0))
            per_species_data[m] = {sp: {"f1": sum(v) / len(v)} for sp, v in species_f1.items()}

        if per_species_data:
            from pollinator_classifier.evaluation.plots import plot_per_species_f1

            plot_per_species_f1(
                per_species_data,
                output_path=results_dir / "per_species_f1.png",
            )
            print(f"  Per-species F1 → {results_dir / 'per_species_f1.png'}")


def cmd_predict(args: argparse.Namespace) -> None:
    from collections import Counter

    import numpy as np

    from pollinator_classifier.config import IDX_TO_LABEL
    from pollinator_classifier.data.loader import load_single
    from pollinator_classifier.data.preprocessing import apply_scaler, load_scaler
    from pollinator_classifier.data.windowing import create_windows

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    record = load_single(args.input)
    windows = create_windows([record], window_size=args.window, overlap=args.overlap)

    if not windows:
        print("ERROR: No windows could be created from the input signal.", file=sys.stderr)
        sys.exit(1)

    X = np.stack([w.window for w in windows])

    # Load scaler saved alongside the model (same stem, .scaler.pkl)
    scaler_path = model_path.parent / (model_path.stem + ".scaler.pkl")
    if scaler_path.exists():
        scaler = load_scaler(scaler_path)
        X_scaled = apply_scaler(X, scaler)
        print(f"  Scaler loaded from: {scaler_path}")
    else:
        from pollinator_classifier.data.preprocessing import fit_scaler

        print(
            "  WARNING: No scaler file found alongside the model. "
            "Fitting scaler on input signal (inference mode).",
            file=sys.stderr,
        )
        scaler = fit_scaler(X)
        X_scaled = apply_scaler(X, scaler)

    suffix = model_path.suffix.lower()

    if suffix in {".h5", ".keras"}:
        import tensorflow as tf

        keras_model = tf.keras.models.load_model(str(model_path))
        probs = keras_model.predict(X_scaled, verbose=0)
        y_pred = np.argmax(probs, axis=1)

    elif suffix == ".pkl":
        import pickle

        from pollinator_classifier.models.svm_baseline import extract_features

        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        feats = extract_features(X_scaled)
        y_pred = clf.predict(feats)

    else:
        print(
            f"ERROR: Unsupported model format '{suffix}'. Use .keras, .h5, or .pkl.",
            file=sys.stderr,
        )
        sys.exit(1)

    vote_counts = Counter(y_pred.tolist())
    majority_idx = vote_counts.most_common(1)[0][0]

    print(f"\n  Input file : {args.input}")
    print(f"  Model      : {args.model}")
    print(f"  Windows    : {len(windows)}")
    print(f"\n  PREDICTION (majority vote): {IDX_TO_LABEL[majority_idx]}")
    print(f"\n  Per-window breakdown:")
    for cls_idx, count in sorted(vote_counts.items()):
        pct = count / len(windows) * 100
        print(f"    {IDX_TO_LABEL[cls_idx]:<25}  {count:>4} windows  ({pct:.1f}%)")
    print()


# ── Parser ────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pollinator_classifier.main",
        description="Pollinator micro-Doppler radar classifier — LOSO LSTM pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable DEBUG logging"
    )

    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # ── inventory ──────────────────────────────────────────────────────────
    inv = sub.add_parser("inventory", help="Scan dataset folder and print summary")
    inv.add_argument("--dataset", required=True, metavar="DIR", help="Dataset folder")
    inv.add_argument(
        "--output",
        metavar="CSV",
        default=None,
        help="Output CSV path (default: <dataset>/inventory.csv)",
    )

    # ── preprocess ─────────────────────────────────────────────────────────
    pre = sub.add_parser("preprocess", help="Preview windowing statistics (dry-run)")
    pre.add_argument("--dataset", required=True, metavar="DIR")
    pre.add_argument(
        "--window",
        type=int,
        default=1000,
        metavar="N",
        help="Window size in samples (1000 = 1 s; 500 = 0.5 s for H4)",
    )
    pre.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        metavar="F",
        help="Fractional overlap between windows (0–1)",
    )

    # ── train ──────────────────────────────────────────────────────────────
    tr = sub.add_parser(
        "train", help="Train model(s) with Leave-One-Subject-Out cross-validation"
    )
    tr.add_argument(
        "--model",
        required=True,
        choices=["lstm", "bilstm", "cnn1d", "svm", "all"],
        help="Model to train ('all' trains all four)",
    )
    tr.add_argument("--dataset", required=True, metavar="DIR")
    tr.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help="Root directory for results (sub-dirs per model are created)",
    )
    tr.add_argument("--window", type=int, default=1000, metavar="N")
    tr.add_argument("--overlap", type=float, default=0.5, metavar="F")
    tr.add_argument(
        "--no-class-weight",
        action="store_true",
        help="Disable inverse-frequency class weighting",
    )

    # ── evaluate ───────────────────────────────────────────────────────────
    ev = sub.add_parser(
        "evaluate", help="Load saved results, compare models, and generate plots"
    )
    ev.add_argument(
        "--results",
        required=True,
        metavar="DIR",
        help="Results directory (must contain model sub-dirs with *_results.json)",
    )
    ev.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save PNG plots (confusion matrix, learning curves, boxplot)",
    )

    # ── predict ────────────────────────────────────────────────────────────
    pr = sub.add_parser(
        "predict", help="Predict species from a single signal CSV (majority-vote)"
    )
    pr.add_argument(
        "--model",
        required=True,
        metavar="PATH",
        help="Saved model file (.keras / .h5 for neural nets, .pkl for SVM)",
    )
    pr.add_argument(
        "--input",
        required=True,
        metavar="CSV",
        help="Input signal CSV (must have 'amplitude' and 'phase' columns)",
    )
    pr.add_argument("--window", type=int, default=1000, metavar="N")
    pr.add_argument("--overlap", type=float, default=0.5, metavar="F")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)

    dispatch = {
        "inventory": cmd_inventory,
        "preprocess": cmd_preprocess,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "predict": cmd_predict,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
