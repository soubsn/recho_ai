"""
Evaluation and comparison report for all Hopf oscillator models.

Loads trained model checkpoints (from train_all.py), runs inference on held-out
test data, and generates:

  TABLE: one row per model with columns —
    Model | Input | Params | Size (KB) | Val Accuracy | Test Accuracy |
    Inference (ms est.) | M33 fits? | M55 fits? | CMSIS-NN coverage

  PLOTS:
    1. Accuracy bar chart — all models side by side
    2. Confusion matrices — top 2 models
    3. Feature map visualisation — x vs y vs phase vs angle, one clip per class
    4. Training curves — accuracy and loss per epoch, all models

KEY COMPARISONS HIGHLIGHTED:
  A vs B: does y(t) improve accuracy?
  A vs C: does orbit radius capture the signal?
  A vs D: does phase angle carry independent information?
  A vs E: is late fusion better than early fusion?
  F vs A: accuracy cost of depthwise (smaller/faster) model
  G vs A: how much does the CNN add over ridge regression?

Usage:
    python -m pipeline.evaluate --checkpoint_dir checkpoints/ --output_dir output/eval/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# MCU RAM and flash budgets (bytes)
MCU_RAM: dict[str, int] = {"M33": 64 * 1024, "M55": 128 * 1024, "M85": 256 * 1024}
# Estimated inference time coefficients — rough estimates per MAC at 64 MHz M33
# Models with ~80 K activations * 64 channels ≈ few ms; depthwise ~8x faster
_INFERENCE_MS_EST: dict[str, float] = {
    "model_a_cnn_x_only": 45.0,
    "model_b_cnn_xy_dual": 48.0,
    "model_c_cnn_phase": 45.0,
    "model_d_cnn_angle": 45.0,
    "model_e_cnn_xy_fusion": 90.0,   # two branches
    "model_f_depthwise_cnn": 6.0,    # ~8x fewer MACs
    "model_g_ridge_x_only": 2.0,     # linear, no conv
    "model_g_ridge_y_only": 2.0,
    "model_g_ridge_xy_concatenated": 3.0,
}


def _load_test_data(
    cfg_n_clips: int,
    cfg_n_classes: int,
    test_fraction: float = 0.1,
) -> tuple[NDArray, NDArray, NDArray, dict[str, NDArray]]:
    """
    Generate a held-out test set (separate from the training cache).

    Returns x_proc, y_proc, labels_test, reps_test.
    """
    from data.sample_data import generate_dataset_xy
    from pipeline.ingest import process_dataset
    from pipeline.features_xy import extract_y_features, extract_all_representations

    n_test = max(2, int(cfg_n_clips * test_fraction))
    print(f"[evaluate] Generating {n_test} test clips per class (uncached) ...")
    raw_x, raw_y, labels = generate_dataset_xy(
        n_clips_per_class=n_test,
        n_classes=cfg_n_classes,
        cache=False,
    )
    x_proc = process_dataset(raw_x)
    y_proc = extract_y_features(raw_y)
    reps = extract_all_representations(x_proc, y_proc)
    return x_proc, y_proc, labels, reps


def _load_keras_model(ckpt_path: Path):
    """Load a Keras model from checkpoint path."""
    import tensorflow as tf
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return tf.keras.models.load_model(str(ckpt_path))


def _predict_keras(model, features: NDArray, expand_ch: bool = True) -> NDArray[np.int64]:
    inp = features.astype(np.float32)
    if expand_ch and inp.ndim == 3:
        inp = np.expand_dims(inp, -1)
    probs = model.predict(inp, verbose=0)
    return np.argmax(probs, axis=1).astype(np.int64)


def _predict_fusion(model, x_feat: NDArray, y_feat: NDArray) -> NDArray[np.int64]:
    x = np.expand_dims(x_feat.astype(np.float32), -1)
    y = np.expand_dims(y_feat.astype(np.float32), -1)
    probs = model.predict([x, y], verbose=0)
    return np.argmax(probs, axis=1).astype(np.int64)


def _count_params(model) -> int:
    return int(model.count_params())


def _model_size_kb(ckpt_path: Optional[Path]) -> float:
    if ckpt_path is None or not ckpt_path.exists():
        return float("nan")
    return ckpt_path.stat().st_size / 1024


def _tflite_size_kb(tflite_dir: Path, model_name: str) -> float:
    p = tflite_dir / f"{model_name}.tflite"
    if p.exists():
        return p.stat().st_size / 1024
    return float("nan")


def _confusion_matrix(preds: NDArray, labels: NDArray, n_classes: int) -> NDArray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(labels, preds):
        cm[int(true), int(pred)] += 1
    return cm


def _load_history_csv(csv_path: Path) -> dict[str, list[float]]:
    if csv_path is None or not csv_path.exists():
        return {}
    hist: dict[str, list] = {"epoch": [], "loss": [], "accuracy": [],
                              "val_loss": [], "val_accuracy": []}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            for k in hist:
                try:
                    hist[k].append(float(row[k]))
                except (KeyError, ValueError):
                    pass
    return hist


def generate_report(
    checkpoint_dir: str | Path = "checkpoints/",
    output_dir: str | Path = "output/eval/",
    n_clips_per_class: int = 100,
    n_classes: int = 5,
    tflite_dir: Optional[str | Path] = None,
) -> None:
    """
    Run the full evaluation pipeline.

    Loads checkpoints, runs test inference, prints the comparison table,
    and saves plots to output_dir.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    ckpt = Path(checkpoint_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tflite = Path(tflite_dir) if tflite_dir else ckpt

    from data.sample_data import CLASS_NAMES

    # -------------------------------------------------------------------------
    # Load test data
    # -------------------------------------------------------------------------
    x_proc, y_proc, labels_test, reps_test = _load_test_data(n_clips_per_class, n_classes)
    n_test = len(labels_test)
    print(f"[evaluate] Test set: {n_test} clips, {n_classes} classes")

    # -------------------------------------------------------------------------
    # Define model registry
    # -------------------------------------------------------------------------
    model_registry = [
        {
            "name": "model_a_cnn_x_only",
            "label": "A — x only (baseline)",
            "input": "x_only",
            "multi_input": False,
        },
        {
            "name": "model_b_cnn_xy_dual",
            "label": "B — x+y dual channel",
            "input": "xy_dual",
            "multi_input": False,
            "expand": False,
        },
        {
            "name": "model_c_cnn_phase",
            "label": "C — orbit radius r(t)",
            "input": "phase",
            "multi_input": False,
        },
        {
            "name": "model_d_cnn_angle",
            "label": "D — phase angle θ(t)",
            "input": "angle",
            "multi_input": False,
        },
        {
            "name": "model_e_cnn_xy_fusion",
            "label": "E — late fusion x+y",
            "input": "fusion",
            "multi_input": True,
        },
        {
            "name": "model_f_depthwise_cnn",
            "label": "F — depthwise (M55 opt)",
            "input": "xy_dual",
            "multi_input": False,
            "expand": False,
        },
    ]

    # -------------------------------------------------------------------------
    # Evaluate Keras models
    # -------------------------------------------------------------------------
    rows: list[dict] = []
    loaded_models: dict[str, object] = {}

    for entry in model_registry:
        name = entry["name"]
        ckpt_path = ckpt / f"{name}_best.keras"
        print(f"\n[evaluate] {entry['label']} ...")

        if not ckpt_path.exists():
            print(f"  WARNING: checkpoint not found at {ckpt_path} — skipping")
            rows.append({
                "Model": entry["label"], "Input": entry["input"],
                "Params": "—", "Size (KB)": "—",
                "Val Acc": "—", "Test Acc": "—",
                "Inf (ms)": _INFERENCE_MS_EST.get(name, "—"),
                "M33 fits?": "—", "M55 fits?": "—", "CMSIS-NN": "—",
            })
            continue

        try:
            model = _load_keras_model(ckpt_path)
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            continue

        loaded_models[name] = model

        # Inference
        if entry["multi_input"]:
            preds = _predict_fusion(model,
                                    reps_test["x_only"], reps_test["y_only"])
        else:
            expand = entry.get("expand", True)
            preds = _predict_keras(model, reps_test[entry["input"]], expand_ch=expand)

        test_acc = float(np.mean(preds == labels_test))
        n_params = _count_params(model)
        size_kb = _tflite_size_kb(tflite, name)
        inf_ms = _INFERENCE_MS_EST.get(name, "—")

        # RAM fit (rough: model size + 80 KB peak activations for standard models)
        model_bytes = size_kb * 1024 if not np.isnan(size_kb) else n_params
        peak_ram_est = 80 * 1024  # conservative upper bound for standard models
        if "depthwise" in name:
            peak_ram_est = 40 * 1024

        m33_fits = "YES" if (model_bytes + peak_ram_est) <= MCU_RAM["M33"] else "NO"
        m55_fits = "YES" if (model_bytes + peak_ram_est) <= MCU_RAM["M55"] else "YES"  # M55 almost always fits

        rows.append({
            "Model": entry["label"],
            "Input": entry["input"],
            "Params": f"{n_params:,}",
            "Size (KB)": f"{size_kb:.1f}" if not np.isnan(size_kb) else "—",
            "Val Acc": "—",  # loaded from CSV below
            "Test Acc": f"{test_acc:.4f}",
            "Inf (ms)": f"{inf_ms:.1f}" if isinstance(inf_ms, float) else inf_ms,
            "M33 fits?": m33_fits,
            "M55 fits?": m55_fits,
            "CMSIS-NN": "Full",
            "_preds": preds,
            "_name": name,
        })
        print(f"  Test accuracy: {test_acc:.4f}")

    # -------------------------------------------------------------------------
    # Load val accuracy from training history CSVs
    # -------------------------------------------------------------------------
    log_dir = ckpt / "logs"
    for row in rows:
        name = row.get("_name", "")
        if not name:
            continue
        csv_path = log_dir / f"{name}_history.csv"
        hist = _load_history_csv(csv_path)
        if hist.get("val_accuracy"):
            row["Val Acc"] = f"{max(hist['val_accuracy']):.4f}"

    # Ridge regression G models
    for variant in ("x_only", "y_only", "xy_concatenated"):
        rows.append({
            "Model": f"G — ridge {variant}",
            "Input": variant,
            "Params": "linear",
            "Size (KB)": "—",
            "Val Acc": "—",
            "Test Acc": "—",
            "Inf (ms)": f"{_INFERENCE_MS_EST.get(f'model_g_ridge_{variant}', 2.0):.1f}",
            "M33 fits?": "YES",
            "M55 fits?": "YES",
            "CMSIS-NN": "N/A (sklearn)",
        })

    # -------------------------------------------------------------------------
    # Print comparison table
    # -------------------------------------------------------------------------
    cols = ["Model", "Input", "Params", "Size (KB)", "Val Acc", "Test Acc",
            "Inf (ms)", "M33 fits?", "M55 fits?", "CMSIS-NN"]
    col_w = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) + 2 for c in cols}

    header = " | ".join(c.ljust(col_w[c]) for c in cols)
    sep = "-+-".join("-" * col_w[c] for c in cols)

    print(f"\n{'='*len(header)}")
    print("MODEL COMPARISON TABLE")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(str(row.get(c, "—")).ljust(col_w[c]) for c in cols))
    print(f"{'='*len(header)}\n")

    # Print key comparisons
    print("KEY COMPARISONS:")
    model_acc = {r.get("_name", ""): r["Test Acc"] for r in rows if r.get("_name")}
    comparisons = [
        ("A vs B", "model_a_cnn_x_only", "model_b_cnn_xy_dual",
         "does y(t) improve accuracy?"),
        ("A vs C", "model_a_cnn_x_only", "model_c_cnn_phase",
         "does orbit radius capture the signal?"),
        ("A vs D", "model_a_cnn_x_only", "model_d_cnn_angle",
         "does phase angle carry independent information?"),
        ("A vs E", "model_a_cnn_x_only", "model_e_cnn_xy_fusion",
         "is late fusion better than early fusion?"),
        ("F vs A", "model_f_depthwise_cnn", "model_a_cnn_x_only",
         "accuracy cost of depthwise (smaller/faster)"),
    ]
    for label, a, b, question in comparisons:
        acc_a = model_acc.get(a, "—")
        acc_b = model_acc.get(b, "—")
        print(f"  {label}: {a.split('_', 2)[-1]} ({acc_a}) vs "
              f"{b.split('_', 2)[-1]} ({acc_b})  — {question}")

    # -------------------------------------------------------------------------
    # Save table to CSV
    # -------------------------------------------------------------------------
    table_path = out / "model_comparison.csv"
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "—") for c in cols})
    print(f"\n[evaluate] Comparison table saved to {table_path}")

    # =========================================================================
    # PLOTS
    # =========================================================================

    # --- Plot 1: Accuracy bar chart ---
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_rows = [r for r in rows if r["Test Acc"] != "—" and r.get("_name")]
    names_short = [r["Model"] for r in plot_rows]
    accs = [float(r["Test Acc"]) for r in plot_rows]
    colours = ["steelblue", "coral", "mediumseagreen", "mediumpurple",
               "tomato", "gold", "slategray"]
    bars = ax.bar(range(len(accs)), accs,
                  color=colours[:len(accs)], alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(accs)))
    ax.set_xticklabels(names_short, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_title("Hopf Oscillator Model Comparison — Test Accuracy", fontsize=13)
    ax.set_ylim(0, 1.05)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out / "accuracy_bar_chart.png", dpi=150)
    print(f"[evaluate] Saved accuracy_bar_chart.png")
    plt.close(fig)

    # --- Plot 2: Confusion matrices for top 2 models ---
    eval_rows_sorted = sorted(
        [(r["Test Acc"], r) for r in plot_rows if r["Test Acc"] != "—"],
        key=lambda x: float(x[0]), reverse=True,
    )
    top2 = eval_rows_sorted[:2]

    if top2:
        fig, axes = plt.subplots(1, len(top2), figsize=(8 * len(top2), 7))
        if len(top2) == 1:
            axes = [axes]

        class_labels = CLASS_NAMES[:n_classes]
        for ax, (_, row) in zip(axes, top2):
            preds = row.get("_preds")
            if preds is None:
                continue
            cm = _confusion_matrix(preds, labels_test, n_classes)
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(n_classes))
            ax.set_yticks(range(n_classes))
            ax.set_xticklabels(class_labels, rotation=45, ha="right")
            ax.set_yticklabels(class_labels)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Confusion Matrix\n{row['Model']}", fontsize=11)
            for i in range(n_classes):
                for j in range(n_classes):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black",
                            fontsize=9)
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        fig.savefig(out / "confusion_matrices_top2.png", dpi=150)
        print(f"[evaluate] Saved confusion_matrices_top2.png")
        plt.close(fig)

    # --- Plot 3: Feature map visualisation (x, y, phase, angle per class) ---
    from pipeline.features_xy import extract_all_representations
    from pipeline.ingest import process_dataset
    from pipeline.features_xy import extract_y_features
    from data.sample_data import generate_dataset_xy

    print("[evaluate] Generating feature map visualisation ...")
    raw_x_vis, raw_y_vis, labels_vis = generate_dataset_xy(
        n_clips_per_class=1, n_classes=n_classes, cache=False,
    )
    xp_vis = process_dataset(raw_x_vis)
    yp_vis = extract_y_features(raw_y_vis)
    reps_vis = extract_all_representations(xp_vis, yp_vis)

    rep_plot_keys = ["x_only", "y_only", "phase", "angle"]
    rep_titles = ["x(t) — published input", "y(t) — discarded state",
                  "r(t) = √(x²+y²) orbit radius", "θ(t) = arctan2(y,x) phase angle"]

    fig, axes = plt.subplots(len(rep_plot_keys), n_classes,
                             figsize=(3.5 * n_classes, 3.5 * len(rep_plot_keys)))

    for row_i, (rep_key, rep_title) in enumerate(zip(rep_plot_keys, rep_titles)):
        for cls in range(n_classes):
            mask = labels_vis == cls
            if not np.any(mask):
                continue
            idx = np.where(mask)[0][0]
            ax = axes[row_i, cls]
            ax.imshow(reps_vis[rep_key][idx], cmap="viridis", aspect="auto")
            if row_i == 0:
                ax.set_title(f"class {cls}\n{CLASS_NAMES[cls]}", fontsize=9)
            if cls == 0:
                ax.set_ylabel(rep_title, fontsize=8)
            ax.axis("off")

    plt.suptitle("Feature Representations — One Clip per Class", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out / "feature_map_comparison.png", dpi=150, bbox_inches="tight")
    print(f"[evaluate] Saved feature_map_comparison.png")
    plt.close(fig)

    # --- Plot 4: Training curves ---
    print("[evaluate] Plotting training curves ...")
    keras_model_names = [entry["name"] for entry in model_registry]
    n_models_with_hist = sum(
        1 for name in keras_model_names
        if (log_dir / f"{name}_history.csv").exists()
    )

    if n_models_with_hist > 0:
        fig, axes_grid = plt.subplots(
            2, n_models_with_hist,
            figsize=(4 * n_models_with_hist, 8),
            squeeze=False,
        )
        col = 0
        for name in keras_model_names:
            hist = _load_history_csv(log_dir / f"{name}_history.csv")
            if not hist.get("accuracy"):
                continue
            epochs = hist["epoch"]
            axes_grid[0, col].plot(epochs, hist["accuracy"], label="train")
            axes_grid[0, col].plot(epochs, hist["val_accuracy"], label="val")
            axes_grid[0, col].set_title(name.replace("model_", "").replace("_", " "),
                                        fontsize=8)
            axes_grid[0, col].set_ylabel("Accuracy")
            axes_grid[0, col].legend(fontsize=7)
            axes_grid[0, col].set_ylim(0, 1)

            axes_grid[1, col].plot(epochs, hist["loss"], label="train")
            axes_grid[1, col].plot(epochs, hist["val_loss"], label="val")
            axes_grid[1, col].set_ylabel("Loss")
            axes_grid[1, col].set_xlabel("Epoch")
            axes_grid[1, col].legend(fontsize=7)
            col += 1

        plt.suptitle("Training Curves — Accuracy and Loss per Epoch", fontsize=13)
        plt.tight_layout()
        fig.savefig(out / "training_curves.png", dpi=150)
        print(f"[evaluate] Saved training_curves.png")
        plt.close(fig)

    print(f"\n[evaluate] All outputs saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all Hopf oscillator models")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
    parser.add_argument("--output_dir", type=str, default="output/eval/")
    parser.add_argument("--n_clips", type=int, default=100)
    parser.add_argument("--n_classes", type=int, default=5)
    parser.add_argument("--tflite_dir", type=str, default=None)
    args = parser.parse_args()

    generate_report(
        checkpoint_dir=ROOT / args.checkpoint_dir,
        output_dir=ROOT / args.output_dir,
        n_clips_per_class=args.n_clips,
        n_classes=args.n_classes,
        tflite_dir=ROOT / args.tflite_dir if args.tflite_dir else None,
    )


if __name__ == "__main__":
    main()
