"""
Train all Hopf oscillator models across all input representations.

Loads (or generates) the full x(t)/y(t) dataset, computes all five input
representations, and trains each model with quantisation-aware training (QAT).
Results — per-epoch training/validation accuracy and loss — are logged to CSV.

Models trained:
  A — cnn_x_only         input: x_only
  B — cnn_xy_dual        input: xy_dual
  C — cnn_phase          input: phase
  D — cnn_angle          input: angle
  E — cnn_xy_fusion      input: (x_only, y_only) — dual-input
  F — depthwise_cnn      input: xy_dual
  G — reservoir_readout  input: x_only, y_only, xy_concat — sklearn, no QAT

Usage:
    python -m pipeline.train_all
    python -m pipeline.train_all --epochs 50 --n_clips 50

Config via TrainConfig dataclass — all hyperparameters in one place.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@dataclass
class TrainConfig:
    """All training hyperparameters in one place."""
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    val_split: float = 0.2
    n_classes: int = 5
    use_qat: bool = True
    save_dir: str = "checkpoints/"
    n_clips_per_class: int = 100
    cache_data: bool = True
    ridge_alpha: float = 0.01


def _prepare_split(
    features: NDArray,
    labels: NDArray[np.int64],
    val_split: float,
    n_classes: int,
    expand_channel: bool = True,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Split feature maps into train/val sets and prepare for Keras.

    Args:
        features: (n, 200, 100) uint8 or (n, 200, 100, 2) uint8
        labels: (n,) integer class labels
        val_split: fraction of data to use for validation
        n_classes: number of output classes
        expand_channel: if True and features.ndim == 3, add channel dim

    Returns:
        x_train, y_train, x_val, y_val (float32)
    """
    import tensorflow as tf
    from tensorflow import keras

    x = features.astype(np.float32)
    if expand_channel and x.ndim == 3:
        x = np.expand_dims(x, axis=-1)  # (n, 200, 100, 1)

    y = keras.utils.to_categorical(labels, num_classes=n_classes)

    n = len(x)
    n_val = int(n * val_split)
    idx = np.random.default_rng(0).permutation(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def _apply_qat(model):
    """Attempt QAT; fall back to base model if tfmot not installed."""
    try:
        import tensorflow_model_optimization as tfmot
        qat_model = tfmot.quantization.keras.quantize_model(model)
        print("  [QAT] Applied quantisation-aware training")
        return qat_model
    except ImportError:
        print("  [QAT] tensorflow-model-optimization not installed — training without QAT")
        return model


def _train_keras_model(
    model,
    x_train: NDArray,
    y_train: NDArray,
    x_val: NDArray,
    y_val: NDArray,
    cfg: TrainConfig,
    ckpt_path: Path,
) -> tuple:
    """Compile, train, and checkpoint a Keras model. Returns (history, model)."""
    import tensorflow as tf
    from tensorflow import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0,
        )
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return history, model


def _save_history_csv(
    history,
    csv_path: Path,
    model_name: str,
) -> None:
    """Write per-epoch metrics to CSV for later evaluation."""
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "epoch", "loss",
                                               "accuracy", "val_loss", "val_accuracy"])
        writer.writeheader()
        for ep in epochs:
            writer.writerow({
                "model": model_name,
                "epoch": ep,
                "loss": hist["loss"][ep - 1],
                "accuracy": hist["accuracy"][ep - 1],
                "val_loss": hist["val_loss"][ep - 1],
                "val_accuracy": hist["val_accuracy"][ep - 1],
            })
    print(f"  [train_all] History saved to {csv_path}")


def train_all(cfg: TrainConfig) -> dict[str, dict]:
    """
    Train all models and return a results dict.

    Returns:
        dict mapping model_name → {
            "val_acc": float,
            "history_csv": Path,
            "checkpoint": Path,
        }
    """
    import tensorflow as tf
    from data.sample_data import generate_dataset_xy
    from pipeline.ingest import process_dataset
    from pipeline.features_xy import extract_y_features, extract_all_representations
    from pipeline.models.cnn_x_only import build_model as build_a
    from pipeline.models.cnn_xy_dual import build_model as build_b
    from pipeline.models.cnn_phase import build_model as build_c
    from pipeline.models.cnn_angle import build_model as build_d
    from pipeline.models.cnn_xy_fusion import build_model as build_e
    from pipeline.models.depthwise_cnn import build_model as build_f
    from pipeline.models.reservoir_readout import fit_all_readouts

    save_dir = ROOT / cfg.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = save_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Generate / load data
    # -------------------------------------------------------------------------
    print("\n[train_all] Loading dataset ...")
    raw_x, raw_y, labels = generate_dataset_xy(
        n_clips_per_class=cfg.n_clips_per_class,
        n_classes=cfg.n_classes,
        cache=cfg.cache_data,
    )
    print(f"  raw_x: {raw_x.shape}, raw_y: {raw_y.shape}, labels: {labels.shape}")

    # -------------------------------------------------------------------------
    # 2. Ingest: downsample → normalise → atanh → reshape
    # -------------------------------------------------------------------------
    print("[train_all] Processing x(t) ...")
    x_processed = process_dataset(raw_x)

    print("[train_all] Processing y(t) ...")
    y_processed = extract_y_features(raw_y)

    # -------------------------------------------------------------------------
    # 3. Build all input representations
    # -------------------------------------------------------------------------
    print("[train_all] Building input representations ...")
    reps = extract_all_representations(x_processed, y_processed)
    # reps: {"x_only", "y_only", "xy_dual", "phase", "angle"} — all uint8

    # Train/val split indices (shared across all models for fair comparison)
    n = len(labels)
    n_val = int(n * cfg.val_split)
    rng_idx = np.random.default_rng(0).permutation(n)
    val_idx, train_idx = rng_idx[:n_val], rng_idx[n_val:]
    labels_train, labels_val = labels[train_idx], labels[val_idx]

    results: dict[str, dict] = {}

    # -------------------------------------------------------------------------
    # Helper: split + train a single-input Sequential model
    # -------------------------------------------------------------------------
    def run_sequential(model_name: str, build_fn, rep_key: str) -> None:
        print(f"\n{'='*60}")
        print(f"[train_all] Training {model_name} (input: {rep_key})")
        print(f"{'='*60}")

        x_tr, y_tr, x_vl, y_vl = _prepare_split(
            reps[rep_key], labels, cfg.val_split, cfg.n_classes, expand_channel=True,
        )
        model = build_fn(n_classes=cfg.n_classes)
        if cfg.use_qat:
            model = _apply_qat(model)

        ckpt = save_dir / f"{model_name}_best.keras"
        history, _ = _train_keras_model(model, x_tr, y_tr, x_vl, y_vl, cfg, ckpt)
        _save_history_csv(history, log_dir / f"{model_name}_history.csv", model_name)

        val_acc = max(history.history["val_accuracy"])
        print(f"  Best val_accuracy: {val_acc:.4f}")
        results[model_name] = {
            "val_acc": val_acc,
            "history_csv": log_dir / f"{model_name}_history.csv",
            "checkpoint": ckpt,
        }

    # -------------------------------------------------------------------------
    # Model A — x(t) only baseline
    # -------------------------------------------------------------------------
    run_sequential("model_a_cnn_x_only", build_a, "x_only")

    # -------------------------------------------------------------------------
    # Model B — two-channel x+y
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"[train_all] Training model_b_cnn_xy_dual (input: xy_dual)")
    print(f"{'='*60}")
    x_tr_b, y_tr_b, x_vl_b, y_vl_b = _prepare_split(
        reps["xy_dual"], labels, cfg.val_split, cfg.n_classes, expand_channel=False,
    )
    model_b = build_b(n_classes=cfg.n_classes)
    if cfg.use_qat:
        model_b = _apply_qat(model_b)
    ckpt_b = save_dir / "model_b_cnn_xy_dual_best.keras"
    history_b, _ = _train_keras_model(model_b, x_tr_b, y_tr_b, x_vl_b, y_vl_b, cfg, ckpt_b)
    _save_history_csv(history_b, log_dir / "model_b_cnn_xy_dual_history.csv", "model_b_cnn_xy_dual")
    val_acc_b = max(history_b.history["val_accuracy"])
    print(f"  Best val_accuracy: {val_acc_b:.4f}")
    results["model_b_cnn_xy_dual"] = {
        "val_acc": val_acc_b,
        "history_csv": log_dir / "model_b_cnn_xy_dual_history.csv",
        "checkpoint": ckpt_b,
    }

    # -------------------------------------------------------------------------
    # Model C — orbit radius (phase)
    # -------------------------------------------------------------------------
    run_sequential("model_c_cnn_phase", build_c, "phase")

    # -------------------------------------------------------------------------
    # Model D — phase angle
    # -------------------------------------------------------------------------
    run_sequential("model_d_cnn_angle", build_d, "angle")

    # -------------------------------------------------------------------------
    # Model E — late fusion (dual-input)
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"[train_all] Training model_e_cnn_xy_fusion (dual-input)")
    print(f"{'='*60}")
    import tensorflow as tf
    from tensorflow import keras

    x_only_f32 = reps["x_only"].astype(np.float32)
    y_only_f32 = reps["y_only"].astype(np.float32)
    x_only_f32 = np.expand_dims(x_only_f32, axis=-1)
    y_only_f32 = np.expand_dims(y_only_f32, axis=-1)

    y_cat = keras.utils.to_categorical(labels, num_classes=cfg.n_classes)
    x_tr_e_x = x_only_f32[train_idx]
    x_tr_e_y = y_only_f32[train_idx]
    y_tr_e = y_cat[train_idx]
    x_vl_e_x = x_only_f32[val_idx]
    x_vl_e_y = y_only_f32[val_idx]
    y_vl_e = y_cat[val_idx]

    model_e = build_e(n_classes=cfg.n_classes)
    if cfg.use_qat:
        model_e = _apply_qat(model_e)
    model_e.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    ckpt_e = save_dir / "model_e_cnn_xy_fusion_best.keras"
    callbacks_e = [keras.callbacks.ModelCheckpoint(
        filepath=str(ckpt_e), monitor="val_accuracy", save_best_only=True, verbose=0,
    )]
    history_e = model_e.fit(
        [x_tr_e_x, x_tr_e_y], y_tr_e,
        validation_data=([x_vl_e_x, x_vl_e_y], y_vl_e),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks_e,
        verbose=1,
    )
    _save_history_csv(history_e, log_dir / "model_e_cnn_xy_fusion_history.csv", "model_e_cnn_xy_fusion")
    val_acc_e = max(history_e.history["val_accuracy"])
    print(f"  Best val_accuracy: {val_acc_e:.4f}")
    results["model_e_cnn_xy_fusion"] = {
        "val_acc": val_acc_e,
        "history_csv": log_dir / "model_e_cnn_xy_fusion_history.csv",
        "checkpoint": ckpt_e,
    }

    # -------------------------------------------------------------------------
    # Model F — depthwise separable CNN
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"[train_all] Training model_f_depthwise_cnn (input: xy_dual)")
    print(f"{'='*60}")
    x_tr_f, y_tr_f, x_vl_f, y_vl_f = _prepare_split(
        reps["xy_dual"], labels, cfg.val_split, cfg.n_classes, expand_channel=False,
    )
    model_f = build_f(n_classes=cfg.n_classes)
    if cfg.use_qat:
        model_f = _apply_qat(model_f)
    ckpt_f = save_dir / "model_f_depthwise_cnn_best.keras"
    history_f, _ = _train_keras_model(model_f, x_tr_f, y_tr_f, x_vl_f, y_vl_f, cfg, ckpt_f)
    _save_history_csv(history_f, log_dir / "model_f_depthwise_cnn_history.csv", "model_f_depthwise_cnn")
    val_acc_f = max(history_f.history["val_accuracy"])
    print(f"  Best val_accuracy: {val_acc_f:.4f}")
    results["model_f_depthwise_cnn"] = {
        "val_acc": val_acc_f,
        "history_csv": log_dir / "model_f_depthwise_cnn_history.csv",
        "checkpoint": ckpt_f,
    }

    # -------------------------------------------------------------------------
    # Model G — ridge regression readout (no QAT, no epochs)
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"[train_all] Training model_g_reservoir_readout (ridge regression)")
    print(f"{'='*60}")
    ridge_results = fit_all_readouts(
        x_features=reps["x_only"][train_idx],
        y_features=reps["y_only"][train_idx],
        labels_train=labels_train,
        x_val=reps["x_only"][val_idx],
        y_val=reps["y_only"][val_idx],
        labels_val=labels_val,
        alpha=cfg.ridge_alpha,
    )
    for variant, res in ridge_results.items():
        key = f"model_g_ridge_{variant}"
        results[key] = {
            "val_acc": res["val_acc"],
            "history_csv": None,
            "checkpoint": None,
            "model_obj": res["model"],
        }

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<35s} {'Val Accuracy':>14s}")
    print(f"  {'-'*50}")
    for name, res in results.items():
        print(f"  {name:<35s} {res['val_acc']:>14.4f}")
    print(f"{'='*60}\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all Hopf oscillator models")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_clips", type=int, default=100)
    parser.add_argument("--n_classes", type=int, default=5)
    parser.add_argument("--no_qat", action="store_true")
    parser.add_argument("--save_dir", type=str, default="checkpoints/")
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_classes=args.n_classes,
        use_qat=not args.no_qat,
        save_dir=args.save_dir,
        n_clips_per_class=args.n_clips,
    )
    print("[train_all] Config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    train_all(cfg)


if __name__ == "__main__":
    main()
