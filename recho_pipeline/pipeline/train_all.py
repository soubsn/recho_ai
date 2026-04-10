"""
Train all Hopf oscillator models across all input representations.

Loads (or generates) the full x(t)/y(t) dataset, computes all input
representations, and trains every model in the zoo.

Results are saved to results/model_comparison.csv with columns:
    Model | Category | Input | Accuracy | F1 | Train_time_s |
    Inference_ms_est | Model_size_KB | RAM_KB | M4 | M33 | M55 | M85

Models grouped by category:
    === CLASSICAL SIGNAL METHODS ===    (no neural network, all arithmetic)
    === ML CLASSIFIERS ===              (sklearn, PCA + feature maps)
    === ANOMALY DETECTORS ===           (unsupervised, no fault labels needed)
    === SEQUENCE MODELS ===             (raw time series input)
    === FEW-SHOT METHODS ===            (1-5 examples per class)

Usage:
    python -m pipeline.train_all
    python -m pipeline.train_all --epochs 50 --n_clips 50

Config via TrainConfig dataclass — all hyperparameters in one place.

References:
  "Recognising sound signals with a Hopf physical reservoir computer"
  (Shougat et al., Scientific Reports 2021) — paper 1
  "Hopf physical reservoir computer for reconfigurable sound recognition"
  (Shougat et al., Scientific Reports 2023) — paper 2
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"


@dataclass
class ModelTrainConfig:
    """Per-model metadata used in the comparison table."""
    model_name: str
    category: str
    input_type: str   # 'x_only'|'y_only'|'xy_dual'|'phase'|'angle'|
                      # 'raw_x'|'raw_xy'|'handcrafted'
    requires_normal_only: bool = False  # True for unsupervised anomaly detectors
    target_chips: list = field(default_factory=lambda: ["M4", "M33", "M55", "M85"])
    estimated_ram_kb: int = 64
    estimated_inference_ms: float = 5.0


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
    # New model toggles — set False to skip heavy models during development
    run_classical: bool = True
    run_ml: bool = True
    run_anomaly: bool = True
    run_sequence: bool = True
    run_fewshot: bool = True
    run_keras: bool = True  # run original CNN models A-F


# ─────────────────────────────────────────────────────────────────────────────
# Model registry — one entry per model
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY: list[ModelTrainConfig] = [
    # --- CLASSICAL ---
    ModelTrainConfig("spc_monitor",         "Classical", "raw_x",      requires_normal_only=True,  target_chips=["M4", "M33", "M55", "M85"], estimated_ram_kb=1,   estimated_inference_ms=0.001),
    ModelTrainConfig("phase_portrait",      "Classical", "raw_xy",     requires_normal_only=False, target_chips=["M4", "M33", "M55", "M85"], estimated_ram_kb=4,   estimated_inference_ms=0.1),
    ModelTrainConfig("recurrence",          "Classical", "raw_x",      requires_normal_only=False, target_chips=["M55", "M85"],   estimated_ram_kb=8,   estimated_inference_ms=50.0),
    ModelTrainConfig("hilbert",             "Classical", "raw_x",      requires_normal_only=False, target_chips=["M4", "M33", "M55", "M85"], estimated_ram_kb=8,   estimated_inference_ms=0.5),
    ModelTrainConfig("autocorrelation",     "Classical", "raw_x",      requires_normal_only=False, target_chips=["M4", "M33", "M55", "M85"], estimated_ram_kb=8,   estimated_inference_ms=1.0),
    # --- ML CLASSIFIERS ---
    ModelTrainConfig("svm_x_only",          "ML",        "x_only",     requires_normal_only=False, target_chips=["M85"],           estimated_ram_kb=16,  estimated_inference_ms=0.5),
    ModelTrainConfig("random_forest",       "ML",        "handcrafted",requires_normal_only=False, target_chips=["M4", "M33", "M55", "M85"], estimated_ram_kb=8,   estimated_inference_ms=0.2),
    ModelTrainConfig("knn",                 "ML",        "x_only",     requires_normal_only=False, target_chips=["M4", "M33", "M55", "M85"], estimated_ram_kb=12,  estimated_inference_ms=0.3),
    # --- ANOMALY DETECTORS ---
    ModelTrainConfig("gmm_anomaly",         "Anomaly",   "x_only",     requires_normal_only=True,  target_chips=["M55", "M85"],   estimated_ram_kb=20,  estimated_inference_ms=1.0),
    ModelTrainConfig("isolation_forest",    "Anomaly",   "handcrafted",requires_normal_only=False, target_chips=["M4", "M33", "M55", "M85"], estimated_ram_kb=8,   estimated_inference_ms=0.2),
    ModelTrainConfig("one_class_svm",       "Anomaly",   "x_only",     requires_normal_only=True,  target_chips=["M55", "M85"],   estimated_ram_kb=16,  estimated_inference_ms=0.5),
    ModelTrainConfig("autoencoder",         "Anomaly",   "x_only",     requires_normal_only=True,  target_chips=["M55", "M85"],   estimated_ram_kb=64,  estimated_inference_ms=20.0),
    ModelTrainConfig("vae",                 "Anomaly",   "x_only",     requires_normal_only=True,  target_chips=["M85"],           estimated_ram_kb=128, estimated_inference_ms=40.0),
    # --- KERAS CNN MODELS ---
    ModelTrainConfig("model_a_cnn_x_only",  "ML",        "x_only",     requires_normal_only=False, target_chips=["M55", "M85"],   estimated_ram_kb=80,  estimated_inference_ms=45.0),
    ModelTrainConfig("model_b_cnn_xy_dual", "ML",        "xy_dual",    requires_normal_only=False, target_chips=["M55", "M85"],   estimated_ram_kb=80,  estimated_inference_ms=48.0),
    ModelTrainConfig("model_c_cnn_phase",   "ML",        "phase",      requires_normal_only=False, target_chips=["M55", "M85"],   estimated_ram_kb=80,  estimated_inference_ms=45.0),
    ModelTrainConfig("model_d_cnn_angle",   "ML",        "angle",      requires_normal_only=False, target_chips=["M55", "M85"],   estimated_ram_kb=80,  estimated_inference_ms=45.0),
    ModelTrainConfig("model_e_cnn_fusion",  "ML",        "xy_dual",    requires_normal_only=False, target_chips=["M55", "M85"],   estimated_ram_kb=160, estimated_inference_ms=90.0),
    ModelTrainConfig("model_f_depthwise",   "ML",        "xy_dual",    requires_normal_only=False, target_chips=["M33", "M55", "M85"], estimated_ram_kb=40,  estimated_inference_ms=6.0),
    ModelTrainConfig("model_g_ridge",       "ML",        "x_only",     requires_normal_only=False, target_chips=["M33", "M55", "M85"], estimated_ram_kb=4,   estimated_inference_ms=2.0),
    # --- SEQUENCE MODELS ---
    ModelTrainConfig("tcn",                 "Sequence",  "raw_x",      requires_normal_only=False, target_chips=["M55", "M85"],   estimated_ram_kb=48,  estimated_inference_ms=15.0),
    ModelTrainConfig("lstm",                "Sequence",  "raw_xy",     requires_normal_only=False, target_chips=["M85"],           estimated_ram_kb=128, estimated_inference_ms=60.0),
    ModelTrainConfig("esn",                 "Sequence",  "x_only",     requires_normal_only=False, target_chips=["M4", "M33", "M55", "M85"], estimated_ram_kb=32,  estimated_inference_ms=5.0),
    # --- FEW-SHOT ---
    ModelTrainConfig("contrastive",         "FewShot",   "x_only",     requires_normal_only=False, target_chips=["M85"],           estimated_ram_kb=80,  estimated_inference_ms=45.0),
    ModelTrainConfig("prototypical",        "FewShot",   "x_only",     requires_normal_only=False, target_chips=["M4", "M33", "M55", "M85"], estimated_ram_kb=16,  estimated_inference_ms=0.5),
]


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (unchanged from original train_all.py)
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_split(
    features: NDArray,
    labels: NDArray[np.int64],
    val_split: float,
    n_classes: int,
    expand_channel: bool = True,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Split feature maps into train/val sets and prepare for Keras."""
    import tensorflow as tf
    from tensorflow import keras

    x = features.astype(np.float32)
    if expand_channel and x.ndim == 3:
        x = np.expand_dims(x, axis=-1)
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
        print("  [QAT] tensorflow-model-optimization not installed — skipping QAT")
        return model


def _train_keras_model(model, x_train, y_train, x_val, y_val, cfg, ckpt_path):
    """Compile, train, and checkpoint a Keras model."""
    import tensorflow as tf
    from tensorflow import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(ckpt_path), monitor="val_accuracy",
            save_best_only=True, verbose=0,
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


def _save_history_csv(history, csv_path: Path, model_name: str) -> None:
    """Write per-epoch metrics to CSV."""
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "epoch", "loss", "accuracy",
                           "val_loss", "val_accuracy"],
        )
        writer.writeheader()
        for ep in epochs:
            writer.writerow({
                "model": model_name, "epoch": ep,
                "loss": hist["loss"][ep - 1],
                "accuracy": hist["accuracy"][ep - 1],
                "val_loss": hist["val_loss"][ep - 1],
                "val_accuracy": hist["val_accuracy"][ep - 1],
            })
    print(f"  [train_all] History saved to {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_all(cfg: TrainConfig) -> dict[str, dict]:
    """
    Train all models and return a results dict.

    Returns:
        dict mapping model_name → {
            "val_acc": float,
            "category": str,
            "history_csv": Path or None,
            "checkpoint": Path or None,
        }
    """
    import tensorflow as tf
    from sklearn.metrics import f1_score as sklearn_f1

    from data.sample_data import generate_dataset_xy, CLASS_NAMES
    from pipeline.ingest import process_dataset
    from pipeline.features_xy import extract_y_features, extract_all_representations

    save_dir = ROOT / cfg.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = save_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
    # 2. Ingest: processed feature maps
    # -------------------------------------------------------------------------
    print("[train_all] Processing x(t) ...")
    x_processed = process_dataset(raw_x)
    print("[train_all] Processing y(t) ...")
    y_processed = extract_y_features(raw_y)

    # -------------------------------------------------------------------------
    # 3. All input representations
    # -------------------------------------------------------------------------
    print("[train_all] Building input representations ...")
    reps = extract_all_representations(x_processed, y_processed)
    # reps: {"x_only", "y_only", "xy_dual", "phase", "angle"} — all uint8

    # Downsampled raw signals (for classical and sequence models)
    x_ds = raw_x[:, ::25].astype(np.float64)   # (n, 4000) at 4 kHz
    y_ds = raw_y[:, ::25].astype(np.float64)

    # Train/val split (shared index for fair comparison)
    n = len(labels)
    n_val = int(n * cfg.val_split)
    rng_idx = np.random.default_rng(0).permutation(n)
    val_idx, train_idx = rng_idx[:n_val], rng_idx[n_val:]
    labels_train, labels_val = labels[train_idx], labels[val_idx]

    # Normal-only subset (class 0 training clips — for unsupervised detectors)
    normal_train_mask = labels_train == 0
    normal_clips_x = reps["x_only"][train_idx][normal_train_mask]

    results: dict[str, dict] = {}

    def _record(name: str, category: str, val_acc: float,
                ckpt: Optional[Path] = None, hist_csv: Optional[Path] = None,
                train_time: float = 0.0) -> None:
        results[name] = {
            "val_acc": val_acc,
            "category": category,
            "checkpoint": ckpt,
            "history_csv": hist_csv,
            "train_time_s": train_time,
        }

    # =========================================================================
    # === ORIGINAL KERAS CNN MODELS (A-G) ===
    # =========================================================================
    if cfg.run_keras:
        from pipeline.models.cnn_x_only import build_model as build_a
        from pipeline.models.cnn_xy_dual import build_model as build_b
        from pipeline.models.cnn_phase import build_model as build_c
        from pipeline.models.cnn_angle import build_model as build_d
        from pipeline.models.cnn_xy_fusion import build_model as build_e
        from pipeline.models.depthwise_cnn import build_model as build_f
        from pipeline.models.reservoir_readout import fit_all_readouts

        def _run_seq(model_name: str, build_fn, rep_key: str,
                     expand: bool = True) -> None:
            print(f"\n{'='*60}\n[train_all] {model_name} (input: {rep_key})\n{'='*60}")
            x_tr, y_tr, x_vl, y_vl = _prepare_split(
                reps[rep_key], labels, cfg.val_split, cfg.n_classes,
                expand_channel=expand,
            )
            model = build_fn(n_classes=cfg.n_classes)
            if cfg.use_qat:
                model = _apply_qat(model)
            ckpt = save_dir / f"{model_name}_best.keras"
            t0 = time.time()
            history, _ = _train_keras_model(
                model, x_tr, y_tr, x_vl, y_vl, cfg, ckpt,
            )
            t1 = time.time()
            _save_history_csv(history, log_dir / f"{model_name}_history.csv", model_name)
            val_acc = max(history.history["val_accuracy"])
            print(f"  Best val_accuracy: {val_acc:.4f}")
            _record(model_name, "ML", val_acc, ckpt,
                    log_dir / f"{model_name}_history.csv", t1 - t0)

        _run_seq("model_a_cnn_x_only", build_a, "x_only")
        _run_seq("model_b_cnn_xy_dual", build_b, "xy_dual", expand=False)
        _run_seq("model_c_cnn_phase", build_c, "phase")
        _run_seq("model_d_cnn_angle", build_d, "angle")

        # Model E — late fusion (dual input)
        print(f"\n{'='*60}\n[train_all] model_e_cnn_xy_fusion\n{'='*60}")
        from tensorflow import keras as keras_tf
        x_x = np.expand_dims(reps["x_only"].astype(np.float32), -1)
        x_y = np.expand_dims(reps["y_only"].astype(np.float32), -1)
        y_cat = keras_tf.utils.to_categorical(labels, cfg.n_classes)
        x_tr_e = [x_x[train_idx], x_y[train_idx]]
        x_vl_e = [x_x[val_idx], x_y[val_idx]]
        model_e = build_e(n_classes=cfg.n_classes)
        if cfg.use_qat:
            model_e = _apply_qat(model_e)
        model_e.compile(
            optimizer=keras_tf.optimizers.Adam(cfg.learning_rate),
            loss="categorical_crossentropy", metrics=["accuracy"],
        )
        ckpt_e = save_dir / "model_e_cnn_xy_fusion_best.keras"
        t0 = time.time()
        hist_e = model_e.fit(
            x_tr_e, y_cat[train_idx],
            validation_data=(x_vl_e, y_cat[val_idx]),
            epochs=cfg.epochs, batch_size=cfg.batch_size,
            callbacks=[keras_tf.callbacks.ModelCheckpoint(
                str(ckpt_e), monitor="val_accuracy", save_best_only=True)],
            verbose=1,
        )
        t1 = time.time()
        _save_history_csv(hist_e, log_dir / "model_e_cnn_xy_fusion_history.csv",
                          "model_e_cnn_xy_fusion")
        _record("model_e_cnn_xy_fusion", "ML",
                max(hist_e.history["val_accuracy"]), ckpt_e,
                log_dir / "model_e_cnn_xy_fusion_history.csv", t1 - t0)

        _run_seq("model_f_depthwise_cnn", build_f, "xy_dual", expand=False)

        # Model G — ridge regression
        print(f"\n{'='*60}\n[train_all] model_g_reservoir_readout\n{'='*60}")
        t0 = time.time()
        ridge_res = fit_all_readouts(
            x_features=reps["x_only"][train_idx],
            y_features=reps["y_only"][train_idx],
            labels_train=labels_train,
            x_val=reps["x_only"][val_idx],
            y_val=reps["y_only"][val_idx],
            labels_val=labels_val,
            alpha=cfg.ridge_alpha,
        )
        t1 = time.time()
        for variant, res in ridge_res.items():
            _record(f"model_g_ridge_{variant}", "ML", res["val_acc"],
                    train_time=t1 - t0)

    # =========================================================================
    # === CLASSICAL SIGNAL METHODS ===
    # =========================================================================
    if cfg.run_classical:
        print(f"\n{'='*60}")
        print("=== CLASSICAL SIGNAL METHODS ===")
        print(f"{'='*60}")

        # SPC Monitor (anomaly detection — no classification accuracy)
        from pipeline.models.classical.spc import SPCMonitor
        print("\n[train_all] SPC Monitor ...")
        t0 = time.time()
        monitor = SPCMonitor(sigma_n=3.0)
        x_norm = x_ds[train_idx][labels_train == 0].flatten()
        y_norm = y_ds[train_idx][labels_train == 0].flatten()
        monitor.fit(x_norm, y_norm)
        # Evaluate as binary anomaly detector: class 0 = normal, rest = anomaly
        correct = 0
        total = 0
        for i in val_idx:
            results_stream = monitor.process_stream(x_ds[i], y_ds[i])
            anom_frac = np.mean([r["anomaly"] for r in results_stream])
            pred_anom = anom_frac > 0.05  # >5% anomalous samples = anomalous clip
            true_anom = labels[i] != 0
            if pred_anom == true_anom:
                correct += 1
            total += 1
        val_acc_spc = correct / max(total, 1)
        _record("spc_monitor", "Classical", val_acc_spc,
                train_time=time.time() - t0)
        print(f"  SPC anomaly accuracy: {val_acc_spc:.4f}")

        # Phase Portrait
        from pipeline.models.classical.phase_portrait import PhasePortraitClassifier
        print("\n[train_all] Phase Portrait Classifier ...")
        t0 = time.time()
        pp = PhasePortraitClassifier()
        pp.fit(x_ds[train_idx], y_ds[train_idx], labels_train)
        val_acc_pp = pp.score(x_ds[val_idx], y_ds[val_idx], labels_val)
        pp.save()
        _record("phase_portrait", "Classical", val_acc_pp,
                CHECKPOINT_DIR if (ROOT / "checkpoints/phase_portrait.pkl").exists()
                else None, train_time=time.time() - t0)
        print(f"  Phase portrait val_acc: {val_acc_pp:.4f}")

        # Hilbert
        from pipeline.models.classical.hilbert import HilbertClassifier
        print("\n[train_all] Hilbert Classifier ...")
        t0 = time.time()
        hil = HilbertClassifier(C=10.0, gamma="scale")
        hil.fit(x_ds[train_idx], labels_train)
        val_acc_hil = hil.score(x_ds[val_idx], labels_val)
        hil.save()
        _record("hilbert", "Classical", val_acc_hil, train_time=time.time() - t0)
        print(f"  Hilbert val_acc: {val_acc_hil:.4f}")

        # Autocorrelation
        from pipeline.models.classical.autocorrelation import AutocorrClassifier
        print("\n[train_all] Autocorrelation Classifier ...")
        t0 = time.time()
        acorr = AutocorrClassifier(n_estimators=50)
        acorr.fit(x_ds[train_idx], labels_train)
        val_acc_ac = acorr.score(x_ds[val_idx], labels_val)
        acorr.save()
        _record("autocorrelation", "Classical", val_acc_ac,
                train_time=time.time() - t0)
        print(f"  Autocorrelation val_acc: {val_acc_ac:.4f}")

        # Recurrence (slow — use small dataset subset)
        from pipeline.models.classical.recurrence import RecurrenceClassifier
        print("\n[train_all] Recurrence Classifier (truncated to 200 samples) ...")
        t0 = time.time()
        rec = RecurrenceClassifier(n_estimators=50, max_samples=200)
        rec.fit(x_ds[train_idx], labels_train)
        val_acc_rec = rec.score(x_ds[val_idx], labels_val)
        rec.save()
        _record("recurrence", "Classical", val_acc_rec, train_time=time.time() - t0)
        print(f"  Recurrence val_acc: {val_acc_rec:.4f}")

    # =========================================================================
    # === ML CLASSIFIERS ===
    # =========================================================================
    if cfg.run_ml:
        print(f"\n{'='*60}")
        print("=== ML CLASSIFIERS ===")
        print(f"{'='*60}")

        # SVM
        from pipeline.models.ml.svm_classifier import SVMClassifier
        print("\n[train_all] SVM Classifier (x_only) ...")
        t0 = time.time()
        svm = SVMClassifier(n_components=50, use_grid_search=False)
        svm.fit(reps["x_only"][train_idx], labels_train)
        val_acc_svm = svm.score(reps["x_only"][val_idx], labels_val)
        svm.save(name="x_only")
        _record("svm_x_only", "ML", val_acc_svm, train_time=time.time() - t0)
        print(f"  SVM val_acc: {val_acc_svm:.4f}")

        # Random Forest
        from pipeline.models.ml.random_forest import RandomForestModel
        from data.sample_data import CLASS_NAMES
        print("\n[train_all] Random Forest (28 features) ...")
        t0 = time.time()
        rf = RandomForestModel(n_estimators=100, max_depth=8)
        rf.fit(x_ds[train_idx], y_ds[train_idx], labels_train)
        val_acc_rf = rf.score(x_ds[val_idx], y_ds[val_idx], labels_val)
        rf.save()
        rf.export_firmware_header(class_names=CLASS_NAMES)
        _record("random_forest", "ML", val_acc_rf, train_time=time.time() - t0)
        print(f"  Random Forest val_acc: {val_acc_rf:.4f}")

        # KNN
        from pipeline.models.ml.knn_classifier import KNNClassifier
        print("\n[train_all] KNN (k=5) ...")
        t0 = time.time()
        knn = KNNClassifier(k=5, n_components=50)
        knn.fit(reps["x_only"][train_idx], labels_train)
        val_acc_knn = knn.score(reps["x_only"][val_idx], labels_val)
        knn.save()
        knn.export_firmware_header()
        _record("knn", "ML", val_acc_knn, train_time=time.time() - t0)
        print(f"  KNN val_acc: {val_acc_knn:.4f}")

    # =========================================================================
    # === ANOMALY DETECTORS ===
    # =========================================================================
    if cfg.run_anomaly:
        print(f"\n{'='*60}")
        print("=== ANOMALY DETECTORS ===")
        print(f"{'='*60}")

        def _anom_acc(detector, clips_all, labels_all, normal_cls=0):
            """Binary accuracy: class=normal_cls as +1, rest as -1."""
            from pipeline.models.ml.gmm_anomaly import GMMDetector
            from pipeline.models.anomaly.one_class_svm import OneClassSVMDetector
            from pipeline.models.anomaly.autoencoder import AnomalyAutoencoder
            from pipeline.models.anomaly.vae import VAEDetector
            from pipeline.models.ml.isolation_forest import IsolationForestModel

            correct = 0
            for i in range(len(clips_all)):
                if isinstance(detector, (GMMDetector, OneClassSVMDetector,
                                         AnomalyAutoencoder, VAEDetector)):
                    is_anom = detector.is_anomaly(clips_all[i])
                    pred_normal = not is_anom
                else:
                    pred_normal = True  # fallback
                true_normal = labels_all[i] == normal_cls
                if pred_normal == true_normal:
                    correct += 1
            return correct / len(clips_all)

        # GMM
        from pipeline.models.ml.gmm_anomaly import GMMDetector
        print("\n[train_all] GMM Anomaly Detector ...")
        t0 = time.time()
        gmm = GMMDetector(n_components=4, n_pca_components=50)
        gmm.fit(normal_clips_x)
        val_acc_gmm = _anom_acc(gmm, reps["x_only"][val_idx], labels_val)
        gmm.save()
        _record("gmm_anomaly", "Anomaly", val_acc_gmm, train_time=time.time() - t0)
        print(f"  GMM anomaly accuracy: {val_acc_gmm:.4f}")

        # Isolation Forest
        from pipeline.models.ml.isolation_forest import IsolationForestModel
        print("\n[train_all] Isolation Forest ...")
        t0 = time.time()
        iso = IsolationForestModel(n_estimators=50, contamination=0.2)
        iso.fit(x_ds[train_idx], y_ds[train_idx])
        preds_iso = iso.batch_predict(x_ds[val_idx], y_ds[val_idx])
        val_acc_iso = float(np.mean(
            np.where(labels_val == 0, 1, -1) == preds_iso
        ))
        iso.save()
        iso.export_firmware_header()
        _record("isolation_forest", "Anomaly", val_acc_iso,
                train_time=time.time() - t0)
        print(f"  Isolation Forest anomaly accuracy: {val_acc_iso:.4f}")

        # One-Class SVM
        from pipeline.models.anomaly.one_class_svm import OneClassSVMDetector
        print("\n[train_all] One-Class SVM ...")
        t0 = time.time()
        ocsvm = OneClassSVMDetector(nu=0.05, n_pca_components=50)
        ocsvm.fit(normal_clips_x)
        val_acc_ocsvm = _anom_acc(ocsvm, reps["x_only"][val_idx], labels_val)
        ocsvm.save()
        _record("one_class_svm", "Anomaly", val_acc_ocsvm,
                train_time=time.time() - t0)
        print(f"  One-Class SVM anomaly accuracy: {val_acc_ocsvm:.4f}")

        # Autoencoder (quick training for completeness)
        from pipeline.models.anomaly.autoencoder import AnomalyAutoencoder
        print("\n[train_all] Convolutional Autoencoder ...")
        t0 = time.time()
        ae_epochs = min(cfg.epochs, 10)
        ae = AnomalyAutoencoder(epochs=ae_epochs, batch_size=cfg.batch_size)
        ae.fit(normal_clips_x)
        val_acc_ae = _anom_acc(ae, reps["x_only"][val_idx], labels_val)
        ae.save()
        _record("autoencoder", "Anomaly", val_acc_ae,
                ROOT / "checkpoints" / "autoencoder.keras",
                train_time=time.time() - t0)
        print(f"  Autoencoder anomaly accuracy: {val_acc_ae:.4f}")

    # =========================================================================
    # === SEQUENCE MODELS ===
    # =========================================================================
    if cfg.run_sequence:
        print(f"\n{'='*60}")
        print("=== SEQUENCE MODELS ===")
        print(f"{'='*60}")

        # ESN
        from pipeline.models.sequence.esn_readout import EchoStateReadout
        print("\n[train_all] Echo State Network ...")
        t0 = time.time()
        esn = EchoStateReadout(reservoir_size=200, spectral_radius=0.9)
        esn.fit(reps["x_only"][train_idx], labels_train)
        val_acc_esn = esn.score(reps["x_only"][val_idx], labels_val)
        esn.save()
        _record("esn", "Sequence", val_acc_esn, train_time=time.time() - t0)
        print(f"  ESN val_acc: {val_acc_esn:.4f}")

        # TCN (fewer epochs for speed)
        from pipeline.models.sequence.tcn import TCNClassifier
        print("\n[train_all] TCN Classifier ...")
        tcn_epochs = min(cfg.epochs, 20)
        t0 = time.time()
        tcn = TCNClassifier(
            n_classes=cfg.n_classes, epochs=tcn_epochs,
            batch_size=cfg.batch_size, val_split=cfg.val_split,
        )
        tcn.fit(x_ds[train_idx], labels_train)
        val_acc_tcn = tcn.score(x_ds[val_idx], labels_val)
        _record("tcn", "Sequence", val_acc_tcn,
                ROOT / "checkpoints" / "tcn.keras",
                train_time=time.time() - t0)
        print(f"  TCN val_acc: {val_acc_tcn:.4f}")

        # LSTM (fewer epochs, short clip for RAM)
        from pipeline.models.sequence.lstm_classifier import LSTMClassifier
        print("\n[train_all] LSTM Classifier (400-sample clips for RAM) ...")
        lstm_epochs = min(cfg.epochs, 10)
        x_short = x_ds[:, :400]
        y_short = y_ds[:, :400]
        t0 = time.time()
        lstm = LSTMClassifier(
            n_classes=cfg.n_classes, epochs=lstm_epochs,
            batch_size=cfg.batch_size, val_split=cfg.val_split,
        )
        lstm.fit(x_short[train_idx], y_short[train_idx], labels_train)
        val_acc_lstm = lstm.score(x_short[val_idx], y_short[val_idx], labels_val)
        _record("lstm", "Sequence", val_acc_lstm,
                ROOT / "checkpoints" / "lstm.keras",
                train_time=time.time() - t0)
        print(f"  LSTM val_acc: {val_acc_lstm:.4f}")

    # =========================================================================
    # === FEW-SHOT METHODS ===
    # =========================================================================
    if cfg.run_fewshot:
        print(f"\n{'='*60}")
        print("=== FEW-SHOT METHODS ===")
        print(f"{'='*60}")

        # Prototypical Network (PCA encoder)
        from pipeline.models.fewshot.prototypical import PrototypicalNetwork
        from data.sample_data import CLASS_NAMES as _CN
        print("\n[train_all] Prototypical Network (5-shot) ...")
        t0 = time.time()
        support_set: dict[str, NDArray] = {}
        for cls, name in enumerate(_CN[:cfg.n_classes]):
            mask_tr = labels_train == cls
            clips_cls = reps["x_only"][train_idx][mask_tr]
            support_set[name] = clips_cls[:5]
        net = PrototypicalNetwork(encoder=None, n_pca_components=50)
        net.build_prototypes(support_set)
        val_preds = net.classify_batch(reps["x_only"][val_idx])
        val_true_names = [_CN[l] for l in labels_val]
        val_acc_proto = float(sum(p == t for p, t in zip(val_preds, val_true_names))
                               / max(len(val_preds), 1))
        net.export_firmware_header()
        _record("prototypical", "FewShot", val_acc_proto,
                train_time=time.time() - t0)
        print(f"  Prototypical val_acc (5-shot): {val_acc_proto:.4f}")

        # Contrastive (unsupervised pretraining + few-shot)
        from pipeline.models.anomaly.contrastive import ContrastiveClassifier
        print("\n[train_all] Contrastive Classifier (pretrain + 5-shot) ...")
        cont_epochs = min(cfg.epochs, 5)
        t0 = time.time()
        cont = ContrastiveClassifier(
            embedding_dim=64, epochs=cont_epochs,
            batch_size=cfg.batch_size,
        )
        cont.pretrain(reps["x_only"][train_idx])
        cont_support: dict[str, NDArray] = {}
        for cls, name in enumerate(_CN[:cfg.n_classes]):
            mask_tr = labels_train == cls
            clips_cls = reps["x_only"][train_idx][mask_tr]
            cont_support[name] = clips_cls[:5]
        cont.build_prototypes(cont_support)
        cont_preds = [cont.few_shot_classify(c)
                      for c in reps["x_only"][val_idx]]
        val_acc_cont = float(sum(p == t for p, t in zip(cont_preds, val_true_names))
                              / max(len(cont_preds), 1))
        _record("contrastive", "FewShot", val_acc_cont,
                ROOT / "checkpoints" / "contrastive_encoder.keras",
                train_time=time.time() - t0)
        print(f"  Contrastive val_acc (5-shot): {val_acc_cont:.4f}")

    # =========================================================================
    # Summary table
    # =========================================================================
    print(f"\n{'='*70}")
    print("FULL TRAINING SUMMARY")
    print(f"{'='*70}")

    categories = ["Classical", "ML", "Anomaly", "Sequence", "FewShot"]
    for cat in categories:
        cat_results = {k: v for k, v in results.items()
                       if v.get("category") == cat}
        if not cat_results:
            continue
        print(f"\n  === {cat.upper()} ===")
        for name, res in cat_results.items():
            print(f"  {'  ' + name:<40s} val_acc={res['val_acc']:.4f}  "
                  f"({res.get('train_time_s', 0):.1f}s)")

    # =========================================================================
    # Save comparison CSV
    # =========================================================================
    csv_path = RESULTS_DIR / "model_comparison.csv"
    csv_cols = [
        "Model", "Category", "Input", "Accuracy", "F1",
        "Train_time_s", "Inference_ms_est", "Model_size_KB",
        "RAM_KB", "M4", "M33", "M55", "M85",
    ]
    rows = []
    for mcfg in MODEL_REGISTRY:
        name = mcfg.model_name
        # Match result by partial name
        matched = None
        for k, v in results.items():
            if k == name or k.startswith(name):
                matched = v
                break
        acc = matched["val_acc"] if matched else "—"
        rows.append({
            "Model": name,
            "Category": mcfg.category,
            "Input": mcfg.input_type,
            "Accuracy": f"{acc:.4f}" if isinstance(acc, float) else acc,
            "F1": "—",
            "Train_time_s": f"{matched.get('train_time_s', 0):.1f}" if matched else "—",
            "Inference_ms_est": f"{mcfg.estimated_inference_ms:.1f}",
            "Model_size_KB": "—",
            "RAM_KB": str(mcfg.estimated_ram_kb),
            "M4": "✓" if "M4" in mcfg.target_chips else "—",
            "M33": "✓" if "M33" in mcfg.target_chips else "—",
            "M55": "✓" if "M55" in mcfg.target_chips else "—",
            "M85": "✓" if "M85" in mcfg.target_chips else "—",
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[train_all] Comparison CSV saved to {csv_path}")
    print(f"{'='*70}\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all Hopf oscillator models")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_clips", type=int, default=100)
    parser.add_argument("--n_classes", type=int, default=5)
    parser.add_argument("--no_qat", action="store_true")
    parser.add_argument("--save_dir", type=str, default="checkpoints/")
    parser.add_argument("--skip_keras", action="store_true",
                        help="Skip original CNN models A-G")
    parser.add_argument("--skip_classical", action="store_true")
    parser.add_argument("--skip_ml", action="store_true")
    parser.add_argument("--skip_anomaly", action="store_true")
    parser.add_argument("--skip_sequence", action="store_true")
    parser.add_argument("--skip_fewshot", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_classes=args.n_classes,
        use_qat=not args.no_qat,
        save_dir=args.save_dir,
        n_clips_per_class=args.n_clips,
        run_keras=not args.skip_keras,
        run_classical=not args.skip_classical,
        run_ml=not args.skip_ml,
        run_anomaly=not args.skip_anomaly,
        run_sequence=not args.skip_sequence,
        run_fewshot=not args.skip_fewshot,
    )
    print("[train_all] Config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    train_all(cfg)


if __name__ == "__main__":
    main()


# Make CHECKPOINT_DIR available for use by model training functions
CHECKPOINT_DIR = ROOT / "checkpoints"
