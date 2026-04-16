"""
Quantisation-aware training (QAT) for the Hopf reservoir CNN.

Applies TensorFlow Model Optimization Toolkit's quantize_model() to insert
fake-quantisation nodes during training. This simulates INT8 rounding so the
trained weights already account for the fixed-point arithmetic used by
CMSIS-NN kernels on Arm Cortex-M targets.

The QAT-trained model converts cleanly to a fully INT8 TFLite model where
every operator maps to a CMSIS-NN kernel function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import tensorflow as tf
from tensorflow import keras


def apply_qat(model: keras.Model) -> keras.Model:
    """
    Apply quantisation-aware training to the model.

    QAT inserts fake quantisation nodes during training that simulate INT8
    rounding, so the trained weights already account for CMSIS-NN's
    fixed-point arithmetic.

    Returns:
        QAT-wrapped model (must be recompiled before training).
    """
    import tensorflow_model_optimization as tfmot
    qat_model = tfmot.quantization.keras.quantize_model(model)
    return qat_model


def representative_data_gen(
    feature_maps: NDArray[np.uint8],
    n_samples: int = 100,
):
    """
    Generator yielding representative samples for TFLite INT8 calibration.

    Yields samples in the same uint8 scaling [0, 255] used in feature
    extraction so calibration matches runtime input.

    CMSIS-NN NOTE: The representative dataset must cover the expected
    input distribution so the converter computes accurate zero-point
    and scale values for cmsis_nn_conv_params / cmsis_nn_fc_params.
    """
    indices = np.random.default_rng(42).choice(
        len(feature_maps), size=min(n_samples, len(feature_maps)), replace=False
    )
    for i in indices:
        sample = feature_maps[i].astype(np.float32)
        sample = np.expand_dims(sample, axis=(0, -1))  # (1, 200, 100, 1)
        yield [sample]


def balance_binary(
    feature_maps: NDArray[np.uint8],
    labels: NDArray[np.int64],
    target_positive_rate: float = 0.5,
    seed: int = 0,
) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
    """
    Undersample the negative class so positives make up `target_positive_rate`
    of the returned dataset.

    All positives are kept; negatives are randomly dropped. If the current
    positive rate already meets or exceeds the target, nothing is dropped.

    Args:
        target_positive_rate: desired positive fraction in (0, 1). 0.5 = 50/50.
        seed: RNG seed for reproducible negative selection.

    Returns:
        (feature_maps, labels) — subsampled and shuffled jointly.
    """
    if not 0.0 < target_positive_rate < 1.0:
        raise ValueError(f"target_positive_rate must be in (0, 1), got {target_positive_rate}")
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    if n_pos == 0:
        raise ValueError("no positive examples — cannot balance")

    n_neg_target = int(round(n_pos * (1.0 - target_positive_rate) / target_positive_rate))
    n_neg_keep = min(n_neg_target, n_neg)

    rng = np.random.default_rng(seed)
    neg_keep = rng.choice(neg_idx, size=n_neg_keep, replace=False)
    keep = np.concatenate([pos_idx, neg_keep])
    rng.shuffle(keep)

    actual_rate = n_pos / (n_pos + n_neg_keep)
    print(
        f"[train] Balanced: kept {n_pos} pos + {n_neg_keep}/{n_neg} neg "
        f"(target_rate={target_positive_rate:.2f}, actual={actual_rate:.2f})"
    )
    return feature_maps[keep], labels[keep]


def prepare_data(
    feature_maps: NDArray[np.uint8],
    labels: NDArray[np.int64],
    n_classes: int = 5,
    val_split: float = 0.1,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """
    Prepare feature maps and labels for training.

    Converts uint8 feature maps to float32 (preserving [0, 255] range),
    adds channel dimension, and one-hot encodes labels.
    """
    x = feature_maps.astype(np.float32)
    x = np.expand_dims(x, axis=-1)  # (n, 200, 100, 1)

    y = keras.utils.to_categorical(labels, num_classes=n_classes)

    n = len(x)
    n_val = int(n * val_split)
    indices = np.random.default_rng(0).permutation(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def train(
    model: keras.Model,
    x_train: NDArray[np.float32],
    y_train: NDArray[np.float32],
    x_val: NDArray[np.float32],
    y_val: NDArray[np.float32],
    epochs: int = 20,
    batch_size: int = 16,
    checkpoint_dir: Optional[str | Path] = None,
) -> keras.callbacks.History:
    """
    Compile and train the (optionally QAT) model.

    Uses Adam optimiser with categorical crossentropy, saves best
    checkpoint by validation accuracy.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = []
    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(ckpt_path / "best_model.h5"),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            )
        )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    # Print per-layer output shape and parameter count
    print("\n--- Post-Training Layer Summary ---")
    for layer in model.layers:
        if hasattr(layer, "output"):
            out_shape = layer.output.shape
        else:
            out_shape = "N/A"
        print(f"  {layer.name:45s}  output={str(out_shape):25s}  params={layer.count_params():>8,}")

    return history


ESC50_HOPF_TEXT_CACHE: Path = Path(
    "/Users/nic-spect/data/recho_ai/Kaggle_Environmental_Sound_Classification_50/hopf_text"
)
TARGET_CLASS: str = "sheep"
# Positive fraction after balancing. 0.5 = 50/50; None = no undersampling.
TARGET_POSITIVE_RATE: float | None = 0.5
# Subtract the dataset-wide mean clip before feature extraction so the
# audio-driven residual isn't buried under the Hopf oscillator's limit cycle.
SUBTRACT_COMMON_MODE: bool = True


def main() -> None:
    """
    Binary QAT training on the ESC-50 hopf_text cache.

    Labels are relabeled to 1 for TARGET_CLASS and 0 for every other class,
    giving a one-vs-all detector. Class imbalance is steep (40/2000 ≈ 2%);
    val_accuracy alone is not a useful metric — inspect per-class recall
    in the fit history or add class_weight/metrics as a follow-up.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.sample_data import load_dataset_from_text_cache
    from pipeline.ingest import process_dataset, FS_HW, FS_TARGET
    from pipeline.features import extract_features
    from pipeline.model import build_model

    print(f"[train] Loading hopf_text cache from {ESC50_HOPF_TEXT_CACHE} ...")
    raw_x, labels, class_names, fs = load_dataset_from_text_cache(
        cache_dir=ESC50_HOPF_TEXT_CACHE,
        target_class=TARGET_CLASS,
    )
    print(f"  raw_x: {raw_x.shape}, fs={fs} Hz, classes={class_names}")

    ds_factor = 1 if fs == FS_TARGET else FS_HW // fs
    print(
        f"[train] Processing clips (downsample_factor={ds_factor}, "
        f"subtract_common_mode={SUBTRACT_COMMON_MODE}) ..."
    )
    processed = process_dataset(
        raw_x,
        downsample_factor=ds_factor,
        subtract_common_mode=SUBTRACT_COMMON_MODE,
    )
    feature_maps, labels = extract_features(processed, labels)

    if TARGET_POSITIVE_RATE is not None:
        feature_maps, labels = balance_binary(
            feature_maps, labels, target_positive_rate=TARGET_POSITIVE_RATE
        )

    n_classes = 2
    print(f"[train] Building model (n_classes={n_classes}) ...")
    model = build_model(n_classes=n_classes)

    print("[train] Applying quantisation-aware training ...")
    try:
        qat_model = apply_qat(model)
    except ImportError:
        print("  WARNING: tensorflow-model-optimization not installed.")
        print("  Training without QAT. Install with: pip install tensorflow-model-optimization")
        qat_model = model

    x_train, y_train, x_val, y_val = prepare_data(feature_maps, labels, n_classes=n_classes)
    print(f"  Train: {x_train.shape}, Val: {x_val.shape}")

    ckpt_dir = Path(__file__).resolve().parent.parent / "output" / "checkpoints"
    history = train(
        qat_model, x_train, y_train, x_val, y_val,
        epochs=20, batch_size=4, checkpoint_dir=ckpt_dir,
    )

    print(f"\n[train] Final val_accuracy: {history.history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()
