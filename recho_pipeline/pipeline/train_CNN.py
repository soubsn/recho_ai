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

import os
from pathlib import Path
from typing import Callable, Optional

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
    val_split: float = 0.2,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """
    Prepare feature maps and labels for training.

    Converts uint8 feature maps to float32 (preserving [0, 255] range),
    adds channel dimension, and one-hot encodes labels. The val split
    is stratified by class so both splits preserve the dataset's class
    ratio — important at low positive rates where a random split can
    easily under- or over-sample the minority class.
    """
    x = feature_maps.astype(np.float32)
    # Single-channel reps are (n, 200, 100); xy_dual is already (n, 200, 100, 2).
    if x.ndim == 3:
        x = np.expand_dims(x, axis=-1)  # (n, 200, 100, 1)

    y = keras.utils.to_categorical(labels, num_classes=n_classes)

    rng = np.random.default_rng(0)
    val_idx_parts = []
    train_idx_parts = []
    for c in np.unique(labels):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)
        n_val_c = int(round(len(class_idx) * val_split))
        val_idx_parts.append(class_idx[:n_val_c])
        train_idx_parts.append(class_idx[n_val_c:])
    val_idx = np.concatenate(val_idx_parts)
    train_idx = np.concatenate(train_idx_parts)
    rng.shuffle(val_idx)
    rng.shuffle(train_idx)

    print(
        f"[train] Stratified split — train: {len(train_idx)} "
        f"(pos={int(np.sum(labels[train_idx] == 1))}), "
        f"val: {len(val_idx)} "
        f"(pos={int(np.sum(labels[val_idx] == 1))})"
    )
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def save_feature_maps_preview(
    x_val: NDArray[np.float32],
    y_val: NDArray[np.float32],
    output_dir: str | Path,
    class_names: Optional[list[str]] = None,
    n_per_class: int = 10,
    seed: int = 0,
) -> None:
    """
    Dump a visual sanity-check of the validation feature maps.

    Saves, per class, up to `n_per_class` individual feature-map PNGs plus a
    summary grid and a text file with intensity stats. Multi-channel inputs
    (xy_dual) are saved once per channel. Intended to be called right before
    train() so the maps the network will actually see can be eyeballed.
    """
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels = np.argmax(y_val, axis=1)
    n_cls = int(y_val.shape[1])
    names = class_names if class_names is not None else [f"class_{c}" for c in range(n_cls)]
    rng = np.random.default_rng(seed)

    n_channels = 1 if x_val.ndim == 3 else int(x_val.shape[-1])
    stats_lines = [
        f"# feature-map preview — n_classes={n_cls}, n_per_class={n_per_class}, "
        f"shape={x_val.shape}, dtype={x_val.dtype}",
        f"# global: min={x_val.min():.3f} max={x_val.max():.3f} "
        f"mean={x_val.mean():.3f} std={x_val.std():.3f}",
        "",
    ]

    for c in range(n_cls):
        class_idx = np.where(labels == c)[0]
        if len(class_idx) == 0:
            stats_lines.append(f"class {c} ({names[c]}): no samples in val set")
            continue
        n_take = min(n_per_class, len(class_idx))
        picks = rng.choice(class_idx, size=n_take, replace=False)

        class_dir = out / f"class_{c}_{names[c]}"
        class_dir.mkdir(parents=True, exist_ok=True)

        for i, idx in enumerate(picks):
            fm = x_val[idx]  # (H, W) or (H, W, C)
            if fm.ndim == 2:
                fig, ax = plt.subplots(1, 1, figsize=(4, 6))
                ax.imshow(fm, cmap="gray", aspect="auto", vmin=0, vmax=255)
                ax.set_title(f"{names[c]} #{i} (val idx {idx})")
                ax.set_xlabel("virtual node"); ax.set_ylabel("time step")
            else:
                fig, axes = plt.subplots(1, n_channels, figsize=(4 * n_channels, 6))
                if n_channels == 1:
                    axes = [axes]
                for ch in range(n_channels):
                    axes[ch].imshow(fm[..., ch], cmap="gray", aspect="auto", vmin=0, vmax=255)
                    axes[ch].set_title(f"{names[c]} #{i} ch{ch}")
                    axes[ch].set_xlabel("virtual node"); axes[ch].set_ylabel("time step")
            fig.tight_layout()
            fig.savefig(class_dir / f"sample_{i:02d}.png", dpi=100)
            plt.close(fig)

        class_fm = x_val[picks]
        stats_lines.append(
            f"class {c} ({names[c]}): n={len(class_idx)} total, saved {n_take} — "
            f"min={class_fm.min():.1f} max={class_fm.max():.1f} "
            f"mean={class_fm.mean():.2f} std={class_fm.std():.2f}"
        )

    # One combined grid: n_per_class columns × n_cls rows, first channel only.
    cols = min(n_per_class, max((np.sum(labels == c) for c in range(n_cls)), default=0))
    if cols > 0:
        fig, axes = plt.subplots(n_cls, cols, figsize=(2 * cols, 2.5 * n_cls), squeeze=False)
        for c in range(n_cls):
            class_idx = np.where(labels == c)[0]
            if len(class_idx) == 0:
                for j in range(cols):
                    axes[c, j].axis("off")
                continue
            n_take = min(cols, len(class_idx))
            picks = rng.choice(class_idx, size=n_take, replace=False)
            for j in range(cols):
                ax = axes[c, j]
                if j < n_take:
                    fm = x_val[picks[j]]
                    img = fm if fm.ndim == 2 else fm[..., 0]
                    ax.imshow(img, cmap="gray", aspect="auto", vmin=0, vmax=255)
                    if j == 0:
                        ax.set_ylabel(names[c])
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Validation feature-map preview (n_per_class={cols}, channel 0)")
        fig.tight_layout()
        fig.savefig(out / "preview_grid.png", dpi=120)
        plt.close(fig)

    (out / "stats.txt").write_text("\n".join(stats_lines) + "\n")
    print(f"[preview] Saved feature-map preview for {n_cls} classes to {out}")


def train(
    model: keras.Model,
    x_train: NDArray[np.float32],
    y_train: NDArray[np.float32],
    x_val: NDArray[np.float32],
    y_val: NDArray[np.float32],
    epochs: int = 20,
    batch_size: int = 16,
    checkpoint_dir: Optional[str | Path] = None,
    peak_lr: float = 3e-6,
) -> keras.callbacks.History:
    """
    Compile and train the (optionally QAT) model.

    Uses Adam with a cosine-decay learning-rate schedule that warms up from 0
    to `peak_lr` over one epoch, then cosine-decays to `peak_lr * alpha` over
    the remaining epochs. Early stopping monitors val_loss (patience=20) so
    the model has time to escape the trivial-majority-class basin — during
    that phase val_accuracy is pinned at the majority rate while val_loss
    is still decreasing. Best checkpoint is selected by val_accuracy.
    Computes class_weight from y_train for the imbalanced case.
    """
    y_int = np.argmax(y_train, axis=1)
    n_classes_seen = int(y_train.shape[1])
    n = len(y_int)
    class_weight = {
        int(c): n / (n_classes_seen * int(np.sum(y_int == c)))
        for c in np.unique(y_int)
    }
    print(f"[train] class_weight = {class_weight}")

    steps_per_epoch = max(1, len(x_train) // batch_size)
    # CosineDecay's warmup_target type stub is int | Tensor | None, but the
    # runtime accepts (and expects) a float. Cast to silence the checker.
    from typing import Any, cast
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=steps_per_epoch * epochs,
        alpha=0.1,
        warmup_target=cast(Any, peak_lr),
        warmup_steps=steps_per_epoch,  # 1 epoch warmup
    )
    print(f"[train] peak_lr={peak_lr:.2e}, batch_size={batch_size}, epochs={epochs}")
    # Legacy AdamW runs ~3× faster than the v2.11+ AdamW on Apple silicon.
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(
            learning_rate=lr_schedule,
            clipnorm=1.0,
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # model.compile(
    #     optimizer=keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
    #     loss="categorical_crossentropy",
    #     metrics=["accuracy"],
    # )

    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(ckpt_path / "best_model.h5"),
                monitor="val_accuracy",
                mode="max",
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
        class_weight=class_weight,
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

    # Confusion matrix + per-class precision/recall on the restored best model.
    # Under class imbalance, val_accuracy can match the always-negative baseline
    # and hide a collapsed model — this tells us if the model actually detects
    # the positive class.
    y_val_true = np.argmax(y_val, axis=1)
    y_val_probs = model.predict(x_val, verbose=0)
    y_val_pred = np.argmax(y_val_probs, axis=1)
    n_cls = y_val.shape[1]
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, p in zip(y_val_true, y_val_pred):
        cm[t, p] += 1

    if n_cls <= 10:
        print("\n--- Validation Confusion Matrix (rows=true, cols=pred) ---")
        header = "       " + " ".join(f"pred={c:>2d}" for c in range(n_cls))
        print(header)
        for c in range(n_cls):
            row = " ".join(f"{cm[c, p]:>7d}" for p in range(n_cls))
            print(f"true={c:>2d} {row}")
        print("\n--- Per-class Precision / Recall ---")
        for c in range(n_cls):
            tp = int(cm[c, c])
            fn = int(cm[c, :].sum() - tp)
            fp = int(cm[:, c].sum() - tp)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            print(f"  class {c}: precision={prec:.3f}  recall={rec:.3f}  "
                  f"(tp={tp}, fp={fp}, fn={fn})")
    else:
        # 50-class case: full CM / per-class P/R is too noisy. Summarize with
        # top-1 accuracy, top-5 accuracy, and the macro-averaged recall.
        top1 = float(np.mean(y_val_pred == y_val_true))
        top5_idx = np.argsort(y_val_probs, axis=1)[:, -5:]
        top5 = float(np.mean([t in row for t, row in zip(y_val_true, top5_idx)]))
        recalls = []
        for c in range(n_cls):
            n_c = int(np.sum(y_val_true == c))
            if n_c == 0:
                continue
            recalls.append(int(cm[c, c]) / n_c)
        macro_recall = float(np.mean(recalls)) if recalls else 0.0
        print(f"\n--- Validation ({n_cls} classes) ---")
        print(f"  top-1 accuracy: {top1:.3f}")
        print(f"  top-5 accuracy: {top5:.3f}")
        print(f"  macro recall:   {macro_recall:.3f}")

    return history


ESC50_HOPF_TEXT_CACHE: Path = Path(
    #"/Users/nic-spect/data/recho_ai/Kaggle_Environmental_Sound_Classification_50/hopf_text"
    "/Users/nic-spect/data/recho_ai/Kaggle_Dog_vs_Cats/hopf_text"
)
# Pipeline mode:
#   "pretrain" — 50-class softmax on all 2,000 ESC-50 clips. Saves the full
#                model to PRETRAIN_CHECKPOINT for later fine-tuning.
#   "finetune" — load the pretrained 50-class model, relabel animals-vs-not,
#                and 2-stage fine-tune a new binary head (freeze → unfreeze).
MODE: str = "pretrain"
# The 12 ESC-50 animal classes used as the positive set during fine-tuning.
# 480 positives / 1520 negatives in the raw dataset — 12x more positive
# supervision than the sheep-only task.
ANIMAL_CLASSES: tuple[str, ...] = (
    "cat", "chirping_birds", "cow", "crickets", "crow", "dog",
    "frog", "hen", "insects", "pig", "rooster", "sheep",
)
# Pretrained 50-class model written here by MODE="pretrain", read by MODE="finetune".
PRETRAIN_CHECKPOINT: Path = (
    Path(__file__).resolve().parent.parent
    / "output" / "checkpoints" / "pretrain_50class.h5"
)
# Fine-tune schedule.
# Stage 1 — freeze every Conv/Pool/Flatten, train only the new Dense head.
# Stage 2 — unfreeze everything and continue at UNFROZEN_LR_DIVISOR x lower LR.
FROZEN_EPOCHS: int = 10
UNFROZEN_EPOCHS: int = 20
UNFROZEN_LR_DIVISOR: float = 10.0
# Peak learning rate for pretraining (post-warmup, before cosine decay).
# The previous value of 3e-6 was far too small — Adam's usual starting range
# for conv training is 1e-4 to 1e-3.
PEAK_LR: float = 1e-3
BATCH_SIZE: int = 32
# When True (and MODE="pretrain"), run a learning-rate sweep over
# TUNE_LR_VALUES for TUNE_LR_EPOCHS each, print a summary, and exit without
# saving a checkpoint. Pick the best LR and set PEAK_LR before the real run.
TUNE_LR: bool = False
TUNE_LR_VALUES: tuple[float, ...] = (1e-4, 3e-4, 1e-3)
TUNE_LR_EPOCHS: int = 30
# Positive fraction after balancing during fine-tuning. 0.5 = 50/50 (drops
# ~1040 non-animals to match 480 animals → 960 clips total). None = keep all
# 2000 clips and rely on class_weight="balanced".
TARGET_POSITIVE_RATE: float | None = 0.5
# Subtract the dataset-wide mean clip before feature extraction so the
# audio-driven residual isn't buried under the Hopf oscillator's limit cycle.
SUBTRACT_COMMON_MODE: bool = True
# Input representation fed to the CNN:
#   "x_only"  — paper baseline; uses pipeline.model.build_model (single channel)
#   "y_only"  — y(t) through the same pipeline; uses cnn_x_only (same arch)
#   "xy_dual" — x and y stacked as two input channels; uses cnn_xy_dual
#   "phase"   — orbit radius r(t) = sqrt(x^2+y^2); uses cnn_phase
#   "angle"   — unwrapped arctan2(y, x); uses cnn_angle
# Anything other than "x_only" loads y(t) from the cache (slower first pass).
INPUT_REP: str = "xy_dual"
# When True with INPUT_REP="x_only", use pipeline.models.cnn_regularized — same
# conv trunk but GlobalAveragePooling2D + Dropout head (481K -> ~76K params).
# Ignored for other INPUT_REP values.
USE_REGULARIZED: bool = True


def _resolve_build_model() -> "Callable[..., keras.Model]":
    """Pick the CNN architecture matching INPUT_REP."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    valid_reps = {"x_only", "y_only", "xy_dual", "phase", "angle"}
    if INPUT_REP not in valid_reps:
        raise ValueError(f"INPUT_REP must be one of {valid_reps}, got {INPUT_REP!r}")
    if INPUT_REP == "x_only":
        if USE_REGULARIZED:
            from pipeline.models.cnn_regularized import build_model
        else:
            from pipeline.model import build_model
    elif INPUT_REP == "xy_dual":
        from pipeline.models.cnn_xy_dual import build_model
    elif INPUT_REP == "phase":
        from pipeline.models.cnn_phase import build_model
    elif INPUT_REP == "angle":
        from pipeline.models.cnn_angle import build_model
    else:  # y_only — same single-channel CNN shape as x_only
        from pipeline.models.cnn_x_only import build_model
    return build_model


def _load_features_50class() -> tuple[NDArray[np.uint8], NDArray[np.int64], list[str]]:
    """
    Load all 2,000 ESC-50 clips at the original 50 class labels and run
    them through the Hopf ingestion + feature-extraction pipeline.

    Both pretraining and fine-tuning start from this 50-class view; the
    fine-tune step then relabels animals vs non-animals in Python so the
    same feature maps can be reused without re-processing.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.sample_data import (
        load_dataset_from_text_cache,
        load_xy_dataset_from_text_cache,
    )
    from pipeline.ingest import process_dataset, FS_HW, FS_TARGET
    from pipeline.features import extract_features

    print(f"[data] Loading hopf_text cache from {ESC50_HOPF_TEXT_CACHE} "
          f"(INPUT_REP={INPUT_REP}, 50-class labels) ...")
    if INPUT_REP == "x_only":
        raw_x, labels, class_names, fs = load_dataset_from_text_cache(
            cache_dir=ESC50_HOPF_TEXT_CACHE, target_class=None,
        )
        raw_y = None
    else:
        raw_x, raw_y, labels, class_names, fs = load_xy_dataset_from_text_cache(
            cache_dir=ESC50_HOPF_TEXT_CACHE, target_class=None,
        )
    print(f"  raw_x: {raw_x.shape}, fs={fs} Hz, n_classes={len(class_names)}")

    ds_factor = 1 if fs == FS_TARGET else FS_HW // fs
    print(
        f"[data] Processing clips (downsample_factor={ds_factor}, "
        f"subtract_common_mode={SUBTRACT_COMMON_MODE}) ..."
    )
    x_processed = process_dataset(
        raw_x, downsample_factor=ds_factor, subtract_common_mode=SUBTRACT_COMMON_MODE,
    )

    if INPUT_REP == "x_only":
        feature_maps, labels = extract_features(x_processed, labels)
    else:
        y_processed = process_dataset(
            raw_y, downsample_factor=ds_factor,
            subtract_common_mode=SUBTRACT_COMMON_MODE,
        )
        from pipeline.features_xy import extract_all_representations
        reps = extract_all_representations(x_processed, y_processed)
        feature_maps = reps[INPUT_REP]
        print(f"  {INPUT_REP}: shape={feature_maps.shape}, dtype={feature_maps.dtype}")

    return feature_maps, labels, class_names


def _animal_mask(labels: NDArray[np.int64], class_names: list[str]) -> NDArray[np.int64]:
    """Convert 50-class labels → binary: 1 if class name is in ANIMAL_CLASSES."""
    animal_indices = [class_names.index(c) for c in ANIMAL_CLASSES if c in class_names]
    missing = [c for c in ANIMAL_CLASSES if c not in class_names]
    if missing:
        print(f"[data] Warning: animal classes missing from cache: {missing}")
    print(f"[data] Animal classes used ({len(animal_indices)} of {len(ANIMAL_CLASSES)}): "
          f"{[class_names[i] for i in animal_indices]}")
    return np.isin(labels, animal_indices).astype(np.int64)


def _pretrain() -> None:
    """Train the 50-class softmax and save the full model to disk."""
    build_model = _resolve_build_model()
    feature_maps, labels, class_names = _load_features_50class()

    n_classes = len(class_names)
    print(f"[pretrain] Building 50-class model (n_classes={n_classes}) ...")
    model = build_model(n_classes=n_classes)

    x_train, y_train, x_val, y_val = prepare_data(feature_maps, labels, n_classes=n_classes)
    print(f"  Train: {x_train.shape}, Val: {x_val.shape}")

    preview_dir = Path(__file__).resolve().parent.parent / "output" / "feature_maps_preview"
    save_feature_maps_preview(
        x_val, y_val, preview_dir, class_names=class_names, n_per_class=10,
    )

    ckpt_dir = Path(__file__).resolve().parent.parent / "output" / "checkpoints"
    history = train(
        model, x_train, y_train, x_val, y_val,
        epochs=1000, batch_size=BATCH_SIZE, checkpoint_dir=ckpt_dir,
        peak_lr=PEAK_LR,
    )
    print(f"\n[pretrain] Final val_accuracy (top-1 over {n_classes} classes): "
          f"{history.history['val_accuracy'][-1]:.4f}")

    PRETRAIN_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(PRETRAIN_CHECKPOINT))
    print(f"[pretrain] Saved 50-class model to {PRETRAIN_CHECKPOINT}")

    # Free readout: animals-vs-not-animals detector from summed softmax mass.
    # Sum P(c) over the 12 animal classes and threshold at 0.5 — tells us the
    # post-pretrain animal-detection baseline before any fine-tuning.
    animal_indices = [class_names.index(c) for c in ANIMAL_CLASSES if c in class_names]
    if animal_indices:
        val_probs = model.predict(x_val, verbose=0)
        val_true = np.argmax(y_val, axis=1)
        is_animal_true = np.isin(val_true, animal_indices).astype(np.int64)
        p_animal = val_probs[:, animal_indices].sum(axis=1)
        is_animal_pred = (p_animal > 0.5).astype(np.int64)
        tp = int(((is_animal_true == 1) & (is_animal_pred == 1)).sum())
        fp = int(((is_animal_true == 0) & (is_animal_pred == 1)).sum())
        fn = int(((is_animal_true == 1) & (is_animal_pred == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        acc = float((is_animal_pred == is_animal_true).mean())
        print(
            f"\n--- Animal detector from 50-class softmax (sum of 12 animal probs > 0.5) ---\n"
            f"  accuracy={acc:.3f}  precision={prec:.3f}  recall={rec:.3f}  "
            f"(tp={tp}, fp={fp}, fn={fn})"
        )


def _compile_and_fit_stage(
    model: keras.Model,
    x_train: NDArray[np.float32],
    y_train: NDArray[np.float32],
    x_val: NDArray[np.float32],
    y_val: NDArray[np.float32],
    epochs: int,
    batch_size: int,
    lr: float,
    stage_label: str,
) -> None:
    """One fine-tune stage: compile at the given LR and fit for `epochs`."""
    y_int = np.argmax(y_train, axis=1)
    n_cls = int(y_train.shape[1])
    n = len(y_int)
    class_weight = {
        int(c): n / (n_cls * int(np.sum(y_int == c)))
        for c in np.unique(y_int)
    }
    print(f"\n[finetune:{stage_label}] lr={lr:.2e}, epochs={epochs}, "
          f"class_weight={class_weight}")

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=lr, clipnorm=1.0),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        class_weight=class_weight,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=20, restore_best_weights=True, verbose=1,
            ),
        ],
        verbose=2,
    )


def _finetune_animals() -> None:
    """
    Load pretrained 50-class model, relabel animals vs non-animals, and run
    a 2-stage fine-tune: freeze conv layers + train new head, then unfreeze
    and continue at a reduced learning rate.
    """
    if not PRETRAIN_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"No pretrained model at {PRETRAIN_CHECKPOINT}. "
            f"Run with MODE='pretrain' first."
        )

    build_model = _resolve_build_model()
    feature_maps, labels_50, class_names = _load_features_50class()
    binary_labels = _animal_mask(labels_50, class_names)
    class_names_bin = ["not_animal", "animal"]
    print(f"[finetune] Binary label counts — "
          f"animal={int((binary_labels == 1).sum())}, "
          f"not_animal={int((binary_labels == 0).sum())}")

    if TARGET_POSITIVE_RATE is not None:
        feature_maps, binary_labels = balance_binary(
            feature_maps, binary_labels, target_positive_rate=TARGET_POSITIVE_RATE,
        )

    print(f"[finetune] Building fresh 2-class model ...")
    model = build_model(n_classes=2)
    pretrained = keras.models.load_model(str(PRETRAIN_CHECKPOINT))
    # Architecture is identical except for the final Dense output size; copy
    # weights for every layer except the last.
    transferred = 0
    for pre_layer, new_layer in zip(pretrained.layers[:-1], model.layers[:-1]):
        pre_w = pre_layer.get_weights()
        if pre_w:
            new_layer.set_weights(pre_w)
            transferred += 1
    print(f"[finetune] Transferred weights for {transferred} layers from {PRETRAIN_CHECKPOINT}")

    # Stage 1 — freeze every non-output layer so only the fresh head learns.
    for layer in model.layers[:-1]:
        layer.trainable = False
    print("[finetune] Stage 1: conv/pool/dense layers frozen, new head trains alone")

    x_train, y_train, x_val, y_val = prepare_data(feature_maps, binary_labels, n_classes=2)
    print(f"  Train: {x_train.shape}, Val: {x_val.shape}")

    stage1_lr = 1e-3
    _compile_and_fit_stage(
        model, x_train, y_train, x_val, y_val,
        epochs=FROZEN_EPOCHS, batch_size=32, lr=stage1_lr,
        stage_label="stage-1 head-only",
    )

    # Stage 2 — unfreeze everything, continue at reduced LR.
    for layer in model.layers:
        layer.trainable = True
    stage2_lr = stage1_lr / UNFROZEN_LR_DIVISOR
    _compile_and_fit_stage(
        model, x_train, y_train, x_val, y_val,
        epochs=UNFROZEN_EPOCHS, batch_size=32, lr=stage2_lr,
        stage_label="stage-2 full-unfreeze",
    )

    # Final binary metrics on the val set.
    val_probs = model.predict(x_val, verbose=0)
    val_true = np.argmax(y_val, axis=1)
    val_pred = np.argmax(val_probs, axis=1)
    tp = int(((val_true == 1) & (val_pred == 1)).sum())
    fp = int(((val_true == 0) & (val_pred == 1)).sum())
    fn = int(((val_true == 1) & (val_pred == 0)).sum())
    tn = int(((val_true == 0) & (val_pred == 0)).sum())
    acc = (tp + tn) / max(1, tp + fp + fn + tn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print(
        f"\n--- Final animal-detector metrics ({class_names_bin}) ---\n"
        f"  accuracy={acc:.3f}  precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}\n"
        f"  confusion: tp={tp}, fp={fp}, fn={fn}, tn={tn}"
    )

    ft_ckpt = PRETRAIN_CHECKPOINT.parent / "finetune_animals.h5"
    model.save(str(ft_ckpt))
    print(f"[finetune] Saved fine-tuned binary model to {ft_ckpt}")


def _tune_lr_pretrain() -> None:
    """
    Short LR sweep for the 50-class pretraining task. Trains a fresh model at
    every LR in TUNE_LR_VALUES for TUNE_LR_EPOCHS epochs on the same data
    split, records best val_accuracy per run, and prints a summary. No
    checkpoint is saved — pick the winning LR, set PEAK_LR, flip TUNE_LR off,
    and rerun MODE="pretrain" for the full training run.
    """
    build_model = _resolve_build_model()
    feature_maps, labels, class_names = _load_features_50class()
    n_classes = len(class_names)
    x_train, y_train, x_val, y_val = prepare_data(
        feature_maps, labels, n_classes=n_classes
    )
    print(f"[tune-lr] sweep over {list(TUNE_LR_VALUES)} at {TUNE_LR_EPOCHS} "
          f"epochs each, batch_size={BATCH_SIZE}")
    print(f"  Train: {x_train.shape}, Val: {x_val.shape}")

    preview_dir = Path(__file__).resolve().parent.parent / "output" / "feature_maps_preview"
    save_feature_maps_preview(
        x_val, y_val, preview_dir, class_names=class_names, n_per_class=10,
    )

    # RECHO_SINGLE_LR — when set by parallel_lr_sweep.py, override the
    # sweep list so this process runs exactly one LR. The [RESULT] line
    # below is machine-parsed by the parallel driver.
    single_lr_env = os.environ.get("RECHO_SINGLE_LR")
    sweep_values = (float(single_lr_env),) if single_lr_env else TUNE_LR_VALUES
    if single_lr_env:
        print(f"[tune-lr] RECHO_SINGLE_LR={single_lr_env} — running one LR only")

    results: list[tuple[float, float, float, int]] = []
    for lr in sweep_values:
        print(f"\n========== [tune-lr] peak_lr={lr:.2e} ==========")
        # Fresh model per LR so weights don't leak between runs.
        model = build_model(n_classes=n_classes)
        history = train(
            model, x_train, y_train, x_val, y_val,
            epochs=TUNE_LR_EPOCHS,
            batch_size=BATCH_SIZE,
            checkpoint_dir=None,  # don't overwrite the real pretrain checkpoint
            peak_lr=lr,
        )
        best_val_acc = float(max(history.history["val_accuracy"]))
        best_val_loss = float(min(history.history["val_loss"]))
        epochs_ran = len(history.history["val_accuracy"])
        results.append((lr, best_val_acc, best_val_loss, epochs_ran))
        print(
            f"[RESULT] lr={lr:.6e} best_val_acc={best_val_acc:.6f} "
            f"best_val_loss={best_val_loss:.6f} epochs={epochs_ran}"
        )

    print("\n--- LR sweep summary (sorted by best val_accuracy desc) ---")
    results.sort(key=lambda r: r[1], reverse=True)
    print(f"{'peak_lr':>10}  {'best_val_acc':>13}  {'best_val_loss':>14}  {'epochs':>7}")
    print("-" * 50)
    for lr, acc, loss, ep in results:
        print(f"{lr:>10.2e}  {acc:>13.4f}  {loss:>14.4f}  {ep:>7d}")
    best_lr = results[0][0]
    print(f"\n[tune-lr] Best peak_lr: {best_lr:.2e} "
          f"(val_accuracy={results[0][1]:.4f}). "
          f"Set PEAK_LR={best_lr:.2e}, TUNE_LR=False, and rerun to pretrain.")


def main() -> None:
    if MODE == "pretrain":
        if TUNE_LR:
            _tune_lr_pretrain()
        else:
            _pretrain()
    elif MODE == "finetune":
        _finetune_animals()
    else:
        raise ValueError(f"MODE must be 'pretrain' or 'finetune', got {MODE!r}")


if __name__ == "__main__":
    main()
