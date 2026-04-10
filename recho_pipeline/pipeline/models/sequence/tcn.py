"""
Temporal Convolutional Network (TCN) on raw x(t) time series.

Processes the raw downsampled x(t) signal directly — no feature map reshape.
Uses dilated causal convolutions so no future samples are used at inference,
making it valid for real-time streaming classification.

Input:  [n_clips, 4000, 1] — raw x(t) at 4 kHz, 1 second = 4000 samples
Output: [n_clips, n_classes]

Architecture:
    Conv1D(32, kernel=3, dilation=1, causal, relu)
    Conv1D(32, kernel=3, dilation=2, causal, relu)
    Conv1D(32, kernel=3, dilation=4, causal, relu)
    Conv1D(64, kernel=3, dilation=8, causal, relu)
    GlobalAveragePooling1D()
    Dense(64, relu)
    Dense(n_classes, softmax)

Causal convolution: padding = (kernel_size-1)*dilation on the LEFT only.
This ensures no future samples contaminate the current prediction.
Dilation = 1,2,4,8 gives receptive field of (3-1)*(1+2+4+8) + 1 = 31 samples
= 7.75 ms at 4 kHz.

CMSIS-NN target:
    arm_convolve_1_x_n_s8() — 1D convolution kernel (specialised for width=1)
    TFLite converts Conv1D to Conv2D internally with height=1
    QAT + TFLite INT8 conversion supported

Advantage over CNN: processes raw signal, no feature map reshape needed.
The Hopf reservoir output x(t) is directly usable as a time series, and the
TCN can learn both temporal and frequency structure natively.

Keras checkpoint: checkpoints/tcn.keras

Reference:
  Bai, S., Kolter, J.Z. & Koltun, V. (2018) An Empirical Evaluation of
  Generic Convolutional and Recurrent Networks for Sequence Modelling.
  Shougat et al., Scientific Reports 2023 (paper 2) — x(t) as reservoir output.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

CHECKPOINT_DIR = ROOT / "checkpoints"
INPUT_TIMESTEPS: int = 4000   # 1 second at 4 kHz
INPUT_FEATURES: int = 1       # x(t) only


def build_model(n_classes: int = 5, input_timesteps: int = INPUT_TIMESTEPS) -> "object":
    """
    Build the TCN model with dilated causal convolutions.

    Each layer maps to CMSIS-NN arm_convolve_1_x_n_s8() for 1D convolution.

    Args:
        n_classes: number of output classes.
        input_timesteps: sequence length (default 4000 = 1 sec at 4 kHz).

    Returns:
        Compiled Keras model with Adam optimiser and categorical crossentropy.
    """
    import tensorflow as tf
    from tensorflow import keras

    inp = keras.Input(shape=(input_timesteps, INPUT_FEATURES))
    x = inp

    # Dilated causal convolution blocks
    # arm_convolve_1_x_n_s8() — 1D causal conv, dilation=1
    # CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.c
    # causal padding: pad (kernel_size-1)*dilation = 2 zeros on the left
    x = keras.layers.ZeroPadding1D(padding=(2, 0))(x)  # causal pad
    x = keras.layers.Conv1D(
        32, kernel_size=3, dilation_rate=1, padding="valid",
        activation="relu", use_bias=True,
        name="tcn_conv1_d1_arm_convolve_1_x_n_s8",
    )(x)

    # arm_convolve_1_x_n_s8() — dilation=2
    # CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.c
    x = keras.layers.ZeroPadding1D(padding=(4, 0))(x)  # causal pad: (3-1)*2=4
    x = keras.layers.Conv1D(
        32, kernel_size=3, dilation_rate=2, padding="valid",
        activation="relu", use_bias=True,
        name="tcn_conv2_d2_arm_convolve_1_x_n_s8",
    )(x)

    # arm_convolve_1_x_n_s8() — dilation=4
    # CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.c
    x = keras.layers.ZeroPadding1D(padding=(8, 0))(x)  # causal pad: (3-1)*4=8
    x = keras.layers.Conv1D(
        32, kernel_size=3, dilation_rate=4, padding="valid",
        activation="relu", use_bias=True,
        name="tcn_conv3_d4_arm_convolve_1_x_n_s8",
    )(x)

    # arm_convolve_1_x_n_s8() — dilation=8, wider channel expansion
    # CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.c
    x = keras.layers.ZeroPadding1D(padding=(16, 0))(x)  # causal pad: (3-1)*8=16
    x = keras.layers.Conv1D(
        64, kernel_size=3, dilation_rate=8, padding="valid",
        activation="relu", use_bias=True,
        name="tcn_conv4_d8_arm_convolve_1_x_n_s8",
    )(x)

    # Global average pooling — reduces temporal dim to 1
    # Maps to arm_average_pool_s8() with global=True
    x = keras.layers.GlobalAveragePooling1D(name="tcn_gap_arm_average_pool_s8")(x)

    # arm_fully_connected_s8() — 64 units (multiple of 4)
    # CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.c
    x = keras.layers.Dense(
        64, activation="relu", use_bias=True,
        name="tcn_dense1_arm_fully_connected_s8",
    )(x)

    # arm_fully_connected_s8() + arm_softmax_s8()
    # CMSIS-NN/Source/SoftmaxFunctions/arm_softmax_s8.c
    out = keras.layers.Dense(
        n_classes, activation="softmax", use_bias=True,
        name="tcn_output_arm_softmax_s8",
    )(x)

    model = keras.Model(inputs=inp, outputs=out, name="tcn_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class TCNClassifier:
    """
    Temporal Convolutional Network wrapper with train/predict interface.

    Example:
        clf = TCNClassifier(n_classes=5, epochs=20)
        clf.fit(x_raw_ds, labels)
        preds = clf.predict(x_raw_ds_test)
    """

    def __init__(
        self,
        n_classes: int = 5,
        epochs: int = 50,
        batch_size: int = 16,
        val_split: float = 0.2,
    ) -> None:
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_split = val_split
        self._model = None
        self._is_fitted = False

    def fit(
        self,
        x_clips: NDArray[np.float64],
        labels: NDArray[np.int64],
    ) -> "TCNClassifier":
        """
        Build and train TCN on downsampled x(t) clips.

        Args:
            x_clips: (n_clips, n_samples) downsampled x(t) at 4 kHz
            labels: (n_clips,) integer class labels

        Returns:
            self
        """
        import tensorflow as tf
        from tensorflow import keras

        n_steps = x_clips.shape[1]
        self._model = build_model(
            n_classes=self.n_classes, input_timesteps=n_steps,
        )

        X = x_clips.astype(np.float32)[:, :, np.newaxis]  # (n, T, 1)
        y = keras.utils.to_categorical(labels, num_classes=self.n_classes)

        ckpt_path = CHECKPOINT_DIR / "tcn.keras"
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                str(ckpt_path), monitor="val_accuracy",
                save_best_only=True, verbose=0,
            )
        ]

        self._model.fit(
            X, y, epochs=self.epochs, batch_size=self.batch_size,
            validation_split=self.val_split, callbacks=callbacks, verbose=1,
        )
        self._is_fitted = True
        return self

    def predict(self, x_clips: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict class labels."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")
        X = x_clips.astype(np.float32)[:, :, np.newaxis]
        probs = self._model.predict(X, verbose=0)
        return np.argmax(probs, axis=1).astype(np.int64)

    def score(
        self,
        x_clips: NDArray[np.float64],
        labels: NDArray[np.int64],
    ) -> float:
        """Return classification accuracy."""
        from sklearn.metrics import accuracy_score
        return float(accuracy_score(labels, self.predict(x_clips)))


def main() -> None:
    """Train TCN on synthetic Hopf oscillator data."""
    from data.sample_data import generate_dataset_xy, CLASS_NAMES

    print("[tcn] Generating synthetic data ...")
    raw_x, raw_y, labels = generate_dataset_xy(
        n_clips_per_class=10, n_classes=5, cache=False,
    )

    # Downsample to 4 kHz (factor 25): 4000 samples per clip
    x_ds = raw_x[:, ::25].astype(np.float64)
    print(f"  Input shape: {x_ds.shape}")

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(labels))
    n_val = int(0.2 * len(labels))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    clf = TCNClassifier(n_classes=5, epochs=5, batch_size=8, val_split=0.2)
    clf.fit(x_ds[train_idx], labels[train_idx])

    val_acc = clf.score(x_ds[val_idx], labels[val_idx])
    print(f"\n  Val accuracy: {val_acc:.4f}")
    print(f"\nModel summary:")
    clf._model.summary()
    print("[tcn] Done.")


if __name__ == "__main__":
    main()
