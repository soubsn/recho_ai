"""
LSTM Classifier on x(t) and y(t) as a two-feature time series.

Input:  [n_clips, 4000, 2] — x and y as features at each time step
Output: [n_clips, n_classes]

y(t) as a second feature alongside x(t) gives the LSTM access to the full
phase space trajectory of the Hopf oscillator, not just one projection.
The LSTM can learn to exploit the timing relationship between x(t) and y(t)
(they are in quadrature: y lags x by pi/2 at the limit cycle frequency).

Architecture:
    LSTM(64, return_sequences=True)
    LSTM(32, return_sequences=False)
    Dense(64, relu)
    Dense(n_classes, softmax)

MCU deployment note:
    TFLite Micro supports LSTM via the LSTMFull op.
    LSTM is computationally heavy: 4 gates × 2 matmuls per timestep.
    Recommended for M55 or M85 targets (256 KB+ RAM).
    On M33 (64 KB), this model likely exceeds available RAM.
    For M33, use TCN or depthwise CNN instead.

    RAM estimate (rough): LSTM(64) × 4 gates × (input + hidden) × int8
        ≈ (64+2) × 64 × 4 × 1 byte = 16,896 bytes per layer
        Plus activation buffer: 4000 × 2 × 4 bytes = 32,000 bytes
        Total ≈ 100+ KB — tight for M33, comfortable for M55.

Keras checkpoint: checkpoints/lstm.keras

Reference:
  Shougat et al., Scientific Reports 2023 (paper 2) — x(t) and y(t) states
  of the Hopf oscillator; y(t) "likely stores information" (paper 2, p.3).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

CHECKPOINT_DIR = ROOT / "checkpoints"
INPUT_TIMESTEPS: int = 4000  # 1 second at 4 kHz
INPUT_FEATURES: int = 2      # x(t) and y(t) as two channels


def build_model(n_classes: int = 5, input_timesteps: int = INPUT_TIMESTEPS) -> "object":
    """
    Build the two-layer LSTM classifier.

    Note: TFLite Micro supports LSTM via the LSTMFull op.
    Recommend M55 or M85 targets due to RAM requirements.

    Args:
        n_classes: number of output classes.
        input_timesteps: sequence length (default 4000 = 1 sec at 4 kHz).

    Returns:
        Compiled Keras model.
    """
    import tensorflow as tf
    from tensorflow import keras

    model = keras.Sequential([
        keras.Input(shape=(input_timesteps, INPUT_FEATURES)),

        # LSTM layer 1 — TFLite Micro: LSTMFull op
        # 4 gates: i, f, g, o — each is a dense(input+hidden) multiply-add
        # On M55: Helium MVE acceleration via arm_nn_vec_mat_mult_t_s8()
        keras.layers.LSTM(
            64, return_sequences=True,
            name="lstm1_tflite_lstmfull_op",
        ),

        # LSTM layer 2 — TFLite Micro: LSTMFull op
        # return_sequences=False: only final hidden state passed forward
        keras.layers.LSTM(
            32, return_sequences=False,
            name="lstm2_tflite_lstmfull_op",
        ),

        # arm_fully_connected_s8() — 64 units (multiple of 4)
        # CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.c
        keras.layers.Dense(
            64, activation="relu", use_bias=True,
            name="dense1_arm_fully_connected_s8",
        ),

        # arm_fully_connected_s8() + arm_softmax_s8()
        # CMSIS-NN/Source/SoftmaxFunctions/arm_softmax_s8.c
        keras.layers.Dense(
            n_classes, activation="softmax", use_bias=True,
            name="output_arm_softmax_s8",
        ),
    ], name="lstm_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class LSTMClassifier:
    """
    LSTM classifier on raw x(t) + y(t) time series.

    Passes x(t) and y(t) as two features at each time step, giving the
    LSTM access to the full phase space trajectory.

    Memory note: LSTM is heavy — recommend M55 or M85 for deployment.
    On M33 (64 KB RAM), use TCN or depthwise CNN instead.

    Example:
        clf = LSTMClassifier(n_classes=5, epochs=30)
        clf.fit(x_ds, y_ds, labels)
        preds = clf.predict(x_ds_test, y_ds_test)
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

    def _prepare(
        self,
        x_clips: NDArray[np.float64],
        y_clips: NDArray[np.float64],
    ) -> NDArray[np.float32]:
        """Stack x and y as two features, return (n, T, 2) float32."""
        return np.stack([x_clips, y_clips], axis=-1).astype(np.float32)

    def fit(
        self,
        x_clips: NDArray[np.float64],
        y_clips: NDArray[np.float64],
        labels: NDArray[np.int64],
    ) -> "LSTMClassifier":
        """
        Build and train LSTM on x(t) + y(t) clips.

        Args:
            x_clips: (n_clips, n_samples) downsampled x(t) at 4 kHz
            y_clips: (n_clips, n_samples) downsampled y(t) at 4 kHz
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

        X = self._prepare(x_clips, y_clips)
        y = keras.utils.to_categorical(labels, num_classes=self.n_classes)

        ckpt_path = CHECKPOINT_DIR / "lstm.keras"
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

    def predict(
        self,
        x_clips: NDArray[np.float64],
        y_clips: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        """Predict class labels."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")
        X = self._prepare(x_clips, y_clips)
        probs = self._model.predict(X, verbose=0)
        return np.argmax(probs, axis=1).astype(np.int64)

    def score(
        self,
        x_clips: NDArray[np.float64],
        y_clips: NDArray[np.float64],
        labels: NDArray[np.int64],
    ) -> float:
        """Return classification accuracy."""
        from sklearn.metrics import accuracy_score
        return float(accuracy_score(labels, self.predict(x_clips, y_clips)))


def main() -> None:
    """Train LSTMClassifier on synthetic Hopf data (small set for demo)."""
    from data.sample_data import generate_dataset_xy, CLASS_NAMES

    print("[lstm] Generating synthetic data ...")
    # Use very short clips (first 400 samples) to keep demo fast
    raw_x, raw_y, labels = generate_dataset_xy(
        n_clips_per_class=10, n_classes=5, cache=False,
    )

    # Downsample to 4 kHz and use first 400 samples (0.1 sec) for speed
    x_ds = raw_x[:, ::25, ][:, :400].astype(np.float64)
    y_ds = raw_y[:, ::25][:, :400].astype(np.float64)
    print(f"  Input shape: {x_ds.shape} (using first 400 of 4000 samples for demo)")

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(labels))
    n_val = int(0.2 * len(labels))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    clf = LSTMClassifier(n_classes=5, epochs=3, batch_size=8, val_split=0.2)
    clf.fit(x_ds[train_idx], y_ds[train_idx], labels[train_idx])

    val_acc = clf.score(x_ds[val_idx], y_ds[val_idx], labels[val_idx])
    print(f"\n  Val accuracy: {val_acc:.4f}")
    print(f"  RAM note: LSTM is heavy — recommend M55/M85 for deployment")
    print("[lstm] Done.")


if __name__ == "__main__":
    main()
