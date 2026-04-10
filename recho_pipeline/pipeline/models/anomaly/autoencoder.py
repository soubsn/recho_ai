"""
Convolutional Autoencoder — Unsupervised Anomaly Detection.

Trained ONLY on normal operation data. Anomaly = high reconstruction error.
The autoencoder learns to compress and reconstruct the normal Hopf feature map;
fault conditions produce feature maps the encoder cannot compress well, resulting
in high MSE between input and reconstruction.

Input:  [n_clips, 200, 100, 1] — x(t) feature map (uint8 scaled to float32)
Output: [n_clips, 200, 100, 1] — reconstructed feature map

Encoder:
    Conv2D(16, 3×3, relu, stride=2)   → (100, 50, 16)
    # arm_convolve_s8()
    Conv2D(32, 3×3, relu, stride=2)   → (50, 25, 32)
    # arm_convolve_s8()
    Flatten → Dense(32)               # bottleneck (32-dimensional)

Decoder:
    Dense(50*25*32) → Reshape(50, 25, 32)
    Conv2DTranspose(16, 3×3, relu, stride=2)  → (~100, ~50, 16)
    Conv2DTranspose(1, 3×3, sigmoid, stride=2) → (~200, ~100, 1)

Anomaly score: MSE between input and reconstruction.
Threshold: 95th percentile of reconstruction error on training set.

Deployment note: the encoder can be deployed standalone on M33/M55 as a
feature compressor (32-dim bottleneck). The reconstruction step is not needed
at inference — only the encoder + MSE comparison to stored normal templates.

Keras checkpoint: checkpoints/autoencoder.keras

Reference:
  Shougat et al., Scientific Reports 2021/2023 — Hopf oscillator feature map
  structure. The autoencoder learns the manifold of normal feature maps.
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
INPUT_HEIGHT: int = 200   # time steps — multiple of 4
INPUT_WIDTH: int = 100    # virtual nodes — multiple of 4
BOTTLENECK_DIM: int = 32  # compressed representation (multiple of 4)
ANOMALY_PERCENTILE: float = 95.0


def build_autoencoder(input_shape: tuple = (INPUT_HEIGHT, INPUT_WIDTH, 1)) -> tuple:
    """
    Build the convolutional autoencoder (encoder + decoder).

    Args:
        input_shape: (H, W, C) input tensor shape.

    Returns:
        (autoencoder, encoder) — both are Keras Model objects.
    """
    import tensorflow as tf
    from tensorflow import keras

    inp = keras.Input(shape=input_shape, name="ae_input")

    # -------------------------------------------------------------------------
    # Encoder
    # -------------------------------------------------------------------------
    # arm_convolve_s8() — stride=2 reduces spatial dims by 2x
    # CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c
    x = keras.layers.Conv2D(
        16, (3, 3), strides=2, padding="same", activation="relu",
        use_bias=True, name="enc_conv1_arm_convolve_s8",
    )(inp)  # → (100, 50, 16)

    # arm_convolve_s8()
    # CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c
    x = keras.layers.Conv2D(
        32, (3, 3), strides=2, padding="same", activation="relu",
        use_bias=True, name="enc_conv2_arm_convolve_s8",
    )(x)  # → (50, 25, 32)

    # arm_reshape_s8() + arm_fully_connected_s8()
    x = keras.layers.Flatten(name="enc_flatten")(x)
    bottleneck = keras.layers.Dense(
        BOTTLENECK_DIM, use_bias=True, name="bottleneck_arm_fully_connected_s8",
    )(x)  # 32-dimensional compressed representation

    encoder = keras.Model(inputs=inp, outputs=bottleneck, name="encoder")

    # -------------------------------------------------------------------------
    # Decoder
    # -------------------------------------------------------------------------
    dec_inp = keras.Input(shape=(BOTTLENECK_DIM,), name="dec_input")
    y = keras.layers.Dense(
        50 * 25 * 32, use_bias=True, name="dec_dense_arm_fully_connected_s8",
    )(dec_inp)
    y = keras.layers.Reshape((50, 25, 32), name="dec_reshape")(y)

    # Conv2DTranspose upsamples by stride=2
    # Note: Conv2DTranspose is not directly CMSIS-NN; use for M55/M85 only.
    # For M33: approximate with UpSampling2D + Conv2D.
    y = keras.layers.Conv2DTranspose(
        16, (3, 3), strides=2, padding="same", activation="relu",
        use_bias=True, name="dec_deconv1",
    )(y)  # → (100, 50, 16)

    y = keras.layers.Conv2DTranspose(
        1, (3, 3), strides=2, padding="same", activation="sigmoid",
        use_bias=True, name="dec_deconv2",
    )(y)  # → (200, 100, 1)

    decoder = keras.Model(inputs=dec_inp, outputs=y, name="decoder")

    # Full autoencoder
    ae_out = decoder(encoder(inp))
    autoencoder = keras.Model(inputs=inp, outputs=ae_out, name="autoencoder")
    autoencoder.compile(
        optimizer="adam", loss="mse",
    )
    return autoencoder, encoder


class AnomalyAutoencoder:
    """
    Convolutional autoencoder for unsupervised Hopf oscillator anomaly detection.

    Train on normal clips only. At inference, compute the MSE between the
    input and its reconstruction. High MSE → anomaly.

    Example:
        ae = AnomalyAutoencoder()
        ae.fit(normal_feature_maps)
        score = ae.reconstruction_error(test_clip)
        if ae.is_anomaly(test_clip):
            alert()
        ae.visualise_reconstruction(test_clip)
    """

    def __init__(
        self,
        epochs: int = 50,
        batch_size: int = 16,
        anomaly_percentile: float = ANOMALY_PERCENTILE,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.anomaly_percentile = anomaly_percentile
        self._autoencoder = None
        self._encoder = None
        self._threshold: float = 0.0
        self._is_fitted = False

    def _prepare(self, features: NDArray) -> NDArray[np.float32]:
        """Scale uint8 feature maps to [0, 1] float32 with channel dim."""
        x = features.astype(np.float32) / 255.0
        if x.ndim == 3:
            x = x[:, :, :, np.newaxis]
        return x

    def fit(self, normal_clips: NDArray) -> "AnomalyAutoencoder":
        """
        Build and train autoencoder on NORMAL clips only.

        Args:
            normal_clips: (n_clips, 200, 100) uint8 feature maps from normal operation

        Returns:
            self
        """
        self._autoencoder, self._encoder = build_autoencoder()
        X = self._prepare(normal_clips)

        ckpt_path = CHECKPOINT_DIR / "autoencoder.keras"
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        import tensorflow.keras as keras
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                str(ckpt_path), monitor="val_loss",
                save_best_only=True, verbose=0,
            )
        ]

        self._autoencoder.fit(
            X, X, epochs=self.epochs, batch_size=self.batch_size,
            validation_split=0.1, callbacks=callbacks, verbose=1,
        )

        # Compute threshold from training set reconstruction errors
        recon = self._autoencoder.predict(X, verbose=0)
        errors = np.mean((X - recon) ** 2, axis=(1, 2, 3))
        self._threshold = float(np.percentile(errors, self.anomaly_percentile))
        print(f"[AnomalyAutoencoder] Fitted. Threshold={self._threshold:.6f} "
              f"(p{self.anomaly_percentile:.0f})")
        self._is_fitted = True
        return self

    def reconstruction_error(self, clip: NDArray) -> float:
        """
        Compute MSE reconstruction error for a single clip.

        Args:
            clip: (200, 100) uint8 feature map for one clip

        Returns:
            Mean squared reconstruction error (float).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before reconstruction_error()")
        X = self._prepare(np.expand_dims(clip, 0))
        recon = self._autoencoder.predict(X, verbose=0)
        return float(np.mean((X - recon) ** 2))

    def is_anomaly(self, clip: NDArray) -> bool:
        """Return True if reconstruction error exceeds threshold."""
        return self.reconstruction_error(clip) > self._threshold

    def visualise_reconstruction(self, clip: NDArray, title: str = "") -> None:
        """
        Plot the input feature map and its reconstruction side by side.

        Args:
            clip: (200, 100) uint8 feature map
            title: optional title prefix
        """
        import matplotlib.pyplot as plt

        X = self._prepare(np.expand_dims(clip, 0))
        recon = self._autoencoder.predict(X, verbose=0)
        mse = float(np.mean((X - recon) ** 2))

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        axes[0].imshow(X[0, :, :, 0], cmap="viridis", aspect="auto")
        axes[0].set_title(f"{title}Input")
        axes[1].imshow(recon[0, :, :, 0], cmap="viridis", aspect="auto")
        axes[1].set_title("Reconstruction")
        diff = np.abs(X[0, :, :, 0] - recon[0, :, :, 0])
        axes[2].imshow(diff, cmap="hot", aspect="auto")
        axes[2].set_title(f"Absolute Difference\nMSE={mse:.6f}")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def save(self, path: Optional[str | Path] = None) -> Path:
        """Save autoencoder to checkpoints/autoencoder.keras."""
        p = Path(path) if path else CHECKPOINT_DIR / "autoencoder.keras"
        p.parent.mkdir(parents=True, exist_ok=True)
        self._autoencoder.save(str(p))
        print(f"[AnomalyAutoencoder] Saved to {p}")
        return p


def main() -> None:
    """Train AnomalyAutoencoder on normal-only Hopf data."""
    from data.sample_data import generate_dataset_xy, CLASS_NAMES
    from pipeline.ingest import process_dataset
    from pipeline.features_xy import extract_y_features, extract_all_representations

    print("[autoencoder] Generating synthetic data ...")
    raw_x, raw_y, labels = generate_dataset_xy(
        n_clips_per_class=15, n_classes=5, cache=False,
    )
    x_proc = process_dataset(raw_x)
    y_proc = extract_y_features(raw_y)
    reps = extract_all_representations(x_proc, y_proc)

    # Train on class 0 only (sine = normal)
    normal_mask = labels == 0
    normal_clips = reps["x_only"][normal_mask]
    print(f"[autoencoder] Training on {normal_mask.sum()} normal clips ...")

    ae = AnomalyAutoencoder(epochs=3, batch_size=4)
    ae.fit(normal_clips)

    # Test anomaly detection
    print("\nAnomaly detection per class:")
    for cls in range(5):
        mask = labels == cls
        clip = reps["x_only"][mask][0]
        is_anom = ae.is_anomaly(clip)
        err = ae.reconstruction_error(clip)
        print(f"  Class {cls} ({CLASS_NAMES[cls]}): "
              f"error={err:.6f}, anomaly={is_anom}")

    ae.save()
    print("[autoencoder] Done.")


if __name__ == "__main__":
    main()
