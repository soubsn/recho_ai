"""
Variational Autoencoder (VAE) — Probabilistic Anomaly Detection.

Same encoder/decoder architecture as autoencoder.py but with a learned
latent distribution (mean + log_variance) instead of a deterministic bottleneck.

The reparameterisation trick: z = mu + epsilon * exp(0.5 * log_var)
allows gradients to flow through the sampling operation.

Loss = reconstruction_loss + KL_divergence
     = MSE(x, x_hat) + 0.5 * sum(exp(log_var) + mu^2 - 1 - log_var)

Anomaly score = reconstruction_loss + KL_divergence (ELBO, negated)

Advantage over plain autoencoder:
    The latent space is smooth and continuous (KL divergence regularises it
    towards a standard normal), making anomaly scores more reliable and less
    sensitive to the exact threshold value.

Input:  [n_clips, 200, 100, 1] — x(t) feature map
Output: [n_clips, 200, 100, 1] — reconstructed feature map

MCU note: heavier than plain autoencoder — M85 recommended.
The encoder alone can run on M55 (for 32-dim bottleneck).

Keras checkpoint: checkpoints/vae.keras

Reference:
  Kingma, D.P. & Welling, M. (2014) Auto-Encoding Variational Bayes. ICLR.
  Shougat et al., Scientific Reports 2023 (paper 2) — x(t) feature maps as
  input representation for the Hopf oscillator reservoir computer.
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
INPUT_HEIGHT: int = 200
INPUT_WIDTH: int = 100
LATENT_DIM: int = 32  # multiple of 4, matching autoencoder bottleneck
ANOMALY_PERCENTILE: float = 95.0


def build_vae(
    input_shape: tuple = (INPUT_HEIGHT, INPUT_WIDTH, 1),
    latent_dim: int = LATENT_DIM,
) -> tuple:
    """
    Build the VAE with encoder (mean/log_var) and decoder.

    Args:
        input_shape: (H, W, C) input tensor shape.
        latent_dim: latent space dimensionality.

    Returns:
        (vae_model, encoder_model) — both are Keras Model objects.
    """
    import tensorflow as tf
    from tensorflow import keras

    # -------------------------------------------------------------------------
    # Encoder
    # -------------------------------------------------------------------------
    enc_inp = keras.Input(shape=input_shape, name="vae_input")

    # arm_convolve_s8() — stride=2 downsamples spatial dims
    # CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c
    x = keras.layers.Conv2D(
        16, (3, 3), strides=2, padding="same", activation="relu",
        use_bias=True, name="vae_enc_conv1_arm_convolve_s8",
    )(enc_inp)  # → (100, 50, 16)

    # arm_convolve_s8()
    x = keras.layers.Conv2D(
        32, (3, 3), strides=2, padding="same", activation="relu",
        use_bias=True, name="vae_enc_conv2_arm_convolve_s8",
    )(x)  # → (50, 25, 32)

    x = keras.layers.Flatten(name="vae_enc_flatten")(x)

    # Reparameterisation: output mean and log_variance
    # arm_fully_connected_s8()
    z_mean = keras.layers.Dense(
        latent_dim, use_bias=True, name="z_mean_arm_fully_connected_s8",
    )(x)
    z_log_var = keras.layers.Dense(
        latent_dim, use_bias=True, name="z_log_var_arm_fully_connected_s8",
    )(x)

    # Sampling layer with reparameterisation trick
    class Sampling(keras.layers.Layer):
        """Reparameterisation trick: z = mu + epsilon * exp(0.5 * log_var)."""

        def call(self, inputs: tuple) -> tf.Tensor:
            mu, log_var = inputs
            epsilon = tf.random.normal(shape=tf.shape(mu))
            return mu + tf.exp(0.5 * log_var) * epsilon

    z = Sampling(name="vae_sampling")([z_mean, z_log_var])
    encoder = keras.Model(
        inputs=enc_inp,
        outputs=[z_mean, z_log_var, z],
        name="vae_encoder",
    )

    # -------------------------------------------------------------------------
    # Decoder
    # -------------------------------------------------------------------------
    dec_inp = keras.Input(shape=(latent_dim,), name="vae_dec_input")
    y = keras.layers.Dense(
        50 * 25 * 32, use_bias=True, name="vae_dec_dense",
    )(dec_inp)
    y = keras.layers.Reshape((50, 25, 32), name="vae_dec_reshape")(y)
    y = keras.layers.Conv2DTranspose(
        16, (3, 3), strides=2, padding="same", activation="relu",
        use_bias=True, name="vae_dec_deconv1",
    )(y)
    y = keras.layers.Conv2DTranspose(
        1, (3, 3), strides=2, padding="same", activation="sigmoid",
        use_bias=True, name="vae_dec_deconv2",
    )(y)
    decoder = keras.Model(inputs=dec_inp, outputs=y, name="vae_decoder")

    # -------------------------------------------------------------------------
    # VAE model with custom loss
    # -------------------------------------------------------------------------
    class VAEModel(keras.Model):
        """VAE with KL-regularised training loss."""

        def __init__(self, encoder: keras.Model, decoder: keras.Model) -> None:
            super().__init__(name="vae")
            self.encoder = encoder
            self.decoder = decoder

        def call(self, inputs: tf.Tensor) -> tf.Tensor:
            z_mean, z_log_var, z = self.encoder(inputs)
            return self.decoder(z)

        def train_step(self, data: tf.Tensor) -> dict:
            x = data
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(x, training=True)
                x_recon = self.decoder(z, training=True)
                # Reconstruction loss (MSE × pixel count)
                recon_loss = tf.reduce_mean(
                    tf.reduce_sum(tf.square(x - x_recon), axis=[1, 2, 3])
                )
                # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
                kl_loss = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(
                        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                        axis=1,
                    )
                )
                total_loss = recon_loss + kl_loss

            grads = tape.gradient(total_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    vae = VAEModel(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))
    return vae, encoder


class VAEDetector:
    """
    VAE-based anomaly detector for Hopf oscillator feature maps.

    The ELBO (Evidence Lower Bound) anomaly score = reconstruction error + KL.
    Smooth latent space makes the threshold more reliable than plain autoencoder.

    Example:
        detector = VAEDetector(epochs=30)
        detector.fit(normal_feature_maps)
        score = detector.anomaly_score(test_clip)
        if detector.is_anomaly(test_clip):
            alert()
    """

    def __init__(
        self,
        epochs: int = 50,
        batch_size: int = 16,
        latent_dim: int = LATENT_DIM,
        anomaly_percentile: float = ANOMALY_PERCENTILE,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.anomaly_percentile = anomaly_percentile
        self._vae = None
        self._encoder = None
        self._threshold: float = 0.0
        self._is_fitted = False

    def _prepare(self, features: NDArray) -> NDArray[np.float32]:
        """Scale uint8 → [0,1] float32 with channel dim."""
        x = features.astype(np.float32) / 255.0
        if x.ndim == 3:
            x = x[:, :, :, np.newaxis]
        return x

    def _elbo_score(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        """Compute ELBO anomaly scores for a batch (higher = more anomalous)."""
        import tensorflow as tf
        z_mean, z_log_var, z = self._encoder(X, training=False)
        x_recon = self._vae.decoder(z, training=False)
        recon = tf.reduce_mean(
            tf.reduce_sum(tf.square(X - x_recon), axis=[1, 2, 3]),
            keepdims=True,
        )
        kl = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
            axis=1, keepdims=True,
        )
        return (recon + kl).numpy().flatten().astype(np.float64)

    def fit(self, normal_clips: NDArray) -> "VAEDetector":
        """
        Train VAE on normal clips only.

        Args:
            normal_clips: (n_clips, 200, 100) uint8 feature maps

        Returns:
            self
        """
        self._vae, self._encoder = build_vae(latent_dim=self.latent_dim)
        X = self._prepare(normal_clips)

        self._vae.fit(
            X, epochs=self.epochs, batch_size=self.batch_size,
            validation_split=0.1, verbose=1,
        )

        # Compute threshold from training ELBO scores
        scores = self._elbo_score(X)
        self._threshold = float(np.percentile(scores, self.anomaly_percentile))
        print(f"[VAEDetector] Fitted. Threshold={self._threshold:.4f} "
              f"(p{self.anomaly_percentile:.0f})")
        self._is_fitted = True
        return self

    def anomaly_score(self, clip: NDArray) -> float:
        """
        Compute ELBO anomaly score for a single clip.

        Higher score = more anomalous.

        Args:
            clip: (200, 100) uint8 feature map

        Returns:
            ELBO anomaly score (float).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before anomaly_score()")
        X = self._prepare(np.expand_dims(clip, 0))
        return float(self._elbo_score(X)[0])

    def is_anomaly(self, clip: NDArray) -> bool:
        """Return True if ELBO score exceeds threshold."""
        return self.anomaly_score(clip) > self._threshold

    def save(self, path: Optional[str | Path] = None) -> Path:
        """Save VAE to checkpoints/vae.keras."""
        p = Path(path) if path else CHECKPOINT_DIR / "vae.keras"
        p.parent.mkdir(parents=True, exist_ok=True)
        self._vae.save_weights(str(p).replace(".keras", "_weights.h5"))
        print(f"[VAEDetector] Saved weights to {p}")
        return p


def main() -> None:
    """Train VAEDetector on synthetic Hopf data."""
    from data.sample_data import generate_dataset_xy, CLASS_NAMES
    from pipeline.ingest import process_dataset
    from pipeline.features_xy import extract_y_features, extract_all_representations

    print("[vae] Generating synthetic data ...")
    raw_x, raw_y, labels = generate_dataset_xy(
        n_clips_per_class=15, n_classes=5, cache=False,
    )
    x_proc = process_dataset(raw_x)
    y_proc = extract_y_features(raw_y)
    reps = extract_all_representations(x_proc, y_proc)

    normal_mask = labels == 0
    normal_clips = reps["x_only"][normal_mask]
    print(f"[vae] Training VAE on {normal_mask.sum()} normal clips ...")

    detector = VAEDetector(epochs=3, batch_size=4, latent_dim=16)
    detector.fit(normal_clips)

    print("\nAnomaly scores per class:")
    for cls in range(5):
        mask = labels == cls
        clip = reps["x_only"][mask][0]
        score = detector.anomaly_score(clip)
        is_anom = detector.is_anomaly(clip)
        print(f"  Class {cls} ({CLASS_NAMES[cls]}): score={score:.2f}, "
              f"anomaly={is_anom}")

    print("[vae] Done. (Note: M85 recommended for full deployment)")


if __name__ == "__main__":
    main()
