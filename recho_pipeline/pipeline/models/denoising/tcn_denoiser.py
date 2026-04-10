"""
Causal TCN denoiser on Hopf reservoir x(t)/y(t) sequences.

Input:  [n_clips, T, 2] — noisy reservoir sequence with x(t), y(t)
Output: [n_clips, T, 1] — clean waveform target aligned to the same time base

This is intentionally separate from the sequence classifiers. It uses causal
residual Conv1D blocks to reconstruct a clean target waveform from the noisy
reservoir response, enabling streaming denoising on M55/M85-class MCUs.
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
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "denoiser_tcn.keras"
INPUT_FEATURES: int = 2
DEFAULT_DILATIONS: tuple[int, ...] = (1, 2, 4, 8)


def receptive_field(
    kernel_size: int = 3,
    dilations: tuple[int, ...] = DEFAULT_DILATIONS,
    convs_per_block: int = 2,
) -> int:
    """Return the causal receptive field in samples."""
    return 1 + convs_per_block * (kernel_size - 1) * sum(dilations)


def si_sdr_db_numpy(
    y_true: NDArray[np.float32] | NDArray[np.float64],
    y_pred: NDArray[np.float32] | NDArray[np.float64],
    eps: float = 1e-8,
) -> NDArray[np.float64]:
    """Compute SI-SDR in dB for a batch of waveforms."""
    ref = np.asarray(y_true, dtype=np.float64)
    est = np.asarray(y_pred, dtype=np.float64)
    if ref.ndim == 3 and ref.shape[-1] == 1:
        ref = ref[..., 0]
    if est.ndim == 3 and est.shape[-1] == 1:
        est = est[..., 0]

    dot = np.sum(est * ref, axis=1, keepdims=True)
    ref_energy = np.sum(ref ** 2, axis=1, keepdims=True) + eps
    s_target = dot / ref_energy * ref
    e_noise = est - s_target
    ratio = (np.sum(s_target ** 2, axis=1) + eps) / (np.sum(e_noise ** 2, axis=1) + eps)
    return 10.0 * np.log10(ratio)


def snr_db_numpy(
    y_true: NDArray[np.float32] | NDArray[np.float64],
    y_pred: NDArray[np.float32] | NDArray[np.float64],
    eps: float = 1e-8,
) -> NDArray[np.float64]:
    """Compute SNR in dB for a batch of waveforms."""
    ref = np.asarray(y_true, dtype=np.float64)
    est = np.asarray(y_pred, dtype=np.float64)
    if ref.ndim == 3 and ref.shape[-1] == 1:
        ref = ref[..., 0]
    if est.ndim == 3 and est.shape[-1] == 1:
        est = est[..., 0]

    noise = ref - est
    signal_power = np.sum(ref ** 2, axis=1) + eps
    noise_power = np.sum(noise ** 2, axis=1) + eps
    return 10.0 * np.log10(signal_power / noise_power)


def _custom_objects() -> dict[str, object]:
    """TensorFlow custom objects needed when loading the denoiser."""
    import tensorflow as tf

    def si_sdr_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        ref = tf.squeeze(tf.cast(y_true, tf.float32), axis=-1)
        est = tf.squeeze(tf.cast(y_pred, tf.float32), axis=-1)
        eps = tf.constant(1e-8, dtype=tf.float32)
        dot = tf.reduce_sum(est * ref, axis=1, keepdims=True)
        ref_energy = tf.reduce_sum(tf.square(ref), axis=1, keepdims=True) + eps
        s_target = dot / ref_energy * ref
        e_noise = est - s_target
        ratio = (tf.reduce_sum(tf.square(s_target), axis=1) + eps) / (
            tf.reduce_sum(tf.square(e_noise), axis=1) + eps
        )
        return tf.reduce_mean(10.0 / tf.math.log(10.0) * tf.math.log(ratio))

    def combined_denoising_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        si_sdr = si_sdr_metric(y_true, y_pred)
        si_sdr_penalty = 1.0 - (si_sdr / 20.0)
        return 0.5 * mae + 0.5 * si_sdr_penalty

    return {
        "combined_denoising_loss": combined_denoising_loss,
        "si_sdr_metric": si_sdr_metric,
    }


def build_model(
    input_timesteps: int,
    input_features: int = INPUT_FEATURES,
    base_channels: int = 32,
    dilations: tuple[int, ...] = DEFAULT_DILATIONS,
) -> "object":
    """
    Build a causal residual TCN denoiser.

    Implementation note:
    `ZeroPadding1D(..., 0)` + `Conv1D(..., padding="valid")` is used instead of
    `padding="causal"` so the graph stays closer to the CMSIS/TFLite kernel map
    used elsewhere in the package.
    """
    import tensorflow as tf
    from tensorflow import keras

    custom = _custom_objects()
    combined_denoising_loss = custom["combined_denoising_loss"]
    si_sdr_metric = custom["si_sdr_metric"]

    inp = keras.Input(shape=(input_timesteps, input_features), name="denoise_input")
    x = keras.layers.Conv1D(
        base_channels, kernel_size=1, padding="same", activation="relu",
        name="denoise_proj_arm_convolve_1_x_n_s8",
    )(inp)

    for idx, dilation in enumerate(dilations, start=1):
        residual = x
        pad = (3 - 1) * dilation
        y = keras.layers.ZeroPadding1D(
            padding=(pad, 0),
            name=f"denoise_tcn_block{idx}_pad1_d{dilation}",
        )(x)
        y = keras.layers.Conv1D(
            base_channels, kernel_size=3, dilation_rate=dilation, padding="valid",
            activation="relu", name=f"denoise_tcn_block{idx}_conv1_d{dilation}",
        )(y)
        y = keras.layers.ZeroPadding1D(
            padding=(pad, 0),
            name=f"denoise_tcn_block{idx}_pad2_d{dilation}",
        )(y)
        y = keras.layers.Conv1D(
            base_channels, kernel_size=3, dilation_rate=dilation, padding="valid",
            activation="relu", name=f"denoise_tcn_block{idx}_conv2_d{dilation}",
        )(y)
        x = keras.layers.Add(name=f"denoise_tcn_block{idx}_residual")([residual, y])

    out = keras.layers.Conv1D(
        1, kernel_size=1, padding="same", activation="tanh",
        name="denoise_output_arm_convolve_1_x_n_s8",
    )(x)

    model = keras.Model(inputs=inp, outputs=out, name="tcn_denoiser")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=combined_denoising_loss,
        metrics=[keras.metrics.MeanAbsoluteError(name="mae"), si_sdr_metric],
    )
    return model


def representative_data_gen(
    noisy_inputs: NDArray[np.float32] | NDArray[np.float64],
    n_samples: int = 100,
):
    """Representative dataset generator for TFLite calibration."""
    indices = np.random.default_rng(42).choice(
        len(noisy_inputs), size=min(n_samples, len(noisy_inputs)), replace=False,
    )
    for idx in indices:
        sample = noisy_inputs[idx].astype(np.float32)[np.newaxis, ...]
        yield [sample]


class TCNDenoiser:
    """Wrapper around the causal TCN denoiser with train/predict utilities."""

    def __init__(
        self,
        epochs: int = 30,
        batch_size: int = 16,
        base_channels: int = 32,
        dilations: tuple[int, ...] = DEFAULT_DILATIONS,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.base_channels = base_channels
        self.dilations = dilations
        self._model = None
        self._is_fitted = False
        self.history_: dict[str, list[float]] = {}

    def _prepare_inputs(self, noisy_inputs: NDArray) -> NDArray[np.float32]:
        x = noisy_inputs.astype(np.float32)
        if x.ndim == 2:
            x = x[np.newaxis, ...]
        return x

    def _prepare_targets(self, clean_targets: NDArray) -> NDArray[np.float32]:
        y = clean_targets.astype(np.float32)
        if y.ndim == 2:
            y = y[:, :, np.newaxis]
        return y

    def fit(
        self,
        noisy_inputs: NDArray,
        clean_targets: NDArray,
        validation_data: Optional[tuple[NDArray, NDArray]] = None,
        checkpoint_path: Optional[str | Path] = None,
    ) -> "TCNDenoiser":
        """Train the denoiser on paired noisy/clean sequences."""
        from tensorflow import keras

        x = self._prepare_inputs(noisy_inputs)
        y = self._prepare_targets(clean_targets)
        self._model = build_model(
            input_timesteps=x.shape[1],
            input_features=x.shape[2],
            base_channels=self.base_channels,
            dilations=self.dilations,
        )

        callbacks = []
        ckpt = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                str(ckpt),
                monitor="val_loss" if validation_data is not None else "loss",
                save_best_only=True,
                verbose=0,
            )
        )

        val = None
        if validation_data is not None:
            val = (
                self._prepare_inputs(validation_data[0]),
                self._prepare_targets(validation_data[1]),
            )

        history = self._model.fit(
            x,
            y,
            validation_data=val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
            callbacks=callbacks,
        )
        self.history_ = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        self._is_fitted = True
        return self

    def predict(self, noisy_inputs: NDArray) -> NDArray[np.float32]:
        """Predict denoised waveforms."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() or load() before predict()")
        x = self._prepare_inputs(noisy_inputs)
        return self._model.predict(x, verbose=0).astype(np.float32)

    def score(self, noisy_inputs: NDArray, clean_targets: NDArray) -> float:
        """Return mean SI-SDR in dB."""
        preds = self.predict(noisy_inputs)
        y = self._prepare_targets(clean_targets)
        return float(np.mean(si_sdr_db_numpy(y, preds)))

    def predict_streaming(
        self,
        chunk: NDArray,
        state: Optional[dict[str, NDArray[np.float32]]] = None,
    ) -> tuple[NDArray[np.float32], dict[str, NDArray[np.float32]]]:
        """
        Predict a denoised chunk with causal state carried in a left context.

        The state stores the previous `receptive_field()-1` input samples.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() or load() before predict_streaming()")

        x = np.asarray(chunk, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected chunk shape (T, C), got {x.shape}")

        context_len = receptive_field(dilations=self.dilations) - 1
        if state is None:
            context = np.zeros((context_len, x.shape[1]), dtype=np.float32)
        else:
            context = state["context"].astype(np.float32)
            if context.shape != (context_len, x.shape[1]):
                raise ValueError(
                    f"Expected state context {(context_len, x.shape[1])}, got {context.shape}"
                )

        model_in = np.concatenate([context, x], axis=0)[np.newaxis, ...]
        pred = self._model.predict(model_in, verbose=0)[0]
        out = pred[-len(x):].astype(np.float32)
        new_state = {
            "context": model_in[0, -context_len:].astype(np.float32),
        }
        return out, new_state

    def save(self, path: Optional[str | Path] = None) -> Path:
        """Save the trained denoiser checkpoint."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before save()")
        p = Path(path) if path else DEFAULT_CHECKPOINT
        p.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(p))
        return p

    @classmethod
    def load(cls, path: str | Path) -> "TCNDenoiser":
        """Load a trained denoiser from disk."""
        import tensorflow as tf

        obj = cls()
        obj._model = tf.keras.models.load_model(str(path), custom_objects=_custom_objects())
        obj._is_fitted = True
        return obj


def main() -> None:
    """Train a tiny denoiser on synthetic data for a smoke test."""
    from data.denoise_data import generate_synthetic_paired_dataset
    from pipeline.denoise_ingest import prepare_denoising_dataset

    clean, _, mixture = generate_synthetic_paired_dataset(n_clips=8, duration_s=0.15, seed=3)
    noisy_inputs, targets = prepare_denoising_dataset(mixture, clean)

    model = TCNDenoiser(epochs=2, batch_size=4)
    model.fit(noisy_inputs, targets)
    score = model.score(noisy_inputs, targets)
    print(f"[tcn_denoiser] mean SI-SDR: {score:.3f} dB")


if __name__ == "__main__":
    main()
