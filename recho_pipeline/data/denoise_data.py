"""
Paired data utilities for Hopf-RPU denoising experiments.

This module creates aligned `(clean, noise, mixture)` waveform triplets for the
denoising pipeline. The default dataset is synthetic so the package can be run
end-to-end without any external audio corpora, but the same helpers also load
paired arrays from disk for future real-data work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.signal import chirp, sawtooth


FS_AUDIO: int = 4_000
DEFAULT_DURATION_S: float = 1.0
DEFAULT_SNR_DB_RANGE: tuple[float, float] = (-5.0, 20.0)


def _time_axis(duration_s: float, fs: int) -> NDArray[np.float64]:
    n_samples = int(duration_s * fs)
    return np.linspace(0.0, duration_s, n_samples, endpoint=False, dtype=np.float64)


def _normalise_peak(x: NDArray[np.float64], peak: float = 0.8) -> NDArray[np.float64]:
    xmax = float(np.max(np.abs(x)))
    if xmax < 1e-12:
        return np.zeros_like(x)
    return x * (peak / xmax)


def _clean_waveform(kind: int, t: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64]:
    """Generate a synthetic clean target waveform."""
    duration = float(t[-1] + (t[1] - t[0]))
    if kind == 0:
        x = 0.45 * np.sin(2.0 * np.pi * rng.uniform(180.0, 520.0) * t)
    elif kind == 1:
        f1 = rng.uniform(150.0, 350.0)
        x = 0.25 * np.sin(2.0 * np.pi * f1 * t)
        x += 0.18 * np.sin(2.0 * np.pi * 2.0 * f1 * t + rng.uniform(0.0, np.pi))
    elif kind == 2:
        f0 = rng.uniform(60.0, 180.0)
        f1 = rng.uniform(700.0, 1200.0)
        x = 0.4 * chirp(t, f0=f0, f1=f1, t1=duration, method="linear")
    elif kind == 3:
        carrier = rng.uniform(200.0, 500.0)
        mod = rng.uniform(2.0, 8.0)
        envelope = 0.55 + 0.35 * np.sin(2.0 * np.pi * mod * t)
        x = 0.45 * envelope * np.sin(2.0 * np.pi * carrier * t)
    else:
        pulse_rate = rng.uniform(4.0, 10.0)
        duty = rng.uniform(0.15, 0.35)
        gate = (np.mod(t * pulse_rate, 1.0) < duty).astype(np.float64)
        tone = np.sin(2.0 * np.pi * rng.uniform(220.0, 420.0) * t)
        x = 0.4 * gate * tone + 0.12 * sawtooth(2.0 * np.pi * 40.0 * t, width=0.5)
    return _normalise_peak(x)


def _colored_noise(n_samples: int, rng: np.random.Generator, exponent: float) -> NDArray[np.float64]:
    """Generate approximate 1/f^exponent noise in the frequency domain."""
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / FS_AUDIO)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(freqs))
    mag = 1.0 / np.power(freqs, exponent / 2.0)
    spectrum = mag * np.exp(1j * phases)
    noise = np.fft.irfft(spectrum, n=n_samples)
    return noise.astype(np.float64)


def _noise_waveform(kind: int, t: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64]:
    """Generate a synthetic noise waveform."""
    n_samples = len(t)
    if kind == 0:
        x = rng.standard_normal(n_samples)
    elif kind == 1:
        x = _colored_noise(n_samples, rng, exponent=1.0)
    elif kind == 2:
        hum_freq = rng.choice([50.0, 60.0, 120.0, 180.0])
        x = np.sin(2.0 * np.pi * hum_freq * t)
        x += 0.4 * np.sin(2.0 * np.pi * 2.0 * hum_freq * t + rng.uniform(0.0, np.pi))
    elif kind == 3:
        x = rng.standard_normal(n_samples) * 0.15
        impulses = rng.choice(n_samples, size=max(2, n_samples // 200), replace=False)
        x[impulses] += rng.uniform(-2.0, 2.0, size=len(impulses))
    else:
        competitor = _clean_waveform(rng.integers(0, 5), t, rng)
        x = competitor + 0.15 * rng.standard_normal(n_samples)
    return _normalise_peak(x)


def mix_at_snr(
    clean: NDArray[np.float64],
    noise: NDArray[np.float64],
    snr_db: float,
    peak_limit: float = 0.95,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Mix clean and noise at a target SNR.

    Returns:
        clean_scaled, noise_scaled, mixture
    """
    clean = clean.astype(np.float64)
    noise = noise.astype(np.float64) - np.mean(noise)
    clean_rms = float(np.sqrt(np.mean(clean ** 2)) + 1e-12)
    noise_rms = float(np.sqrt(np.mean(noise ** 2)) + 1e-12)
    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    noise_scaled = noise * (target_noise_rms / noise_rms)
    mixture = clean + noise_scaled

    peak = max(
        float(np.max(np.abs(clean))),
        float(np.max(np.abs(noise_scaled))),
        float(np.max(np.abs(mixture))),
        1e-12,
    )
    scale = min(1.0, peak_limit / peak)
    return clean * scale, noise_scaled * scale, mixture * scale


def generate_synthetic_paired_dataset(
    n_clips: int = 100,
    duration_s: float = DEFAULT_DURATION_S,
    fs: int = FS_AUDIO,
    snr_db_range: tuple[float, float] = DEFAULT_SNR_DB_RANGE,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Generate paired `(clean, noise, mixture)` waveforms.

    Returns:
        clean, noise, mixture with shape `(n_clips, n_samples)`.
    """
    rng = np.random.default_rng(seed)
    t = _time_axis(duration_s, fs)
    n_samples = len(t)

    clean = np.zeros((n_clips, n_samples), dtype=np.float64)
    noise = np.zeros((n_clips, n_samples), dtype=np.float64)
    mixture = np.zeros((n_clips, n_samples), dtype=np.float64)

    for idx in range(n_clips):
        clean_kind = int(rng.integers(0, 5))
        noise_kind = int(rng.integers(0, 5))
        clean_wave = _clean_waveform(clean_kind, t, rng)
        noise_wave = _noise_waveform(noise_kind, t, rng)
        snr_db = float(rng.uniform(*snr_db_range))
        clean[idx], noise[idx], mixture[idx] = mix_at_snr(clean_wave, noise_wave, snr_db)

    return clean, noise, mixture


def _load_array(path: str | Path) -> NDArray[np.float64]:
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p).astype(np.float64)
    if p.suffix == ".csv":
        return np.loadtxt(p, delimiter=",", dtype=np.float64)
    raise ValueError(f"Unsupported file format: {p}")


def load_paired_waveforms(
    clean_path: str | Path,
    noise_path: Optional[str | Path] = None,
    mixture_path: Optional[str | Path] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Load paired waveform arrays from disk.

    `mixture_path` is optional; if omitted, the mixture is computed as clean + noise.
    """
    clean = _load_array(clean_path)
    if clean.ndim == 1:
        clean = clean[np.newaxis, :]

    if noise_path is None and mixture_path is None:
        raise ValueError("Provide at least noise_path or mixture_path")

    noise = np.zeros_like(clean)
    if noise_path is not None:
        noise = _load_array(noise_path)
        if noise.ndim == 1:
            noise = noise[np.newaxis, :]

    if mixture_path is not None:
        mixture = _load_array(mixture_path)
        if mixture.ndim == 1:
            mixture = mixture[np.newaxis, :]
    else:
        mixture = clean + noise

    if clean.shape != mixture.shape:
        raise ValueError(f"Shape mismatch: clean={clean.shape}, mixture={mixture.shape}")
    if noise.shape != clean.shape:
        noise = np.zeros_like(clean) if noise_path is None else noise
        if noise.shape != clean.shape:
            raise ValueError(f"Shape mismatch: clean={clean.shape}, noise={noise.shape}")
    return clean, noise, mixture


def train_val_test_split(
    clean: NDArray[np.float64],
    noise: NDArray[np.float64],
    mixture: NDArray[np.float64],
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> dict[str, dict[str, NDArray[np.float64]]]:
    """Split paired arrays into train/val/test partitions with aligned indices."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 <= test_fraction < 1.0):
        raise ValueError("Split fractions must lie in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1")

    n = len(clean)
    idx = np.random.default_rng(seed).permutation(n)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)

    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    def _pack(indices: NDArray[np.int64]) -> dict[str, NDArray[np.float64]]:
        return {
            "clean": clean[indices],
            "noise": noise[indices],
            "mixture": mixture[indices],
        }

    return {
        "train": _pack(train_idx),
        "val": _pack(val_idx),
        "test": _pack(test_idx),
    }


def main() -> None:
    """Generate and summarise a small paired dataset."""
    clean, noise, mixture = generate_synthetic_paired_dataset(n_clips=8, seed=7)
    print("[denoise_data] Generated synthetic paired dataset")
    print(f"  clean shape:   {clean.shape}")
    print(f"  noise shape:   {noise.shape}")
    print(f"  mixture shape: {mixture.shape}")
    print(f"  clean range:   [{clean.min():.3f}, {clean.max():.3f}]")
    print(f"  mix range:     [{mixture.min():.3f}, {mixture.max():.3f}]")


if __name__ == "__main__":
    main()
