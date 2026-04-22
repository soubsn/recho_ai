"""
Pre-Hopf audio augmentation primitives for the cat-vs-rest merged dataset.

All ops run on float64 mono signals in [-1, 1] at the source sample rate.
They are waveform-level (not spectrogram) so the augmented clip can pass
through the Hopf reservoir exactly like a real recording would at deploy
time on the MCU.

The set of transforms — gain, colored noise, background SNR mix, time
shift — was picked to match what a microphone + room could plausibly
produce. Pitch/tempo shift and room IR convolution were deliberately
left out: they change the spectral signature in ways the Hopf reservoir
is sensitive to, and we don't have a reason yet to model reverb.

SNR convention: target SNR is measured in dB as
    20 * log10(rms_signal / rms_noise_scaled)
so a larger target SNR means a quieter noise/background.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


EPS: float = 1e-12


def _rms(x: NDArray[np.float64]) -> float:
    """Root-mean-square amplitude; returns EPS on silence to keep SNR math safe."""
    r = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return max(r, EPS)


def gain_scale(
    x: NDArray[np.float64],
    gain: float,
) -> NDArray[np.float64]:
    """Multiply by a linear gain then soft-clip to [-1, 1]."""
    return np.clip(x * gain, -1.0, 1.0)


def _white_noise(
    n: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Zero-mean unit-variance Gaussian noise."""
    return rng.standard_normal(n)


def _pink_noise(
    n: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """
    Pink (1/f) noise via FFT shaping.

    Generate white noise, scale each positive-frequency bin by 1/sqrt(f),
    iFFT back. DC bin is zeroed so the output has no offset.
    """
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.arange(len(spectrum))
    freqs[0] = 1.0
    spectrum = spectrum / np.sqrt(freqs)
    spectrum[0] = 0.0
    pink = np.fft.irfft(spectrum, n=n)
    pink = pink / (np.std(pink) + EPS)
    return pink


def _brown_noise(
    n: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """
    Brown (1/f^2) noise as the cumulative sum of white noise, detrended
    and normalised to unit variance.
    """
    white = rng.standard_normal(n)
    brown = np.cumsum(white)
    brown = brown - np.linspace(brown[0], brown[-1], n)
    brown = brown / (np.std(brown) + EPS)
    return brown


_NOISE_KINDS: tuple[str, ...] = ("white", "pink", "brown")


def add_noise(
    x: NDArray[np.float64],
    snr_db: float,
    kind: str,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """
    Add colored noise scaled so the mix has the requested signal-to-noise
    ratio in dB.
    """
    if kind == "white":
        noise = _white_noise(len(x), rng)
    elif kind == "pink":
        noise = _pink_noise(len(x), rng)
    elif kind == "brown":
        noise = _brown_noise(len(x), rng)
    else:
        raise ValueError(f"unknown noise kind: {kind}")

    sig_rms = _rms(x)
    noise_rms = _rms(noise)
    target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    noise = noise * (target_noise_rms / noise_rms)
    return np.clip(x + noise, -1.0, 1.0)


def mix_background(
    x: NDArray[np.float64],
    bg: NDArray[np.float64],
    snr_db: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """
    Mix a real background clip behind x at the target SNR in dB.

    If the background is shorter than the signal it loops; if longer, a
    random window is sampled. Background is normalised to match sig RMS
    at the chosen SNR before summing.
    """
    n = len(x)
    if len(bg) == 0:
        return x
    if len(bg) < n:
        repeats = int(np.ceil(n / len(bg)))
        bg = np.tile(bg, repeats)[:n]
    elif len(bg) > n:
        start = int(rng.integers(0, len(bg) - n + 1))
        bg = bg[start:start + n]

    sig_rms = _rms(x)
    bg_rms = _rms(bg)
    target_bg_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    bg = bg * (target_bg_rms / bg_rms)
    return np.clip(x + bg, -1.0, 1.0)


def time_shift(
    x: NDArray[np.float64],
    shift_samples: int,
) -> NDArray[np.float64]:
    """
    Circular time shift. Positive = delay (wrap tail to head).

    Circular (not zero-pad) keeps the clip length and RMS stable so the
    SNR math upstream stays accurate.
    """
    return np.roll(x, shift_samples)


def random_combination(
    x: NDArray[np.float64],
    sr: int,
    bg_pool: list[NDArray[np.float64]],
    rng: np.random.Generator,
    gain_range: tuple[float, float] = (0.5, 2.0),
    noise_snr_range: tuple[float, float] = (10.0, 25.0),
    bg_snr_range: tuple[float, float] = (10.0, 20.0),
    max_shift_s: float = 0.5,
    p_each: float = 0.5,
) -> tuple[NDArray[np.float64], list[str]]:
    """
    Apply a random subset of {gain, noise, background, time_shift}.

    Each transform is rolled independently at probability `p_each`; if
    none are selected, one is forced so the output always differs from
    the input. Returns (augmented, applied_ops) — the ops list is logged
    into the manifest so we can audit what got applied per file.
    """
    picked: list[str] = []
    if rng.random() < p_each:
        picked.append("gain")
    if rng.random() < p_each:
        picked.append("noise")
    if rng.random() < p_each and bg_pool:
        picked.append("background")
    if rng.random() < p_each:
        picked.append("shift")
    if not picked:
        candidates = ["gain", "noise", "shift"] + (["background"] if bg_pool else [])
        picked.append(str(rng.choice(candidates)))

    y = x.copy()
    for op in picked:
        if op == "gain":
            g = float(rng.uniform(*gain_range))
            y = gain_scale(y, g)
        elif op == "noise":
            snr = float(rng.uniform(*noise_snr_range))
            kind = str(rng.choice(_NOISE_KINDS))
            y = add_noise(y, snr, kind, rng)
        elif op == "background":
            snr = float(rng.uniform(*bg_snr_range))
            bg = bg_pool[int(rng.integers(0, len(bg_pool)))]
            y = mix_background(y, bg, snr, rng)
        elif op == "shift":
            max_samples = int(max_shift_s * sr)
            s = int(rng.integers(-max_samples, max_samples + 1))
            y = time_shift(y, s)
    return y, picked
