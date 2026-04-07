"""
Data ingestion for the Hopf reservoir computer pipeline.

Loads raw x(t) time series, downsamples from 100 kHz to 4 kHz,
normalises to [-1, +1], and applies the inverse hyperbolic tangent
activation from eq. 6 of:
  "Hopf physical reservoir computer for reconfigurable sound recognition"
  (Shougat et al., Scientific Reports 2023)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray


# Hardware and target sample rates
FS_HW: int = 100_000
FS_TARGET: int = 4_000
DOWNSAMPLE_FACTOR: int = FS_HW // FS_TARGET  # 25

# Feature map dimensions — paper 2, figure 2
N_TIME_STEPS: int = 200
N_VIRTUAL_NODES: int = 100
SAMPLES_PER_CLIP: int = N_TIME_STEPS * N_VIRTUAL_NODES  # 20,000 at 4 kHz = 5 sec
# With 1-sec clips at 4 kHz we get 4000 samples → reshape to 200x100 requires
# exactly 20,000 samples, so we use 5-sec windows or tile shorter clips.
# For 1-sec clips (4000 samples) we pad/tile to 20,000.


def load_csv(path: str | Path) -> NDArray[np.float64]:
    """Load raw x(t) time series from a CSV file (single column)."""
    return np.loadtxt(path, delimiter=",", dtype=np.float64)


def downsample(
    x: NDArray[np.float64],
    factor: int = DOWNSAMPLE_FACTOR,
) -> NDArray[np.float64]:
    """
    Skip-sample downsample (no anti-alias filter).

    The physical reservoir's bandwidth is limited by the oscillator dynamics,
    so aliasing from skip-sampling is negligible in practice.
    """
    return x[::factor].copy()


def normalise(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalise to [-1, +1]: x_norm = x / max(|x|)."""
    peak = np.max(np.abs(x))
    if peak < 1e-12:
        return np.zeros_like(x)
    return x / peak


def atanh_activation(x_norm: NDArray[np.float64], eps: float = 1e-6) -> NDArray[np.float64]:
    """
    Inverse hyperbolic tangent nonlinear activation (eq. 6, paper 2).

    X = atanh((x_norm - mean) / std)

    Clamps the argument to (-1+eps, 1-eps) to avoid infinities.
    """
    mu = np.mean(x_norm)
    sigma = np.std(x_norm)
    if sigma < 1e-12:
        return np.zeros_like(x_norm)
    z = (x_norm - mu) / sigma
    z = np.clip(z, -1.0 + eps, 1.0 - eps)
    return np.arctanh(z)


def _tile_to_length(x: NDArray[np.float64], target_len: int) -> NDArray[np.float64]:
    """Tile or truncate x to exactly target_len samples."""
    if len(x) >= target_len:
        return x[:target_len]
    repeats = (target_len // len(x)) + 1
    return np.tile(x, repeats)[:target_len]


def process_clip(
    raw_x: NDArray[np.float64],
    target_samples: int = SAMPLES_PER_CLIP,
) -> NDArray[np.float64]:
    """
    Full ingestion pipeline for a single clip of raw x(t).

    Returns:
        2-D array of shape (N_TIME_STEPS, N_VIRTUAL_NODES) = (200, 100)
    """
    ds = downsample(raw_x)
    ds = _tile_to_length(ds, target_samples)
    normed = normalise(ds)
    activated = atanh_activation(normed)
    return activated.reshape(N_TIME_STEPS, N_VIRTUAL_NODES)


def process_dataset(
    raw_clips: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Process a batch of raw x(t) clips.

    Args:
        raw_clips: shape (n_clips, n_hw_samples)

    Returns:
        shape (n_clips, 200, 100) — feature maps ready for feature extraction
    """
    n_clips = raw_clips.shape[0]
    out = np.zeros((n_clips, N_TIME_STEPS, N_VIRTUAL_NODES), dtype=np.float64)
    for i in range(n_clips):
        out[i] = process_clip(raw_clips[i])
    return out


def main() -> None:
    """Run ingestion on synthetic data for verification."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.sample_data import generate_dataset

    print("[ingest] Loading synthetic dataset ...")
    raw_x, labels = generate_dataset(n_clips_per_class=5, n_classes=5, cache=False)
    print(f"  Raw shape: {raw_x.shape}")

    features = process_dataset(raw_x)
    print(f"  Processed shape: {features.shape}")
    print(f"  Value range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"  Mean: {features.mean():.6f}, Std: {features.std():.6f}")


if __name__ == "__main__":
    main()
