"""
Reservoir ingestion for denoising.

Unlike the classifier pipeline, this module keeps the data as a causal
sequence. It feeds the noisy mixture waveform through the Hopf reservoir front
end, downsamples the resulting `x(t)` and `y(t)` states, and aligns them with
the clean target waveform for sequence-to-sequence denoising.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.sample_data import A_DRIVE, MU, OMEGA, OMEGA_DRIVE
from pipeline.ingest import DOWNSAMPLE_FACTOR, FS_HW, FS_TARGET, downsample


def _upsample_input(
    waveform: NDArray[np.float64],
    fs_input: int,
    fs_hw: int,
) -> NDArray[np.float64]:
    """Upsample an input waveform to the reservoir hardware rate."""
    if fs_hw % fs_input == 0:
        return np.repeat(waveform, fs_hw // fs_input).astype(np.float64)

    n_hw = int(round(len(waveform) * fs_hw / fs_input))
    t_in = np.arange(len(waveform), dtype=np.float64) / fs_input
    t_hw = np.arange(n_hw, dtype=np.float64) / fs_hw
    return np.interp(t_hw, t_in, waveform).astype(np.float64)


def simulate_hopf_reservoir(
    input_waveform: NDArray[np.float64],
    fs_input: int = FS_TARGET,
    fs_hw: int = FS_HW,
    mu: float = MU,
    omega: float = OMEGA,
    omega_drive: float = OMEGA_DRIVE,
    a_drive: float = A_DRIVE,
    initial_state: tuple[float, float] = (0.1, 0.0),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Simulate the Hopf reservoir on a waveform input using explicit Euler steps.

    Returns:
        `x(t), y(t)` at the hardware sample rate.
    """
    a_hw = np.clip(_upsample_input(input_waveform, fs_input=fs_input, fs_hw=fs_hw), -0.95, 0.95)
    dt = 1.0 / fs_hw

    x_hw = np.zeros(len(a_hw), dtype=np.float64)
    y_hw = np.zeros(len(a_hw), dtype=np.float64)
    x_prev, y_prev = initial_state

    for idx, a_t in enumerate(a_hw):
        f_t = 1.0 + a_t
        r2 = x_prev * x_prev + y_prev * y_prev
        t = idx * dt
        dx = (mu * f_t - r2) * x_prev - omega * y_prev + a_drive * f_t * np.sin(omega_drive * t)
        dy = (mu * f_t - r2) * y_prev + omega * x_prev
        x_prev = x_prev + dt * dx
        y_prev = y_prev + dt * dy
        x_hw[idx] = x_prev
        y_hw[idx] = y_prev

    return x_hw, y_hw


def prepare_denoising_example(
    mixture_waveform: NDArray[np.float64],
    clean_waveform: NDArray[np.float64],
    fs_input: int = FS_TARGET,
    fs_hw: int = FS_HW,
    fs_target: int = FS_TARGET,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Convert one `(mixture, clean)` pair into `(reservoir_sequence, clean_target)`.

    Returns:
        noisy_inputs: `(T, 2)` float32 for `x,y`
        clean_target: `(T, 1)` float32 clean waveform
    """
    if len(mixture_waveform) != len(clean_waveform):
        raise ValueError("Mixture and clean waveform lengths must match")

    x_hw, y_hw = simulate_hopf_reservoir(mixture_waveform, fs_input=fs_input, fs_hw=fs_hw)
    factor = max(1, fs_hw // fs_target)
    x_ds = downsample(x_hw, factor=factor)
    y_ds = downsample(y_hw, factor=factor)

    target_len = min(len(clean_waveform), len(x_ds), len(y_ds))
    x_seq = x_ds[:target_len].astype(np.float64)
    y_seq = y_ds[:target_len].astype(np.float64)
    peak = max(float(np.max(np.abs(x_seq))), float(np.max(np.abs(y_seq))), 1e-12)
    x_seq = x_seq / peak
    y_seq = y_seq / peak
    noisy_inputs = np.stack([x_seq, y_seq], axis=-1).astype(np.float32)
    clean_target = clean_waveform[:target_len].astype(np.float32)[:, np.newaxis]
    return noisy_inputs, clean_target


def prepare_denoising_dataset(
    mixture_waveforms: NDArray[np.float64],
    clean_waveforms: NDArray[np.float64],
    fs_input: int = FS_TARGET,
    fs_hw: int = FS_HW,
    fs_target: int = FS_TARGET,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Vectorised wrapper around `prepare_denoising_example()` for a dataset."""
    n = len(mixture_waveforms)
    inputs: list[NDArray[np.float32]] = []
    targets: list[NDArray[np.float32]] = []
    for idx in range(n):
        x, y = prepare_denoising_example(
            mixture_waveforms[idx],
            clean_waveforms[idx],
            fs_input=fs_input,
            fs_hw=fs_hw,
            fs_target=fs_target,
        )
        inputs.append(x)
        targets.append(y)

    return np.stack(inputs, axis=0), np.stack(targets, axis=0)


def main() -> None:
    """Run a small denoising-ingest demo."""
    from data.denoise_data import generate_synthetic_paired_dataset

    clean, noise, mixture = generate_synthetic_paired_dataset(n_clips=2, seed=11)
    x, y = prepare_denoising_dataset(mixture, clean)
    print("[denoise_ingest] Prepared denoising dataset")
    print(f"  inputs:  {x.shape}")
    print(f"  targets: {y.shape}")


if __name__ == "__main__":
    main()
