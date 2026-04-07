"""
Synthetic Hopf oscillator data generator.

Numerically integrates the Hopf equations from:
  "Hopf physical reservoir computer for reconfigurable sound recognition"
  (Shougat et al., Scientific Reports 2023)

  dx/dt = (mu*f(t) - x^2 - y^2)*x - omega*y + A*f(t)*sin(Omega*t)
  dy/dt = (mu*f(t) - x^2 - y^2)*y + omega*x
  f(t)  = 1 + a(t)

Parameters: mu=5, A=0.5, Omega=40*pi, omega=40*pi
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


# Hopf oscillator parameters — exact values from paper 2
MU: float = 5.0
A_DRIVE: float = 0.5
OMEGA: float = 40.0 * np.pi      # natural frequency
OMEGA_DRIVE: float = 40.0 * np.pi  # driving frequency

# Simulation settings
FS_HW: int = 100_000   # hardware sample rate (Hz)
FS_TARGET: int = 4_000  # target sample rate after downsample (Hz)
CLIP_DURATION: float = 1.0  # seconds
N_CLIPS_PER_CLASS: int = 100
N_CLASSES: int = 5

CACHE_DIR: Path = Path(__file__).resolve().parent / "cache"


def _hopf_rhs(
    t: float,
    state: NDArray[np.float64],
    a_func: Callable[[float], float],
) -> list[float]:
    """Right-hand side of the Hopf ODE system."""
    x, y = state
    a_t = a_func(t)
    f_t = 1.0 + a_t
    r2 = x * x + y * y
    dxdt = (MU * f_t - r2) * x - OMEGA * y + A_DRIVE * f_t * np.sin(OMEGA_DRIVE * t)
    dydt = (MU * f_t - r2) * y + OMEGA * x
    return [dxdt, dydt]


def integrate_hopf(
    a_func: Callable[[float], float],
    duration: float = CLIP_DURATION,
    fs: int = FS_HW,
) -> NDArray[np.float64]:
    """
    Integrate the Hopf ODE and return x(t) sampled at *fs* Hz.

    Returns:
        1-D array of x(t) values, length = int(duration * fs).
    """
    n_samples = int(duration * fs)
    t_eval = np.linspace(0.0, duration, n_samples, endpoint=False)
    sol = solve_ivp(
        fun=lambda t, y: _hopf_rhs(t, y, a_func),
        t_span=(0.0, duration),
        y0=[0.1, 0.0],
        t_eval=t_eval,
        method="RK45",
        max_step=1e-5,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    return sol.y[0]  # x(t) only


# ---------------------------------------------------------------------------
# Input signal generators — five synthetic sound classes
# ---------------------------------------------------------------------------

def _make_sine(freq: float = 300.0, amp: float = 0.3) -> Callable[[float], float]:
    """Class 0: single sine tone."""
    def a(t: float) -> float:
        return amp * np.sin(2.0 * np.pi * freq * t)
    return a


def _make_two_sines(f1: float = 300.0, f2: float = 600.0, amp: float = 0.2) -> Callable[[float], float]:
    """Class 1: two overlapping harmonics."""
    def a(t: float) -> float:
        return amp * (np.sin(2.0 * np.pi * f1 * t) + np.sin(2.0 * np.pi * f2 * t))
    return a


def _make_square(freq: float = 200.0, amp: float = 0.3) -> Callable[[float], float]:
    """Class 2: square wave (sharp transient)."""
    def a(t: float) -> float:
        return amp * np.sign(np.sin(2.0 * np.pi * freq * t))
    return a


def _make_chirp(f0: float = 100.0, f1: float = 1000.0, amp: float = 0.3) -> Callable[[float], float]:
    """Class 3: linear frequency sweep."""
    def a(t: float) -> float:
        freq = f0 + (f1 - f0) * t / CLIP_DURATION
        return amp * np.sin(2.0 * np.pi * freq * t)
    return a


def _make_noise(amp: float = 0.2, seed: int = 0) -> Callable[[float], float]:
    """Class 4: band-limited noise (pre-generated, interpolated)."""
    rng = np.random.default_rng(seed)
    n = int(CLIP_DURATION * FS_TARGET)
    noise_samples = rng.standard_normal(n) * amp
    ts = np.linspace(0.0, CLIP_DURATION, n, endpoint=False)

    def a(t: float) -> float:
        idx = int(t * FS_TARGET)
        idx = min(idx, n - 1)
        return float(noise_samples[idx])
    return a


CLASS_NAMES: list[str] = ["sine", "two_sines", "square", "chirp", "noise"]


def _class_factory(class_id: int, variation_seed: int) -> Callable[[float], float]:
    """Return an input signal function for the given class with slight variation."""
    rng = np.random.default_rng(variation_seed)
    jitter = 1.0 + 0.1 * rng.standard_normal()  # +-10% parameter variation
    if class_id == 0:
        return _make_sine(freq=300.0 * jitter)
    elif class_id == 1:
        return _make_two_sines(f1=300.0 * jitter, f2=600.0 * jitter)
    elif class_id == 2:
        return _make_square(freq=200.0 * jitter)
    elif class_id == 3:
        return _make_chirp(f0=100.0 * jitter, f1=1000.0 * jitter)
    elif class_id == 4:
        return _make_noise(seed=variation_seed)
    else:
        raise ValueError(f"Unknown class_id={class_id}")


def generate_dataset(
    n_clips_per_class: int = N_CLIPS_PER_CLASS,
    n_classes: int = N_CLASSES,
    cache: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Generate the full synthetic dataset.

    Returns:
        x_data: (n_clips_per_class * n_classes, n_hw_samples) raw x(t) at 100 kHz
        labels: (n_clips_per_class * n_classes,) integer class labels
    """
    cache_key = f"hopf_n{n_clips_per_class}_c{n_classes}"
    cache_x = CACHE_DIR / f"{cache_key}_x.npy"
    cache_y = CACHE_DIR / f"{cache_key}_y.npy"

    if cache and cache_x.exists() and cache_y.exists():
        print(f"[sample_data] Loading cached dataset from {CACHE_DIR}")
        return np.load(cache_x), np.load(cache_y)

    print(f"[sample_data] Generating {n_classes} classes x {n_clips_per_class} clips ...")
    total = n_clips_per_class * n_classes
    n_samples = int(CLIP_DURATION * FS_HW)
    x_data = np.zeros((total, n_samples), dtype=np.float64)
    labels = np.zeros(total, dtype=np.int64)

    idx = 0
    for cls in range(n_classes):
        for clip in range(n_clips_per_class):
            seed = cls * 10_000 + clip
            a_func = _class_factory(cls, seed)
            x_data[idx] = integrate_hopf(a_func)
            labels[idx] = cls
            idx += 1
            if idx % 50 == 0:
                print(f"  [{idx}/{total}] clips generated")

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache_x, x_data)
        np.save(cache_y, labels)
        print(f"[sample_data] Cached to {CACHE_DIR}")

    return x_data, labels


def main() -> None:
    """Generate and summarise the synthetic dataset."""
    x_data, labels = generate_dataset()
    print(f"\nDataset shape: x={x_data.shape}, labels={labels.shape}")
    for cls in range(N_CLASSES):
        count = int(np.sum(labels == cls))
        print(f"  Class {cls} ({CLASS_NAMES[cls]}): {count} clips")
    print(f"  Sample range: [{x_data.min():.4f}, {x_data.max():.4f}]")


if __name__ == "__main__":
    main()
