"""
Build a Hopf hopf_text cache for the Kaggle Dog-vs-Cats audio dataset.

Also performs window-based augmentation: each wav is sliced into multiple
WINDOW_S-long audio windows at stride HOP_S before integration, so one 10-second
wav becomes ~3 independent 5-second clips through the Hopf ODE. Wavs shorter
than WINDOW_S yield a single padded clip (same behavior as before).

Each clip's source_filename is recorded in manifest.json so downstream code can
split by source file, preventing leakage between train and val.

Layout produced (mirrors the ESC-50 cache so load_xy_dataset_from_text_cache
works unchanged):

    <out_dir>/
        manifest.json
        labels.txt
        classes.txt
        clips/
            clip_0000_x.txt
            clip_0000_y.txt
            clip_0001_x.txt
            ...

Run:
    python recho_pipeline/data/generate_hopf_cache_cats_dogs.py
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile
from scipy.signal import resample_poly


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.sample_data import (  # noqa: E402
    FS_HW,
    FS_TARGET,
    export_dataset_text,
    integrate_hopf_xy,
    _make_audio_signal,
)


DATASET_ROOT: Path = Path("/Users/nic-spect/data/recho_ai/Kaggle_Dog_vs_Cats")
WAV_DIR: Path = DATASET_ROOT / "cats_dogs"
OUT_DIR: Path = DATASET_ROOT / "hopf_text"
# 5 s keeps samples_per_clip = 20_000 at 4 kHz so the existing feature extractor
# + CNN input shape (200, 100) are unchanged.
WINDOW_S: float = 5.0
# Stride between consecutive windows. Smaller = more overlap = more clips but
# higher correlation between neighbours. 2.5 s gives 50% overlap.
HOP_S: float = 2.5
CLASS_NAMES: list[str] = ["cat", "dog"]

# Audio-space augmentation. Each base window additionally produces
# N_AUGMENT_PER_WINDOW copies with randomized gain / time shift / additive
# noise BEFORE the Hopf integration, so every augmented copy is a genuinely
# new reservoir trajectory (not just a pixel-space perturbation).
# Set to 0 to disable and get the base-only cache.
N_AUGMENT_PER_WINDOW: int = 2
# Multiplicative gain applied to the audio amplitude. Hopf is nonlinear in
# drive amplitude so gain changes real dynamics, not just loudness.
GAIN_RANGE: tuple[float, float] = (0.6, 1.4)
# Max time-shift (in seconds) for the roll-with-zero-fill augmentation.
# The transient lands at a different time in the 5 s window; Hopf's
# settling / bifurcation response shifts accordingly.
MAX_SHIFT_S: float = 0.5
# Gaussian noise std, as a fraction of the augmented-audio RMS.
# 0.03 ~ 30 dB SNR — audible to the reservoir, inaudible as "corruption".
NOISE_SNR_FACTOR: float = 0.03
# Master seed for augmentation. Every clip derives a deterministic per-clip
# RNG from (AUGMENT_SEED, out_idx) so reruns produce byte-identical output.
AUGMENT_SEED: int = 0


def _label_from_filename(name: str) -> int:
    """`cat_…` -> 0, `dog_…` (incl. dog_barking_…) -> 1."""
    lower = name.lower()
    if lower.startswith("cat"):
        return 0
    if lower.startswith("dog"):
        return 1
    raise ValueError(f"Unexpected filename (no cat/dog prefix): {name}")


def _load_wav_at_fs(wav_path: Path, fs: int = FS_TARGET) -> NDArray[np.float64]:
    """Load one wav as mono float64 in [-1, +1] at the target sample rate."""
    sr, x = wavfile.read(wav_path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        max_val = float(np.iinfo(x.dtype).max)
        x = x.astype(np.float64) / max_val
    else:
        x = x.astype(np.float64)
    if sr != fs:
        from math import gcd
        g = gcd(sr, fs)
        x = resample_poly(x, fs // g, sr // g)
    return x


def _window_starts(n_audio: int, n_window: int, n_hop: int) -> list[int]:
    """
    Start indices of all WINDOW_S windows in a wav of n_audio samples.

    - If audio is shorter than one window, returns [0] and the caller pads.
    - Otherwise slides with stride n_hop, always including a final flush-right
      window so the tail of long wavs isn't dropped.
    """
    if n_audio <= n_window:
        return [0]
    starts = list(range(0, n_audio - n_window + 1, n_hop))
    tail = n_audio - n_window
    if starts[-1] != tail:
        starts.append(tail)
    return starts


def _zero_shift(audio: NDArray[np.float64], shift: int) -> NDArray[np.float64]:
    """Shift audio by `shift` samples, filling the vacated end with zeros."""
    out = np.zeros_like(audio)
    if shift > 0:
        out[shift:] = audio[:-shift]
    elif shift < 0:
        out[:shift] = audio[-shift:]
    else:
        out[:] = audio
    return out


def _augment_audio(
    audio: NDArray[np.float64], rng: np.random.Generator
) -> NDArray[np.float64]:
    """Apply gain jitter + time shift + low-SNR additive noise, in that order."""
    gain = rng.uniform(*GAIN_RANGE)
    audio = audio * gain

    max_shift = int(MAX_SHIFT_S * FS_TARGET)
    if max_shift > 0:
        shift = int(rng.integers(-max_shift, max_shift + 1))
        audio = _zero_shift(audio, shift)

    rms = float(np.sqrt(np.mean(audio * audio)) + 1e-12)
    noise = rng.standard_normal(len(audio)) * (rms * NOISE_SNR_FACTOR)
    audio = audio + noise

    np.clip(audio, -1.0, 1.0, out=audio)
    return audio


def _integrate_window_worker(
    args: tuple[int, str, int, int, int]
) -> tuple[int, NDArray[np.float64], NDArray[np.float64]]:
    """Worker: load wav, extract window, maybe augment, integrate Hopf."""
    out_idx, wav_path_str, start_sample, n_window, aug_seed = args
    full = _load_wav_at_fs(Path(wav_path_str), fs=FS_TARGET)
    if len(full) <= n_window:
        audio = np.zeros(n_window, dtype=np.float64)
        audio[: len(full)] = full
    else:
        audio = full[start_sample : start_sample + n_window]
    if aug_seed >= 0:
        audio = _augment_audio(audio, np.random.default_rng(aug_seed))
    a_func = _make_audio_signal(audio, fs=FS_TARGET)
    x, y = integrate_hopf_xy(a_func, duration=WINDOW_S, fs=FS_HW)
    return out_idx, x, y


def _plan_clips() -> tuple[list[dict[str, str]], list[tuple[int, str, int, int, int]], list[int]]:
    """
    Enumerate (wav, window, augmentation) triples into:
      - rows: {filename, category} per clip (filename = source wav).
      - tasks: (out_idx, wav_path, start_sample_4k, n_window_4k, aug_seed).
               aug_seed = -1 marks the base (un-augmented) clip for each window;
               aug_seed >= 0 is the RNG seed for an augmented variant.
      - labels_per_clip: 0/1 per clip.
    """
    n_window = int(WINDOW_S * FS_TARGET)
    n_hop = int(HOP_S * FS_TARGET)

    rows: list[dict[str, str]] = []
    tasks: list[tuple[int, str, int, int, int]] = []
    labels_per_clip: list[int] = []

    out_idx = 0
    short = 0
    long_wavs = 0
    windows_from_long = 0
    for p in sorted(WAV_DIR.glob("*.wav")):
        try:
            label = _label_from_filename(p.name)
        except ValueError as e:
            print(f"[skip] {e}")
            continue
        audio = _load_wav_at_fs(p, fs=FS_TARGET)
        starts = _window_starts(len(audio), n_window, n_hop)
        if len(starts) == 1 and len(audio) <= n_window:
            short += 1
        else:
            long_wavs += 1
            windows_from_long += len(starts)
        for s in starts:
            # Base (un-augmented) clip.
            rows.append({"filename": p.name, "category": CLASS_NAMES[label]})
            tasks.append((out_idx, str(p), s, n_window, -1))
            labels_per_clip.append(label)
            out_idx += 1
            # N_AUGMENT_PER_WINDOW augmented copies. Each gets a deterministic
            # seed derived from (AUGMENT_SEED, out_idx) so reruns reproduce.
            for a in range(N_AUGMENT_PER_WINDOW):
                rows.append({"filename": p.name, "category": CLASS_NAMES[label]})
                tasks.append((
                    out_idx, str(p), s, n_window,
                    AUGMENT_SEED * 1_000_003 + out_idx,
                ))
                labels_per_clip.append(label)
                out_idx += 1

    n_base = len(tasks) // (1 + N_AUGMENT_PER_WINDOW) if N_AUGMENT_PER_WINDOW else len(tasks)
    print(f"[plan] {short} short wavs (1 padded window each)")
    print(f"[plan] {long_wavs} long wavs -> {windows_from_long} windows "
          f"(avg {windows_from_long/max(1, long_wavs):.1f}/wav)")
    if N_AUGMENT_PER_WINDOW > 0:
        print(f"[plan] audio aug: ×{1 + N_AUGMENT_PER_WINDOW} per window "
              f"(gain {GAIN_RANGE}, shift ±{MAX_SHIFT_S}s, noise_snr={NOISE_SNR_FACTOR})")
    print(f"[plan] total clips: {len(tasks)} "
          f"({n_base} base + {len(tasks) - n_base} augmented)")
    return rows, tasks, labels_per_clip


def _default_workers() -> int:
    import os
    return max(1, (os.cpu_count() or 4) - 2)


def main() -> None:
    rows, tasks, labels_list = _plan_clips()
    n = len(tasks)
    n_cat = sum(1 for r in rows if r["category"] == "cat")
    n_dog = n - n_cat
    print(f"[cats_dogs] {n} clips ({n_cat} cat, {n_dog} dog) "
          f"from {WAV_DIR} [WINDOW_S={WINDOW_S}, HOP_S={HOP_S}]")
    if n == 0:
        raise SystemExit(f"No .wav files under {WAV_DIR}")

    labels = np.asarray(labels_list, dtype=np.int64)
    n_hw_samples = int(WINDOW_S * FS_HW)
    x_data = np.zeros((n, n_hw_samples), dtype=np.float64)
    y_data = np.zeros((n, n_hw_samples), dtype=np.float64)

    workers = _default_workers()
    print(f"[cats_dogs] Integrating Hopf ODE at {FS_HW} Hz, "
          f"duration={WINDOW_S}s, workers={workers} ...")
    t0 = time.time()
    done = 0
    if workers <= 1:
        for t in tasks:
            idx, x, y = _integrate_window_worker(t)
            x_data[idx] = x
            y_data[idx] = y
            done += 1
            if done % 10 == 0 or done == n:
                dt = time.time() - t0
                print(f"  [integrate {done}/{n}] {dt:.0f}s "
                      f"(~{dt/done:.1f}s/clip)")
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for idx, x, y in pool.imap_unordered(
                _integrate_window_worker, tasks, chunksize=1,
            ):
                x_data[idx] = x
                y_data[idx] = y
                done += 1
                if done % 10 == 0 or done == n:
                    dt = time.time() - t0
                    print(f"  [integrate {done}/{n}] {dt:.0f}s "
                          f"(~{dt/done:.1f}s/clip, wall)")

    print(f"[cats_dogs] Integration done in {time.time() - t0:.1f}s. "
          f"Exporting to {OUT_DIR} ...")
    export_dataset_text(
        out_dir=OUT_DIR,
        x_data=x_data,
        labels=labels,
        class_names=CLASS_NAMES,
        y_data=y_data,
        source_rows=rows,
        source="cats_dogs",
        export_fs=FS_TARGET,
        hw_fs=FS_HW,
        clip_duration_s=WINDOW_S,
    )
    print(f"[cats_dogs] Done. Cache: {OUT_DIR}")


if __name__ == "__main__":
    main()
