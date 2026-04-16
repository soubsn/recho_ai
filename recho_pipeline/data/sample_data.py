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

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.io import wavfile
from scipy.signal import resample_poly


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


def integrate_hopf_xy(
    a_func: Callable[[float], float],
    duration: float = CLIP_DURATION,
    fs: int = FS_HW,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Integrate the Hopf ODE and return both x(t) and y(t) sampled at *fs* Hz.

    y(t) is the quadrature state of the oscillator — the "spinning dot"
    companion to x(t). The published papers (Shougat et al. 2021, 2023)
    discard y(t), noting it "likely stores information". This function
    captures both states from the same single integration pass.

    Returns:
        x: 1-D array of x(t) values, length = int(duration * fs)
        y: 1-D array of y(t) values, same length
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
    return sol.y[0], sol.y[1]  # x(t), y(t)


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


# ---------------------------------------------------------------------------
# ESC-50 audio loader — real-world environmental sounds as Hopf input
# ---------------------------------------------------------------------------

ESC50_DEFAULT_ROOT: Path = Path(
    "/Users/nic-spect/data/recho_ai/Kaggle_Environmental_Sound_Classification_50"
)
ESC50_NATIVE_FS: int = 44_100  # native ESC-50 sample rate
ESC50_AUDIO_FS: int = FS_TARGET  # 4 kHz, per Shougat et al. 2023
ESC50_CLIP_DURATION: float = 5.0  # seconds — full ESC-50 clip length


def default_esc50_cache_dir(esc50_root: Path | str = ESC50_DEFAULT_ROOT) -> Path:
    """Default cache location for ESC-50 Hopf outputs — lives alongside the
    source dataset so multiple training runs can reuse the cached integration."""
    return Path(esc50_root) / "hopf_cache"


def _integrate_clip_worker(args: tuple[int, str, bool]) -> tuple:
    """
    Pool worker: load one wav, integrate Hopf, return (idx, x[, y]).

    Declared at module scope so it pickles cleanly for multiprocessing.
    Uses ESC50_CLIP_DURATION and ESC50_AUDIO_FS constants.
    """
    idx, wav_path_str, want_xy = args
    audio = _load_wav_at_4khz(Path(wav_path_str))
    a_func = _make_audio_signal(audio, fs=ESC50_AUDIO_FS)
    if want_xy:
        x, y = integrate_hopf_xy(a_func, duration=ESC50_CLIP_DURATION)
        return idx, x, y
    x = integrate_hopf(a_func, duration=ESC50_CLIP_DURATION)
    return idx, x


def _read_esc50_csv(csv_path: Path) -> list[dict[str, str]]:
    """Read esc50.csv as a list of row dicts."""
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _load_wav_at_4khz(wav_path: Path) -> NDArray[np.float64]:
    """
    Load a wav file and resample to 4 kHz audio (the rate fed to the
    Hopf reservoir in Shougat et al. 2023).

    Returns mono float64 audio in [-1, +1], length = 4000 * 5 = 20000.
    """
    sr, x = wavfile.read(wav_path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        max_val = float(np.iinfo(x.dtype).max)
        x = x.astype(np.float64) / max_val
    else:
        x = x.astype(np.float64)
    if sr != ESC50_AUDIO_FS:
        # 44100 -> 4000 has gcd 100, so up=40 down=441
        from math import gcd
        g = gcd(sr, ESC50_AUDIO_FS)
        x = resample_poly(x, ESC50_AUDIO_FS // g, sr // g)
    return x


def _make_audio_signal(audio: NDArray[np.float64], fs: int = ESC50_AUDIO_FS) -> Callable[[float], float]:
    """
    Wrap a discrete audio waveform as the continuous a(t) drive function
    consumed by the Hopf integrator. Uses linear interpolation between
    samples — closer to a physical analog signal than zero-order hold.
    """
    n = len(audio)
    duration = n / fs

    def a(t: float) -> float:
        if t <= 0.0:
            return float(audio[0])
        if t >= duration:
            return float(audio[-1])
        pos = t * fs
        i0 = int(pos)
        if i0 >= n - 1:
            return float(audio[-1])
        frac = pos - i0
        return float(audio[i0] * (1.0 - frac) + audio[i0 + 1] * frac)

    return a


def _select_esc50_rows(
    rows: list[dict[str, str]],
    esc10: bool,
    max_clips_per_class: int | None,
) -> tuple[list[dict[str, str]], list[str]]:
    """
    Filter rows for esc10 (curated 10-class subset) if requested,
    cap clips per class, and return (filtered_rows, class_names).
    Class IDs in the returned rows are remapped to a contiguous 0..N-1.
    """
    if esc10:
        rows = [r for r in rows if r["esc10"].strip().lower() == "true"]

    # Build sorted unique categories so label IDs are stable across runs.
    categories = sorted({r["category"] for r in rows})
    cat_to_id = {c: i for i, c in enumerate(categories)}

    # Cap per class.
    by_cls: dict[int, list[dict[str, str]]] = {}
    for r in rows:
        cid = cat_to_id[r["category"]]
        r = dict(r)
        r["_class_id"] = str(cid)
        by_cls.setdefault(cid, []).append(r)

    selected: list[dict[str, str]] = []
    for cid in sorted(by_cls):
        clips = by_cls[cid]
        if max_clips_per_class is not None:
            clips = clips[:max_clips_per_class]
        selected.extend(clips)

    return selected, categories


def _default_workers() -> int:
    """Default worker count: leave 2 cores free for the OS / main process."""
    return max(1, (os.cpu_count() or 4) - 2)


def _run_integration_pool(
    rows: list[dict[str, str]],
    audio_dir: Path,
    want_xy: bool,
    workers: int,
):
    """
    Run Hopf integration across ESC-50 rows, optionally in parallel.

    Yields (idx, x) or (idx, x, y) tuples as they complete. Order is not
    guaranteed — caller must write results by idx.
    """
    tasks = [(i, str(audio_dir / r["filename"]), want_xy) for i, r in enumerate(rows)]
    if workers <= 1:
        for t in tasks:
            yield _integrate_clip_worker(t)
        return
    ctx = mp.get_context("spawn")  # safe on macOS; avoids fork+numpy/BLAS issues
    with ctx.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(_integrate_clip_worker, tasks, chunksize=1):
            yield result


def generate_dataset_esc50(
    esc50_root: Path | str = ESC50_DEFAULT_ROOT,
    esc10: bool = True,
    max_clips_per_class: int | None = 10,
    cache: bool = True,
    workers: int | None = None,
    cache_dir: Path | str | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[str]]:
    """
    Run real ESC-50 audio through the Hopf reservoir.

    For each ESC-50 clip: resample 44.1 kHz wav to 4 kHz, drive the
    Hopf ODE with that audio, return raw x(t) at 100 kHz.

    Memory note: each 5-s clip at 100 kHz is ~4 MB float64. Default
    max_clips_per_class=10 keeps the run small for first iteration.
    Set max_clips_per_class=None for full ESC-50 (~8 GB, hours).

    Args:
        workers: parallel integration workers. None -> cpu_count - 2.
        cache_dir: where to read/write .npy caches. None -> <esc50_root>/hopf_cache/.

    Returns:
        x_data: (n_clips, 500_000) raw x(t) at 100 kHz
        labels: (n_clips,) integer class labels (contiguous 0..N-1)
        class_names: ESC-50 category names indexed by label
    """
    esc50_root = Path(esc50_root)
    csv_path = esc50_root / "esc50.csv"
    audio_dir = esc50_root / "audio"
    cache_root = Path(cache_dir) if cache_dir is not None else default_esc50_cache_dir(esc50_root)
    if workers is None:
        workers = _default_workers()

    rows = _read_esc50_csv(csv_path)
    rows, class_names = _select_esc50_rows(rows, esc10, max_clips_per_class)
    n_classes = len(class_names)
    total = len(rows)

    cap = "all" if max_clips_per_class is None else str(max_clips_per_class)
    subset = "esc10" if esc10 else "esc50"
    cache_key = f"esc50_{subset}_n{cap}_c{n_classes}"
    cache_x = cache_root / f"{cache_key}_x.npy"
    cache_labels = cache_root / f"{cache_key}_labels.npy"
    cache_names = cache_root / f"{cache_key}_classes.txt"

    if cache and cache_x.exists() and cache_labels.exists() and cache_names.exists():
        print(f"[sample_data] Loading cached ESC-50 dataset from {cache_root}")
        names = cache_names.read_text().splitlines()
        return np.load(cache_x), np.load(cache_labels), names

    print(
        f"[sample_data] Generating ESC-50 dataset: {n_classes} classes, "
        f"{total} clips ({subset}, cap={cap}, workers={workers})"
    )
    n_samples = int(ESC50_CLIP_DURATION * FS_HW)
    x_data = np.zeros((total, n_samples), dtype=np.float64)
    labels = np.zeros(total, dtype=np.int64)
    for i, row in enumerate(rows):
        labels[i] = int(row["_class_id"])

    done = 0
    for result in _run_integration_pool(rows, audio_dir, want_xy=False, workers=workers):
        idx, x = result
        x_data[idx] = x
        done += 1
        if done % 10 == 0 or done == total:
            print(f"  [{done}/{total}] last: {rows[idx]['filename']} "
                  f"-> class {labels[idx]} ({rows[idx]['category']})")

    if cache:
        cache_root.mkdir(parents=True, exist_ok=True)
        np.save(cache_x, x_data)
        np.save(cache_labels, labels)
        cache_names.write_text("\n".join(class_names))
        print(f"[sample_data] Cached ESC-50 dataset to {cache_root}")

    return x_data, labels, class_names


def generate_dataset_xy_esc50(
    esc50_root: Path | str = ESC50_DEFAULT_ROOT,
    esc10: bool = True,
    max_clips_per_class: int | None = 10,
    cache: bool = True,
    workers: int | None = None,
    cache_dir: Path | str | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], list[str]]:
    """
    Same as generate_dataset_esc50 but captures both x(t) and y(t)
    quadrature states from the Hopf integrator in a single pass.

    Memory: ~2x generate_dataset_esc50 (both x and y stored).
    """
    esc50_root = Path(esc50_root)
    csv_path = esc50_root / "esc50.csv"
    audio_dir = esc50_root / "audio"
    cache_root = Path(cache_dir) if cache_dir is not None else default_esc50_cache_dir(esc50_root)
    if workers is None:
        workers = _default_workers()

    rows = _read_esc50_csv(csv_path)
    rows, class_names = _select_esc50_rows(rows, esc10, max_clips_per_class)
    n_classes = len(class_names)
    total = len(rows)

    cap = "all" if max_clips_per_class is None else str(max_clips_per_class)
    subset = "esc10" if esc10 else "esc50"
    cache_key = f"esc50_xy_{subset}_n{cap}_c{n_classes}"
    cache_x = cache_root / f"{cache_key}_x.npy"
    cache_y_state = cache_root / f"{cache_key}_y_state.npy"
    cache_labels = cache_root / f"{cache_key}_labels.npy"
    cache_names = cache_root / f"{cache_key}_classes.txt"

    if (cache and cache_x.exists() and cache_y_state.exists()
            and cache_labels.exists() and cache_names.exists()):
        print(f"[sample_data] Loading cached ESC-50 XY dataset from {cache_root}")
        names = cache_names.read_text().splitlines()
        return np.load(cache_x), np.load(cache_y_state), np.load(cache_labels), names

    print(
        f"[sample_data] Generating ESC-50 XY dataset: {n_classes} classes, "
        f"{total} clips ({subset}, cap={cap}, workers={workers})"
    )
    n_samples = int(ESC50_CLIP_DURATION * FS_HW)
    x_data = np.zeros((total, n_samples), dtype=np.float64)
    y_data = np.zeros((total, n_samples), dtype=np.float64)
    labels = np.zeros(total, dtype=np.int64)
    for i, row in enumerate(rows):
        labels[i] = int(row["_class_id"])

    done = 0
    for result in _run_integration_pool(rows, audio_dir, want_xy=True, workers=workers):
        idx, x, y = result
        x_data[idx] = x
        y_data[idx] = y
        done += 1
        if done % 10 == 0 or done == total:
            print(f"  [{done}/{total}] last: {rows[idx]['filename']} "
                  f"-> class {labels[idx]} ({rows[idx]['category']})")

    if cache:
        cache_root.mkdir(parents=True, exist_ok=True)
        np.save(cache_x, x_data)
        np.save(cache_y_state, y_data)
        np.save(cache_labels, labels)
        cache_names.write_text("\n".join(class_names))
        print(f"[sample_data] Cached ESC-50 XY dataset to {cache_root}")

    return x_data, y_data, labels, class_names


# ---------------------------------------------------------------------------
# Portable text/JSON export — for reuse across notebooks and other frameworks
# ---------------------------------------------------------------------------

def export_dataset_text(
    out_dir: Path | str,
    x_data: NDArray[np.float64],
    labels: NDArray[np.int64],
    class_names: list[str],
    y_data: NDArray[np.float64] | None = None,
    source_rows: list[dict[str, str]] | None = None,
    source: str = "esc50",
    export_fs: int = FS_TARGET,
    hw_fs: int = FS_HW,
) -> Path:
    """
    Write the integrated dataset as per-clip ASCII files plus manifest.json.

    Layout:
        out_dir/
          manifest.json     -- params, class_names, label per clip, file index
          labels.txt        -- one label int per line (clip-aligned)
          classes.txt       -- one class name per line (label-aligned)
          clips/
            clip_0000_x.txt -- one float per line (single-column x(t))
            clip_0000_y.txt -- present only if y_data was passed
            ...

    `export_fs` controls the on-disk sample rate. The default 4 kHz matches
    Shougat et al. 2023 and the rate `pipeline/ingest.py` consumes — and keeps
    each clip's text file ~240 KB (vs ~12 MB at 100 kHz).
    Pass export_fs=hw_fs to keep the full 100 kHz rate (warning: ~12 MB/clip).
    """
    out_dir = Path(out_dir)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    if hw_fs % export_fs != 0:
        raise ValueError(
            f"export_fs ({export_fs}) must divide hw_fs ({hw_fs}); "
            f"current value gives non-integer downsample factor."
        )
    ds_factor = hw_fs // export_fs
    n = len(labels)
    samples_per_clip = x_data.shape[1] // ds_factor

    files = []
    for i in range(n):
        x_path = clips_dir / f"clip_{i:04d}_x.txt"
        np.savetxt(x_path, x_data[i, ::ds_factor], fmt="%.6e")
        entry: dict = {
            "idx": i,
            "label": int(labels[i]),
            "class_name": class_names[int(labels[i])],
            "x_path": f"clips/{x_path.name}",
        }
        if y_data is not None:
            y_path = clips_dir / f"clip_{i:04d}_y.txt"
            np.savetxt(y_path, y_data[i, ::ds_factor], fmt="%.6e")
            entry["y_path"] = f"clips/{y_path.name}"
        if source_rows is not None:
            entry["source_filename"] = source_rows[i]["filename"]
            entry["source_category"] = source_rows[i]["category"]
        files.append(entry)
        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"  [export {i + 1}/{n}] wrote {x_path.name}")

    np.savetxt(out_dir / "labels.txt", labels.astype(int), fmt="%d")
    (out_dir / "classes.txt").write_text("\n".join(class_names))

    manifest = {
        "source": source,
        "n_clips": n,
        "n_classes": len(class_names),
        "class_names": class_names,
        "label_to_class": {str(i): name for i, name in enumerate(class_names)},
        "audio_fs": ESC50_AUDIO_FS,
        "hw_fs": hw_fs,
        "export_fs": export_fs,
        "downsample_factor": ds_factor,
        "samples_per_clip": int(samples_per_clip),
        "clip_duration_s": ESC50_CLIP_DURATION if source == "esc50" else CLIP_DURATION,
        "hopf": {
            "mu": MU, "A_drive": A_DRIVE,
            "omega": float(OMEGA), "omega_drive": float(OMEGA_DRIVE),
        },
        "files": files,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[sample_data] Exported {n} clips ({export_fs} Hz) to {out_dir}")
    return out_dir


def load_dataset_from_text_cache(
    cache_dir: Path | str,
    target_class: str | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[str], int]:
    """
    Load the Hopf-integrated ESC-50 dataset from the exported text cache.

    Reads per-clip `clips/clip_XXXX_x.txt` files plus `labels.txt`,
    `classes.txt`, and `manifest.json` as written by `export_dataset_text()`.
    Clips are returned at the export sample rate recorded in the manifest
    (typically 4 kHz — already downsampled from the 100 kHz integrator
    output). Feed downstream with `process_dataset(..., downsample_factor=1)`.

    Args:
        cache_dir: directory containing manifest.json, labels.txt, classes.txt, clips/.
        target_class: if set, relabel as a binary task — 1 where the source
            class name equals target_class, else 0. The returned class_names
            becomes [f"not_{target_class}", target_class].

    Returns:
        x_data: (n_clips, samples_per_clip) float64 at export_fs.
        labels: (n_clips,) int64 — either the original 0..N-1 labels or binary.
        class_names: list[str] — original names, or the two-element binary pair.
        fs: int — sample rate of x_data (from manifest.export_fs).
    """
    cache_dir = Path(cache_dir)
    with open(cache_dir / "manifest.json") as f:
        manifest = json.load(f)

    n_clips = int(manifest["n_clips"])
    samples_per_clip = int(manifest["samples_per_clip"])
    fs = int(manifest.get("export_fs", FS_TARGET))
    class_names = list(manifest["class_names"])

    labels = np.loadtxt(cache_dir / "labels.txt", dtype=np.int64)
    if labels.shape != (n_clips,):
        raise ValueError(
            f"labels.txt has {labels.shape[0]} entries but manifest says {n_clips}"
        )

    print(f"[sample_data] Loading {n_clips} clips from {cache_dir} (fs={fs} Hz) ...")
    x_data = np.zeros((n_clips, samples_per_clip), dtype=np.float64)
    for i in range(n_clips):
        x_path = cache_dir / "clips" / f"clip_{i:04d}_x.txt"
        x_data[i] = np.loadtxt(x_path, dtype=np.float64)
        if (i + 1) % 200 == 0 or i + 1 == n_clips:
            print(f"  [{i + 1}/{n_clips}] loaded")

    if target_class is not None:
        if target_class not in class_names:
            raise ValueError(
                f"target_class={target_class!r} not in class_names={class_names}"
            )
        target_id = class_names.index(target_class)
        labels = (labels == target_id).astype(np.int64)
        pos = int(labels.sum())
        print(
            f"[sample_data] Binary relabel: {pos} positive ({target_class}), "
            f"{n_clips - pos} negative — positive rate {pos / n_clips:.1%}"
        )
        class_names = [f"not_{target_class}", target_class]

    return x_data, labels, class_names, fs


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


def generate_dataset_xy(
    n_clips_per_class: int = N_CLIPS_PER_CLASS,
    n_classes: int = N_CLASSES,
    cache: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """
    Generate the full synthetic dataset, returning both x(t) and y(t).

    Both states come from the same single numerical integration — no extra
    computation compared to generate_dataset(). y(t) is the quadrature
    state of the Hopf oscillator.

    Returns:
        x_data: (n_clips, n_hw_samples) raw x(t) at 100 kHz
        y_data: (n_clips, n_hw_samples) raw y(t) at 100 kHz
        labels: (n_clips,) integer class labels
    """
    cache_key = f"hopf_xy_n{n_clips_per_class}_c{n_classes}"
    cache_x = CACHE_DIR / f"{cache_key}_x.npy"
    cache_y_state = CACHE_DIR / f"{cache_key}_y_state.npy"
    cache_labels = CACHE_DIR / f"{cache_key}_labels.npy"

    if cache and cache_x.exists() and cache_y_state.exists() and cache_labels.exists():
        print(f"[sample_data] Loading cached XY dataset from {CACHE_DIR}")
        return np.load(cache_x), np.load(cache_y_state), np.load(cache_labels)

    print(f"[sample_data] Generating XY dataset: {n_classes} classes x {n_clips_per_class} clips ...")
    total = n_clips_per_class * n_classes
    n_samples = int(CLIP_DURATION * FS_HW)
    x_data = np.zeros((total, n_samples), dtype=np.float64)
    y_data = np.zeros((total, n_samples), dtype=np.float64)
    labels = np.zeros(total, dtype=np.int64)

    idx = 0
    for cls in range(n_classes):
        for clip in range(n_clips_per_class):
            seed = cls * 10_000 + clip
            a_func = _class_factory(cls, seed)
            x_data[idx], y_data[idx] = integrate_hopf_xy(a_func)
            labels[idx] = cls
            idx += 1
            if idx % 50 == 0:
                print(f"  [{idx}/{total}] clips generated")

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache_x, x_data)
        np.save(cache_y_state, y_data)
        np.save(cache_labels, labels)
        print(f"[sample_data] Cached XY dataset to {CACHE_DIR}")

    return x_data, y_data, labels


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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Hopf reservoir input dataset.")
    p.add_argument(
        "--source", choices=["synthetic", "esc50"], default="synthetic",
        help="synthetic: 5-class sine/square/chirp/etc. esc50: real audio from ESC-50.",
    )
    p.add_argument(
        "--esc50-root", type=Path, default=ESC50_DEFAULT_ROOT,
        help="Path to ESC-50 dataset root (containing esc50.csv and audio/).",
    )
    p.add_argument(
        "--esc10", action="store_true",
        help="Use the curated 10-class ESC-10 subset instead of full 50-class ESC-50.",
    )
    p.add_argument(
        "--max-clips-per-class", type=int, default=10,
        help="Cap clips per class. Use -1 for all (warning: full ESC-50 is ~8 GB and hours).",
    )
    p.add_argument(
        "--workers", type=int, default=-1,
        help="Parallel integration workers. -1 = cpu_count-2 (default).",
    )
    p.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Directory for .npy caches. Default: <esc50-root>/hopf_cache/ for esc50, "
             "<repo>/data/cache/ for synthetic.",
    )
    p.add_argument("--no-cache", action="store_true", help="Disable .npy cache load/store.")
    p.add_argument(
        "--xy", action="store_true",
        help="Also integrate y(t) (uses generate_dataset_xy*). Doubles memory.",
    )
    p.add_argument(
        "--export-dir", type=Path, default=None,
        help="Also write per-clip .txt files + manifest.json to this directory "
             "(portable for other tools/notebooks).",
    )
    p.add_argument(
        "--export-fs", type=int, default=FS_TARGET,
        help=f"Sample rate for .txt export (default {FS_TARGET} Hz, must divide {FS_HW}). "
             "Use 100000 for full 100 kHz (~12 MB per clip).",
    )
    return p.parse_args()


def main() -> None:
    """Generate and summarise the dataset (synthetic or ESC-50)."""
    args = _parse_args()
    cache = not args.no_cache
    cap = None if args.max_clips_per_class < 0 else args.max_clips_per_class
    workers = None if args.workers < 0 else args.workers

    y_data: NDArray[np.float64] | None = None
    source_rows: list[dict[str, str]] | None = None

    if args.source == "synthetic":
        if args.xy:
            x_data, y_data, labels = generate_dataset_xy(cache=cache)
        else:
            x_data, labels = generate_dataset(cache=cache)
        class_names = list(CLASS_NAMES[:N_CLASSES])
        print(f"\nDataset shape: x={x_data.shape}, labels={labels.shape}")
        for cls in range(N_CLASSES):
            count = int(np.sum(labels == cls))
            print(f"  Class {cls} ({CLASS_NAMES[cls]}): {count} clips")
    else:
        if args.xy:
            x_data, y_data, labels, class_names = generate_dataset_xy_esc50(
                esc50_root=args.esc50_root,
                esc10=args.esc10,
                max_clips_per_class=cap,
                cache=cache,
                workers=workers,
                cache_dir=args.cache_dir,
            )
        else:
            x_data, labels, class_names = generate_dataset_esc50(
                esc50_root=args.esc50_root,
                esc10=args.esc10,
                max_clips_per_class=cap,
                cache=cache,
                workers=workers,
                cache_dir=args.cache_dir,
            )
        if args.export_dir is not None:
            # Re-derive the row list (in the same selection/order used by the
            # generators) so the manifest can include source filenames.
            all_rows = _read_esc50_csv(Path(args.esc50_root) / "esc50.csv")
            source_rows, _ = _select_esc50_rows(all_rows, args.esc10, cap)
        print(f"\nDataset shape: x={x_data.shape}, labels={labels.shape}")
        for cls, name in enumerate(class_names):
            count = int(np.sum(labels == cls))
            print(f"  Class {cls} ({name}): {count} clips")
    print(f"  Sample range: [{x_data.min():.4f}, {x_data.max():.4f}]")

    if args.export_dir is not None:
        export_dataset_text(
            out_dir=args.export_dir,
            x_data=x_data,
            labels=labels,
            class_names=class_names,
            y_data=y_data,
            source_rows=source_rows,
            source=args.source,
            export_fs=args.export_fs,
        )


if __name__ == "__main__":
    main()
