"""
Build a single Hopf hopf_text cache for the merged cat-vs-rest dataset.

Expects the on-disk layout produced by build_merged_cat_vs_rest.py:

    <merged_root>/audio/{test,val,train}/{cat,control}/*.wav

Each wav is already exactly 5 s at 44.1 kHz (resampled + segmented +
augmented upstream), so this script does no windowing and no extra
augmentation — it just loads, resamples to FS_TARGET (4 kHz for the
Hopf drive), integrates x(t) and y(t), and exports one combined cache
with a per-clip `split` field so the loader can filter train/val/test.

Output:

    <merged_root>/hopf_text/
        manifest.json   (each file entry has split + cls)
        labels.txt
        classes.txt
        clips/
            clip_0000_x.txt
            clip_0000_y.txt
            ...

Run:
    python recho_pipeline/data/generate_hopf_cache_merged.py
    python recho_pipeline/data/generate_hopf_cache_merged.py --workers 8
"""

from __future__ import annotations

import argparse
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


DEFAULT_MERGED_ROOT: Path = Path(
    "/Users/nic-spect/data/recho_ai/merged_cat_vs_rest"
)
# Label 0 = control, label 1 = cat — putting the positive last keeps
# binary-relabel shims that do `target_class="cat"` mapping to id 1.
CLASS_NAMES: list[str] = ["control", "cat"]
SPLITS: tuple[str, ...] = ("test", "val", "train")
WINDOW_S: float = 5.0


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
    n_window = int(WINDOW_S * fs)
    if len(x) >= n_window:
        return x[:n_window]
    padded = np.zeros(n_window, dtype=np.float64)
    padded[: len(x)] = x
    return padded


def _integrate_clip_worker(
    args: tuple[int, str]
) -> tuple[int, NDArray[np.float64], NDArray[np.float64]]:
    """Load one wav, run Hopf integrator, return (idx, x(t), y(t))."""
    out_idx, wav_path_str = args
    audio = _load_wav_at_fs(Path(wav_path_str), fs=FS_TARGET)
    a_func = _make_audio_signal(audio, fs=FS_TARGET)
    x, y = integrate_hopf_xy(a_func, duration=WINDOW_S, fs=FS_HW)
    return out_idx, x, y


def _scan_merged(audio_root: Path) -> list[dict[str, str]]:
    """
    Walk <audio_root>/{test,val,train}/{cat,control}/*.wav.

    Returns a list of row dicts with filename, category (cat/control),
    split, and path — in a stable order (split, cls, filename).
    """
    rows: list[dict[str, str]] = []
    for split in SPLITS:
        for cls in CLASS_NAMES:
            cls_dir = audio_root / split / cls
            if not cls_dir.is_dir():
                continue
            for w in sorted(cls_dir.glob("*.wav")):
                rows.append({
                    "filename": w.name,
                    "category": cls,
                    "split": split,
                    "path": str(w),
                })
    return rows


def _default_workers() -> int:
    import os
    return max(1, (os.cpu_count() or 4) - 2)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--merged-root", type=Path, default=DEFAULT_MERGED_ROOT)
    p.add_argument(
        "--workers", type=int, default=-1,
        help="Integration workers. -1 = cpu_count-2 (default).",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Cap clips (first N in scan order) for smoke tests.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    audio_root = args.merged_root / "audio"
    out_dir = args.merged_root / "hopf_text"
    if not audio_root.is_dir():
        raise SystemExit(f"Missing audio dir: {audio_root}")

    rows = _scan_merged(audio_root)
    if args.limit is not None:
        rows = rows[: args.limit]

    n = len(rows)
    if n == 0:
        raise SystemExit(f"No wavs found under {audio_root}")

    cls_counts: dict[str, dict[str, int]] = {s: {c: 0 for c in CLASS_NAMES} for s in SPLITS}
    for r in rows:
        cls_counts[r["split"]][r["category"]] += 1
    print(f"[merged] {n} clips from {audio_root}")
    for s in SPLITS:
        cc = cls_counts[s]
        print(f"  {s:>5}: total={sum(cc.values())}  "
              + "  ".join(f"{c}={cc[c]}" for c in CLASS_NAMES))

    labels = np.array(
        [CLASS_NAMES.index(r["category"]) for r in rows],
        dtype=np.int64,
    )
    n_hw_samples = int(WINDOW_S * FS_HW)
    x_data = np.zeros((n, n_hw_samples), dtype=np.float64)
    y_data = np.zeros((n, n_hw_samples), dtype=np.float64)

    workers = _default_workers() if args.workers < 0 else args.workers
    tasks = [(i, r["path"]) for i, r in enumerate(rows)]
    print(f"[merged] Integrating Hopf ODE at {FS_HW} Hz, "
          f"duration={WINDOW_S}s, workers={workers} ...")
    t0 = time.time()
    done = 0
    if workers <= 1:
        for t in tasks:
            idx, x, y = _integrate_clip_worker(t)
            x_data[idx] = x
            y_data[idx] = y
            done += 1
            if done % 20 == 0 or done == n:
                dt = time.time() - t0
                print(f"  [integrate {done}/{n}] {dt:.0f}s (~{dt/done:.2f}s/clip)")
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for idx, x, y in pool.imap_unordered(
                _integrate_clip_worker, tasks, chunksize=1,
            ):
                x_data[idx] = x
                y_data[idx] = y
                done += 1
                if done % 20 == 0 or done == n:
                    dt = time.time() - t0
                    print(f"  [integrate {done}/{n}] {dt:.0f}s "
                          f"(~{dt/done:.2f}s/clip wall)")

    print(f"[merged] Integration done in {time.time() - t0:.1f}s. "
          f"Exporting to {out_dir} ...")
    export_dataset_text(
        out_dir=out_dir,
        x_data=x_data,
        labels=labels,
        class_names=CLASS_NAMES,
        y_data=y_data,
        source_rows=rows,
        source="merged_cat_vs_rest",
        export_fs=FS_TARGET,
        hw_fs=FS_HW,
        clip_duration_s=WINDOW_S,
    )
    print(f"[merged] Done. Cache: {out_dir}")


if __name__ == "__main__":
    main()
