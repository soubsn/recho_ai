"""
Pre-Hopf audio mixture generator for ESC-50.

Reads esc50.csv, produces mixtures of 2+ clips at the native 44.1 kHz
sample rate, and writes an extended CSV plus a sibling audio directory:

  <esc50_root>/
    audio/               (untouched)
    esc50.csv            (untouched)
    audio_mixed/         (new — symlinks to originals + mix_NNNNN.wav)
    esc50_mixed.csv      (new — originals + mixtures, pipe-joined categories)

Mixture rows use the same ESC-50 schema with pipe-joined values for the
multi-label fields:

    filename              mix_00000.wav
    fold                  (taken from primary component)
    target                8|5|26
    category              sheep|dog|laughing
    esc10                 False
    src_file              <pipe-joined originals>
    take                  M

The binary relabel path in data/sample_data.load_dataset_from_text_cache
treats `target_class` as positive when it appears in the pipe-split list,
so this works unchanged for any class the user points it at.

Mixing recipe:
  1. Load each component wav, convert to mono float64 in [-1, 1].
  2. Peak-normalize each to 0.7 (so components are on the same scale).
  3. Sum, divide by sqrt(n_components) to keep energy roughly constant.
  4. Clip to [-1, 1] and write as int16 mono at the source sample rate.

Guarantee: every ESC-50 category appears in at least one mixture — this
ensures no class can fall entirely into the validation split with only
single-label examples.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile


DEFAULT_ESC50_ROOT: Path = Path(
    "/Users/nic-spect/data/recho_ai/Kaggle_Environmental_Sound_Classification_50"
)
DEFAULT_MIX_SIZE_DIST: str = "2:0.6,3:0.3,4:0.1"

ESC50_CSV_COLUMNS: list[str] = [
    "filename", "fold", "target", "category", "esc10", "src_file", "take",
]


def _parse_mix_size_dist(s: str) -> dict[int, float]:
    """Parse '2:0.6,3:0.3,4:0.1' into normalized {size: probability}."""
    dist: dict[int, float] = {}
    for part in s.split(","):
        k, v = part.split(":")
        size = int(k.strip())
        prob = float(v.strip())
        if size < 2:
            raise ValueError(f"mix size must be >= 2, got {size}")
        if prob < 0:
            raise ValueError(f"probability must be >= 0, got {prob}")
        dist[size] = prob
    total = sum(dist.values())
    if total <= 0:
        raise ValueError("mix size distribution has zero total probability")
    return {k: v / total for k, v in dist.items()}


def _load_mono_float(wav_path: Path) -> tuple[int, NDArray[np.float64]]:
    """Load wav as (sr, mono float64 in [-1, 1])."""
    sr, x = wavfile.read(wav_path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        max_val = float(np.iinfo(x.dtype).max)
        x = x.astype(np.float64) / max_val
    else:
        x = x.astype(np.float64)
    return sr, x


def mix_clips(
    paths: list[Path],
    target_peak: float = 0.7,
) -> tuple[int, NDArray[np.int16]]:
    """
    Mix >=2 wavs. All inputs must share a sample rate.

    Returns (sr, int16 mono mixture). Components are peak-normalized to
    `target_peak` before summing so loud clips don't drown quiet ones.
    """
    if len(paths) < 2:
        raise ValueError(f"need >=2 components to mix, got {len(paths)}")
    signals: list[NDArray[np.float64]] = []
    sr_ref: int | None = None
    for p in paths:
        sr, x = _load_mono_float(p)
        if sr_ref is None:
            sr_ref = sr
        elif sr != sr_ref:
            raise ValueError(f"sample rate mismatch: {sr} vs {sr_ref} ({p})")
        peak = float(np.max(np.abs(x)))
        if peak > 0:
            x = x * (target_peak / peak)
        signals.append(x)

    min_len = min(len(s) for s in signals)
    mix = np.sum([s[:min_len] for s in signals], axis=0) / np.sqrt(len(signals))
    mix = np.clip(mix, -1.0, 1.0)
    assert sr_ref is not None
    return sr_ref, (mix * 32767.0).astype(np.int16)


def _pipe_join(values: list[str]) -> str:
    return "|".join(values)


def _symlink_originals(
    rows: list[dict[str, str]],
    src_audio_dir: Path,
    out_audio_dir: Path,
) -> int:
    """Symlink each original wav into out_audio_dir. Returns count created."""
    created = 0
    for r in rows:
        src = (src_audio_dir / r["filename"]).resolve()
        dst = out_audio_dir / r["filename"]
        if dst.exists() or dst.is_symlink():
            continue
        dst.symlink_to(src)
        created += 1
    return created


def _build_mixture(
    idx: int,
    components: list[dict[str, str]],
    src_audio_dir: Path,
    out_audio_dir: Path,
) -> dict[str, str]:
    """Mix components, write wav, return CSV row dict."""
    paths = [src_audio_dir / c["filename"] for c in components]
    sr, audio = mix_clips(paths)
    out_name = f"mix_{idx:05d}.wav"
    wavfile.write(out_audio_dir / out_name, sr, audio)
    return {
        "filename": out_name,
        "fold": components[0]["fold"],
        "target": _pipe_join([c["target"] for c in components]),
        "category": _pipe_join([c["category"] for c in components]),
        "esc10": "False",
        "src_file": _pipe_join([c["src_file"] for c in components]),
        "take": "M",
    }


def generate_mixtures(
    esc50_root: Path,
    n_mixtures: int,
    mix_size_dist: dict[int, float],
    seed: int,
    out_audio_subdir: str,
    out_csv_name: str,
    include_originals: bool,
    source_csv_name: str,
    source_audio_subdir: str,
) -> Path:
    """
    Generate mixtures and write new CSV + audio directory.

    Phase 1 guarantees each of the N categories appears in at least one
    mixture of size >=2. Phase 2 fills the remainder by sampling per
    `mix_size_dist` with each component drawn from a distinct category.

    Returns:
        Path to the new CSV.
    """
    src_csv = esc50_root / source_csv_name
    src_audio_dir = esc50_root / source_audio_subdir
    out_audio_dir = esc50_root / out_audio_subdir
    out_csv = esc50_root / out_csv_name

    with open(src_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if fieldnames != ESC50_CSV_COLUMNS:
        raise ValueError(
            f"Unexpected CSV header in {src_csv}: {fieldnames} "
            f"(expected {ESC50_CSV_COLUMNS})"
        )

    by_category: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by_category.setdefault(r["category"], []).append(r)
    all_categories = sorted(by_category)

    if n_mixtures < len(all_categories):
        raise ValueError(
            f"n_mixtures ({n_mixtures}) < n_categories ({len(all_categories)}); "
            f"cannot guarantee one mixture per class."
        )

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    sizes = list(mix_size_dist.keys())
    probs = [mix_size_dist[s] for s in sizes]

    def _sample_size() -> int:
        return int(np_rng.choice(sizes, p=probs))

    out_audio_dir.mkdir(parents=True, exist_ok=True)

    if include_originals:
        n_linked = _symlink_originals(rows, src_audio_dir, out_audio_dir)
        print(f"[mix_esc50] Symlinked {n_linked} originals into {out_audio_dir}")

    mixture_records: list[dict[str, str]] = []

    print(f"[mix_esc50] Phase 1: guaranteed coverage for {len(all_categories)} categories")
    for i, cat in enumerate(all_categories):
        size = _sample_size()
        primary = rng.choice(by_category[cat])
        other_cats = [c for c in all_categories if c != cat]
        chosen_other = rng.sample(other_cats, k=size - 1)
        components = [primary] + [rng.choice(by_category[c]) for c in chosen_other]
        mixture_records.append(
            _build_mixture(i, components, src_audio_dir, out_audio_dir)
        )
        if (i + 1) % 10 == 0 or i + 1 == len(all_categories):
            print(f"  [phase1 {i + 1}/{len(all_categories)}] covered {cat}")

    n_remaining = n_mixtures - len(all_categories)
    print(f"[mix_esc50] Phase 2: {n_remaining} random mixtures")
    for j in range(n_remaining):
        size = _sample_size()
        size = min(size, len(all_categories))
        chosen_cats = rng.sample(all_categories, k=size)
        components = [rng.choice(by_category[c]) for c in chosen_cats]
        idx = len(all_categories) + j
        mixture_records.append(
            _build_mixture(idx, components, src_audio_dir, out_audio_dir)
        )
        if (j + 1) % 100 == 0 or j + 1 == n_remaining:
            print(f"  [phase2 {j + 1}/{n_remaining}] mixtures written")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ESC50_CSV_COLUMNS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        if include_originals:
            for r in rows:
                writer.writerow({k: r[k] for k in ESC50_CSV_COLUMNS})
        for m in mixture_records:
            writer.writerow(m)

    print(
        f"[mix_esc50] Wrote {len(mixture_records)} mixtures "
        f"({'+' + str(len(rows)) + ' originals' if include_originals else 'mixtures only'}) "
        f"to {out_csv}"
    )
    return out_csv


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate pre-Hopf ESC-50 audio mixtures with pipe-joined multi-label CSV."
    )
    p.add_argument("--esc50-root", type=Path, default=DEFAULT_ESC50_ROOT)
    p.add_argument("--n-mixtures", type=int, default=1000)
    p.add_argument(
        "--mix-sizes", type=str, default=DEFAULT_MIX_SIZE_DIST,
        help="Comma-separated size:probability pairs, e.g. '2:0.6,3:0.3,4:0.1'. "
             "Probabilities are renormalized.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source-csv-name", type=str, default="esc50.csv")
    p.add_argument("--source-audio-subdir", type=str, default="audio")
    p.add_argument("--out-audio-subdir", type=str, default="audio_mixed")
    p.add_argument("--out-csv-name", type=str, default="esc50_mixed.csv")
    p.add_argument(
        "--no-originals", action="store_true",
        help="Omit original rows/symlinks from the output (default: include them).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    dist = _parse_mix_size_dist(args.mix_sizes)
    print(
        f"[mix_esc50] n_mixtures={args.n_mixtures}, size_dist={dist}, "
        f"seed={args.seed}"
    )
    generate_mixtures(
        esc50_root=args.esc50_root,
        n_mixtures=args.n_mixtures,
        mix_size_dist=dist,
        seed=args.seed,
        out_audio_subdir=args.out_audio_subdir,
        out_csv_name=args.out_csv_name,
        include_originals=not args.no_originals,
        source_csv_name=args.source_csv_name,
        source_audio_subdir=args.source_audio_subdir,
    )


if __name__ == "__main__":
    main()
