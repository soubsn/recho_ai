"""
Build the merged cat-vs-rest dataset from Kaggle Cats&Dogs + ESC-50.

Output layout:
    <output_root>/audio/
        test/cat/*.wav
        test/control/*.wav
        val/cat/*.wav
        val/control/*.wav
        train/cat/*.wav
        train/control/*.wav
    <output_root>/manifest.json

Pipeline order (source-aware — no leakage from sliding windows or
augmented copies):
    1. Scan all source files and tag each with (class, source_id).
    2. Carve a fixed test holdout: 50 Kaggle cats + 25 Kaggle dogs + 25
       ESC-50 non-animal sources.
    3. Stratified 80/20 source split on the remainder (train vs val).
    4. Resample Kaggle (16 kHz) to the common target rate and chop into
       non-overlapping 5 s segments; ESC-50 clips are already 5 s.
    5. Apply 4x random-combination augmentation to train segments only;
       val/test stay clean originals.

Notes:
  - Kaggle Cats&Dogs has 164 cat + 113 dog mono wavs at 16 kHz (variable
    length). Sibling `test/` and `train/` subdirs are ignored — we only
    walk the flat `.wav` files.
  - ESC-50 gives 40 clips per class, 5 s at 44.1 kHz. The 12 animal
    classes (defined below) are excluded from the background pool so we
    don't mix animal sounds behind animal sounds.
  - Background pool = ESC-50 non-animal files that landed in the train
    split (not in test, not in val), which keeps test/val audio free of
    train-side leakage.

Run:
    python recho_pipeline/data/build_merged_cat_vs_rest.py
    python recho_pipeline/data/build_merged_cat_vs_rest.py --dry-run
    python recho_pipeline/data/build_merged_cat_vs_rest.py --seed 1
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile
from scipy.signal import resample_poly

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from augmentation import random_combination  # noqa: E402


TARGET_SR: int = 44100
SEGMENT_S: float = 5.0
SEGMENT_SAMPLES: int = int(TARGET_SR * SEGMENT_S)

TEST_N_CAT: int = 50
TEST_N_DOG: int = 25
TEST_N_ESC50_NON_ANIMAL: int = 25

VAL_RATIO: float = 0.20
N_AUG_PER_TRAIN: int = 4
DEFAULT_SEED: int = 0

KAGGLE_ROOT: Path = Path(
    "/Users/nic-spect/data/recho_ai/Kaggle_Dog_vs_Cats/cats_dogs"
)
ESC50_ROOT: Path = Path(
    "/Users/nic-spect/data/recho_ai/Kaggle_Environmental_Sound_Classification_50"
)
ESC50_AUDIO_DIR: Path = ESC50_ROOT / "audio"
ESC50_CSV: Path = ESC50_ROOT / "esc50.csv"

DEFAULT_OUTPUT_ROOT: Path = Path(
    "/Users/nic-spect/data/recho_ai/merged_cat_vs_rest"
)

ESC50_ANIMAL: frozenset[str] = frozenset({
    "cat", "chirping_birds", "cow", "crickets", "crow", "dog", "frog",
    "hen", "insects", "pig", "rooster", "sheep",
})


@dataclass
class Source:
    """One original audio file before segmentation."""
    path: Path
    cls: str                       # "cat" or "control"
    source_id: str                 # unique id for dedup / manifest
    origin: str                    # "kaggle_cat" | "kaggle_dog" | "esc50"
    esc50_category: str | None = None


@dataclass
class Segment:
    """One 5 s segment derived from a Source."""
    audio: NDArray[np.float64]
    cls: str
    source_id: str
    origin: str
    seg_index: int
    esc50_category: str | None = None


@dataclass
class ManifestEntry:
    path: str
    cls: str
    source_id: str
    origin: str
    seg_index: int
    aug_index: int | None = None
    augmentations: list[str] = field(default_factory=list)


def load_mono(path: Path) -> tuple[int, NDArray[np.float64]]:
    """Load wav as (sr, mono float64 in [-1, 1])."""
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        max_val = float(np.iinfo(x.dtype).max)
        x = x.astype(np.float64) / max_val
    else:
        x = x.astype(np.float64)
    return sr, x


def resample_to(x: NDArray[np.float64], sr_in: int, sr_out: int) -> NDArray[np.float64]:
    """Polyphase resample; identity if rates already match."""
    if sr_in == sr_out:
        return x
    from math import gcd
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float64)


def save_wav(path: Path, x: NDArray[np.float64], sr: int = TARGET_SR) -> None:
    """Write int16 mono wav. Clips to [-1, 1] defensively."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(x, -1.0, 1.0)
    wavfile.write(path, sr, (x * 32767.0).astype(np.int16))


def segment_5s(x: NDArray[np.float64]) -> list[NDArray[np.float64]]:
    """Chop into non-overlapping 5 s chunks at TARGET_SR. Drops partial tail."""
    n_full = len(x) // SEGMENT_SAMPLES
    return [x[i * SEGMENT_SAMPLES:(i + 1) * SEGMENT_SAMPLES] for i in range(n_full)]


def scan_kaggle() -> list[Source]:
    """Flat `.wav` files only — ignores the nested test/ and train/ dirs."""
    sources: list[Source] = []
    for w in sorted(KAGGLE_ROOT.glob("*.wav")):
        head = w.stem.split("_")[0].lower()
        if head == "cat":
            sources.append(Source(w, "cat", f"kaggle_{w.stem}", "kaggle_cat"))
        elif head == "dog":
            sources.append(Source(w, "control", f"kaggle_{w.stem}", "kaggle_dog"))
        # stray files (e.g. .DS_Store) silently ignored
    return sources


def scan_esc50() -> list[Source]:
    """Read esc50.csv; cat → cat, everything else → control."""
    sources: list[Source] = []
    with open(ESC50_CSV, newline="") as f:
        for row in csv.DictReader(f):
            category = row["category"]
            cls = "cat" if category == "cat" else "control"
            sources.append(Source(
                path=ESC50_AUDIO_DIR / row["filename"],
                cls=cls,
                source_id=f"esc50_{row['filename']}",
                origin="esc50",
                esc50_category=category,
            ))
    return sources


def select_test_and_split(
    kaggle: list[Source],
    esc50: list[Source],
    rng: np.random.Generator,
) -> tuple[list[Source], list[Source], list[Source]]:
    """
    Carve test holdout, then source-aware stratified 80/20 on the rest.

    Returns (test, train, val) as lists of Sources. A source placed in
    val never contributes to train or to the background pool.
    """
    kaggle_cats = [s for s in kaggle if s.cls == "cat"]
    kaggle_dogs = [s for s in kaggle if s.cls == "control"]
    esc50_non_animal = [
        s for s in esc50 if s.esc50_category not in ESC50_ANIMAL
    ]

    if len(kaggle_cats) < TEST_N_CAT:
        raise ValueError(f"need {TEST_N_CAT} kaggle cats, have {len(kaggle_cats)}")
    if len(kaggle_dogs) < TEST_N_DOG:
        raise ValueError(f"need {TEST_N_DOG} kaggle dogs, have {len(kaggle_dogs)}")
    if len(esc50_non_animal) < TEST_N_ESC50_NON_ANIMAL:
        raise ValueError(
            f"need {TEST_N_ESC50_NON_ANIMAL} esc50 non-animal, "
            f"have {len(esc50_non_animal)}"
        )

    def _pick(pool: list[Source], n: int) -> tuple[list[Source], list[Source]]:
        idx = rng.permutation(len(pool))
        picked = [pool[i] for i in idx[:n]]
        rest = [pool[i] for i in idx[n:]]
        return picked, rest

    test_cats, rest_kaggle_cats = _pick(kaggle_cats, TEST_N_CAT)
    test_dogs, rest_kaggle_dogs = _pick(kaggle_dogs, TEST_N_DOG)
    test_esc_na, rest_esc_na = _pick(esc50_non_animal, TEST_N_ESC50_NON_ANIMAL)

    test = test_cats + test_dogs + test_esc_na

    esc50_cats = [s for s in esc50 if s.esc50_category == "cat"]
    esc50_animal_non_cat = [
        s for s in esc50
        if s.esc50_category in ESC50_ANIMAL and s.esc50_category != "cat"
    ]

    cats_pool = rest_kaggle_cats + esc50_cats
    control_pool = rest_kaggle_dogs + esc50_animal_non_cat + rest_esc_na

    def _train_val_split(pool: list[Source]) -> tuple[list[Source], list[Source]]:
        idx = rng.permutation(len(pool))
        n_val = int(round(len(pool) * VAL_RATIO))
        val = [pool[i] for i in idx[:n_val]]
        train = [pool[i] for i in idx[n_val:]]
        return train, val

    cat_train, cat_val = _train_val_split(cats_pool)
    ctrl_train, ctrl_val = _train_val_split(control_pool)

    return test, cat_train + ctrl_train, cat_val + ctrl_val


def sources_to_segments(
    sources: list[Source],
) -> list[Segment]:
    """Load, resample, and 5 s-chunk every source."""
    segments: list[Segment] = []
    for src in sources:
        sr, x = load_mono(src.path)
        x = resample_to(x, sr, TARGET_SR)
        chunks = segment_5s(x)
        for i, chunk in enumerate(chunks):
            segments.append(Segment(
                audio=chunk,
                cls=src.cls,
                source_id=src.source_id,
                origin=src.origin,
                seg_index=i,
                esc50_category=src.esc50_category,
            ))
    return segments


def build_bg_pool(train_segments: list[Segment]) -> list[NDArray[np.float64]]:
    """
    Background clips = train-split ESC-50 non-animal segments.

    Keeping the pool to non-animal avoids contaminating the cat signal
    with another animal class; limiting to train-split ensures test/val
    segments never appear as background material.
    """
    return [
        seg.audio for seg in train_segments
        if seg.origin == "esc50"
        and seg.esc50_category is not None
        and seg.esc50_category not in ESC50_ANIMAL
    ]


def _segment_filename(seg: Segment, aug_index: int | None) -> str:
    """Stable, collision-free filename for a segment."""
    suffix = f"_aug{aug_index}" if aug_index is not None else ""
    return f"{seg.source_id}_seg{seg.seg_index:03d}{suffix}.wav"


def write_split(
    split_name: str,
    segments: list[Segment],
    output_root: Path,
    rng: np.random.Generator | None = None,
    bg_pool: list[NDArray[np.float64]] | None = None,
    n_aug: int = 0,
) -> list[ManifestEntry]:
    """
    Write segments to <output_root>/audio/<split>/<cls>/*.wav.

    For train splits pass `rng`, `bg_pool`, and `n_aug` so each original
    segment is emitted alongside N_AUG_PER_TRAIN augmented copies.
    """
    entries: list[ManifestEntry] = []
    for seg in segments:
        out_dir = output_root / "audio" / split_name / seg.cls
        fname = _segment_filename(seg, aug_index=None)
        out_path = out_dir / fname
        save_wav(out_path, seg.audio)
        entries.append(ManifestEntry(
            path=str(out_path.relative_to(output_root)),
            cls=seg.cls,
            source_id=seg.source_id,
            origin=seg.origin,
            seg_index=seg.seg_index,
        ))
        if n_aug > 0:
            assert rng is not None and bg_pool is not None
            for k in range(n_aug):
                aug_audio, ops = random_combination(
                    x=seg.audio, sr=TARGET_SR, bg_pool=bg_pool, rng=rng,
                )
                aug_name = _segment_filename(seg, aug_index=k)
                aug_path = out_dir / aug_name
                save_wav(aug_path, aug_audio)
                entries.append(ManifestEntry(
                    path=str(aug_path.relative_to(output_root)),
                    cls=seg.cls,
                    source_id=seg.source_id,
                    origin=seg.origin,
                    seg_index=seg.seg_index,
                    aug_index=k,
                    augmentations=ops,
                ))
    return entries


def _summary(name: str, entries: list[ManifestEntry]) -> str:
    n_cat = sum(1 for e in entries if e.cls == "cat")
    n_ctrl = sum(1 for e in entries if e.cls == "control")
    return f"  {name:>6}: {len(entries):>5} total  |  cat={n_cat}  control={n_ctrl}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument(
        "--dry-run", action="store_true",
        help="Scan + plan only; don't write audio.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Cap sources-per-bucket for quick smoke tests.",
    )
    p.add_argument(
        "--clobber", action="store_true",
        help="Remove any existing <output_root>/audio before writing.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    kaggle = scan_kaggle()
    esc50 = scan_esc50()
    print(f"[build] scanned {len(kaggle)} kaggle + {len(esc50)} esc50 sources")

    if args.limit is not None:
        kaggle = kaggle[: args.limit]
        esc50 = esc50[: args.limit]
        print(f"[build] --limit {args.limit} → {len(kaggle)} kaggle, {len(esc50)} esc50")

    test_src, train_src, val_src = select_test_and_split(kaggle, esc50, rng)
    print(
        f"[build] sources → test={len(test_src)}  "
        f"train={len(train_src)}  val={len(val_src)}"
    )

    print("[build] segmenting test ...")
    test_segs = sources_to_segments(test_src)
    print("[build] segmenting val  ...")
    val_segs = sources_to_segments(val_src)
    print("[build] segmenting train ...")
    train_segs = sources_to_segments(train_src)

    print(
        f"[build] segments → test={len(test_segs)}  "
        f"train={len(train_segs)}  val={len(val_segs)}"
    )

    bg_pool = build_bg_pool(train_segs)
    print(f"[build] background pool = {len(bg_pool)} esc50 non-animal train segments")

    if args.dry_run:
        print("[build] --dry-run set; not writing audio.")
        return

    audio_root = args.output_root / "audio"
    if audio_root.exists():
        if args.clobber:
            print(f"[build] --clobber: removing {audio_root}")
            shutil.rmtree(audio_root)
        else:
            raise SystemExit(
                f"[build] refuse to overwrite {audio_root} — pass --clobber"
            )

    test_entries = write_split("test", test_segs, args.output_root)
    val_entries = write_split("val", val_segs, args.output_root)
    train_entries = write_split(
        "train", train_segs, args.output_root,
        rng=rng, bg_pool=bg_pool, n_aug=N_AUG_PER_TRAIN,
    )

    manifest = {
        "target_sr": TARGET_SR,
        "segment_s": SEGMENT_S,
        "seed": args.seed,
        "n_aug_per_train": N_AUG_PER_TRAIN,
        "val_ratio": VAL_RATIO,
        "test": [e.__dict__ for e in test_entries],
        "val": [e.__dict__ for e in val_entries],
        "train": [e.__dict__ for e in train_entries],
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    elapsed = time.time() - t0
    print(f"\n[build] wrote manifest → {manifest_path}")
    print(_summary("test", test_entries))
    print(_summary("val", val_entries))
    print(_summary("train", train_entries))
    print(f"[build] done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
