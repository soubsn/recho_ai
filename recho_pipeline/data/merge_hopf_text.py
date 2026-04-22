"""
Concatenate multiple Hopf text-cache exports into one combined cache.

Typical flow when you generate originals and mixtures separately:

  1. Hopf-integrate originals once (slow):
       sample_data.py --source esc50 --export-dir .../hopf_text

  2. Generate mixtures only (mix_esc50.py --no-originals), then
     Hopf-integrate just the mixtures:
       sample_data.py --source esc50 --csv-name esc50_mixed.csv \\
           --audio-subdir audio_mixed --export-dir .../hopf_text_mix_only

  3. Merge into a single cache the trainer can load:
       python -m recho_pipeline.data.merge_hopf_text \\
           --source .../hopf_text --source .../hopf_text_mix_only \\
           --out-dir .../hopf_text_combined

The merged directory has the same layout as a single export_dataset_text()
output: manifest.json + labels.txt + classes.txt + clips/clip_NNNN_x.txt,
so load_dataset_from_text_cache() reads it without modification.

Compatibility requirements across sources:
  - identical audio_fs, hw_fs, export_fs, samples_per_clip, clip_duration_s
  - identical hopf parameters (mu, A_drive, omega, omega_drive)
  - identical class_names (so integer labels are directly concatenable)

`mix_esc50.py`'s Phase 1 guarantees every ESC-50 category appears in at
least one mixture, so the mixture-only cache and the originals cache
produce identical class_names lists.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


_META_KEYS: tuple[str, ...] = (
    "audio_fs", "hw_fs", "export_fs", "downsample_factor",
    "samples_per_clip", "clip_duration_s",
)
_HOPF_KEYS: tuple[str, ...] = ("mu", "A_drive", "omega", "omega_drive")


def _load_manifest(cache_dir: Path) -> dict:
    with open(cache_dir / "manifest.json") as f:
        return json.load(f)


def _validate_compatible(manifests: list[dict]) -> None:
    """Raise ValueError if the caches can't be safely concatenated."""
    ref = manifests[0]
    for key in _META_KEYS:
        ref_val = ref.get(key)
        for m in manifests[1:]:
            if m.get(key) != ref_val:
                raise ValueError(
                    f"manifest mismatch on {key!r}: {ref_val!r} vs {m.get(key)!r}"
                )
    ref_hopf = ref.get("hopf", {})
    for m in manifests[1:]:
        m_hopf = m.get("hopf", {})
        for k in _HOPF_KEYS:
            if m_hopf.get(k) != ref_hopf.get(k):
                raise ValueError(
                    f"hopf parameter mismatch on {k!r}: "
                    f"{ref_hopf.get(k)!r} vs {m_hopf.get(k)!r}"
                )
    ref_classes = list(ref.get("class_names", []))
    for m in manifests[1:]:
        m_classes = list(m.get("class_names", []))
        if m_classes != ref_classes:
            raise ValueError(
                "class_names mismatch between caches — integer labels would not "
                "be consistent. Re-export both from a CSV that covers the same "
                "full category set (mix_esc50.py's Phase 1 guarantee normally "
                "ensures this).\n"
                f"  source 0: {ref_classes}\n"
                f"  other:    {m_classes}"
            )


def _place(src: Path, dst: Path, mode: str) -> None:
    """Put src at dst via symlink or copy. Overwrites existing dst."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def merge_text_caches(
    source_dirs: list[Path],
    out_dir: Path,
    mode: str = "symlink",
) -> Path:
    """
    Concatenate >=2 Hopf text-cache exports into out_dir.

    Clip files are re-indexed contiguously — source B's clip_0000 becomes
    clip_<n_A> in the merged output. Per-file metadata (source_category,
    source_filename) is preserved in the merged manifest so the binary
    relabel path in load_dataset_from_text_cache keeps working.

    Args:
        source_dirs: ordered list of cache dirs to merge.
        out_dir: destination. Must not already contain a different cache.
        mode: "symlink" (fast, sources must stay put) or "copy"
              (self-contained, doubles disk usage for clip files).

    Returns:
        out_dir (as a Path).
    """
    if len(source_dirs) < 2:
        raise ValueError(f"need >=2 source caches, got {len(source_dirs)}")
    if mode not in ("symlink", "copy"):
        raise ValueError(f"mode must be 'symlink' or 'copy', got {mode!r}")

    source_dirs = [Path(d) for d in source_dirs]
    manifests = [_load_manifest(d) for d in source_dirs]
    _validate_compatible(manifests)
    ref = manifests[0]
    class_names = list(ref["class_names"])

    out_dir = Path(out_dir)
    clips_out = out_dir / "clips"
    clips_out.mkdir(parents=True, exist_ok=True)

    merged_files: list[dict] = []
    merged_labels: list[int] = []
    source_summaries: list[str] = []
    offset = 0

    for src_dir, m in zip(source_dirs, manifests):
        files = m.get("files")
        if not files:
            raise ValueError(
                f"{src_dir}/manifest.json has no 'files' entries; cannot re-index. "
                f"Re-export with the current export_dataset_text()."
            )
        n_src = int(m["n_clips"])
        if len(files) != n_src:
            raise ValueError(
                f"{src_dir}: manifest n_clips={n_src} but files list has {len(files)}"
            )

        src_labels = np.loadtxt(src_dir / "labels.txt", dtype=np.int64)
        if src_labels.ndim == 0:
            src_labels = src_labels.reshape(1)
        if src_labels.shape[0] != n_src:
            raise ValueError(
                f"{src_dir}/labels.txt has {src_labels.shape[0]} entries but "
                f"manifest n_clips={n_src}"
            )

        for entry in files:
            new_idx = offset + int(entry["idx"])
            x_dst_name = f"clip_{new_idx:04d}_x.txt"
            _place(src_dir / entry["x_path"], clips_out / x_dst_name, mode)
            new_entry = dict(entry)
            new_entry["idx"] = new_idx
            new_entry["x_path"] = f"clips/{x_dst_name}"
            if "y_path" in entry:
                y_dst_name = f"clip_{new_idx:04d}_y.txt"
                _place(src_dir / entry["y_path"], clips_out / y_dst_name, mode)
                new_entry["y_path"] = f"clips/{y_dst_name}"
            merged_files.append(new_entry)

        merged_labels.extend(src_labels.tolist())
        source_summaries.append(f"{src_dir.name}={n_src}")
        offset += n_src

    np.savetxt(
        out_dir / "labels.txt",
        np.array(merged_labels, dtype=np.int64),
        fmt="%d",
    )
    (out_dir / "classes.txt").write_text("\n".join(class_names))

    merged_manifest: dict = {
        "source": "merged",
        "n_clips": len(merged_files),
        "n_classes": len(class_names),
        "class_names": class_names,
        "label_to_class": {str(i): name for i, name in enumerate(class_names)},
        "hopf": ref.get("hopf", {}),
        "merged_from": [str(d.resolve()) for d in source_dirs],
        "merge_mode": mode,
        "files": merged_files,
    }
    for key in _META_KEYS:
        if key in ref:
            merged_manifest[key] = ref[key]

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(merged_manifest, f, indent=2)

    print(
        f"[merge_hopf_text] Merged {len(source_dirs)} caches "
        f"({' + '.join(source_summaries)} = {len(merged_files)} clips) "
        f"into {out_dir} (mode={mode})"
    )
    return out_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Concatenate Hopf text-cache exports into one directory."
    )
    p.add_argument(
        "--source", type=Path, action="append", required=True,
        help="Source cache dir. Repeat --source for each cache to include; "
             "order is preserved in the merged output.",
    )
    p.add_argument(
        "--out-dir", type=Path, required=True,
        help="Destination directory for the merged cache.",
    )
    p.add_argument(
        "--mode", choices=("symlink", "copy"), default="symlink",
        help="symlink (default, fast, sources must stay put) or copy "
             "(self-contained, duplicates clip files on disk).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    merge_text_caches(
        source_dirs=args.source,
        out_dir=args.out_dir,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
