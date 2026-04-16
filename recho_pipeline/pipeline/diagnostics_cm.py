"""
Diagnostic: strip the oscillator's common-mode limit cycle before feature
extraction and see whether the audio-driven residual is class-discriminative.

Rationale: raw x(t) from the hopf_text cache is ~1464x dominated by a shared
limit cycle across all clips. The audio-driven deviation sits ~0.1% on top and
gets crushed by the uint8 min-max scaling in features.extract_features. Here we
subtract the mean clip across the whole dataset (the common oscillation) and
feed the residual through the same pipeline.

Output: output/features_cm_removed/feature_maps_overview.png
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.sample_data import load_dataset_from_text_cache
from pipeline.ingest import process_dataset, FS_HW, FS_TARGET
from pipeline.features import extract_features, visualise_features


ESC50_HOPF_TEXT_CACHE: Path = Path(
    "/Users/nic-spect/data/recho_ai/Kaggle_Environmental_Sound_Classification_50/hopf_text"
)
CLASSES_TO_VISUALISE: list[str] = ["sheep", "dog", "cow", "cat", "rooster"]


def main() -> None:
    print(f"[cm-test] Loading cache from {ESC50_HOPF_TEXT_CACHE} ...")
    raw_x, labels, class_names, fs = load_dataset_from_text_cache(
        cache_dir=ESC50_HOPF_TEXT_CACHE,
    )

    common_mode = raw_x.mean(axis=0)
    residual = raw_x - common_mode
    print(
        f"[cm-test] raw std: {raw_x.std():.4f}  "
        f"residual std: {residual.std():.4f}  "
        f"ratio: {raw_x.std() / residual.std():.1f}x"
    )

    ds_factor = 1 if fs == FS_TARGET else FS_HW // fs
    print(f"[cm-test] Processing residuals (downsample_factor={ds_factor}) ...")
    processed = process_dataset(residual, downsample_factor=ds_factor)

    feature_maps, labels = extract_features(processed, labels)
    print(
        f"[cm-test] feature_maps: shape={feature_maps.shape} "
        f"range=[{feature_maps.min()}, {feature_maps.max()}]"
    )

    missing = [c for c in CLASSES_TO_VISUALISE if c not in class_names]
    if missing:
        raise ValueError(f"classes not in manifest: {missing}")
    orig_ids = [class_names.index(c) for c in CLASSES_TO_VISUALISE]
    mask = np.isin(labels, orig_ids)
    subset_fm = feature_maps[mask]
    subset_labels = np.zeros(int(mask.sum()), dtype=np.int64)
    for new_id, orig_id in enumerate(orig_ids):
        subset_labels[labels[mask] == orig_id] = new_id

    out_dir = Path(__file__).resolve().parent.parent / "output" / "features_cm_removed"
    visualise_features(subset_fm, subset_labels, CLASSES_TO_VISUALISE, output_dir=out_dir)


if __name__ == "__main__":
    main()
