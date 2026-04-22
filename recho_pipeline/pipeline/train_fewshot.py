"""
Few-shot classification on the Hopf feature maps.

Picks K clips per class as the support set, builds one prototype per class as
the mean embedding of its support clips, and classifies the remaining (query)
clips by nearest prototype in embedding space.

Encoder: PCA (PrototypicalNetwork's fallback when no Keras encoder is passed).
PCA is pre-fit on the full query pool so the embedding space is well-conditioned
before the tiny support set is transformed through it — otherwise a 5-shot
support set would fit a degenerate PCA basis.

Metric: K-shot accuracy on the query set. Not directly comparable to the
supervised classifier's val accuracy since the task is different (class
discrimination given few examples, not from-scratch training).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


ESC50_HOPF_TEXT_CACHE: Path = Path(
    "/Users/nic-spect/data/recho_ai/Kaggle_Environmental_Sound_Classification_50/hopf_text"
)
# Classes that form the few-shot task. Must all exist in manifest class_names.
CLASSES_TO_USE: list[str] = ["sheep", "dog", "cow", "cat", "rooster"]
# Shots per class.
K_SHOT: int = 5
# Match the supervised pipeline so comparisons are apples-to-apples.
SUBTRACT_COMMON_MODE: bool = True
# PCA embedding dimensionality. Kept below K_SHOT * len(CLASSES_TO_USE) and
# well below the per-class query pool size so the fit is stable.
PCA_DIM: int = 16
SEED: int = 0


def _split_support_query(
    feature_maps: NDArray[np.uint8],
    labels: NDArray[np.int64],
    k_shot: int,
    seed: int,
) -> tuple[dict[int, NDArray[np.uint8]], NDArray[np.uint8], NDArray[np.int64]]:
    """
    For each class, pick k_shot clips as support; the rest go into the query pool.

    Returns:
        support: dict mapping class_id → (k_shot, 200, 100) support clips.
        query_features: (n_query, 200, 100).
        query_labels: (n_query,) matching class_ids.
    """
    rng = np.random.default_rng(seed)
    support: dict[int, NDArray[np.uint8]] = {}
    query_feat_list = []
    query_lbl_list = []

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) < k_shot + 1:
            raise ValueError(
                f"class {cls} has only {len(cls_idx)} clips — need at least {k_shot + 1}"
            )
        rng.shuffle(cls_idx)
        support[int(cls)] = feature_maps[cls_idx[:k_shot]]
        query_feat_list.append(feature_maps[cls_idx[k_shot:]])
        query_lbl_list.append(labels[cls_idx[k_shot:]])

    return (
        support,
        np.concatenate(query_feat_list, axis=0),
        np.concatenate(query_lbl_list, axis=0),
    )


def main() -> None:
    """Build K-shot prototypes over CLASSES_TO_USE and report query accuracy."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.sample_data import load_dataset_from_text_cache
    from pipeline.ingest import process_dataset, FS_HW, FS_TARGET
    from pipeline.features import extract_features
    from pipeline.models.fewshot.prototypical import PrototypicalNetwork

    print(f"[train_fewshot] Loading hopf_text cache from {ESC50_HOPF_TEXT_CACHE} ...")
    raw_x, labels, class_names, fs = load_dataset_from_text_cache(
        cache_dir=ESC50_HOPF_TEXT_CACHE,
    )

    missing = [c for c in CLASSES_TO_USE if c not in class_names]
    if missing:
        raise ValueError(f"classes not in manifest: {missing}")
    orig_ids = [class_names.index(c) for c in CLASSES_TO_USE]
    mask = np.isin(labels, orig_ids)
    raw_x = raw_x[mask]
    # Relabel to contiguous 0..N-1 so downstream code doesn't need the 50-class gap.
    relabel = np.full(int(mask.sum()), -1, dtype=np.int64)
    for new_id, orig_id in enumerate(orig_ids):
        relabel[labels[mask] == orig_id] = new_id
    labels = relabel
    print(
        f"  Subset to {len(CLASSES_TO_USE)} classes, {len(labels)} clips. "
        f"per-class counts: {[int((labels == i).sum()) for i in range(len(CLASSES_TO_USE))]}"
    )

    ds_factor = 1 if fs == FS_TARGET else FS_HW // fs
    print(
        f"[train_fewshot] Processing clips (downsample_factor={ds_factor}, "
        f"subtract_common_mode={SUBTRACT_COMMON_MODE}) ..."
    )
    processed = process_dataset(
        raw_x,
        downsample_factor=ds_factor,
        subtract_common_mode=SUBTRACT_COMMON_MODE,
    )
    feature_maps, labels = extract_features(processed, labels)

    support, query_features, query_labels = _split_support_query(
        feature_maps, labels, k_shot=K_SHOT, seed=SEED,
    )
    print(
        f"[train_fewshot] Support: {K_SHOT}-shot × {len(CLASSES_TO_USE)} classes. "
        f"Query: {len(query_features)} clips."
    )

    net = PrototypicalNetwork(encoder=None, n_pca_components=PCA_DIM)

    # Pre-fit PCA on the query pool so the embedding basis isn't degenerate.
    # _encode fits on the first call and transforms on subsequent calls.
    print(f"[train_fewshot] Pre-fitting PCA (dim={PCA_DIM}) on query pool ...")
    net._encode(query_features)

    print("[train_fewshot] Building prototypes ...")
    support_named = {CLASSES_TO_USE[cls]: clips for cls, clips in support.items()}
    net.build_prototypes(support_named)

    print("[train_fewshot] Classifying query set ...")
    preds = net.classify_batch(query_features)
    pred_ids = np.array([CLASSES_TO_USE.index(p) for p in preds], dtype=np.int64)
    acc = float(np.mean(pred_ids == query_labels))

    print("\n[train_fewshot] Summary")
    print(f"  classes             : {CLASSES_TO_USE}")
    print(f"  k_shot              : {K_SHOT}")
    print(f"  pca_dim             : {PCA_DIM}")
    print(f"  subtract_common_mode: {SUBTRACT_COMMON_MODE}")
    print(f"  query accuracy      : {acc:.4f}  ({len(query_features)} clips)")

    # Per-class recall gives more signal than overall accuracy when class
    # counts differ (ESC-50 is balanced at 40/class, but still useful to see).
    print("  per-class recall:")
    for cls_id, cls_name in enumerate(CLASSES_TO_USE):
        cls_mask = query_labels == cls_id
        if not np.any(cls_mask):
            continue
        recall = float(np.mean(pred_ids[cls_mask] == cls_id))
        print(f"    {cls_name:<10s} {recall:.4f}  ({int(cls_mask.sum())} clips)")


if __name__ == "__main__":
    main()
