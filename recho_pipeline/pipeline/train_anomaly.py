"""
Unsupervised anomaly detection on the Hopf feature maps.

Trains only on one class designated "normal" (NORMAL_CLASS) and scores every
validation clip by deviation from that distribution. Clips from any other class
count as anomalies at evaluation time.

Metric: ROC AUC on the binary (normal vs rest) task. Unlike accuracy, AUROC is
threshold-free and robust to the steep class imbalance of ESC-50 — a detector
that perfectly ranks normal above anomaly scores 1.0 regardless of where you
put the alarm threshold.

Models trained here:
  - GMMDetector        (pipeline.models.ml.gmm_anomaly)
  - OneClassSVMDetector (pipeline.models.anomaly.one_class_svm)

Both are sklearn-based; no QAT step. Input is the same uint8 (200, 100) feature
map used by the supervised classifier, so the ingest + feature-extraction path
is identical to train.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score


ESC50_HOPF_TEXT_CACHE: Path = Path(
    "/Users/nic-spect/data/recho_ai/Kaggle_Environmental_Sound_Classification_50/hopf_text"
)
# Class treated as the "normal" distribution the detector is fit on.
# Everything else is treated as anomaly at evaluation time.
NORMAL_CLASS: str = "sheep"
# Match the supervised pipeline so comparisons are apples-to-apples.
SUBTRACT_COMMON_MODE: bool = True
# Hold out this fraction of normal clips for the evaluation set so the AUROC
# isn't computed on points the detector was fit on.
VAL_SPLIT: float = 0.2


def _split_normal_val(
    feature_maps: NDArray[np.uint8],
    labels: NDArray[np.int64],
    val_split: float,
    seed: int = 0,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.int64]]:
    """
    Split into (normal_train, eval_features, eval_labels).

    All anomaly clips go into the eval set. Normal clips are split by val_split
    so the detector is fit on one subset and scored on the rest.

    Returns:
        normal_train: (n_normal_train, 200, 100) — normal clips for fit().
        eval_features: (n_eval, 200, 100) — held-out normals + all anomalies.
        eval_labels: (n_eval,) — 0 for normal, 1 for anomaly.
    """
    rng = np.random.default_rng(seed)
    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]

    rng.shuffle(normal_idx)
    n_val = max(1, int(len(normal_idx) * val_split))
    val_normal_idx = normal_idx[:n_val]
    train_normal_idx = normal_idx[n_val:]

    eval_idx = np.concatenate([val_normal_idx, anomaly_idx])
    eval_labels = np.concatenate([
        np.zeros(len(val_normal_idx), dtype=np.int64),
        np.ones(len(anomaly_idx), dtype=np.int64),
    ])
    perm = rng.permutation(len(eval_idx))

    return (
        feature_maps[train_normal_idx],
        feature_maps[eval_idx][perm],
        eval_labels[perm],
    )


def _gmm_scores(detector, eval_features: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Higher score = more anomalous. GMM returns log-likelihood (higher = normal), so negate."""
    log_lik = detector.score_batch(eval_features)
    return -log_lik


def _ocsvm_scores(detector, eval_features: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Higher score = more anomalous. OCSVM decision_function is positive for normal, so negate."""
    scores = np.array([detector.decision_score(c) for c in eval_features])
    return -scores


def main() -> None:
    """Fit GMM and One-Class SVM on NORMAL_CLASS; report AUROC against everything else."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.sample_data import load_dataset_from_text_cache
    from pipeline.ingest import process_dataset, FS_HW, FS_TARGET
    from pipeline.features import extract_features
    from pipeline.models.ml.gmm_anomaly import GMMDetector
    from pipeline.models.anomaly.one_class_svm import OneClassSVMDetector

    print(f"[train_anomaly] Loading hopf_text cache from {ESC50_HOPF_TEXT_CACHE} ...")
    raw_x, labels, class_names, fs = load_dataset_from_text_cache(
        cache_dir=ESC50_HOPF_TEXT_CACHE,
        target_class=NORMAL_CLASS,
    )
    # load_dataset_from_text_cache with target_class returns labels as
    # 1 for target, 0 for everything else. Flip so 0 = normal, 1 = anomaly —
    # matches the convention used in _split_normal_val and roc_auc_score.
    labels = 1 - labels
    n_normal = int((labels == 0).sum())
    n_anomaly = int((labels == 1).sum())
    print(f"  normal ({NORMAL_CLASS}): {n_normal}, anomaly (rest): {n_anomaly}")

    ds_factor = 1 if fs == FS_TARGET else FS_HW // fs
    print(
        f"[train_anomaly] Processing clips (downsample_factor={ds_factor}, "
        f"subtract_common_mode={SUBTRACT_COMMON_MODE}) ..."
    )
    processed = process_dataset(
        raw_x,
        downsample_factor=ds_factor,
        subtract_common_mode=SUBTRACT_COMMON_MODE,
    )
    feature_maps, labels = extract_features(processed, labels)

    normal_train, eval_features, eval_labels = _split_normal_val(
        feature_maps, labels, val_split=VAL_SPLIT,
    )
    print(
        f"[train_anomaly] Fit set: {len(normal_train)} normal clips. "
        f"Eval set: {len(eval_features)} "
        f"({int((eval_labels == 0).sum())} held-out normal + "
        f"{int((eval_labels == 1).sum())} anomaly)."
    )

    print("\n[train_anomaly] Fitting GMMDetector ...")
    gmm = GMMDetector(n_components=4, n_pca_components=50)
    gmm.fit(normal_train)
    gmm_auroc = roc_auc_score(eval_labels, _gmm_scores(gmm, eval_features))
    print(f"  GMM AUROC: {gmm_auroc:.4f}")

    print("\n[train_anomaly] Fitting OneClassSVMDetector ...")
    ocsvm = OneClassSVMDetector(nu=0.05, n_pca_components=50)
    ocsvm.fit(normal_train)
    ocsvm_auroc = roc_auc_score(eval_labels, _ocsvm_scores(ocsvm, eval_features))
    print(f"  One-Class SVM AUROC: {ocsvm_auroc:.4f}")

    print("\n[train_anomaly] Summary")
    print(f"  normal_class        : {NORMAL_CLASS}")
    print(f"  subtract_common_mode: {SUBTRACT_COMMON_MODE}")
    print(f"  GMM AUROC           : {gmm_auroc:.4f}")
    print(f"  OCSVM AUROC         : {ocsvm_auroc:.4f}")
    print("  (AUROC 0.5 = chance, 1.0 = perfect separation)")


if __name__ == "__main__":
    main()
