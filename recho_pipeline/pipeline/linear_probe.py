"""
Linear-probe diagnostic for the Hopf feature maps.

Fits sklearn LogisticRegression on flattened (200*100) uint8 feature maps using
the same stratified 80/20 split (seed=0) as train_CNN.prepare_data, so the
val_accuracy is directly comparable to the CNN sweep.

Interpretation:
  - LogReg val_acc ~= CNN val_acc (~15-18%)
    -> the representation is the ceiling; shrinking/regularizing the CNN
       won't help. Fix the features or narrow to the paper's 10-class task.
  - LogReg val_acc << CNN val_acc (LogReg ~5-8%)
    -> the CNN is extracting nonlinear structure; the overfit is the real
       problem and architecture fixes (Dropout, GAP, L2) are worth it.
  - LogReg val_acc >> CNN val_acc (LogReg ~25%+)
    -> the CNN's training dynamics are broken; a linear model on the same
       pixels beats it. Revisit optimizer/init/schedule before anything else.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


VAL_SPLIT: float = 0.2
SEED: int = 0
# L2 inverse-regularization strength. 1.0 is sklearn's default. At 20,000
# features and 1,600 samples we're overdetermined in the other direction,
# so strong regularization (small C) matters. Try a small grid.
C_VALUES: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)
# Cap iterations so the probe stays under a couple minutes per C.
MAX_ITER: int = 2000


def _stratified_split(
    labels: NDArray[np.int64], val_split: float, seed: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Replicate prepare_data's split exactly: seed=0, same loop, same shuffles."""
    rng = np.random.default_rng(seed)
    val_idx_parts: list[NDArray[np.int64]] = []
    train_idx_parts: list[NDArray[np.int64]] = []
    for c in np.unique(labels):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)
        n_val_c = int(round(len(class_idx) * val_split))
        val_idx_parts.append(class_idx[:n_val_c])
        train_idx_parts.append(class_idx[n_val_c:])
    val_idx = np.concatenate(val_idx_parts)
    train_idx = np.concatenate(train_idx_parts)
    rng.shuffle(val_idx)
    rng.shuffle(train_idx)
    return train_idx, val_idx


def _top_k_accuracy(
    probs: NDArray[np.float64], y_true: NDArray[np.int64], k: int
) -> float:
    top_k = np.argsort(-probs, axis=1)[:, :k]
    hits = np.any(top_k == y_true[:, None], axis=1)
    return float(hits.mean())


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.train_CNN import _load_features_50class

    feature_maps, labels, class_names = _load_features_50class()
    n_classes = len(class_names)
    n, h, w = feature_maps.shape
    print(f"[probe] feature_maps: {feature_maps.shape} dtype={feature_maps.dtype}")
    print(f"[probe] n_classes={n_classes}, pixels per clip={h * w}")

    x = feature_maps.reshape(n, h * w).astype(np.float32)
    train_idx, val_idx = _stratified_split(labels, VAL_SPLIT, SEED)
    x_train, y_train = x[train_idx], labels[train_idx]
    x_val, y_val = x[val_idx], labels[val_idx]
    print(f"[probe] train: {x_train.shape}, val: {x_val.shape}")

    # LogReg on raw pixels is very sensitive to scale; standardize per feature.
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)

    results: list[tuple[float, float, float, float, float]] = []
    for c in C_VALUES:
        print(f"\n========== [probe] LogReg C={c} ==========")
        t0 = time.time()
        clf = LogisticRegression(
            C=c,
            penalty="l2",
            solver="lbfgs",
            multi_class="multinomial",
            max_iter=MAX_ITER,
            n_jobs=-1,
            verbose=0,
        )
        clf.fit(x_train_s, y_train)
        fit_s = time.time() - t0

        train_probs = clf.predict_proba(x_train_s)
        val_probs = clf.predict_proba(x_val_s)
        train_top1 = _top_k_accuracy(train_probs, y_train, 1)
        val_top1 = _top_k_accuracy(val_probs, y_val, 1)
        val_top5 = _top_k_accuracy(val_probs, y_val, 5)
        print(
            f"  train_top1={train_top1:.3f}  val_top1={val_top1:.3f}  "
            f"val_top5={val_top5:.3f}  (fit {fit_s:.1f}s, "
            f"n_iter={getattr(clf, 'n_iter_', np.array([-1]))[0]})"
        )
        results.append((c, train_top1, val_top1, val_top5, fit_s))

    print("\n--- Linear-probe summary (sorted by val_top1 desc) ---")
    results.sort(key=lambda r: r[2], reverse=True)
    print(f"{'C':>8}  {'train_top1':>11}  {'val_top1':>9}  {'val_top5':>9}  {'fit_s':>6}")
    print("-" * 54)
    for c, tr, v1, v5, fs in results:
        print(f"{c:>8.3g}  {tr:>11.3f}  {v1:>9.3f}  {v5:>9.3f}  {fs:>6.1f}")

    best_c, _, best_val, _, _ = results[0]
    print(
        f"\n[probe] Best C={best_c} -> val_top1={best_val:.3f} "
        f"(chance = {1/n_classes:.3f}). CNN best was ~0.162 at peak_lr=3e-4."
    )
    print("Compare val_top1 above to CNN val_acc:")
    print("  ~= CNN (within 0.03)  -> representation is the ceiling; fix features.")
    print("  << CNN                -> CNN extracts real nonlinear structure; fix overfit.")
    print("  >> CNN                -> CNN training dynamics broken; fix optimizer/init.")


if __name__ == "__main__":
    main()
