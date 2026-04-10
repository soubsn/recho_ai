"""
Isolation Forest — Anomaly Detection via Isolation Trees.

Trains on normal + anomalous data (or normal only) and isolates anomalous
samples by requiring fewer random splits to isolate them.

Input: 28 handcrafted features (same feature vector as random_forest.py).
    [8 x features] + [8 y features] + [4 phase] + [4 radius] + [4 extra]

Anomaly score = average path length across all trees.
    Short path → anomaly (isolated quickly)
    Long path → normal (takes many splits to isolate)

Firmware export: generates firmware/isolation_forest.h with the first tree
as nested switch/case — pure integer arithmetic, runs in microseconds on M33.

Why Isolation Forest:
    Isolates anomalies in O(log n) time — extremely fast on MCU.
    Works well with contamination rates 1-5% (typical industrial setting).
    Does not assume a specific distribution (unlike GMM).
    Can be trained with or without fault labels.

Joblib checkpoint: checkpoints/isolation_forest.pkl

Reference:
  Liu, F.T., Ting, K.M. & Zhou, Z-H. (2008) Isolation Forest. ICDM.
  Shougat et al., Scientific Reports 2023 (paper 2) — handcrafted features
  from the Hopf oscillator x(t)/y(t) output.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

CHECKPOINT_DIR = ROOT / "checkpoints"
FIRMWARE_DIR = ROOT / "firmware"


class IsolationForestModel:
    """
    Isolation Forest anomaly detector on 28 handcrafted signal features.

    Uses the same feature extraction as RandomForestModel (random_forest.py)
    — if you have a RandomForestModel fitted, you can reuse its feature matrix.

    Example:
        model = IsolationForestModel(contamination=0.05)
        model.fit(x_all_ds, y_all_ds)
        score = model.score_anomaly(x_clip_ds, y_clip_ds)
        if model.is_anomaly(x_clip_ds, y_clip_ds):
            alert()
        model.export_firmware_header()
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        """
        Args:
            n_estimators: number of isolation trees.
            contamination: expected fraction of anomalies in training data.
            random_state: random seed.
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self._clf = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self._scaler = StandardScaler()
        self._is_fitted = False

    def _get_features(
        self,
        x_clips: NDArray[np.float64],
        y_clips: NDArray[np.float64],
        fs: float = 4000.0,
    ) -> NDArray[np.float64]:
        """Extract 28 handcrafted features."""
        from pipeline.models.ml.random_forest import compute_feature_matrix
        return compute_feature_matrix(x_clips, y_clips, fs)

    def fit(
        self,
        x_clips: NDArray[np.float64],
        y_clips: NDArray[np.float64],
        fs: float = 4000.0,
    ) -> "IsolationForestModel":
        """
        Fit Isolation Forest on 28-feature handcrafted feature vectors.

        Args:
            x_clips: (n_clips, n_samples) downsampled x(t) (normal + anomalous)
            y_clips: (n_clips, n_samples) downsampled y(t)
            fs: sample rate in Hz

        Returns:
            self
        """
        print("[IsolationForest] Extracting 28 features ...")
        X = self._get_features(x_clips, y_clips, fs)
        X = self._scaler.fit_transform(X)
        self._clf.fit(X)
        self._is_fitted = True
        print(f"[IsolationForest] Fitted on {len(x_clips)} clips, "
              f"contamination={self.contamination}")
        return self

    def score_anomaly(
        self,
        x_clip: NDArray[np.float64],
        y_clip: NDArray[np.float64],
        fs: float = 4000.0,
    ) -> float:
        """
        Compute anomaly score for a single clip.

        Returns negative of the raw Isolation Forest score (so higher = more anomalous).
        The raw score is in [-0.5, 0.5]; negative raw scores indicate anomalies.

        Args:
            x_clip: 1-D downsampled x(t) for one clip
            y_clip: 1-D downsampled y(t) for one clip
            fs: sample rate in Hz

        Returns:
            Anomaly score — positive values indicate likely anomaly.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before score_anomaly()")
        X = self._get_features(
            x_clip[np.newaxis, :], y_clip[np.newaxis, :], fs,
        )
        X = self._scaler.transform(X)
        raw_score = float(self._clf.score_samples(X)[0])
        return -raw_score  # flip sign: higher = more anomalous

    def is_anomaly(
        self,
        x_clip: NDArray[np.float64],
        y_clip: NDArray[np.float64],
        fs: float = 4000.0,
    ) -> bool:
        """
        Return True if the clip is classified as an anomaly.

        Args:
            x_clip: 1-D downsampled x(t)
            y_clip: 1-D downsampled y(t)
            fs: sample rate in Hz

        Returns:
            True if anomalous (-1 from IsolationForest.predict).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before is_anomaly()")
        X = self._get_features(
            x_clip[np.newaxis, :], y_clip[np.newaxis, :], fs,
        )
        X = self._scaler.transform(X)
        prediction = int(self._clf.predict(X)[0])
        return prediction == -1  # IsolationForest: -1 = anomaly, 1 = normal

    def batch_predict(
        self,
        x_clips: NDArray[np.float64],
        y_clips: NDArray[np.float64],
        fs: float = 4000.0,
    ) -> NDArray[np.int64]:
        """
        Predict +1 (normal) or -1 (anomaly) for each clip in a batch.

        Args:
            x_clips: (n_clips, n_samples)
            y_clips: (n_clips, n_samples)
            fs: sample rate in Hz

        Returns:
            (n_clips,) array of +1 or -1
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before batch_predict()")
        X = self._get_features(x_clips, y_clips, fs)
        X = self._scaler.transform(X)
        return self._clf.predict(X).astype(np.int64)

    def export_firmware_header(
        self,
        path: Optional[str | Path] = None,
    ) -> Path:
        """
        Export first isolation tree as nested switch/case C code.

        Generates firmware/isolation_forest.h — pure integer arithmetic,
        O(log n) per inference, runs in microseconds on M33.

        Args:
            path: output path (default firmware/isolation_forest.h)

        Returns:
            Path to generated header file.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before export_firmware_header()")

        from pipeline.models.ml.random_forest import FEATURE_NAMES

        fw_path = Path(path) if path else FIRMWARE_DIR / "isolation_forest.h"
        fw_path.parent.mkdir(parents=True, exist_ok=True)

        tree = self._clf.estimators_[0]
        t = tree.tree_

        lines = [
            "/* isolation_forest.h — generated by pipeline/models/ml/isolation_forest.py */",
            "/* First isolation tree: pure integer comparisons, O(log n) on M33 */",
            "/* Returns: path_length (compare to threshold for anomaly decision) */",
            "#pragma once",
            f"#define IF_N_FEATURES  {len(FEATURE_NAMES)}",
            "",
            "static inline int if_path_length(const float *feat) {",
            "  int depth = 0;",
        ]

        def _emit_node(node_id: int, indent: int) -> None:
            prefix = "  " * indent
            if t.feature[node_id] < 0:
                lines.append(f"{prefix}return depth;  /* leaf */")
            else:
                feat_idx = int(t.feature[node_id])
                thresh = float(t.threshold[node_id])
                feat_name = (FEATURE_NAMES[feat_idx]
                             if feat_idx < len(FEATURE_NAMES) else f"feat_{feat_idx}")
                lines.append(f"{prefix}depth++;  /* {feat_name} */")
                lines.append(
                    f"{prefix}if (feat[{feat_idx}] <= {thresh:.6f}f) {{"
                )
                _emit_node(int(t.children_left[node_id]), indent + 1)
                lines.append(f"{prefix}}} else {{")
                _emit_node(int(t.children_right[node_id]), indent + 1)
                lines.append(f"{prefix}}}")

        _emit_node(0, indent=1)
        lines.append("}")

        fw_path.write_text("\n".join(lines))
        print(f"[IsolationForestModel] Firmware header written to {fw_path}")
        return fw_path

    def save(self, path: Optional[str | Path] = None) -> Path:
        """Save to checkpoints/isolation_forest.pkl."""
        import joblib
        p = Path(path) if path else CHECKPOINT_DIR / "isolation_forest.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self._clf, "scaler": self._scaler}, p)
        print(f"[IsolationForestModel] Saved to {p}")
        return p


def main() -> None:
    """Train IsolationForestModel on synthetic Hopf data."""
    from data.sample_data import generate_dataset_xy, CLASS_NAMES

    print("[isolation_forest] Generating synthetic data ...")
    raw_x, raw_y, labels = generate_dataset_xy(
        n_clips_per_class=20, n_classes=5, cache=False,
    )
    x_ds = raw_x[:, ::25].astype(np.float64)
    y_ds = raw_y[:, ::25].astype(np.float64)

    # Train on all data (contamination=0.2 for the 4 "anomalous" classes)
    print("[isolation_forest] Fitting Isolation Forest ...")
    model = IsolationForestModel(n_estimators=50, contamination=0.2)
    model.fit(x_ds, y_ds)

    # Evaluate: treat class 0 as normal, rest as anomalous
    preds = model.batch_predict(x_ds, y_ds)
    # Class 0 should be +1 (normal), rest -1 (anomaly)
    expected = np.where(labels == 0, 1, -1).astype(np.int64)
    acc = float(np.mean(preds == expected))
    print(f"  Anomaly detection accuracy (0=normal vs rest): {acc:.4f}")

    # Per-class breakdown
    print("\nPer-class anomaly rate:")
    for cls in range(5):
        mask = labels == cls
        cls_preds = preds[mask]
        anom_rate = float(np.mean(cls_preds == -1)) * 100
        print(f"  Class {cls} ({CLASS_NAMES[cls]}): {anom_rate:.0f}% anomalous")

    model.save()
    model.export_firmware_header()
    print("[isolation_forest] Done.")


if __name__ == "__main__":
    main()
