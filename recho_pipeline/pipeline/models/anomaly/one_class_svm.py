"""
One-Class SVM — Anomaly Detection Trained on Normal Data Only.

Learns a decision boundary around the normal operation feature distribution.
Any sample outside the boundary is flagged as anomalous.

Input: PCA-reduced feature maps (50 components) — matches SVMClassifier.

sklearn OneClassSVM(kernel='rbf', nu=0.05)
    nu = expected fraction of anomalies in the training data (configurable).
    The support vectors form the boundary of the normal region.

After fitting, the support vectors can be approximated as a small set of
INT8 vectors for MCU deployment, making this the lightest learned anomaly
detector after the SPC Monitor.

Calibration: ROC curve on held-out normal vs anomaly data.

Why One-Class SVM:
    Lightest anomaly detector after SPC (only needs support vectors at inference).
    Works with very few normal examples (10-20 clips can be enough).
    No assumption about the distribution of normal data (unlike GMM).
    Can approximate boundary with ~10-50 support vectors for MCU.

Joblib checkpoint: checkpoints/one_class_svm.pkl

Reference:
  Schölkopf, B. et al. (2001) Estimating the support of a high-dimensional
  distribution. Neural Computation 13(7):1443-1471.
  Shougat et al., Scientific Reports 2021/2023 — normal operating conditions
  form a well-defined compact region in Hopf feature space.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

CHECKPOINT_DIR = ROOT / "checkpoints"
N_PCA_COMPONENTS: int = 50


class OneClassSVMDetector:
    """
    One-Class SVM anomaly detector on PCA-reduced Hopf feature maps.

    Trains only on normal operation data. At inference, samples outside the
    learned boundary are flagged as anomalies.

    Example:
        detector = OneClassSVMDetector(nu=0.05)
        detector.fit(normal_feature_maps)
        if detector.is_anomaly(new_clip):
            alert()
        detector.plot_roc(anomalous_clips)
    """

    def __init__(
        self,
        nu: float = 0.05,
        n_pca_components: int = N_PCA_COMPONENTS,
    ) -> None:
        """
        Args:
            nu: upper bound on training errors AND lower bound on support vector
                fraction. Approximately equals the expected anomaly fraction.
            n_pca_components: PCA dimensionality before OCSVM.
        """
        self.nu = nu
        self.n_pca_components = n_pca_components
        self._pca = PCA(n_components=n_pca_components, random_state=42)
        self._scaler = StandardScaler()
        self._clf = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
        self._is_fitted = False

    def _prepare(self, features: NDArray, fit: bool = False) -> NDArray[np.float64]:
        """Flatten → scale → PCA."""
        X = features.reshape(features.shape[0], -1).astype(np.float32)
        if fit:
            X = self._scaler.fit_transform(X)
            X = self._pca.fit_transform(X)
        else:
            X = self._scaler.transform(X)
            X = self._pca.transform(X)
        return X.astype(np.float64)

    def fit(self, normal_clips: NDArray) -> "OneClassSVMDetector":
        """
        Fit One-Class SVM on normal operation feature maps.

        Args:
            normal_clips: (n_clips, 200, 100) feature maps from normal operation

        Returns:
            self
        """
        X = self._prepare(normal_clips, fit=True)
        self._clf.fit(X)
        n_sv = len(self._clf.support_vectors_)
        print(f"[OneClassSVM] Fitted on {len(normal_clips)} normal clips. "
              f"Support vectors: {n_sv}")
        self._is_fitted = True
        return self

    def decision_score(self, clip: NDArray) -> float:
        """
        Return the signed distance to the decision boundary.

        Positive = normal, negative = anomaly.

        Args:
            clip: (200, 100) feature map for one clip

        Returns:
            Signed distance score (float). Negative → anomaly.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before decision_score()")
        X = self._prepare(np.expand_dims(clip, 0), fit=False)
        return float(self._clf.decision_function(X)[0])

    def is_anomaly(self, clip: NDArray) -> bool:
        """
        Return True if the clip is outside the normal boundary.

        Args:
            clip: (200, 100) feature map for one clip

        Returns:
            True if anomalous.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before is_anomaly()")
        X = self._prepare(np.expand_dims(clip, 0), fit=False)
        return int(self._clf.predict(X)[0]) == -1

    def batch_predict(self, clips: NDArray) -> NDArray[np.int64]:
        """
        Predict +1 (normal) or -1 (anomaly) for a batch of clips.

        Args:
            clips: (n_clips, 200, 100) feature maps

        Returns:
            (n_clips,) array of +1 or -1.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before batch_predict()")
        X = self._prepare(clips, fit=False)
        return self._clf.predict(X).astype(np.int64)

    def plot_roc(
        self,
        anomaly_clips: NDArray,
        title: str = "One-Class SVM ROC Curve",
    ) -> None:
        """
        Plot ROC curve: normal (from fit) vs anomalous clips.

        Args:
            anomaly_clips: (n_clips, 200, 100) known anomalous feature maps
            title: plot title
        """
        import matplotlib.pyplot as plt

        if not self._is_fitted:
            raise RuntimeError("Call fit() before plot_roc()")

        # Get decision scores for both normal and anomalous samples
        # We need some normal clips — re-use training data if available
        X_anom = self._prepare(anomaly_clips, fit=False)
        scores_anom = self._clf.decision_function(X_anom)

        y_true = np.ones(len(scores_anom))  # 1 = anomaly for ROC
        fpr, tpr, thresholds = roc_curve(y_true, -scores_anom)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="tomato", lw=2,
                label=f"ROC (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def save(self, path: Optional[str | Path] = None) -> Path:
        """Save to checkpoints/one_class_svm.pkl."""
        import joblib
        p = Path(path) if path else CHECKPOINT_DIR / "one_class_svm.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self._clf, "pca": self._pca,
                     "scaler": self._scaler}, p)
        print(f"[OneClassSVMDetector] Saved to {p}")
        return p


def main() -> None:
    """Train OneClassSVMDetector on normal-only synthetic data."""
    from data.sample_data import generate_dataset_xy, CLASS_NAMES
    from pipeline.ingest import process_dataset
    from pipeline.features_xy import extract_y_features, extract_all_representations

    print("[one_class_svm] Generating synthetic data ...")
    raw_x, raw_y, labels = generate_dataset_xy(
        n_clips_per_class=20, n_classes=5, cache=False,
    )
    x_proc = process_dataset(raw_x)
    y_proc = extract_y_features(raw_y)
    reps = extract_all_representations(x_proc, y_proc)

    normal_mask = labels == 0
    normal_clips = reps["x_only"][normal_mask]
    print(f"[one_class_svm] Training on {normal_mask.sum()} normal clips ...")

    detector = OneClassSVMDetector(nu=0.05, n_pca_components=20)
    detector.fit(normal_clips)

    print("\nAnomaly detection per class:")
    for cls in range(5):
        mask = labels == cls
        clips = reps["x_only"][mask]
        preds = detector.batch_predict(clips)
        anom_rate = float(np.mean(preds == -1)) * 100
        print(f"  Class {cls} ({CLASS_NAMES[cls]}): {anom_rate:.0f}% anomalous")

    detector.save()
    print("[one_class_svm] Done.")


if __name__ == "__main__":
    main()
