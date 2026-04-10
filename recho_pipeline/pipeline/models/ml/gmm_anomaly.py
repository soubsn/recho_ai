"""
Gaussian Mixture Model — Unsupervised Anomaly Detector.

Trains ONLY on normal operation data (no fault labels needed).
Anomaly score = negative log likelihood of incoming sample under the GMM.
Threshold: 95th percentile of training set log likelihood.

This is the most commercially valuable model for industrial deployment.
The customer does not need to collect fault examples — only normal operation.
Train once during installation, alert on anything unexpected.

Input: PCA-reduced feature maps (50 components) — matches SVMClassifier.

Architecture:
    sklearn GaussianMixture(n_components=4, covariance_type='full')
    Each Gaussian component captures one mode of normal operating behaviour
    (e.g., different load conditions, warm-up vs steady-state).

The Hopf oscillator naturally has multiple normal operating modes depending
on the ambient input — 4 components is a conservative estimate for most
industrial settings. Increase n_components if the log-likelihood distribution
shows multiple well-separated peaks.

Joblib checkpoint: checkpoints/gmm_anomaly.pkl

Reference:
  Shougat et al., Scientific Reports 2021/2023 — the Hopf oscillator is sensitive
  to ANY change in the input signal; GMM exploits this sensitivity for detection.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

CHECKPOINT_DIR = ROOT / "checkpoints"
N_PCA_COMPONENTS: int = 50
N_GMM_COMPONENTS: int = 4
ANOMALY_PERCENTILE: float = 95.0


class GMMDetector:
    """
    Gaussian mixture model for unsupervised anomaly detection.

    Trained on normal-only data. At inference, a low log-likelihood
    indicates an anomalous pattern the model has never seen.

    Example:
        detector = GMMDetector(n_components=4)
        detector.fit(normal_feature_maps)
        score = detector.score(new_clip)
        if detector.is_anomaly(new_clip):
            alert()
        detector.plot_likelihood_distribution()
    """

    def __init__(
        self,
        n_components: int = N_GMM_COMPONENTS,
        n_pca_components: int = N_PCA_COMPONENTS,
        anomaly_percentile: float = ANOMALY_PERCENTILE,
    ) -> None:
        """
        Args:
            n_components: number of Gaussian mixture components.
            n_pca_components: PCA dimensionality before GMM.
            anomaly_percentile: training set percentile that defines the threshold.
        """
        self.n_components = n_components
        self.n_pca_components = n_pca_components
        self.anomaly_percentile = anomaly_percentile
        self._pca = PCA(n_components=n_pca_components, random_state=42)
        self._scaler = StandardScaler()
        self._gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=42,
            n_init=3,
        )
        self._threshold: float = 0.0
        self._train_log_likelihoods: Optional[NDArray[np.float64]] = None
        self._is_fitted = False

    def _prepare(self, features: NDArray, fit: bool = False) -> NDArray[np.float64]:
        """Flatten → scale → PCA → return reduced array."""
        X = features.reshape(features.shape[0], -1).astype(np.float32)
        if fit:
            X = self._scaler.fit_transform(X)
            X = self._pca.fit_transform(X)
        else:
            X = self._scaler.transform(X)
            X = self._pca.transform(X)
        return X.astype(np.float64)

    def fit(self, normal_clips: NDArray) -> "GMMDetector":
        """
        Fit GMM on normal operation feature maps only.

        The anomaly threshold is the anomaly_percentile-th percentile of
        log likelihoods on the training set. Samples below this are flagged.

        Args:
            normal_clips: (n_clips, 200, 100) or (n_clips, 200, 100, 2)
                          feature maps from NORMAL operation only

        Returns:
            self
        """
        X = self._prepare(normal_clips, fit=True)
        self._gmm.fit(X)

        # Compute threshold from training set log likelihoods
        log_liks = self._gmm.score_samples(X)  # shape (n,)
        self._train_log_likelihoods = log_liks
        # Low log likelihood = anomaly; threshold at (100 - percentile)th percentile
        self._threshold = float(np.percentile(
            log_liks, 100.0 - self.anomaly_percentile,
        ))
        print(f"[GMMDetector] Fitted on {len(normal_clips)} normal clips. "
              f"Threshold (log-lik < {self._threshold:.3f}) = "
              f"{self.anomaly_percentile:.0f}th percentile")
        self._is_fitted = True
        return self

    def score(self, clip: NDArray) -> float:
        """
        Compute anomaly score for a single clip.

        Returns the negative log likelihood — higher score = more anomalous.
        A score > |threshold| indicates an anomaly.

        Args:
            clip: (200, 100) or (200, 100, 2) feature map for one clip

        Returns:
            Anomaly score (negative log likelihood, float)
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before score()")
        clip_batch = np.expand_dims(clip, 0)
        X = self._prepare(clip_batch, fit=False)
        log_lik = float(self._gmm.score_samples(X)[0])
        return -log_lik  # return as positive "anomaly score"

    def score_batch(self, clips: NDArray) -> NDArray[np.float64]:
        """Compute anomaly scores for a batch of feature maps."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before score_batch()")
        X = self._prepare(clips, fit=False)
        log_liks = self._gmm.score_samples(X)
        return (-log_liks).astype(np.float64)

    def is_anomaly(self, clip: NDArray) -> bool:
        """
        Return True if the clip's log likelihood is below the threshold.

        Args:
            clip: (200, 100) or (200, 100, 2) feature map for one clip

        Returns:
            True if anomalous.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before is_anomaly()")
        clip_batch = np.expand_dims(clip, 0)
        X = self._prepare(clip_batch, fit=False)
        log_lik = float(self._gmm.score_samples(X)[0])
        return log_lik < self._threshold

    def plot_likelihood_distribution(
        self,
        anomaly_clips: Optional[NDArray] = None,
    ) -> None:
        """
        Plot the training set log-likelihood distribution showing the threshold.

        Optionally overlays the distribution of known anomalous clips.

        Args:
            anomaly_clips: optional (n, 200, 100) anomalous feature maps for
                           visualising class separation
        """
        import matplotlib.pyplot as plt

        if self._train_log_likelihoods is None:
            raise RuntimeError("Call fit() first")

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(self._train_log_likelihoods, bins=40, alpha=0.7,
                color="steelblue", label="Normal (training)")
        ax.axvline(self._threshold, color="tomato", linewidth=2,
                   label=f"Threshold ({self.anomaly_percentile:.0f}th pct)")

        if anomaly_clips is not None:
            X_anom = self._prepare(anomaly_clips, fit=False)
            anom_lls = self._gmm.score_samples(X_anom)
            ax.hist(anom_lls, bins=40, alpha=0.6, color="tomato",
                    label="Anomalous (test)")

        ax.set_xlabel("Log Likelihood")
        ax.set_ylabel("Count")
        ax.set_title("GMM Log-Likelihood Distribution\n"
                     "Lower = more anomalous")
        ax.legend()
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def save(self, path: Optional[str | Path] = None) -> Path:
        """Save to checkpoints/gmm_anomaly.pkl."""
        import joblib
        p = Path(path) if path else CHECKPOINT_DIR / "gmm_anomaly.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "gmm": self._gmm, "pca": self._pca, "scaler": self._scaler,
            "threshold": self._threshold,
            "train_log_likelihoods": self._train_log_likelihoods,
        }, p)
        print(f"[GMMDetector] Saved to {p}")
        return p


def main() -> None:
    """Train GMMDetector on normal-only synthetic data, test on all classes."""
    from data.sample_data import generate_dataset_xy, CLASS_NAMES
    from pipeline.ingest import process_dataset
    from pipeline.features_xy import extract_y_features, extract_all_representations

    print("[gmm] Generating synthetic data ...")
    raw_x, raw_y, labels = generate_dataset_xy(
        n_clips_per_class=20, n_classes=5, cache=False,
    )
    x_proc = process_dataset(raw_x)
    y_proc = extract_y_features(raw_y)
    reps = extract_all_representations(x_proc, y_proc)
    x_feats = reps["x_only"]

    # Train only on class 0 (sine = "normal operation")
    normal_mask = labels == 0
    normal_clips = x_feats[normal_mask]
    print(f"[gmm] Training on {normal_mask.sum()} normal clips (class 0 only) ...")

    detector = GMMDetector(n_components=4, n_pca_components=20)
    detector.fit(normal_clips)

    # Test on all classes
    print("\nAnomaly detection per class:")
    for cls in range(5):
        mask = labels == cls
        clip_batch = x_feats[mask]
        scores = detector.score_batch(clip_batch)
        n_anom = int(np.sum(scores > -detector._threshold))
        frac = n_anom / len(scores) * 100
        print(f"  Class {cls} ({CLASS_NAMES[cls]}): mean_score={scores.mean():.2f}, "
              f"detected={n_anom}/{len(scores)} ({frac:.0f}%)")

    detector.save()
    print("[gmm] Done.")


if __name__ == "__main__":
    main()
