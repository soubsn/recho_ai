"""
Autocorrelation-based Periodicity Detector and Classifier.

Computes the normalised autocorrelation of x(t) up to lag = 0.5 * clip_length
and extracts 3 features per clip:
    dominant_period     — lag of the first prominent autocorrelation peak
    periodicity_strength — peak autocorrelation value (0–1; 1 = perfectly periodic)
    decay_rate          — exponential decay rate of the autocorrelation envelope

These 3 features feed into a sklearn GradientBoostingClassifier.

Anomaly detection mode:
    Compare autocorrelation of an incoming clip to a reference autocorrelation
    (computed from normal-operation clips) using Pearson correlation.
    Alert if correlation < threshold (default 0.85).

Physical interpretation:
    Periodic machine faults (e.g., bearing defects, unbalance) produce sharp
    autocorrelation peaks at the fault frequency and its harmonics. The Hopf
    oscillator responds to periodic inputs by developing a periodic orbit, so
    the autocorrelation of x(t) reflects the periodicity of the input a(t).
    Noise input → rapidly decaying autocorrelation.
    Pure sine → slowly decaying, periodic autocorrelation.

Joblib checkpoint: checkpoints/autocorrelation.pkl

Reference:
  Shougat et al., Scientific Reports 2021 (paper 1) — x(t) as reservoir readout.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

CHECKPOINT_DIR = ROOT / "checkpoints"
ANOMALY_THRESHOLD: float = 0.85  # Pearson correlation below this → anomaly


def _normalised_autocorr(x: NDArray[np.float64], max_lag: int) -> NDArray[np.float64]:
    """
    Compute normalised autocorrelation up to max_lag.

    r(k) = sum_{t=0}^{N-k-1} x(t)*x(t+k) / sum_{t} x(t)^2

    Args:
        x: 1-D time series (zero-mean recommended)
        max_lag: maximum lag to compute

    Returns:
        1-D array of length max_lag, normalised so r(0) = 1.
    """
    x = x - np.mean(x)
    var = float(np.dot(x, x))
    if var < 1e-12:
        return np.zeros(max_lag, dtype=np.float64)
    acf = np.array([float(np.dot(x[:len(x) - k], x[k:])) / var
                    for k in range(max_lag)], dtype=np.float64)
    return acf


def extract_autocorr_features(
    x_clip: NDArray[np.float64],
    fs: float = 4000.0,
) -> NDArray[np.float64]:
    """
    Extract 3 autocorrelation features from a single x(t) clip.

    Args:
        x_clip: 1-D downsampled x(t) at fs Hz
        fs: sample rate in Hz

    Returns:
        1-D float64 array of 3 features:
        [dominant_period_s, periodicity_strength, decay_rate]
    """
    max_lag = len(x_clip) // 2
    acf = _normalised_autocorr(x_clip, max_lag)

    # Find peaks in the autocorrelation
    peaks, properties = find_peaks(acf, height=0.1, distance=5)

    if len(peaks) > 0:
        # Dominant period: lag of first prominent peak
        dominant_period_s = float(peaks[0]) / fs
        periodicity_strength = float(acf[peaks[0]])
    else:
        dominant_period_s = float(max_lag) / fs  # no clear period
        periodicity_strength = 0.0

    # Decay rate: fit exponential decay to |acf| envelope
    t = np.arange(max_lag, dtype=np.float64) / fs
    log_acf = np.log(np.abs(acf) + 1e-12)
    # Simple linear regression in log space → exponential decay
    if max_lag > 1:
        coeffs = np.polyfit(t[:max_lag], log_acf[:max_lag], 1)
        decay_rate = -float(coeffs[0])  # positive = decay
    else:
        decay_rate = 0.0

    return np.array([dominant_period_s, periodicity_strength, decay_rate],
                    dtype=np.float64)


def compute_feature_matrix(
    x_clips: NDArray[np.float64],
    fs: float = 4000.0,
) -> NDArray[np.float64]:
    """
    Compute autocorrelation feature matrix for a batch of clips.

    Args:
        x_clips: (n_clips, n_samples) downsampled x(t)
        fs: sample rate in Hz

    Returns:
        (n_clips, 3) float64 feature matrix
    """
    n = x_clips.shape[0]
    feats = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        feats[i] = extract_autocorr_features(x_clips[i], fs=fs)
    return feats


class AutocorrClassifier:
    """
    Gradient boosting classifier on autocorrelation periodicity features.

    Also supports anomaly detection by comparing the incoming clip's
    autocorrelation to a reference (normal-operation) autocorrelation
    using Pearson correlation. Alert if correlation < threshold.

    Example:
        clf = AutocorrClassifier()
        clf.fit(x_train_ds, labels_train)
        clf.set_reference(x_normal_ds)
        preds = clf.predict(x_test_ds)
        is_anom = clf.is_anomaly(x_new_clip)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        anomaly_threshold: float = ANOMALY_THRESHOLD,
    ) -> None:
        self.n_estimators = n_estimators
        self.anomaly_threshold = anomaly_threshold
        self._clf = GradientBoostingClassifier(
            n_estimators=n_estimators, random_state=42,
        )
        self._scaler = StandardScaler()
        self._reference_acf: Optional[NDArray[np.float64]] = None
        self._is_fitted = False

    def fit(
        self,
        x_clips: NDArray[np.float64],
        labels: NDArray[np.int64],
        fs: float = 4000.0,
    ) -> "AutocorrClassifier":
        """
        Fit gradient boosting classifier on autocorrelation features.

        Args:
            x_clips: (n_clips, n_samples) downsampled x(t)
            labels: (n_clips,) integer class labels
            fs: sample rate in Hz

        Returns:
            self
        """
        X = compute_feature_matrix(x_clips, fs=fs)
        X = self._scaler.fit_transform(X)
        self._clf.fit(X, labels)
        self._is_fitted = True
        return self

    def set_reference(
        self,
        x_normal_clips: NDArray[np.float64],
        fs: float = 4000.0,
    ) -> None:
        """
        Set the reference autocorrelation from normal-operation clips.

        The reference is the mean autocorrelation across all normal clips.
        Used by is_anomaly() for real-time anomaly detection.

        Args:
            x_normal_clips: (n_clips, n_samples) normal operation clips
            fs: sample rate in Hz
        """
        max_lag = x_normal_clips.shape[1] // 2
        acfs = np.stack([_normalised_autocorr(x_normal_clips[i], max_lag)
                         for i in range(x_normal_clips.shape[0])], axis=0)
        self._reference_acf = np.mean(acfs, axis=0)
        print(f"[AutocorrClassifier] Reference set from "
              f"{x_normal_clips.shape[0]} normal clips, max_lag={max_lag}")

    def is_anomaly(
        self,
        x_clip: NDArray[np.float64],
        fs: float = 4000.0,
    ) -> bool:
        """
        Anomaly detection via autocorrelation Pearson correlation.

        Alert if correlation between incoming clip's ACF and the reference
        ACF drops below self.anomaly_threshold (default 0.85).

        Args:
            x_clip: 1-D downsampled x(t) clip
            fs: sample rate in Hz

        Returns:
            True if the clip looks anomalous relative to normal operation.
        """
        if self._reference_acf is None:
            raise RuntimeError("Call set_reference() before is_anomaly()")
        max_lag = len(self._reference_acf)
        acf = _normalised_autocorr(x_clip, max_lag)
        corr = float(np.corrcoef(acf, self._reference_acf)[0, 1])
        return corr < self.anomaly_threshold

    def predict(
        self,
        x_clips: NDArray[np.float64],
        fs: float = 4000.0,
    ) -> NDArray[np.int64]:
        """Predict class labels."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")
        X = compute_feature_matrix(x_clips, fs=fs)
        X = self._scaler.transform(X)
        return self._clf.predict(X).astype(np.int64)

    def score(
        self,
        x_clips: NDArray[np.float64],
        labels: NDArray[np.int64],
        fs: float = 4000.0,
    ) -> float:
        """Return classification accuracy."""
        return float(accuracy_score(labels, self.predict(x_clips, fs=fs)))

    def save(self, path: Optional[str | Path] = None) -> Path:
        """Save to checkpoints/autocorrelation.pkl."""
        import joblib
        p = Path(path) if path else CHECKPOINT_DIR / "autocorrelation.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self._clf, "scaler": self._scaler,
                     "reference_acf": self._reference_acf,
                     "anomaly_threshold": self.anomaly_threshold}, p)
        print(f"[AutocorrClassifier] Saved to {p}")
        return p


def main() -> None:
    """Train and evaluate AutocorrClassifier on synthetic Hopf data."""
    from data.sample_data import generate_dataset_xy, CLASS_NAMES

    print("[autocorrelation] Generating synthetic data ...")
    raw_x, raw_y, labels = generate_dataset_xy(
        n_clips_per_class=20, n_classes=5, cache=False,
    )
    x_ds = raw_x[:, ::25].astype(np.float64)  # 4 kHz

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(labels))
    n_val = int(0.2 * len(labels))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    print("[autocorrelation] Fitting classifier ...")
    clf = AutocorrClassifier(n_estimators=100, anomaly_threshold=0.85)
    clf.fit(x_ds[train_idx], labels[train_idx])

    # Set reference from class 0 (sine = "normal")
    normal_mask_train = labels[train_idx] == 0
    clf.set_reference(x_ds[train_idx][normal_mask_train])

    train_acc = clf.score(x_ds[train_idx], labels[train_idx])
    val_acc = clf.score(x_ds[val_idx], labels[val_idx])
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Val accuracy:   {val_acc:.4f}")

    # Anomaly detection test
    print("\nAnomaly detection (relative to sine reference):")
    for cls in range(5):
        mask = labels == cls
        clip = x_ds[mask][0]
        is_anom = clf.is_anomaly(clip)
        print(f"  Class {cls} ({CLASS_NAMES[cls]}): anomaly={is_anom}")

    clf.save()
    print("[autocorrelation] Done.")


if __name__ == "__main__":
    main()
