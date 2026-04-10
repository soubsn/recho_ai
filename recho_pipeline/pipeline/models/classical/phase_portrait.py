"""
Phase Portrait Classifier — geometric features from x(t) vs y(t) orbit.

Extracts 4 features per clip from the 2-D phase portrait (x vs y trajectory):
    orbit_area        — area enclosed by orbit (shoelace formula)
    orbit_eccentricity — ratio of major to minor axis of fitted ellipse
    centre_drift       — mean distance of orbit centre from origin over time
    orbit_variance     — variance in orbit radius r(t) = sqrt(x^2 + y^2)

These 4 features feed into a sklearn RidgeClassifier.

Physical interpretation (from Hopf limit cycle theory):
    orbit_area        — encodes total energy of the perturbation; larger input
                        amplitude pushes the limit cycle to a bigger orbit
    orbit_eccentricity — encodes asymmetry; symmetric inputs (e.g. sine) give
                        a circular orbit; asymmetric inputs (e.g. square) give
                        an elliptical one
    centre_drift      — encodes DC offset in the input signal a(t)
    orbit_variance    — encodes how much the limit cycle was disrupted; steady-
                        state inputs give low variance; transients give high

Also provides rolling anomaly thresholding via orbit radius:
    threshold = baseline_mean + N * baseline_std   (N configurable, default 3)

All arithmetic — no neural network. Runs in microseconds on any MCU.
Joblib checkpoint: checkpoints/phase_portrait.pkl

Reference:
  Shougat et al., Scientific Reports 2021 (paper 1) — limit cycle radius is the
  natural summary statistic of Hopf oscillator perturbation magnitude.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

CHECKPOINT_DIR = ROOT / "checkpoints"


def _shoelace_area(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """
    Compute area enclosed by a 2-D polygon using the shoelace formula.

    A = 0.5 * |sum_i (x_i * y_{i+1} - x_{i+1} * y_i)|

    Args:
        x: 1-D x-coordinates of the trajectory
        y: 1-D y-coordinates of the trajectory

    Returns:
        Absolute enclosed area (float).
    """
    n = len(x)
    area = 0.0
    for i in range(n - 1):
        area += x[i] * y[i + 1] - x[i + 1] * y[i]
    area += x[n - 1] * y[0] - x[0] * y[n - 1]
    return abs(area) * 0.5


def _fit_ellipse_axes(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[float, float]:
    """
    Estimate major and minor semi-axes of the best-fit ellipse using PCA.

    The covariance matrix eigenvalues give axis lengths:
        semi_axis = 2 * sqrt(eigenvalue)

    Args:
        x: 1-D x trajectory
        y: 1-D y trajectory

    Returns:
        (major_axis, minor_axis) — both non-negative.
    """
    data = np.column_stack([x - x.mean(), y - y.mean()])
    cov = np.cov(data.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    eigvals = np.sort(eigvals)[::-1]  # descending
    major = 2.0 * float(np.sqrt(eigvals[0]) + 1e-12)
    minor = 2.0 * float(np.sqrt(eigvals[1]) + 1e-12)
    return major, minor


def extract_phase_features(
    x_clip: NDArray[np.float64],
    y_clip: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Extract 4 phase portrait features from a single clip.

    Args:
        x_clip: 1-D array, downsampled x(t) for one clip
        y_clip: 1-D array, downsampled y(t), same length

    Returns:
        1-D array of 4 features: [orbit_area, orbit_eccentricity,
                                   centre_drift, orbit_variance]
    """
    r = np.sqrt(x_clip ** 2 + y_clip ** 2)

    # orbit_area: total energy proxy — larger orbit = stronger perturbation
    orbit_area = _shoelace_area(x_clip, y_clip)

    # orbit_eccentricity: asymmetry of the limit cycle perturbation
    major, minor = _fit_ellipse_axes(x_clip, y_clip)
    orbit_eccentricity = major / minor  # 1.0 = circular, >1 = elliptical

    # centre_drift: DC offset indicator — mean distance of centroid from origin
    centroid_x = float(np.mean(x_clip))
    centroid_y = float(np.mean(y_clip))
    centre_drift = float(np.sqrt(centroid_x ** 2 + centroid_y ** 2))

    # orbit_variance: disruption measure — how much r(t) varied from mean
    orbit_variance = float(np.var(r))

    return np.array([orbit_area, orbit_eccentricity, centre_drift, orbit_variance],
                    dtype=np.float64)


def compute_feature_matrix(
    x_clips: NDArray[np.float64],
    y_clips: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute phase portrait features for a batch of clips.

    Args:
        x_clips: (n_clips, n_samples) downsampled x(t)
        y_clips: (n_clips, n_samples) downsampled y(t)

    Returns:
        (n_clips, 4) float64 feature matrix
    """
    n = x_clips.shape[0]
    feats = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        feats[i] = extract_phase_features(x_clips[i], y_clips[i])
    return feats


class PhasePortraitClassifier:
    """
    Ridge regression classifier on phase portrait geometric features.

    Fits a RidgeClassifier on 4 geometric features extracted from the
    x(t) vs y(t) orbit of the Hopf oscillator. Fast to train and interpret.

    Also maintains a rolling anomaly threshold on orbit radius for real-time
    per-sample alerting (threshold = baseline_mean + N * baseline_std).

    Example:
        clf = PhasePortraitClassifier()
        clf.fit(x_train, y_train, labels_train)
        preds = clf.predict(x_test, y_test)
        is_anom = clf.is_anomaly_radius(r_sample)
    """

    def __init__(self, alpha: float = 1.0, anomaly_n: float = 3.0) -> None:
        """
        Args:
            alpha: ridge regularisation parameter.
            anomaly_n: number of std devs for anomaly threshold (default 3).
        """
        self.alpha = alpha
        self.anomaly_n = anomaly_n
        self._clf = RidgeClassifier(alpha=alpha)
        self._scaler = StandardScaler()
        self._baseline_mean: float = 0.0
        self._baseline_std: float = 1.0
        self._anomaly_threshold: float = 3.0
        self._is_fitted = False

    def fit(
        self,
        x_clips: NDArray[np.float64],
        y_clips: NDArray[np.float64],
        labels: NDArray[np.int64],
    ) -> "PhasePortraitClassifier":
        """
        Fit classifier on phase portrait features.

        Args:
            x_clips: (n_clips, n_samples) downsampled x(t)
            y_clips: (n_clips, n_samples) downsampled y(t)
            labels: (n_clips,) integer class labels

        Returns:
            self
        """
        X = compute_feature_matrix(x_clips, y_clips)
        X = self._scaler.fit_transform(X)
        self._clf.fit(X, labels)

        # Baseline anomaly threshold on orbit radius (from all training clips)
        r_all = np.sqrt(x_clips ** 2 + y_clips ** 2).flatten()
        self._baseline_mean = float(np.mean(r_all))
        self._baseline_std = float(np.std(r_all)) + 1e-12
        self._anomaly_threshold = self._baseline_mean + self.anomaly_n * self._baseline_std

        self._is_fitted = True
        return self

    def predict(
        self,
        x_clips: NDArray[np.float64],
        y_clips: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        """Predict class labels for a batch of clips."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")
        X = compute_feature_matrix(x_clips, y_clips)
        X = self._scaler.transform(X)
        return self._clf.predict(X).astype(np.int64)

    def score(
        self,
        x_clips: NDArray[np.float64],
        y_clips: NDArray[np.float64],
        labels: NDArray[np.int64],
    ) -> float:
        """Return classification accuracy."""
        preds = self.predict(x_clips, y_clips)
        return float(accuracy_score(labels, preds))

    def is_anomaly_radius(self, r_sample: float) -> bool:
        """
        Real-time anomaly check on orbit radius.

        Threshold = baseline_mean + N * baseline_std.
        Maps to a simple comparison in firmware: r > threshold.

        Args:
            r_sample: current orbit radius r = sqrt(x^2 + y^2)

        Returns:
            True if r_sample exceeds the anomaly threshold.
        """
        return r_sample > self._anomaly_threshold

    def save(self, path: Optional[str | Path] = None) -> Path:
        """Save classifier to checkpoints/phase_portrait.pkl."""
        import joblib
        p = Path(path) if path else CHECKPOINT_DIR / "phase_portrait.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self._clf, "scaler": self._scaler,
                     "baseline_mean": self._baseline_mean,
                     "baseline_std": self._baseline_std,
                     "anomaly_threshold": self._anomaly_threshold}, p)
        print(f"[PhasePortraitClassifier] Saved to {p}")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "PhasePortraitClassifier":
        """Load a saved classifier."""
        import joblib
        obj = cls()
        data = joblib.load(path)
        obj._clf = data["clf"]
        obj._scaler = data["scaler"]
        obj._baseline_mean = data["baseline_mean"]
        obj._baseline_std = data["baseline_std"]
        obj._anomaly_threshold = data["anomaly_threshold"]
        obj._is_fitted = True
        return obj


def main() -> None:
    """Train and evaluate PhasePortraitClassifier on synthetic data."""
    from data.sample_data import generate_dataset_xy, CLASS_NAMES

    print("[phase_portrait] Generating synthetic data ...")
    raw_x, raw_y, labels = generate_dataset_xy(
        n_clips_per_class=20, n_classes=5, cache=False,
    )

    # Downsample to 4 kHz: factor 25
    x_ds = raw_x[:, ::25].astype(np.float64)
    y_ds = raw_y[:, ::25].astype(np.float64)

    # Train/val split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(labels))
    n_val = int(0.2 * len(labels))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    print("[phase_portrait] Fitting classifier ...")
    clf = PhasePortraitClassifier(alpha=1.0, anomaly_n=3.0)
    clf.fit(x_ds[train_idx], y_ds[train_idx], labels[train_idx])

    train_acc = clf.score(x_ds[train_idx], y_ds[train_idx], labels[train_idx])
    val_acc = clf.score(x_ds[val_idx], y_ds[val_idx], labels[val_idx])
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Val accuracy:   {val_acc:.4f}")

    clf.save()

    # Show a few phase portrait features
    feats = compute_feature_matrix(x_ds[:5], y_ds[:5])
    print("\nSample features (first 5 clips):")
    print(f"  {'orbit_area':>14s} {'eccentricity':>14s} {'centre_drift':>14s} {'orbit_var':>12s}")
    for i, row in enumerate(feats):
        print(f"  {row[0]:14.4f} {row[1]:14.4f} {row[2]:14.4f} {row[3]:12.6f}"
              f"  (class {labels[i]}: {CLASS_NAMES[labels[i]]})")


if __name__ == "__main__":
    main()
