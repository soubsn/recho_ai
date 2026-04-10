"""
Recurrence Quantification Analysis (RQA) classifier.

Builds the recurrence matrix R[i,j] = 1 if ||x(i) - x(j)|| < epsilon and
extracts 5 metrics per clip that characterise the dynamical structure of x(t):
    recurrence_rate  — fraction of recurrent points (density of R)
    determinism      — fraction of recurrent points forming diagonal lines
    laminarity       — fraction of recurrent points forming vertical lines
    entropy          — Shannon entropy of diagonal line lengths
    trapping_time    — mean length of vertical lines (how long system stays near state)

Physical interpretation (from nonlinear dynamics theory):
    high determinism  = periodic/structured input (e.g., pure sine)
    high entropy      = complex broadband input (e.g., noise, chirp)
    laminarity        = how long the Hopf oscillator stays near the same orbit state
    trapping_time     = persistence in a dynamical state

These 5 features feed into a sklearn RandomForestClassifier.

Computational note: O(N^2) recurrence matrix. For MCU deployment, only the
first MAX_SAMPLES samples of each clip are used (configurable, default 500).
At 4 kHz this covers 125 ms of signal — sufficient for RQA features.

Epsilon: configurable threshold, default = 0.1 * std(x)

Joblib checkpoint: checkpoints/recurrence.pkl

Reference:
  Zbilut, J.P. & Webber, C.L. (1992) Embeddings and delays as derived
  from quantification of recurrence plots. Physics Letters A 171(3-4):199-203.
  Shougat et al., Scientific Reports 2021 (paper 1) — x(t) stream analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

CHECKPOINT_DIR = ROOT / "checkpoints"
MAX_SAMPLES: int = 500  # truncate for MCU deployment approximation


def _build_recurrence_matrix(
    x: NDArray[np.float64],
    epsilon: Optional[float] = None,
) -> NDArray[np.bool_]:
    """
    Build binary recurrence matrix R[i,j] = (|x[i] - x[j]| < epsilon).

    Args:
        x: 1-D time series
        epsilon: threshold; if None, uses 0.1 * std(x)

    Returns:
        (N, N) boolean recurrence matrix
    """
    if epsilon is None:
        epsilon = 0.1 * float(np.std(x)) + 1e-12
    diff = np.abs(x[:, None] - x[None, :])
    return diff < epsilon


def _rqa_metrics(R: NDArray[np.bool_]) -> dict[str, float]:
    """
    Compute the 5 RQA metrics from a recurrence matrix.

    Args:
        R: (N, N) boolean recurrence matrix

    Returns:
        dict with keys: recurrence_rate, determinism, laminarity,
                        entropy, trapping_time
    """
    N = R.shape[0]
    n_recurrent = int(np.sum(R)) - N  # exclude main diagonal
    total_pairs = N * (N - 1)
    recurrence_rate = n_recurrent / (total_pairs + 1e-12)

    # Diagonal line lengths (determinism and entropy)
    diag_lengths: list[int] = []
    for k in range(-(N - 2), N - 1):
        diag = np.diag(R, k=k)
        run = 0
        for val in diag:
            if val:
                run += 1
            elif run >= 2:
                diag_lengths.append(run)
                run = 0
        if run >= 2:
            diag_lengths.append(run)

    total_diag_pts = sum(diag_lengths)
    total_non_diag = n_recurrent
    determinism = total_diag_pts / (total_non_diag + 1e-12)

    # Shannon entropy of diagonal line length distribution
    if diag_lengths:
        unique, counts = np.unique(diag_lengths, return_counts=True)
        probs = counts / counts.sum()
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    else:
        entropy = 0.0

    # Vertical line lengths (laminarity and trapping time)
    vert_lengths: list[int] = []
    for j in range(N):
        run = 0
        for i in range(N):
            if R[i, j]:
                run += 1
            elif run >= 2:
                vert_lengths.append(run)
                run = 0
        if run >= 2:
            vert_lengths.append(run)

    total_vert_pts = sum(vert_lengths)
    laminarity = total_vert_pts / (total_non_diag + 1e-12)
    trapping_time = float(np.mean(vert_lengths)) if vert_lengths else 0.0

    return {
        "recurrence_rate": recurrence_rate,
        "determinism": determinism,
        "laminarity": laminarity,
        "entropy": entropy,
        "trapping_time": trapping_time,
    }


def extract_rqa_features(
    x_clip: NDArray[np.float64],
    epsilon: Optional[float] = None,
    max_samples: int = MAX_SAMPLES,
) -> NDArray[np.float64]:
    """
    Extract 5 RQA features from a single x(t) clip.

    Truncates to max_samples for computational tractability.

    Args:
        x_clip: 1-D downsampled x(t) time series
        epsilon: recurrence threshold; None = 0.1 * std(x)
        max_samples: maximum samples to use (default 500)

    Returns:
        1-D float64 array of 5 RQA features
    """
    x = x_clip[:max_samples]
    R = _build_recurrence_matrix(x, epsilon=epsilon)
    metrics = _rqa_metrics(R)
    return np.array([
        metrics["recurrence_rate"],
        metrics["determinism"],
        metrics["laminarity"],
        metrics["entropy"],
        metrics["trapping_time"],
    ], dtype=np.float64)


def compute_feature_matrix(
    x_clips: NDArray[np.float64],
    epsilon: Optional[float] = None,
    max_samples: int = MAX_SAMPLES,
) -> NDArray[np.float64]:
    """
    Compute RQA feature matrix for a batch of clips.

    Args:
        x_clips: (n_clips, n_samples) downsampled x(t)
        epsilon: recurrence threshold (None = per-clip adaptive)
        max_samples: truncation length for speed

    Returns:
        (n_clips, 5) float64 feature matrix
    """
    n = x_clips.shape[0]
    feats = np.zeros((n, 5), dtype=np.float64)
    for i in range(n):
        feats[i] = extract_rqa_features(x_clips[i], epsilon=epsilon,
                                         max_samples=max_samples)
        if (i + 1) % 20 == 0:
            print(f"  [rqa] {i + 1}/{n} clips processed")
    return feats


class RecurrenceClassifier:
    """
    Random forest classifier on RQA features extracted from x(t).

    Physical insight: Hopf oscillator in a periodic attractor (sine input)
    shows high determinism. Broadband noise input shows high entropy and low
    determinism. The RQA features discriminate these regimes naturally.

    Example:
        clf = RecurrenceClassifier()
        clf.fit(x_train_ds, labels_train)
        preds = clf.predict(x_test_ds)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        epsilon: Optional[float] = None,
        max_samples: int = MAX_SAMPLES,
    ) -> None:
        self.n_estimators = n_estimators
        self.epsilon = epsilon
        self.max_samples = max_samples
        self._clf = RandomForestClassifier(n_estimators=n_estimators,
                                           random_state=42, n_jobs=-1)
        self._scaler = StandardScaler()
        self._is_fitted = False

    def fit(
        self,
        x_clips: NDArray[np.float64],
        labels: NDArray[np.int64],
    ) -> "RecurrenceClassifier":
        """
        Fit classifier on RQA features.

        Args:
            x_clips: (n_clips, n_samples) downsampled x(t)
            labels: (n_clips,) integer class labels

        Returns:
            self
        """
        print(f"[RecurrenceClassifier] Extracting RQA features "
              f"(max {self.max_samples} samples per clip) ...")
        X = compute_feature_matrix(x_clips, epsilon=self.epsilon,
                                    max_samples=self.max_samples)
        X = self._scaler.fit_transform(X)
        self._clf.fit(X, labels)
        self._is_fitted = True
        return self

    def predict(self, x_clips: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict class labels."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")
        X = compute_feature_matrix(x_clips, epsilon=self.epsilon,
                                    max_samples=self.max_samples)
        X = self._scaler.transform(X)
        return self._clf.predict(X).astype(np.int64)

    def score(
        self,
        x_clips: NDArray[np.float64],
        labels: NDArray[np.int64],
    ) -> float:
        """Return classification accuracy."""
        return float(accuracy_score(labels, self.predict(x_clips)))

    def save(self, path: Optional[str | Path] = None) -> Path:
        """Save to checkpoints/recurrence.pkl."""
        import joblib
        p = Path(path) if path else CHECKPOINT_DIR / "recurrence.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self._clf, "scaler": self._scaler,
                     "epsilon": self.epsilon, "max_samples": self.max_samples}, p)
        print(f"[RecurrenceClassifier] Saved to {p}")
        return p


def main() -> None:
    """Train and evaluate RecurrenceClassifier on synthetic Hopf data."""
    from data.sample_data import generate_dataset_xy, CLASS_NAMES

    # Use small dataset because RQA is O(N^2)
    print("[recurrence] Generating synthetic data ...")
    raw_x, raw_y, labels = generate_dataset_xy(
        n_clips_per_class=8, n_classes=5, cache=False,
    )

    x_ds = raw_x[:, ::25].astype(np.float64)  # 4 kHz

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(labels))
    n_val = max(2, int(0.2 * len(labels)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    print("[recurrence] Fitting classifier ...")
    clf = RecurrenceClassifier(n_estimators=50, max_samples=200)
    clf.fit(x_ds[train_idx], labels[train_idx])

    train_acc = clf.score(x_ds[train_idx], labels[train_idx])
    val_acc = clf.score(x_ds[val_idx], labels[val_idx])
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Val accuracy:   {val_acc:.4f}")

    clf.save()
    print("[recurrence] Done.")


if __name__ == "__main__":
    main()
