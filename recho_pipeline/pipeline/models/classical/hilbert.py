"""
Hilbert Transform Classifier — instantaneous amplitude and frequency analysis.

Computes the analytic signal z(t) = x(t) + j*H{x(t)} where H{} is the Hilbert
transform, then extracts 5 features per clip:
    instantaneous_amplitude  — envelope |z(t)|; how the amplitude modulates
    instantaneous_frequency  — derivative of the instantaneous phase (rad/s)
    mean_frequency           — mean of instantaneous frequency over clip
    frequency_variance       — variance of instantaneous frequency
    amplitude_modulation_index — std(envelope) / mean(envelope)

Physical interpretation:
    The Hopf oscillator responds to the input signal by modulating both its
    amplitude and frequency. The Hilbert transform exposes this modulation
    directly, making it ideal for detecting changes in input character:
    - Pure sines → stable instantaneous frequency
    - Chirp → monotonically varying instantaneous frequency
    - Noise → high frequency_variance
    - Square wave → amplitude modulation spikes at zero crossings

These 5 features feed into a sklearn SVC (RBF kernel).

Maps to: compute analytic signal z(t) = x(t) + j*H{x(t)}
         envelope = |z(t)|, phase = angle(z(t)), freq = d(phase)/dt

Joblib checkpoint: checkpoints/hilbert.pkl

Reference:
  Shougat et al., Scientific Reports 2023 (paper 2) — x(t) as reservoir output.
  scipy.signal.hilbert implements the Hilbert transform via FFT.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert as scipy_hilbert
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

CHECKPOINT_DIR = ROOT / "checkpoints"


def extract_hilbert_features(
    x_clip: NDArray[np.float64],
    fs: float = 4000.0,
) -> NDArray[np.float64]:
    """
    Extract 5 Hilbert transform features from a single x(t) clip.

    Args:
        x_clip: 1-D downsampled x(t) at fs Hz
        fs: sample rate in Hz (default 4000)

    Returns:
        1-D float64 array of 5 features:
        [mean_freq, freq_variance, mean_envelope, am_index, envelope_std]
    """
    # Analytic signal via Hilbert transform (FFT-based)
    z = scipy_hilbert(x_clip)

    # Instantaneous amplitude (envelope): |z(t)|
    envelope = np.abs(z)

    # Instantaneous phase: unwrap to remove 2pi jumps
    phase = np.unwrap(np.angle(z))

    # Instantaneous frequency: d(phase)/dt  (rad/s → divide by 2pi for Hz)
    inst_freq = np.diff(phase) * fs / (2.0 * np.pi)
    # Clip extreme values (edges can be noisy)
    inst_freq = np.clip(inst_freq, -fs / 2, fs / 2)

    mean_frequency = float(np.mean(inst_freq))
    frequency_variance = float(np.var(inst_freq))
    mean_envelope = float(np.mean(envelope))
    envelope_std = float(np.std(envelope))
    am_index = envelope_std / (mean_envelope + 1e-12)

    return np.array([
        mean_frequency,
        frequency_variance,
        mean_envelope,
        am_index,
        envelope_std,
    ], dtype=np.float64)


def compute_feature_matrix(
    x_clips: NDArray[np.float64],
    fs: float = 4000.0,
) -> NDArray[np.float64]:
    """
    Compute Hilbert feature matrix for a batch of clips.

    Args:
        x_clips: (n_clips, n_samples) downsampled x(t)
        fs: sample rate in Hz

    Returns:
        (n_clips, 5) float64 feature matrix
    """
    n = x_clips.shape[0]
    feats = np.zeros((n, 5), dtype=np.float64)
    for i in range(n):
        feats[i] = extract_hilbert_features(x_clips[i], fs=fs)
    return feats


class HilbertClassifier:
    """
    SVC classifier on Hilbert transform features extracted from x(t).

    Uses RBF kernel SVC, which is well-suited to the non-linear separability
    of amplitude/frequency features from different input signal classes.

    Example:
        clf = HilbertClassifier()
        clf.fit(x_train_ds, labels_train)
        preds = clf.predict(x_test_ds)
        clf.plot_instantaneous(x_clip)  # visual inspection
    """

    def __init__(self, C: float = 1.0, gamma: str = "scale") -> None:
        """
        Args:
            C: SVC regularisation parameter.
            gamma: RBF kernel bandwidth ('scale', 'auto', or float).
        """
        self.C = C
        self.gamma = gamma
        self._clf = SVC(kernel="rbf", C=C, gamma=gamma, probability=True)
        self._scaler = StandardScaler()
        self._is_fitted = False

    def fit(
        self,
        x_clips: NDArray[np.float64],
        labels: NDArray[np.int64],
        fs: float = 4000.0,
    ) -> "HilbertClassifier":
        """
        Fit SVC on Hilbert transform features.

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

    def plot_instantaneous(
        self,
        x_clip: NDArray[np.float64],
        fs: float = 4000.0,
        title: str = "Hilbert Transform Analysis",
    ) -> None:
        """
        Plot instantaneous amplitude and frequency for visual inspection.

        Args:
            x_clip: 1-D downsampled x(t) clip
            fs: sample rate in Hz
            title: plot title
        """
        import matplotlib.pyplot as plt

        z = scipy_hilbert(x_clip)
        envelope = np.abs(z)
        phase = np.unwrap(np.angle(z))
        inst_freq = np.diff(phase) * fs / (2.0 * np.pi)
        t = np.arange(len(x_clip)) / fs

        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        axes[0].plot(t, x_clip, color="steelblue", linewidth=0.8)
        axes[0].plot(t, envelope, color="tomato", linewidth=1.5, label="envelope")
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title(title)
        axes[0].legend()

        axes[1].plot(t, phase, color="mediumseagreen", linewidth=0.8)
        axes[1].set_ylabel("Instantaneous phase (rad)")

        axes[2].plot(t[:-1], inst_freq, color="mediumpurple", linewidth=0.8)
        axes[2].set_ylabel("Instantaneous freq (Hz)")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylim(-fs / 2, fs / 2)

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def save(self, path: Optional[str | Path] = None) -> Path:
        """Save to checkpoints/hilbert.pkl."""
        import joblib
        p = Path(path) if path else CHECKPOINT_DIR / "hilbert.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self._clf, "scaler": self._scaler}, p)
        print(f"[HilbertClassifier] Saved to {p}")
        return p


def main() -> None:
    """Train and evaluate HilbertClassifier on synthetic Hopf data."""
    from data.sample_data import generate_dataset_xy, CLASS_NAMES

    print("[hilbert] Generating synthetic data ...")
    raw_x, raw_y, labels = generate_dataset_xy(
        n_clips_per_class=20, n_classes=5, cache=False,
    )
    x_ds = raw_x[:, ::25].astype(np.float64)  # 4 kHz

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(labels))
    n_val = int(0.2 * len(labels))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    print("[hilbert] Fitting SVC classifier ...")
    clf = HilbertClassifier(C=10.0, gamma="scale")
    clf.fit(x_ds[train_idx], labels[train_idx])

    train_acc = clf.score(x_ds[train_idx], labels[train_idx])
    val_acc = clf.score(x_ds[val_idx], labels[val_idx])
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Val accuracy:   {val_acc:.4f}")

    # Show feature table
    feats = compute_feature_matrix(x_ds[:5])
    print("\nSample Hilbert features (first 5 clips):")
    header = f"  {'mean_freq':>12} {'freq_var':>12} {'mean_env':>12} "
    header += f"{'am_index':>12} {'env_std':>12}"
    print(header)
    for i, row in enumerate(feats):
        vals = "  " + "  ".join(f"{v:12.4f}" for v in row)
        print(f"{vals}  (class {labels[i]}: {CLASS_NAMES[labels[i]]})")

    clf.save()
    print("[hilbert] Done.")


if __name__ == "__main__":
    main()
