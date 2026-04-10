"""
Echo State Network (ESN) Readout — Reservoir of Reservoirs.

The Hopf oscillator is already a physical reservoir. This module adds a
software reservoir on top — effectively a "reservoir of reservoirs" that
processes the Hopf feature maps through a fixed recurrent neural network
before applying a linear readout.

This is closest to the original Hopf PRC paper 1 methodology (Shougat 2021),
which uses a single linear readout layer. The ESN extends this by adding a
fixed nonlinear reservoir between the Hopf features and the output.

Input: x_features [n_clips, 200, 100] flattened to [n_clips, 20000]

Architecture:
    Fixed random recurrent matrix W (never trained) — sparse (10% non-zero)
    Scaled so spectral radius = spectral_radius (default 0.9)
    Drive reservoir with each input sequence
    Collect reservoir states across time steps
    Train output weights with ridge regression (same as paper 1)

Export: output weights as float32 array for MCU deployment.
    The full reservoir is too large for M33 — use a small reservoir (64-128 units)
    or the GMM/KNN models for M33 targets. M55 can handle 200-500 units.

Joblib checkpoint: checkpoints/esn.pkl

Reference:
  Jaeger, H. (2001) The "echo state" approach to analysing and training
  recurrent neural networks. GMD Report 148.
  Shougat et al., Scientific Reports 2021 (paper 1) — linear readout from
  Hopf reservoir, identical in principle to ESN output layer training.
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


class EchoStateReadout:
    """
    Echo State Network with fixed reservoir and trained linear readout.

    The reservoir weight matrix W is fixed at construction (never trained).
    Only the output weights are learned — identical approach to the linear
    readout in Shougat et al. (2021), but with a richer input representation
    from the nonlinear reservoir transformation.

    Example:
        esn = EchoStateReadout(reservoir_size=200, spectral_radius=0.9)
        esn.fit(X_train, labels_train)
        preds = esn.predict(X_test)
    """

    def __init__(
        self,
        reservoir_size: int = 500,
        spectral_radius: float = 0.9,
        sparsity: float = 0.1,
        input_scaling: float = 0.5,
        alpha: float = 0.01,
        random_state: int = 42,
    ) -> None:
        """
        Args:
            reservoir_size: number of reservoir neurons.
            spectral_radius: target spectral radius of W (echo state property
                             requires spectral_radius < 1.0).
            sparsity: fraction of non-zero connections in W (default 10%).
            input_scaling: multiplier on input before driving reservoir.
            alpha: ridge regression regularisation for output weights.
            random_state: seed for reproducible W matrix construction.
        """
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.alpha = alpha

        # Build fixed random reservoir weight matrix W
        # Sparse: only sparsity fraction of weights are non-zero
        rng = np.random.default_rng(random_state)
        W = rng.standard_normal((reservoir_size, reservoir_size))
        # Apply sparsity mask
        mask = rng.random((reservoir_size, reservoir_size)) > sparsity
        W[mask] = 0.0
        # Scale to target spectral radius
        eigvals = np.linalg.eigvals(W)
        sr = float(np.max(np.abs(eigvals)))
        if sr > 1e-8:
            W = W * (spectral_radius / sr)
        self._W: NDArray[np.float64] = W.astype(np.float64)

        self._readout = RidgeClassifier(alpha=alpha)
        self._scaler_in = StandardScaler()
        self._scaler_res = StandardScaler()
        self._is_fitted = False

    def _drive_reservoir(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Drive the reservoir with input X and collect final states.

        Each row of X is one input sample. We run the reservoir for
        len(X) steps and collect the final reservoir state.

        For batch processing (one vector per clip): treat each clip's
        feature vector as a single-step input drive.

        Args:
            X: (n_clips, n_features) input matrix

        Returns:
            (n_clips, reservoir_size) reservoir states
        """
        n = X.shape[0]
        states = np.zeros((n, self.reservoir_size), dtype=np.float64)
        h = np.zeros(self.reservoir_size, dtype=np.float64)

        # Input projection: random fixed matrix
        rng = np.random.default_rng(0)
        W_in = rng.standard_normal((self.reservoir_size, X.shape[1])) * self.input_scaling

        for i in range(n):
            # ESN update: h(t) = tanh(W*h(t-1) + W_in*u(t))
            h = np.tanh(self._W @ h + W_in @ X[i])
            states[i] = h

        return states

    def fit(
        self,
        features: NDArray,
        labels: NDArray[np.int64],
    ) -> "EchoStateReadout":
        """
        Drive reservoir with feature maps and train linear readout.

        Args:
            features: (n_clips, 200, 100) or (n_clips, 200, 100, 2) feature maps
            labels: (n_clips,) integer class labels

        Returns:
            self
        """
        X = features.reshape(features.shape[0], -1).astype(np.float64)
        X = self._scaler_in.fit_transform(X)

        print(f"[EchoStateReadout] Driving reservoir ({self.reservoir_size} neurons) ...")
        reservoir_states = self._drive_reservoir(X)
        reservoir_states = self._scaler_res.fit_transform(reservoir_states)

        print("[EchoStateReadout] Training ridge regression output weights ...")
        self._readout.fit(reservoir_states, labels)
        self._is_fitted = True
        return self

    def predict(self, features: NDArray) -> NDArray[np.int64]:
        """Predict class labels."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")
        X = features.reshape(features.shape[0], -1).astype(np.float64)
        X = self._scaler_in.transform(X)
        states = self._drive_reservoir(X)
        states = self._scaler_res.transform(states)
        return self._readout.predict(states).astype(np.int64)

    def score(
        self,
        features: NDArray,
        labels: NDArray[np.int64],
    ) -> float:
        """Return classification accuracy."""
        return float(accuracy_score(labels, self.predict(features)))

    def export_output_weights(
        self,
        path: Optional[str | Path] = None,
    ) -> Path:
        """
        Export output weights as a float32 C array for MCU deployment.

        The output layer is a simple matrix multiply: class = argmax(W_out @ h).
        This maps to arm_fully_connected_s8() after int8 quantisation.

        Args:
            path: output path (default firmware/esn_output_weights.h)

        Returns:
            Path to generated header file.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before export_output_weights()")

        fw_path = (Path(path) if path else
                   ROOT / "firmware" / "esn_output_weights.h")
        fw_path.parent.mkdir(parents=True, exist_ok=True)

        W_out = self._readout.coef_.astype(np.float32)
        n_classes, n_reservoir = W_out.shape

        lines = [
            "/* esn_output_weights.h — generated by pipeline/models/sequence/esn_readout.py */",
            "/* ESN output layer: class = argmax(W_out @ reservoir_state) */",
            "/* Maps to arm_fully_connected_s8() after int8 quantisation */",
            "#pragma once",
            f"#define ESN_RESERVOIR_SIZE {n_reservoir}",
            f"#define ESN_N_CLASSES      {n_classes}",
            "",
            f"/* W_out[{n_classes}][{n_reservoir}] — float32 output weights */",
            f"static const float esn_output_weights[{n_classes}][{n_reservoir}] = {{",
        ]
        for i in range(n_classes):
            vals = ", ".join(f"{v:.6f}f" for v in W_out[i])
            lines.append(f"  /* class {i} */ {{ {vals} }},")
        lines.append("};")

        fw_path.write_text("\n".join(lines))
        print(f"[EchoStateReadout] Output weights written to {fw_path}")
        return fw_path

    def save(self, path: Optional[str | Path] = None) -> Path:
        """Save to checkpoints/esn.pkl."""
        import joblib
        p = Path(path) if path else CHECKPOINT_DIR / "esn.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "W": self._W,
            "readout": self._readout,
            "scaler_in": self._scaler_in,
            "scaler_res": self._scaler_res,
        }, p)
        print(f"[EchoStateReadout] Saved to {p}")
        return p


def main() -> None:
    """Train EchoStateReadout on processed Hopf feature maps."""
    from data.sample_data import generate_dataset_xy
    from pipeline.ingest import process_dataset
    from pipeline.features_xy import extract_y_features, extract_all_representations

    print("[esn] Generating synthetic data ...")
    raw_x, raw_y, labels = generate_dataset_xy(
        n_clips_per_class=15, n_classes=5, cache=False,
    )
    x_proc = process_dataset(raw_x)
    y_proc = extract_y_features(raw_y)
    reps = extract_all_representations(x_proc, y_proc)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(labels))
    n_val = int(0.2 * len(labels))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    # Use a small reservoir for demo speed
    esn = EchoStateReadout(reservoir_size=100, spectral_radius=0.9, alpha=0.01)
    esn.fit(reps["x_only"][train_idx], labels[train_idx])

    val_acc = esn.score(reps["x_only"][val_idx], labels[val_idx])
    print(f"  Val accuracy: {val_acc:.4f}")

    esn.save()
    esn.export_output_weights()
    print("[esn] Done.")


if __name__ == "__main__":
    main()
