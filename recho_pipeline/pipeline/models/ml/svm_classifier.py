"""
SVM Classifier on PCA-reduced Hopf oscillator feature maps.

Input: x_features flattened to 1D [n_clips, 20000]
PCA reduces to 50 components before SVM to fit the M33's memory constraint —
a 20,000-dim SVM would require enormous RAM for support vectors.

Trains and compares 5 input variants:
    x_only          — flatten(x_features)           [n, 20000]
    y_only          — flatten(y_features)            [n, 20000]
    xy_concatenated — [flatten(x), flatten(y)]       [n, 40000]
    phase           — flatten(phase_features)        [n, 20000]
    angle           — flatten(angle_features)        [n, 20000]

Why SVM for RECHO:
    SVM needs only 5-10 examples per class to generalise well in a PCA-reduced
    space. Most relevant for RECHO's few-shot deployment scenario where a
    customer runs the machine for 5 minutes and expects a trained classifier.
    Paper 1 (Shougat 2021) uses ridge regression — SVM is the natural upgrade.

Joblib checkpoint: checkpoints/svm_[variant].pkl

Reference:
  Shougat et al., Scientific Reports 2021 (paper 1) — RidgeClassifier readout.
  This extends to SVC with PCA preprocessing for higher accuracy.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

CHECKPOINT_DIR = ROOT / "checkpoints"
N_PCA_COMPONENTS: int = 50  # memory constraint: 50 floats per clip at inference


class SVMClassifier:
    """
    SVM classifier with PCA preprocessing on flattened Hopf feature maps.

    Compresses 20,000-dimensional feature maps to 50 PCA components before
    fitting an RBF-kernel SVM. GridSearchCV tunes C and gamma automatically.

    Example:
        clf = SVMClassifier()
        clf.fit(x_features, labels)
        preds = clf.predict(x_features_test)
    """

    def __init__(
        self,
        n_components: int = N_PCA_COMPONENTS,
        use_grid_search: bool = True,
    ) -> None:
        """
        Args:
            n_components: number of PCA components (default 50).
            use_grid_search: if True, tune C and gamma via GridSearchCV.
        """
        self.n_components = n_components
        self.use_grid_search = use_grid_search
        self._pca = PCA(n_components=n_components, random_state=42)
        self._scaler = StandardScaler()
        self._clf: Optional[SVC] = None
        self._is_fitted = False

    def _flatten(self, features: NDArray) -> NDArray[np.float32]:
        """Flatten (n, H, W[, C]) → (n, H*W*[C]) float32."""
        return features.reshape(features.shape[0], -1).astype(np.float32)

    def fit(
        self,
        features: NDArray,
        labels: NDArray[np.int64],
    ) -> "SVMClassifier":
        """
        Fit PCA + SVM on feature maps.

        Args:
            features: (n_clips, 200, 100) or (n_clips, 200, 100, 2) feature maps
            labels: (n_clips,) integer class labels

        Returns:
            self
        """
        X = self._flatten(features)
        X = self._scaler.fit_transform(X)
        X_pca = self._pca.fit_transform(X)

        if self.use_grid_search:
            param_grid = {"C": [0.1, 1.0, 10.0, 100.0],
                          "gamma": ["scale", "auto", 0.01, 0.1]}
            base_svc = SVC(kernel="rbf", probability=True, random_state=42)
            gs = GridSearchCV(base_svc, param_grid, cv=3, n_jobs=-1, verbose=0)
            gs.fit(X_pca, labels)
            self._clf = gs.best_estimator_
            print(f"[SVMClassifier] Best params: {gs.best_params_}")
        else:
            self._clf = SVC(kernel="rbf", C=1.0, gamma="scale",
                            probability=True, random_state=42)
            self._clf.fit(X_pca, labels)

        self._is_fitted = True
        return self

    def predict(self, features: NDArray) -> NDArray[np.int64]:
        """Predict class labels."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")
        X = self._scaler.transform(self._flatten(features))
        X_pca = self._pca.transform(X)
        return self._clf.predict(X_pca).astype(np.int64)

    def score(self, features: NDArray, labels: NDArray[np.int64]) -> float:
        """Return classification accuracy."""
        return float(accuracy_score(labels, self.predict(features)))

    def save(self, path: Optional[str | Path] = None, name: str = "svm") -> Path:
        """Save to checkpoints/svm_[name].pkl."""
        import joblib
        p = Path(path) if path else CHECKPOINT_DIR / f"svm_{name}.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self._clf, "pca": self._pca,
                     "scaler": self._scaler}, p)
        print(f"[SVMClassifier] Saved to {p}")
        return p


def train_all_variants(
    reps: dict[str, NDArray],
    labels: NDArray[np.int64],
    val_split: float = 0.2,
    use_grid_search: bool = False,
) -> dict[str, dict]:
    """
    Train SVM on all 5 input variants and compare accuracy.

    Args:
        reps: dict with keys "x_only", "y_only", "xy_dual", "phase", "angle"
        labels: (n_clips,) integer class labels
        val_split: validation fraction
        use_grid_search: if True, tune hyperparameters (slow)

    Returns:
        dict mapping variant_name → {"model": SVMClassifier, "val_acc": float}
    """
    rng = np.random.default_rng(42)
    n = len(labels)
    idx = rng.permutation(n)
    n_val = int(n * val_split)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    # Build the 5 input variants
    variants = {
        "x_only": reps["x_only"],
        "y_only": reps["y_only"],
        "xy_concatenated": np.concatenate([
            reps["x_only"].reshape(n, -1),
            reps["y_only"].reshape(n, -1),
        ], axis=1).reshape(n, -1),
        "phase": reps["phase"],
        "angle": reps["angle"],
    }

    results: dict[str, dict] = {}
    for name, feat in variants.items():
        print(f"\n[svm] Training SVM on {name} ...")
        clf = SVMClassifier(n_components=N_PCA_COMPONENTS,
                            use_grid_search=use_grid_search)
        clf.fit(feat[train_idx], labels[train_idx])
        val_acc = clf.score(feat[val_idx], labels[val_idx])
        print(f"  {name}: val_accuracy = {val_acc:.4f}")
        clf.save(name=name)
        results[name] = {"model": clf, "val_acc": val_acc}

    # Print comparison table
    print(f"\n{'='*50}")
    print("SVM INPUT VARIANT COMPARISON")
    print(f"{'='*50}")
    for name, res in results.items():
        print(f"  {name:<20s}: val_acc = {res['val_acc']:.4f}")

    return results


def main() -> None:
    """Train SVMClassifier on all 5 input variants using synthetic data."""
    from data.sample_data import generate_dataset_xy
    from pipeline.ingest import process_dataset
    from pipeline.features_xy import extract_y_features, extract_all_representations

    print("[svm] Generating synthetic data ...")
    raw_x, raw_y, labels = generate_dataset_xy(
        n_clips_per_class=20, n_classes=5, cache=False,
    )
    x_proc = process_dataset(raw_x)
    y_proc = extract_y_features(raw_y)
    reps = extract_all_representations(x_proc, y_proc)

    results = train_all_variants(reps, labels, use_grid_search=False)
    print("[svm] Done.")


if __name__ == "__main__":
    main()
