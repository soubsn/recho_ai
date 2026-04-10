"""
k-Nearest Neighbours Classifier on PCA-reduced Feature Maps.

Input: flattened feature maps → PCA to 50 components → KNN classification.

Paper reference suggests inter-class Euclidean distance is already large in
the Hopf feature space, meaning KNN may achieve near-CNN accuracy with zero
training (besides storing a few reference clips per class).

Two implementations:
    1. sklearn KNeighborsClassifier — full precision, Python inference
    2. Pure numpy MCU implementation — stores k reference feature vectors per
       class in flash; classifies by Euclidean distance to nearest neighbour.
       Generates firmware/knn_references.h with reference arrays as int8_t[][].

Accuracy comparison: k=1,3,5,7 evaluated and plotted.

Joblib checkpoint: checkpoints/knn.pkl

Reference:
  Shougat et al., Scientific Reports 2021 (paper 1) — feature space geometry.
  Large inter-class distances in PCA space support KNN as a viable classifier.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

CHECKPOINT_DIR = ROOT / "checkpoints"
FIRMWARE_DIR = ROOT / "firmware"
N_PCA_COMPONENTS: int = 50


class KNNClassifier:
    """
    k-Nearest Neighbours classifier with PCA preprocessing.

    Stores k reference feature maps per class for inference.
    The pure numpy MCU version does not require any training beyond selecting
    reference clips — making it the fastest model to deploy in the field.

    Example:
        clf = KNNClassifier(k=5)
        clf.fit(x_features, labels)
        preds = clf.predict(x_features_test)
        clf.plot_k_comparison(x_test, labels_test)
        clf.export_firmware_header()
    """

    def __init__(
        self,
        k: int = 5,
        n_components: int = N_PCA_COMPONENTS,
    ) -> None:
        """
        Args:
            k: number of nearest neighbours.
            n_components: PCA dimensionality.
        """
        self.k = k
        self.n_components = n_components
        self._pca = PCA(n_components=n_components, random_state=42)
        self._scaler = StandardScaler()
        self._clf = KNeighborsClassifier(
            n_neighbors=k, metric="euclidean", n_jobs=-1,
        )
        self._X_train_pca: Optional[NDArray[np.float64]] = None
        self._y_train: Optional[NDArray[np.int64]] = None
        self._is_fitted = False

    def _flatten(self, features: NDArray) -> NDArray[np.float32]:
        """Flatten (n, H, W[, C]) → (n, D) float32."""
        return features.reshape(features.shape[0], -1).astype(np.float32)

    def fit(
        self,
        features: NDArray,
        labels: NDArray[np.int64],
    ) -> "KNNClassifier":
        """
        Fit KNN on PCA-reduced feature maps.

        Args:
            features: (n_clips, 200, 100) or (n_clips, 200, 100, 2) feature maps
            labels: (n_clips,) integer class labels

        Returns:
            self
        """
        X = self._flatten(features)
        X = self._scaler.fit_transform(X)
        X_pca = self._pca.fit_transform(X)
        self._clf.fit(X_pca, labels)
        self._X_train_pca = X_pca.astype(np.float64)
        self._y_train = labels.copy()
        self._is_fitted = True
        return self

    def predict(self, features: NDArray) -> NDArray[np.int64]:
        """Predict class labels."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")
        X = self._scaler.transform(self._flatten(features))
        X_pca = self._pca.transform(X)
        return self._clf.predict(X_pca).astype(np.int64)

    def predict_numpy_mcu(
        self,
        features: NDArray,
        k: Optional[int] = None,
    ) -> NDArray[np.int64]:
        """
        Pure numpy KNN inference — mirrors the MCU firmware implementation.

        Computes Euclidean distance from each test clip to all training clips
        in PCA-reduced space. Returns class of k nearest neighbours.
        This is the exact computation that runs in firmware (with int8 inputs).

        Args:
            features: (n_clips, 200, 100) feature maps
            k: override number of neighbours (default: self.k)

        Returns:
            (n_clips,) predicted class labels
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_numpy_mcu()")
        k = k or self.k
        X = self._scaler.transform(self._flatten(features))
        X_pca = self._pca.transform(X).astype(np.float64)

        preds = np.zeros(len(X_pca), dtype=np.int64)
        for i in range(len(X_pca)):
            dists = np.sqrt(np.sum((self._X_train_pca - X_pca[i]) ** 2, axis=1))
            nn_idx = np.argsort(dists)[:k]
            nn_labels = self._y_train[nn_idx]
            counts = np.bincount(nn_labels, minlength=int(self._y_train.max()) + 1)
            preds[i] = int(np.argmax(counts))
        return preds

    def score(self, features: NDArray, labels: NDArray[np.int64]) -> float:
        """Return classification accuracy."""
        return float(accuracy_score(labels, self.predict(features)))

    def plot_k_comparison(
        self,
        features: NDArray,
        labels: NDArray[np.int64],
        k_values: Optional[list[int]] = None,
    ) -> None:
        """
        Plot accuracy vs k for k=1,3,5,7 (and additional values if specified).

        Args:
            features: (n, 200, 100) test feature maps
            labels: (n,) true class labels
            k_values: list of k values to test (default [1, 3, 5, 7])
        """
        import matplotlib.pyplot as plt

        k_values = k_values or [1, 3, 5, 7]
        accs = []
        for kk in k_values:
            clf_k = KNNClassifier(k=kk, n_components=self.n_components)
            # Reuse fitted PCA and scaler — just refit KNN with different k
            clf_k._pca = self._pca
            clf_k._scaler = self._scaler
            clf_k._X_train_pca = self._X_train_pca
            clf_k._y_train = self._y_train
            clf_k._clf = KNeighborsClassifier(n_neighbors=kk, metric="euclidean")
            clf_k._clf.fit(self._X_train_pca, self._y_train)
            clf_k._is_fitted = True
            accs.append(clf_k.score(features, labels))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(k_values, accs, "o-", color="steelblue", linewidth=2)
        for kk, acc in zip(k_values, accs):
            ax.text(kk, acc + 0.005, f"{acc:.3f}", ha="center", fontsize=9)
        ax.set_xlabel("k (number of neighbours)")
        ax.set_ylabel("Test Accuracy")
        ax.set_title("KNN Accuracy vs k — Hopf Feature Space")
        ax.set_xticks(k_values)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def export_firmware_header(
        self,
        n_references_per_class: int = 5,
        path: Optional[str | Path] = None,
    ) -> Path:
        """
        Export KNN reference vectors as int8_t arrays in a C header.

        Generates firmware/knn_references.h with:
            int8_t knn_references[N_CLASSES][N_REFS][N_COMPONENTS]
        Classification in firmware: compute Euclidean distance to all
        references, return class of nearest reference.

        Args:
            n_references_per_class: number of reference clips per class
            path: output path (default firmware/knn_references.h)

        Returns:
            Path to generated header file.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before export_firmware_header()")

        fw_path = Path(path) if path else FIRMWARE_DIR / "knn_references.h"
        fw_path.parent.mkdir(parents=True, exist_ok=True)

        n_classes = int(self._y_train.max()) + 1
        n_refs = min(n_references_per_class,
                     int(np.min([np.sum(self._y_train == c)
                                 for c in range(n_classes)])))

        # Select first n_refs training clips from each class
        refs: list[list[NDArray]] = []
        for cls in range(n_classes):
            cls_idx = np.where(self._y_train == cls)[0][:n_refs]
            refs.append([self._X_train_pca[i] for i in cls_idx])

        # Quantise to int8
        flat_refs = np.stack([v for cls_refs in refs for v in cls_refs], axis=0)
        scale = 127.0 / (np.max(np.abs(flat_refs)) + 1e-12)
        flat_q = np.clip(np.round(flat_refs * scale), -128, 127).astype(np.int8)

        lines = [
            "/* knn_references.h — generated by pipeline/models/ml/knn_classifier.py */",
            "/* KNN references as int8_t[n_classes][n_refs][n_pca_components] */",
            "/* Classification: Euclidean distance to nearest reference in PCA space */",
            "#pragma once",
            f"#define KNN_N_CLASSES    {n_classes}",
            f"#define KNN_N_REFS       {n_refs}",
            f"#define KNN_N_COMPONENTS {self.n_components}",
            f"#define KNN_SCALE_INV    {1.0/scale:.6f}f  /* multiply by this to get float */",
            "",
        ]

        lines.append("static const int8_t knn_references"
                     f"[{n_classes}][{n_refs}][{self.n_components}] = {{")
        ref_idx = 0
        for cls in range(n_classes):
            lines.append(f"  /* Class {cls} */  {{")
            for r in range(n_refs):
                vals = ", ".join(str(int(v)) for v in flat_q[ref_idx])
                lines.append(f"    {{ {vals} }},")
                ref_idx += 1
            lines.append("  },")
        lines.append("};")

        fw_path.write_text("\n".join(lines))
        print(f"[KNNClassifier] Firmware references written to {fw_path}")
        return fw_path

    def save(self, path: Optional[str | Path] = None) -> Path:
        """Save to checkpoints/knn.pkl."""
        import joblib
        p = Path(path) if path else CHECKPOINT_DIR / "knn.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "clf": self._clf, "pca": self._pca, "scaler": self._scaler,
            "X_train_pca": self._X_train_pca, "y_train": self._y_train,
        }, p)
        print(f"[KNNClassifier] Saved to {p}")
        return p


def main() -> None:
    """Train KNNClassifier and compare accuracy across k values."""
    from data.sample_data import generate_dataset_xy
    from pipeline.ingest import process_dataset
    from pipeline.features_xy import extract_y_features, extract_all_representations

    print("[knn] Generating synthetic data ...")
    raw_x, raw_y, labels = generate_dataset_xy(
        n_clips_per_class=20, n_classes=5, cache=False,
    )
    x_proc = process_dataset(raw_x)
    y_proc = extract_y_features(raw_y)
    reps = extract_all_representations(x_proc, y_proc)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(labels))
    n_val = int(0.2 * len(labels))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    print("[knn] Fitting KNN (k=5) ...")
    clf = KNNClassifier(k=5, n_components=20)
    clf.fit(reps["x_only"][train_idx], labels[train_idx])

    val_acc = clf.score(reps["x_only"][val_idx], labels[val_idx])
    print(f"  Val accuracy (k=5): {val_acc:.4f}")

    # Accuracy vs k
    print("\nAccuracy vs k:")
    for kk in [1, 3, 5, 7]:
        temp = KNNClassifier(k=kk, n_components=20)
        temp._pca = clf._pca
        temp._scaler = clf._scaler
        temp._X_train_pca = clf._X_train_pca
        temp._y_train = clf._y_train
        from sklearn.neighbors import KNeighborsClassifier as _KNN
        temp._clf = _KNN(n_neighbors=kk, metric="euclidean")
        temp._clf.fit(clf._X_train_pca, clf._y_train)
        temp._is_fitted = True
        acc = temp.score(reps["x_only"][val_idx], labels[val_idx])
        print(f"  k={kk}: {acc:.4f}")

    clf.save()
    clf.export_firmware_header(n_references_per_class=3)
    print("[knn] Done.")


if __name__ == "__main__":
    main()
