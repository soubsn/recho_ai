"""
Model G — Ridge regression readout (paper 1 method).

This is a linear classifier — not a neural network — matching the readout
method used in:
  "Recognising sound signals with a Hopf physical reservoir computer"
  (Shougat et al., 2021) — paper 1.

The feature map is flattened to a 1D vector and a regularised linear
classifier (RidgeClassifier, lambda=0.01) is fitted. This is the simplest
possible readout from the reservoir state.

Fits separate classifiers for:
  - x_only:         flatten(x_features) — 20,000 features
  - y_only:         flatten(y_features) — 20,000 features
  - xy_concatenated: [flatten(x), flatten(y)] — 40,000 features

Purpose:
  1. Establishes the linear upper bound — how much does the CNN add over
     the simple approach from paper 1?
  2. Answers whether y(t) adds signal when combined linearly (no nonlinearity).
  3. Fast to train, no QAT, no GPU needed.

NOT a TFLite/CMSIS-NN model — used for comparison only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


RIDGE_ALPHA: float = 0.01  # regularisation lambda from paper 1


class ReservoirReadout:
    """
    Linear ridge regression readout over flattened Hopf reservoir states.

    Fits one RidgeClassifier per input representation (x_only, y_only,
    xy_concatenated) using scikit-learn.

    Example:
        readout = ReservoirReadout()
        readout.fit(x_train, y_train)
        acc = readout.score(x_val, y_val)
    """

    def __init__(self, alpha: float = RIDGE_ALPHA) -> None:
        self.alpha = alpha
        self._classifier = RidgeClassifier(alpha=alpha)
        self._scaler = StandardScaler()
        self._is_fitted: bool = False

    def _flatten(self, feature_maps: NDArray) -> NDArray[np.float32]:
        """Flatten (n, 200, 100[, 2]) → (n, features) float32."""
        n = feature_maps.shape[0]
        return feature_maps.reshape(n, -1).astype(np.float32)

    def fit(
        self,
        feature_maps: NDArray,
        labels: NDArray[np.int64],
    ) -> "ReservoirReadout":
        """
        Fit the ridge regression classifier.

        Args:
            feature_maps: (n, 200, 100) or (n, 200, 100, 2) — training features
            labels: (n,) integer class labels

        Returns:
            self (for chaining)
        """
        X = self._flatten(feature_maps)
        X = self._scaler.fit_transform(X)
        self._classifier.fit(X, labels)
        self._is_fitted = True
        return self

    def predict(self, feature_maps: NDArray) -> NDArray[np.int64]:
        """Predict class labels."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")
        X = self._flatten(feature_maps)
        X = self._scaler.transform(X)
        return self._classifier.predict(X)

    def predict_proba_approx(self, feature_maps: NDArray) -> NDArray[np.float32]:
        """
        Return approximate class probabilities via softmax over decision function.

        RidgeClassifier has no native predict_proba. We use the decision
        function scores normalised via softmax as a proxy — useful for
        combining with other models in the ensemble.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_proba_approx()")
        X = self._flatten(feature_maps)
        X = self._scaler.transform(X)
        scores = self._classifier.decision_function(X)
        if scores.ndim == 1:
            # Binary case — wrap in (n, 2) for consistency
            scores = np.stack([-scores, scores], axis=-1)
        # Softmax normalisation
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return (exp_scores / exp_scores.sum(axis=1, keepdims=True)).astype(np.float32)

    def score(
        self,
        feature_maps: NDArray,
        labels: NDArray[np.int64],
    ) -> float:
        """Return accuracy on the given feature maps and labels."""
        preds = self.predict(feature_maps)
        return float(accuracy_score(labels, preds))


def fit_all_readouts(
    x_features: NDArray,
    y_features: NDArray,
    labels_train: NDArray[np.int64],
    x_val: NDArray,
    y_val: NDArray,
    labels_val: NDArray[np.int64],
    alpha: float = RIDGE_ALPHA,
) -> dict[str, dict]:
    """
    Fit and evaluate ridge regression for x_only, y_only, and xy_concatenated.

    Returns:
        dict mapping variant name → {"model": ReservoirReadout, "val_acc": float}
    """
    variants = {
        "x_only": (x_features, x_val),
        "y_only": (y_features, y_val),
        "xy_concatenated": (
            np.concatenate([
                x_features.reshape(len(x_features), -1),
                y_features.reshape(len(y_features), -1),
            ], axis=1).reshape(len(x_features), -1),
            np.concatenate([
                x_val.reshape(len(x_val), -1),
                y_val.reshape(len(y_val), -1),
            ], axis=1).reshape(len(x_val), -1),
        ),
    }

    results: dict[str, dict] = {}
    for name, (train_data, val_data) in variants.items():
        print(f"[reservoir_readout] Fitting {name} ...")
        readout = ReservoirReadout(alpha=alpha)
        readout.fit(train_data, labels_train)
        val_acc = readout.score(val_data, labels_val)
        print(f"  {name}: val_accuracy = {val_acc:.4f}")
        results[name] = {"model": readout, "val_acc": val_acc}

    return results
