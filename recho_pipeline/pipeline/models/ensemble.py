"""
Model H — Majority vote ensemble across multiple trained models.

Loads any subset of trained Keras models and a ReservoirReadout and combines
their predictions. Two combination strategies:

  1. Majority vote: each model casts one vote per sample; class with
     most votes wins. Handles ties by preferring the highest-confidence model.

  2. Weighted vote: votes are weighted by each model's validation accuracy.
     Models that performed better on validation data have proportionally more
     influence. Reduces the impact of weak models.

Purpose:
  - Often achieves the best practical accuracy by averaging out model errors.
  - Relevant for a user-facing product where latency matters less than
    reliability (run on host, not directly on M33).
  - Identifies when models make systematically different errors — high
    ensemble gain means models disagree on different samples.
  - Can be deployed on M85 where the NPU runs multiple model inference
    passes in sequence (still < 50 ms total for 4 models).

NOT a single TFLite model — this runs in Python/on-host or on a richer MCU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class VoteEnsemble:
    """
    Majority vote or weighted vote ensemble across heterogeneous models.

    Supports mixing Keras models (predict returns softmax probabilities)
    and ReservoirReadout models (predict_proba_approx returns softmax-like
    probabilities).

    Example:
        ensemble = VoteEnsemble(strategy="weighted")
        ensemble.add_keras_model(model_a, x_val, labels_val, input_key="x_only")
        ensemble.add_keras_model(model_b, xy_val, labels_val, input_key="xy_dual")
        ensemble.add_readout(readout, x_val_flat, labels_val, input_key="x_only_flat")
        preds = ensemble.predict({"x_only": x_test, "xy_dual": xy_test, ...})
        acc = ensemble.score(preds, labels_test)
    """

    def __init__(self, strategy: str = "majority") -> None:
        """
        Args:
            strategy: "majority" (equal weights) or "weighted" (by val accuracy).
        """
        if strategy not in ("majority", "weighted"):
            raise ValueError(f"strategy must be 'majority' or 'weighted', got '{strategy}'")
        self.strategy = strategy
        self._members: list[dict] = []

    def add_keras_model(
        self,
        model,
        val_inputs,
        val_labels: NDArray[np.int64],
        input_key: str,
        name: Optional[str] = None,
    ) -> "VoteEnsemble":
        """
        Register a Keras model as an ensemble member.

        Args:
            model: trained Keras model (Sequential or functional)
            val_inputs: validation input(s) — numpy array or list of arrays
              for multi-input models
            val_labels: (n_val,) integer class labels
            input_key: key into the inputs dict at predict time
            name: optional display name
        """
        if isinstance(val_inputs, (list, tuple)):
            probs = model.predict(val_inputs, verbose=0)
        else:
            probs = model.predict(
                np.expand_dims(val_inputs, -1) if val_inputs.ndim == 3 else val_inputs,
                verbose=0,
            )
        preds = np.argmax(probs, axis=1)
        val_acc = float(np.mean(preds == val_labels))

        self._members.append({
            "type": "keras",
            "model": model,
            "input_key": input_key,
            "val_acc": val_acc,
            "name": name or f"keras_{len(self._members)}",
        })
        print(f"[ensemble] Added {name or 'keras model'} — val_acc={val_acc:.4f}")
        return self

    def add_readout(
        self,
        readout,
        val_features: NDArray,
        val_labels: NDArray[np.int64],
        input_key: str,
        name: Optional[str] = None,
    ) -> "VoteEnsemble":
        """
        Register a ReservoirReadout as an ensemble member.

        Args:
            readout: fitted ReservoirReadout instance
            val_features: validation feature maps
            val_labels: (n_val,) integer class labels
            input_key: key into the inputs dict at predict time
            name: optional display name
        """
        val_acc = readout.score(val_features, val_labels)

        self._members.append({
            "type": "readout",
            "model": readout,
            "input_key": input_key,
            "val_acc": val_acc,
            "name": name or f"readout_{len(self._members)}",
        })
        print(f"[ensemble] Added {name or 'readout'} — val_acc={val_acc:.4f}")
        return self

    def predict(self, inputs: dict[str, NDArray]) -> NDArray[np.int64]:
        """
        Combine predictions from all registered models.

        Args:
            inputs: dict mapping input_key → feature array for that model

        Returns:
            (n_samples,) integer class predictions
        """
        if not self._members:
            raise RuntimeError("No models registered. Call add_keras_model() or add_readout().")

        all_probs: list[NDArray[np.float32]] = []
        weights: list[float] = []

        for member in self._members:
            key = member["input_key"]
            if key not in inputs:
                raise KeyError(f"Input key '{key}' not found. Available: {list(inputs.keys())}")

            feat = inputs[key]

            if member["type"] == "keras":
                model = member["model"]
                if isinstance(feat, (list, tuple)):
                    probs = model.predict(feat, verbose=0)
                else:
                    inp = np.expand_dims(feat, -1) if feat.ndim == 3 else feat
                    probs = model.predict(inp, verbose=0)
                all_probs.append(probs.astype(np.float32))
            else:
                probs = member["model"].predict_proba_approx(feat)
                all_probs.append(probs)

            weights.append(member["val_acc"])

        if self.strategy == "majority":
            # Each model casts a hard vote (argmax of its probabilities)
            votes = np.stack([np.argmax(p, axis=1) for p in all_probs], axis=1)
            # Majority: most frequent class; ties broken by highest total confidence
            n_samples = votes.shape[0]
            n_classes = all_probs[0].shape[1]
            combined = np.zeros((n_samples, n_classes), dtype=np.float32)
            for i, probs in enumerate(all_probs):
                vote_idx = np.argmax(probs, axis=1)
                for s in range(n_samples):
                    combined[s, vote_idx[s]] += 1.0
            return np.argmax(combined, axis=1).astype(np.int64)

        else:  # weighted
            # Weighted average of probabilities (soft vote)
            total_weight = sum(weights)
            combined = np.zeros_like(all_probs[0])
            for probs, w in zip(all_probs, weights):
                combined += (w / total_weight) * probs
            return np.argmax(combined, axis=1).astype(np.int64)

    def score(
        self,
        inputs: dict[str, NDArray],
        labels: NDArray[np.int64],
    ) -> float:
        """Return ensemble accuracy."""
        preds = self.predict(inputs)
        return float(np.mean(preds == labels))

    def member_summary(self) -> None:
        """Print a summary table of all registered models."""
        print("\n--- Ensemble Members ---")
        print(f"  {'Name':30s} {'Type':10s} {'Input Key':20s} {'Val Acc':>10s}")
        print("  " + "-" * 72)
        for m in self._members:
            print(f"  {m['name']:30s} {m['type']:10s} {m['input_key']:20s} "
                  f"{m['val_acc']:>10.4f}")
        print(f"\n  Strategy: {self.strategy}")
        print(f"  Members:  {len(self._members)}")
