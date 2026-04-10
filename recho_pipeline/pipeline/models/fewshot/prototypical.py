"""
Prototype Network — Few-Shot Classification.

Each class is represented by the mean embedding of its support set examples.
Classification = nearest prototype in embedding space.

With as few as 1-5 examples per class, the Prototypical Network achieves
competitive accuracy because the Hopf oscillator feature space has large
inter-class distances (shown in Shougat et al. 2021/2023).

Encoder: any trained CNN encoder (default: cnn_x_only backbone).
    Compatible with ContrastiveClassifier's pretrained encoder.
    Can also use a raw sklearn PCA projection as a simple "encoder."

Online prototype update:
    As more examples arrive, the prototype is updated via running mean.
    No retraining needed — just update the mean embedding.

MCU deployment:
    Deploy prototypes as int8_t arrays in flash memory.
    Classification = arm_fully_connected_s8() distance computation.
    For k prototypes × 128-dim embedding: 128 * k bytes in flash.
    Example: 5 classes × 128-dim int8 = 640 bytes — trivial overhead.

This is the ideal EchoReveal model: user records 5 examples, model updates
immediately. No GPU, no retraining, no data collection campaign needed.

Reference:
  Snell, J., Swersky, K. & Zemel, R. (2017) Prototypical Networks for
  Few-shot Learning. NeurIPS 2017.
  Shougat et al., Scientific Reports 2023 (paper 2) — "trained in a few
  hours" reconfigurability as a key product feature.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

CHECKPOINT_DIR = ROOT / "checkpoints"


class PrototypicalNetwork:
    """
    Prototype Network for few-shot classification on Hopf feature maps.

    Accepts any encoder (Keras model, sklearn PCA, or identity) and builds
    per-class prototypes from a small support set. Classification is then
    pure nearest-neighbour in embedding space.

    Example:
        # Use a pre-trained CNN encoder
        from tensorflow import keras
        encoder = keras.models.load_model("checkpoints/contrastive_encoder.keras")

        net = PrototypicalNetwork(encoder=encoder)
        support = {
            "normal":  x_normal_clips[:5],
            "fault_a": x_fault_clips[:5],
        }
        net.build_prototypes(support)
        label = net.classify(new_clip)
        net.update_prototype("normal", another_normal_clip)
    """

    def __init__(
        self,
        encoder: Optional[object] = None,
        n_pca_components: int = 64,
    ) -> None:
        """
        Args:
            encoder: any callable that maps (n_clips, H, W[, C]) → (n_clips, D).
                     If None, uses PCA as a simple linear encoder.
            n_pca_components: PCA components to use if encoder is None.
        """
        self._encoder_obj = encoder
        self.n_pca_components = n_pca_components
        self._prototypes: dict[str, NDArray[np.float32]] = {}
        self._prototype_counts: dict[str, int] = {}
        self._pca = None
        self._pca_scaler = None
        self._pca_fitted = False

    def _encode(self, clips: NDArray) -> NDArray[np.float32]:
        """
        Encode a batch of feature map clips to embedding vectors.

        Args:
            clips: (n_clips, H, W) or (n_clips, H, W, C) feature maps

        Returns:
            (n_clips, embedding_dim) float32 embeddings
        """
        if self._encoder_obj is not None:
            # Use provided encoder (Keras model or callable)
            x = clips.astype(np.float32) / 255.0
            if x.ndim == 3:
                x = x[:, :, :, np.newaxis]
            return np.array(self._encoder_obj.predict(x, verbose=0),
                            dtype=np.float32)

        # Fallback: PCA encoder
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        X = clips.reshape(clips.shape[0], -1).astype(np.float32)

        if not self._pca_fitted:
            self._pca_scaler = StandardScaler()
            self._pca = PCA(n_components=self.n_pca_components, random_state=42)
            X = self._pca_scaler.fit_transform(X)
            X = self._pca.fit_transform(X).astype(np.float32)
            self._pca_fitted = True
        else:
            X = self._pca_scaler.transform(X)
            X = self._pca.transform(X).astype(np.float32)

        return X

    def build_prototypes(
        self,
        support_set: dict[str, NDArray],
    ) -> None:
        """
        Compute class prototypes from support set examples.

        Prototype[class] = mean(encoder(support_clips_for_class)).
        As few as 1-5 examples per class are sufficient.

        Args:
            support_set: dict mapping class_name → (n_examples, 200, 100) clips
        """
        self._prototypes = {}
        self._prototype_counts = {}

        for class_name, clips in support_set.items():
            embeddings = self._encode(np.array(clips))
            prototype = np.mean(embeddings, axis=0)
            self._prototypes[class_name] = prototype.astype(np.float32)
            self._prototype_counts[class_name] = len(clips)
            print(f"  Prototype '{class_name}': {len(clips)} examples → "
                  f"dim={len(prototype)}")

    def classify(self, query_clip: NDArray) -> str:
        """
        Return class name of nearest prototype in embedding space.

        Distance metric: Euclidean in embedding space.
        Maps to arm_fully_connected_s8() distance computation in firmware.

        Args:
            query_clip: (200, 100) uint8 feature map to classify

        Returns:
            Class name string (key in support_set dict).
        """
        if not self._prototypes:
            raise RuntimeError("Call build_prototypes() before classify()")

        query_emb = self._encode(np.expand_dims(query_clip, 0))[0]

        best_class = ""
        best_dist = float("inf")
        for class_name, proto in self._prototypes.items():
            dist = float(np.linalg.norm(query_emb.astype(np.float64)
                                         - proto.astype(np.float64)))
            if dist < best_dist:
                best_dist = dist
                best_class = class_name

        return best_class

    def classify_batch(self, query_clips: NDArray) -> list[str]:
        """
        Classify a batch of query clips.

        Args:
            query_clips: (n_clips, 200, 100) feature maps

        Returns:
            list of predicted class name strings
        """
        embeddings = self._encode(query_clips)
        proto_names = list(self._prototypes.keys())
        proto_matrix = np.stack([self._prototypes[k] for k in proto_names],
                                 axis=0).astype(np.float64)

        results = []
        for emb in embeddings.astype(np.float64):
            dists = np.linalg.norm(proto_matrix - emb[np.newaxis, :], axis=1)
            results.append(proto_names[int(np.argmin(dists))])
        return results

    def update_prototype(
        self,
        class_name: str,
        new_clip: NDArray,
    ) -> None:
        """
        Online prototype update — running mean as new examples arrive.

        new_prototype = (count * old_proto + new_emb) / (count + 1)
        No retraining needed. Allows the model to improve incrementally.

        Args:
            class_name: class name to update (creates new class if not seen)
            new_clip: (200, 100) new example clip for this class
        """
        new_emb = self._encode(np.expand_dims(new_clip, 0))[0].astype(np.float32)

        if class_name not in self._prototypes:
            # New class: initialise from this example
            self._prototypes[class_name] = new_emb
            self._prototype_counts[class_name] = 1
            print(f"  New class '{class_name}' added with 1 example")
            return

        n = self._prototype_counts[class_name]
        old_proto = self._prototypes[class_name]
        self._prototypes[class_name] = (
            (n * old_proto + new_emb) / (n + 1)
        ).astype(np.float32)
        self._prototype_counts[class_name] = n + 1

    def export_firmware_header(
        self,
        path: Optional[str | Path] = None,
    ) -> Path:
        """
        Export prototypes as int8_t arrays for MCU deployment.

        Classification in firmware:
            int8_t query_emb[EMBEDDING_DIM];
            // compute query embedding
            // find nearest prototype via Euclidean distance
            // arm_fully_connected_s8() for distance computation

        Args:
            path: output path (default firmware/prototypes.h)

        Returns:
            Path to generated header file.
        """
        if not self._prototypes:
            raise RuntimeError("Call build_prototypes() before export_firmware_header()")

        fw_path = (Path(path) if path else ROOT / "firmware" / "prototypes.h")
        fw_path.parent.mkdir(parents=True, exist_ok=True)

        class_names = list(self._prototypes.keys())
        n_classes = len(class_names)
        embedding_dim = len(next(iter(self._prototypes.values())))

        # Quantise to int8
        all_protos = np.stack(list(self._prototypes.values()), axis=0)
        scale = 127.0 / (np.max(np.abs(all_protos)) + 1e-12)
        protos_q = np.clip(np.round(all_protos * scale), -128, 127).astype(np.int8)

        lines = [
            "/* prototypes.h — generated by pipeline/models/fewshot/prototypical.py */",
            "/* Prototype vectors for few-shot classification on MCU */",
            "/* Classification: Euclidean distance to nearest prototype */",
            "/* arm_fully_connected_s8() computes the distance in firmware */",
            "#pragma once",
            f"#define PROTO_N_CLASSES    {n_classes}",
            f"#define PROTO_EMBED_DIM    {embedding_dim}",
            f"#define PROTO_SCALE_INV    {1.0/scale:.6f}f",
            "",
        ]

        for i, name in enumerate(class_names):
            lines.append(f"#define PROTO_CLASS_{i}  \"{name}\"")
        lines.append("")

        lines.append(f"static const int8_t prototypes[{n_classes}][{embedding_dim}] = {{")
        for i, name in enumerate(class_names):
            vals = ", ".join(str(int(v)) for v in protos_q[i])
            lines.append(f"  /* {name} */ {{ {vals} }},")
        lines.append("};")

        fw_path.write_text("\n".join(lines))
        print(f"[PrototypicalNetwork] Firmware header written to {fw_path}")
        return fw_path


def main() -> None:
    """Demonstrate PrototypicalNetwork few-shot classification."""
    from data.sample_data import generate_dataset_xy, CLASS_NAMES
    from pipeline.ingest import process_dataset
    from pipeline.features_xy import extract_y_features, extract_all_representations

    print("[prototypical] Generating synthetic data ...")
    raw_x, raw_y, labels = generate_dataset_xy(
        n_clips_per_class=15, n_classes=5, cache=False,
    )
    x_proc = process_dataset(raw_x)
    y_proc = extract_y_features(raw_y)
    reps = extract_all_representations(x_proc, y_proc)

    # Build support set: 5 examples per class
    support_set: dict[str, NDArray] = {}
    test_clips: list[NDArray] = []
    test_labels: list[str] = []

    for cls, name in enumerate(CLASS_NAMES):
        mask = labels == cls
        class_clips = reps["x_only"][mask]
        support_set[name] = class_clips[:5]
        for clip in class_clips[5:]:
            test_clips.append(clip)
            test_labels.append(name)

    print(f"[prototypical] Support set: 5 examples × 5 classes")
    print(f"[prototypical] Test set: {len(test_clips)} clips")

    # Create network (no pre-trained encoder — uses PCA)
    net = PrototypicalNetwork(encoder=None, n_pca_components=32)
    net.build_prototypes(support_set)

    # Evaluate
    preds = net.classify_batch(np.stack(test_clips, axis=0))
    correct = sum(p == t for p, t in zip(preds, test_labels))
    acc = correct / len(test_labels)
    print(f"\nFew-shot accuracy (5 support examples, PCA encoder): "
          f"{acc:.4f} ({correct}/{len(test_labels)})")

    # Demonstrate online update
    net.update_prototype("sine", reps["x_only"][labels == 0][5])
    print("\n[prototypical] Prototype updated with 1 new example (online update)")

    # Export firmware header
    net.export_firmware_header()
    print("[prototypical] Done.")


if __name__ == "__main__":
    main()
