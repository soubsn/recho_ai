"""
Contrastive Learning Classifier — Few-Shot Reconfigurable Classification.

Learns an embedding space where same-class samples cluster together without
needing class labels during representation learning. After pretraining, new
classes can be added with just 5-10 examples by computing their embedding.

This is the RECHO reconfigurability use case:
    1. Pretrain encoder once (unsupervised, on all available clips)
    2. Deploy to machine
    3. User records 5 examples of a new fault condition
    4. Compute mean embedding (prototype) of the 5 examples
    5. Classification = nearest prototype in embedding space
    No retraining needed.

Architecture:
    Shared encoder: same CNN backbone as cnn_x_only.py
    Projects to 128-dimensional embedding space (multiple of 4)
    Loss: NT-Xent (normalised temperature-scaled cross entropy)
    Positive pairs: two augmented versions of the same clip
    Augmentations:
        time shift: random roll of ±20 rows in the 200-time-step axis
        amplitude scale: multiply by uniform(0.9, 1.1)
        additive noise: Gaussian σ=0.01

Why contrastive learning for RECHO:
    Most relevant for the EchoReveal "train in hours" product vision.
    A factory installs the sensor and it pretains overnight on unlabelled data.
    The next morning, an engineer labels 5 clips per condition.
    Classification starts immediately — no GPU needed for the fine-tuning step.

Keras checkpoint: checkpoints/contrastive_encoder.keras

Reference:
  Chen, T. et al. (2020) A Simple Framework for Contrastive Learning (SimCLR).
  ICML 2020.
  Shougat et al., Scientific Reports 2023 (paper 2) — reconfigurability as
  core feature: "trained in a few hours on laptop hardware."
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
EMBEDDING_DIM: int = 128   # multiple of 4
INPUT_HEIGHT: int = 200
INPUT_WIDTH: int = 100


def _augment(clip: NDArray[np.float32], rng: np.random.Generator) -> NDArray[np.float32]:
    """
    Apply random augmentation to a single feature map clip.

    Augmentations that preserve the Hopf oscillator's physical content:
        time shift ±20 rows: realistic — the oscillator's phase can vary
        amplitude scale ±10%: realistic — sensor gain variation
        additive noise σ=0.01: realistic — ADC noise floor

    Args:
        clip: (200, 100) or (200, 100, 1) float32 feature map
        rng: numpy random generator

    Returns:
        Augmented clip, same shape as input.
    """
    x = clip.copy()
    # Time shift: random roll along time axis (axis=0)
    shift = int(rng.integers(-20, 21))
    x = np.roll(x, shift, axis=0)
    # Amplitude scale
    scale = float(rng.uniform(0.9, 1.1))
    x = x * scale
    # Additive Gaussian noise
    noise = rng.standard_normal(x.shape).astype(np.float32) * 0.01
    x = x + noise
    return np.clip(x, 0.0, 1.0)


def build_encoder(
    input_shape: tuple = (INPUT_HEIGHT, INPUT_WIDTH, 1),
    embedding_dim: int = EMBEDDING_DIM,
):
    """
    Build the CNN encoder with a projection head.

    Backbone matches cnn_x_only.py for compatibility with pre-trained weights.
    Final layer projects to 128-dim embedding space.

    Args:
        input_shape: (H, W, C) input tensor shape.
        embedding_dim: size of the embedding vector.

    Returns:
        Keras Model that maps input clips to embedding vectors.
    """
    import tensorflow as tf
    from tensorflow import keras

    inp = keras.Input(shape=input_shape)

    # Backbone — same as cnn_x_only.py
    # arm_convolve_s8()
    # CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c
    x = keras.layers.Conv2D(
        32, (3, 3), padding="same", activation="relu",
        use_bias=True, name="enc_conv1_arm_convolve_s8",
    )(inp)
    x = keras.layers.Conv2D(
        32, (3, 3), padding="same", activation="relu",
        use_bias=True, name="enc_conv2_arm_convolve_s8",
    )(x)
    # arm_max_pool_s8()
    x = keras.layers.MaxPool2D((2, 2), name="enc_pool1_arm_max_pool_s8")(x)

    x = keras.layers.Conv2D(
        64, (3, 3), padding="same", activation="relu",
        use_bias=True, name="enc_conv3_arm_convolve_s8",
    )(x)
    x = keras.layers.Conv2D(
        64, (3, 3), padding="same", activation="relu",
        use_bias=True, name="enc_conv4_arm_convolve_s8",
    )(x)
    x = keras.layers.MaxPool2D((2, 2), name="enc_pool2_arm_max_pool_s8")(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(
        128, activation="relu", use_bias=True,
        name="enc_dense1_arm_fully_connected_s8",
    )(x)

    # Projection head — L2 normalised embedding
    # arm_fully_connected_s8()
    embedding = keras.layers.Dense(
        embedding_dim, use_bias=True,
        name="embedding_arm_fully_connected_s8",
    )(x)
    embedding = keras.layers.Lambda(
        lambda v: v / (tf.norm(v, axis=1, keepdims=True) + 1e-12),
        name="l2_normalise",
    )(embedding)

    return keras.Model(inputs=inp, outputs=embedding, name="contrastive_encoder")


class ContrastiveClassifier:
    """
    Contrastive learning encoder with few-shot prototype classification.

    Pretrain unsupervised with NT-Xent loss, then classify by nearest
    prototype in embedding space with as few as 5 examples per class.

    This is the RECHO reconfigurability model: pretrain once, retrain
    the classification head in minutes with just 5 new examples.

    Example:
        clf = ContrastiveClassifier()
        clf.pretrain(all_clips)
        support = {"normal": clips_normal[:5], "fault_a": clips_fault[:5]}
        clf.build_prototypes(support)
        label = clf.few_shot_classify(new_clip)
    """

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        temperature: float = 0.5,
        epochs: int = 30,
        batch_size: int = 32,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.epochs = epochs
        self.batch_size = batch_size
        self._encoder = None
        self._prototypes: dict[str, NDArray[np.float32]] = {}
        self._rng = np.random.default_rng(42)
        self._is_pretrained = False

    def _prepare(self, features: NDArray) -> NDArray[np.float32]:
        """Scale uint8 → [0,1] float32 with channel dim."""
        x = features.astype(np.float32) / 255.0
        if x.ndim == 3:
            x = x[:, :, :, np.newaxis]
        return x

    def pretrain(self, all_clips: NDArray) -> "ContrastiveClassifier":
        """
        Pretrain encoder with NT-Xent contrastive loss (unsupervised).

        For each clip, generates two augmented views. The loss pushes
        same-clip views together and different-clip views apart.

        Args:
            all_clips: (n_clips, 200, 100) uint8 feature maps (no labels needed)

        Returns:
            self
        """
        import tensorflow as tf
        from tensorflow import keras

        self._encoder = build_encoder(
            embedding_dim=self.embedding_dim,
        )

        X = self._prepare(all_clips)
        n = len(X)
        temp = self.temperature

        # Simplified contrastive training — generate augmented pairs on the fly
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        @tf.function
        def nt_xent_loss(z_i, z_j):
            """NT-Xent loss for a batch of positive pairs (z_i, z_j)."""
            z = tf.concat([z_i, z_j], axis=0)
            # Cosine similarity matrix
            sim = tf.matmul(z, z, transpose_b=True) / temp
            batch = tf.shape(z_i)[0]
            # Labels: positive pair = (i, batch+i) and vice versa
            labels = tf.concat([
                tf.range(batch, 2 * batch),
                tf.range(batch),
            ], axis=0)
            # Mask out self-similarities
            mask = 1.0 - tf.eye(2 * batch)
            sim = sim * mask - 1e9 * (1.0 - mask)
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=sim,
                )
            )
            return loss

        print(f"[ContrastiveClassifier] Pretraining on {n} clips for "
              f"{self.epochs} epochs ...")
        indices = np.arange(n)
        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                if len(batch_idx) < 2:
                    continue
                x_batch = X[batch_idx]
                x_aug1 = np.stack([_augment(x, self._rng) for x in x_batch])
                x_aug2 = np.stack([_augment(x, self._rng) for x in x_batch])

                with tf.GradientTape() as tape:
                    z1 = self._encoder(x_aug1, training=True)
                    z2 = self._encoder(x_aug2, training=True)
                    loss = nt_xent_loss(z1, z2)

                grads = tape.gradient(loss, self._encoder.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, self._encoder.trainable_variables)
                )
                epoch_loss += float(loss.numpy())
                n_batches += 1

            if (epoch + 1) % max(1, self.epochs // 5) == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}: "
                      f"loss={epoch_loss/max(n_batches,1):.4f}")

        ckpt = CHECKPOINT_DIR / "contrastive_encoder.keras"
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self._encoder.save(str(ckpt))
        print(f"[ContrastiveClassifier] Encoder saved to {ckpt}")
        self._is_pretrained = True
        return self

    def build_prototypes(self, support_set: dict[str, NDArray]) -> None:
        """
        Build class prototypes from support set examples.

        Called after pretraining, with just 5-10 examples per class.
        Prototype = mean embedding of support examples for that class.
        No retraining needed — runs in milliseconds.

        Args:
            support_set: dict mapping class_name → (n_examples, 200, 100) clips
        """
        if not self._is_pretrained:
            raise RuntimeError("Call pretrain() before build_prototypes()")

        self._prototypes = {}
        for class_name, clips in support_set.items():
            X = self._prepare(clips)
            embeddings = self._encoder.predict(X, verbose=0)
            self._prototypes[class_name] = np.mean(embeddings, axis=0)
            print(f"  Prototype '{class_name}': {len(clips)} examples → "
                  f"{EMBEDDING_DIM}-dim embedding")

    def few_shot_classify(self, query_clip: NDArray) -> str:
        """
        Classify a query clip by nearest prototype distance.

        Args:
            query_clip: (200, 100) uint8 feature map to classify

        Returns:
            Class name of nearest prototype.
        """
        if not self._prototypes:
            raise RuntimeError("Call build_prototypes() before few_shot_classify()")

        X = self._prepare(np.expand_dims(query_clip, 0))
        query_emb = self._encoder.predict(X, verbose=0)[0]

        best_class = ""
        best_dist = float("inf")
        for class_name, proto in self._prototypes.items():
            dist = float(np.linalg.norm(query_emb - proto))
            if dist < best_dist:
                best_dist = dist
                best_class = class_name

        return best_class

    def update_prototype(
        self,
        class_name: str,
        new_clip: NDArray,
        momentum: float = 0.9,
    ) -> None:
        """
        Online prototype update using exponential moving average.

        Allows the model to improve as more examples arrive without retraining.
        new_prototype = momentum * old + (1 - momentum) * new_embedding

        Args:
            class_name: class to update
            new_clip: (200, 100) new example clip for this class
            momentum: EMA momentum (default 0.9)
        """
        if class_name not in self._prototypes:
            # New class: create prototype from this one example
            X = self._prepare(np.expand_dims(new_clip, 0))
            self._prototypes[class_name] = self._encoder.predict(X, verbose=0)[0]
            return

        X = self._prepare(np.expand_dims(new_clip, 0))
        new_emb = self._encoder.predict(X, verbose=0)[0]
        old_proto = self._prototypes[class_name]
        self._prototypes[class_name] = (momentum * old_proto
                                         + (1 - momentum) * new_emb)


def main() -> None:
    """Pretrain ContrastiveClassifier and demonstrate few-shot classification."""
    from data.sample_data import generate_dataset_xy, CLASS_NAMES
    from pipeline.ingest import process_dataset
    from pipeline.features_xy import extract_y_features, extract_all_representations

    print("[contrastive] Generating synthetic data ...")
    raw_x, raw_y, labels = generate_dataset_xy(
        n_clips_per_class=10, n_classes=5, cache=False,
    )
    x_proc = process_dataset(raw_x)
    y_proc = extract_y_features(raw_y)
    reps = extract_all_representations(x_proc, y_proc)

    print("[contrastive] Pretraining encoder (unsupervised) ...")
    clf = ContrastiveClassifier(embedding_dim=64, epochs=3, batch_size=16)
    clf.pretrain(reps["x_only"])

    # Build prototypes from 5 examples per class (few-shot)
    support_set = {}
    for cls, name in enumerate(CLASS_NAMES):
        mask = labels == cls
        support_set[name] = reps["x_only"][mask][:5]

    clf.build_prototypes(support_set)

    # Evaluate few-shot classification on held-out clips
    correct = 0
    total = 0
    for cls, name in enumerate(CLASS_NAMES):
        mask = labels == cls
        clips = reps["x_only"][mask][5:]  # use clips not in support set
        for clip in clips:
            pred = clf.few_shot_classify(clip)
            if pred == name:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0.0
    print(f"\nFew-shot accuracy (5 support examples): {acc:.4f} ({correct}/{total})")
    print("[contrastive] Done.")


if __name__ == "__main__":
    main()
