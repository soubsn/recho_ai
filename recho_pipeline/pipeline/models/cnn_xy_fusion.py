"""
Model E — Late-fusion CNN with separate x(t) and y(t) branches.

Input:  [input_x: batch, 200, 100, 1] and [input_y: batch, 200, 100, 1]
Output: [batch, n_classes]

Two independent CNN branches learn separate representations of x(t)
and y(t) before merging at a concatenation layer. This allows each
branch to develop specialised filters tuned to its input modality,
giving the model more capacity than simple channel stacking (Model B).

Architecture:
  Branch X: Conv32 → Conv32 → MaxPool → Conv64 → Flatten
  Branch Y: Conv32 → Conv32 → MaxPool → Conv64 → Flatten
  Merge:    Concatenate([x_branch, y_branch])
  Head:     Dense(128) → Dense(n_classes, softmax)

Comparison target: Model B — is late fusion better than early channel stacking?
Late fusion: x and y learn independent representations then combine.
Early fusion (Model B): joint patterns learned from the first layer onward.

CMSIS-NN NOTE:
  - arm_concatenation_s8() handles the merge layer
    CMSIS-NN/Source/ConcatenationFunctions/arm_concatenation_s8.c
  - TFLite supports multiple inputs natively
  - On M85 dual-core: buffer_x[200*100] and buffer_y[200*100] can be filled
    simultaneously via dual DMA channels from the Hopf oscillator hardware
  - Each input tensor gets its own quantisation parameters (separate zero-points
    and scales for x and y branches)

This is a functional API model, not Sequential.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, Input


INPUT_HEIGHT: int = 200   # time steps — multiple of 4
INPUT_WIDTH: int = 100    # virtual nodes — multiple of 4
INPUT_CHANNELS: int = 1   # each branch is single-channel


def _build_branch(inp: tf.Tensor, prefix: str) -> tf.Tensor:
    """Build one CNN branch (x or y) and return the flattened feature vector."""
    # arm_convolve_s8()
    # CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c
    x = layers.Conv2D(
        32, (3, 3), padding="same", activation="relu", use_bias=True,
        name=f"{prefix}_conv1_arm_convolve_s8",
    )(inp)

    # arm_convolve_s8()
    # CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c
    x = layers.Conv2D(
        32, (3, 3), padding="same", activation="relu", use_bias=True,
        name=f"{prefix}_conv2_arm_convolve_s8",
    )(x)

    # arm_max_pool_s8() — output: (100, 50, 32)
    # CMSIS-NN/Source/PoolingFunctions/arm_max_pool_s8.c
    x = layers.MaxPool2D((2, 2), name=f"{prefix}_pool1_arm_max_pool_s8")(x)

    # arm_convolve_s8()
    # CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c
    x = layers.Conv2D(
        64, (3, 3), padding="same", activation="relu", use_bias=True,
        name=f"{prefix}_conv3_arm_convolve_s8",
    )(x)

    # arm_reshape_s8() — flatten branch output: 100*50*64 = 320,000
    # (per branch before merge — large but symmetric)
    x = layers.Flatten(name=f"{prefix}_flatten_arm_reshape_s8")(x)
    return x


def build_model(n_classes: int = 5) -> Model:
    """
    Build Model E — late fusion x+y CNN.

    Args:
        n_classes: number of output classes.

    Returns:
        Uncompiled Keras functional Model with two inputs.
    """
    input_x = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS),
                    name="input_x")
    input_y = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS),
                    name="input_y")

    branch_x = _build_branch(input_x, prefix="x")
    branch_y = _build_branch(input_y, prefix="y")

    # arm_concatenation_s8() — concatenate branch feature vectors
    # CMSIS-NN/Source/ConcatenationFunctions/arm_concatenation_s8.c
    # On M85: buffer_x and buffer_y fed simultaneously via dual DMA
    merged = layers.Concatenate(name="merge_arm_concatenation_s8")([branch_x, branch_y])

    # arm_fully_connected_s8() — 128 units (multiple of 4)
    # CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.c
    out = layers.Dense(128, activation="relu", use_bias=True,
                       name="dense1_arm_fully_connected_s8")(merged)

    # arm_fully_connected_s8() + arm_softmax_s8()
    # CMSIS-NN/Source/SoftmaxFunctions/arm_softmax_s8.c
    out = layers.Dense(n_classes, activation="softmax", use_bias=True,
                       name="output_arm_softmax_s8")(out)

    return Model(inputs=[input_x, input_y], outputs=out,
                 name="model_e_cnn_xy_fusion")
