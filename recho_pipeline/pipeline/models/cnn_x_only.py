"""
Model A — CNN baseline using x(t) only.

Replicates the classification architecture from:
  "Hopf physical reservoir computer for reconfigurable sound recognition"
  (Shougat et al., Scientific Reports 2023) — paper 2.

Input:  [batch, 200, 100, 1] — x(t) feature map only
Output: [batch, n_classes]   — class probabilities

This is the reference model. All other models are compared against it
to measure what y(t) and derived representations contribute.

CMSIS-NN compatibility:
  - Only Conv2D, MaxPool2D, Flatten, Dense — no unsupported ops
  - Only ReLU activations (maps to arm_relu_s8(), fused into conv output)
  - Bias enabled on all layers (required by cmsis_nn_conv_params)
  - All tensor dimensions are multiples of 4 (SIMD alignment)
  - Channels-last layout: NHWC
  - No batch normalisation
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential


INPUT_HEIGHT: int = 200   # time steps — multiple of 4
INPUT_WIDTH: int = 100    # virtual nodes — multiple of 4
INPUT_CHANNELS: int = 1   # single channel: x(t) only


def build_model(n_classes: int = 5) -> Sequential:
    """
    Build Model A — x(t)-only CNN baseline.

    Identical architecture to pipeline/model.py but defined here for
    uniform handling by train_all.py.

    Args:
        n_classes: number of output classes.

    Returns:
        Uncompiled Keras Sequential model.
    """
    model = Sequential([
        keras.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)),

        # --- Block 1 ---
        # arm_convolve_s8() — 2D INT8 convolution with fused ReLU
        # CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c
        layers.Conv2D(
            32, (3, 3), padding="same", activation="relu",
            use_bias=True,
            name="conv1_arm_convolve_s8",
        ),

        # arm_convolve_s8()
        # CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c
        layers.Conv2D(
            32, (3, 3), padding="same", activation="relu",
            use_bias=True,
            name="conv2_arm_convolve_s8",
        ),

        # arm_max_pool_s8() — output: (100, 50, 32)
        # CMSIS-NN/Source/PoolingFunctions/arm_max_pool_s8.c
        layers.MaxPool2D((2, 2), name="pool1_arm_max_pool_s8"),

        # --- Block 2 ---
        # arm_convolve_s8()
        # CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c
        layers.Conv2D(
            64, (3, 3), padding="same", activation="relu",
            use_bias=True,
            name="conv3_arm_convolve_s8",
        ),

        # arm_convolve_s8()
        # CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c
        layers.Conv2D(
            64, (3, 3), padding="same", activation="relu",
            use_bias=True,
            name="conv4_arm_convolve_s8",
        ),

        # arm_max_pool_s8() — output: (50, 25, 64)
        # CMSIS-NN/Source/PoolingFunctions/arm_max_pool_s8.c
        layers.MaxPool2D((2, 2), name="pool2_arm_max_pool_s8"),

        # arm_reshape_s8() — no-op pointer reinterpret; flattened: 50*25*64 = 80,000
        # CMSIS-NN: reshape has no kernel; it is a view of existing memory
        layers.Flatten(name="flatten_arm_reshape_s8"),

        # arm_fully_connected_s8() — 128 units (multiple of 4)
        # CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.c
        layers.Dense(128, activation="relu", use_bias=True,
                     name="dense1_arm_fully_connected_s8"),

        # arm_fully_connected_s8() + arm_softmax_s8()
        # CMSIS-NN/Source/SoftmaxFunctions/arm_softmax_s8.c
        layers.Dense(n_classes, activation="softmax", use_bias=True,
                     name="output_arm_softmax_s8"),
    ], name="model_a_cnn_x_only")

    return model
