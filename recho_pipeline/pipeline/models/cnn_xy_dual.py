"""
Model B — Two-channel CNN with x(t) and y(t) stacked as input channels.

Input:  [batch, 200, 100, 2] — x(t) and y(t) stacked channel-last
Output: [batch, n_classes]

The first Conv2D layer sees both x and y feature maps simultaneously
and learns cross-channel patterns — relationships between x(t) and y(t)
dynamics at each spatial location. This is the simplest way to incorporate
y(t): zero architectural change beyond increasing input channels from 1 to 2.

Comparison target: Model A (x_only) — does y(t) improve accuracy when
combined at the input level?

CMSIS-NN NOTE: arm_convolve_s8() supports multi-channel input natively.
The input zero_point and scale are per-tensor (not per-channel) for inputs,
so both channels share the same quantisation parameters. Use
scale_dual_channel_to_uint8() from features_xy.py to ensure both channels
have comparable dynamic ranges before training.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential


INPUT_HEIGHT: int = 200   # time steps — multiple of 4
INPUT_WIDTH: int = 100    # virtual nodes — multiple of 4
INPUT_CHANNELS: int = 2   # two channels: x(t) and y(t)


def build_model(n_classes: int = 5) -> Sequential:
    """
    Build Model B — two-channel x+y CNN.

    Same architecture as Model A; only the input channel count changes.

    Args:
        n_classes: number of output classes.

    Returns:
        Uncompiled Keras Sequential model.
    """
    model = Sequential([
        keras.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)),

        # --- Block 1 ---
        # arm_convolve_s8() — processes both x and y channels simultaneously
        # CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c
        # Weights shape: (3, 3, 2, 32) — learns cross-channel x/y patterns
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

        # arm_reshape_s8() — flatten: 50*25*64 = 80,000
        layers.Flatten(name="flatten_arm_reshape_s8"),

        # arm_fully_connected_s8() — 128 units (multiple of 4)
        # CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.c
        layers.Dense(128, activation="relu", use_bias=True,
                     name="dense1_arm_fully_connected_s8"),

        # arm_fully_connected_s8() + arm_softmax_s8()
        # CMSIS-NN/Source/SoftmaxFunctions/arm_softmax_s8.c
        layers.Dense(n_classes, activation="softmax", use_bias=True,
                     name="output_arm_softmax_s8"),
    ], name="model_b_cnn_xy_dual")

    return model
