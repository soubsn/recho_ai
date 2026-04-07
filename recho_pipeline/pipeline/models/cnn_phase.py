"""
Model C — CNN using orbit radius r(t) = sqrt(x^2 + y^2) as input.

Input:  [batch, 200, 100, 1] — orbit radius feature map
Output: [batch, n_classes]

The orbit radius encodes the amplitude of oscillation: how far the
spinning dot is from the centre of the limit cycle at each moment.
When the input signal is strong, the oscillator is pushed off its
natural orbit and r(t) changes significantly. When there is no signal,
r(t) stays near the limit cycle radius sqrt(mu) ≈ 2.24.

This is a pure amplitude measure — it discards rotational direction.
Comparison target: Model A — does amplitude alone carry classification signal?

If Model C ≈ Model A: amplitude dominates over direction.
If Model C << Model A: directional information (x vs y separately) matters.

CMSIS-NN note: identical architecture to Model A, single channel input.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential


INPUT_HEIGHT: int = 200   # time steps — multiple of 4
INPUT_WIDTH: int = 100    # virtual nodes — multiple of 4
INPUT_CHANNELS: int = 1   # single channel: orbit radius r(t)


def build_model(n_classes: int = 5) -> Sequential:
    """
    Build Model C — orbit radius CNN.

    Args:
        n_classes: number of output classes.

    Returns:
        Uncompiled Keras Sequential model.
    """
    model = Sequential([
        keras.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)),

        # --- Block 1 ---
        # arm_convolve_s8() — processes orbit radius r(t) = sqrt(x^2+y^2)
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
    ], name="model_c_cnn_phase")

    return model
