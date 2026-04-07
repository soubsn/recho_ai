"""
Model D — CNN using instantaneous phase angle theta(t) = arctan2(y, x).

Input:  [batch, 200, 100, 1] — unwrapped phase angle feature map
Output: [batch, n_classes]

The instantaneous phase encodes the angular position of the spinning dot
on the limit cycle. It captures rotational information that neither x(t)
nor y(t) captures alone — specifically the timing and frequency structure
of perturbations, independently of their amplitude.

Unwrapped phase is a monotonically evolving signal (mostly) whose local
rate of change reveals instantaneous frequency deviations caused by the
input. This is complementary to Model C (orbit radius), which captures
amplitude perturbations.

If Model D ≈ Model A: phase angle alone carries sufficient classification signal.
If Model C + Model D ≈ Model B: x and y together are no better than their
  derived representations (amplitude + phase).

Comparison target: Model A — does rotational phase carry independent information?

CMSIS-NN note: identical architecture to Model A, single channel input.
Input is scaled to uint8 via global min-max normalisation of unwrapped angle.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential


INPUT_HEIGHT: int = 200   # time steps — multiple of 4
INPUT_WIDTH: int = 100    # virtual nodes — multiple of 4
INPUT_CHANNELS: int = 1   # single channel: instantaneous phase angle theta(t)


def build_model(n_classes: int = 5) -> Sequential:
    """
    Build Model D — phase angle CNN.

    Args:
        n_classes: number of output classes.

    Returns:
        Uncompiled Keras Sequential model.
    """
    model = Sequential([
        keras.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)),

        # --- Block 1 ---
        # arm_convolve_s8() — processes instantaneous phase theta(t)=arctan2(y,x)
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
    ], name="model_d_cnn_angle")

    return model
