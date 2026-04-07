"""
Model F — Depthwise separable CNN, optimised for Cortex-M33/M55.

Input:  [batch, 200, 100, 2] — x(t) and y(t) as two channels
Output: [batch, n_classes]

Depthwise separable convolutions factoise a standard Conv2D(3x3, C_in→C_out)
into two cheaper operations:
  1. DepthwiseConv2D(3x3): filters each input channel independently — C_in kernels
  2. Conv2D(1x1, C_in→C_out): pointwise projection — mixes channel information

This reduces multiply-accumulate operations by ~8-9x compared to standard
Conv2D, at a modest accuracy cost. Critical for M33 where every cycle counts.

On M33: arm_depthwise_conv_s8() uses DSP SMLAD instruction.
On M55: arm_depthwise_conv_s8() uses Helium vmlaq_s8 intrinsics.

Comparison target: Model A — what is the accuracy cost of the smaller, faster model?
The M33 has 64 KB RAM. If Model A is too large, Model F provides a fallback.

CMSIS-NN compatibility:
  DepthwiseConv2D(3x3) → arm_depthwise_conv_s8()
  Conv2D(1x1)          → arm_convolve_1x1_s8()
  MaxPool2D            → arm_max_pool_s8()
  Dense                → arm_fully_connected_s8()
  Softmax              → arm_softmax_s8()
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
    Build Model F — depthwise separable CNN.

    Args:
        n_classes: number of output classes.

    Returns:
        Uncompiled Keras Sequential model.
    """
    model = Sequential([
        keras.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)),

        # --- Block 1 ---
        # arm_depthwise_conv_s8() — 8-9x fewer MACs than standard Conv2D
        # CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.c
        # On M33: DSP SMLAD instruction; on M55: Helium vmlaq_s8
        # depth_multiplier=1 keeps channel count unchanged
        layers.DepthwiseConv2D(
            (3, 3), padding="same", activation="relu",
            use_bias=True, depth_multiplier=1,
            name="dw_conv1_arm_depthwise_conv_s8",
        ),

        # arm_convolve_1x1_s8() — pointwise projection 2→32 channels
        # CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8.c
        # 1x1 convolutions have a dedicated faster kernel on CMSIS-NN
        layers.Conv2D(
            32, (1, 1), padding="same", activation="relu",
            use_bias=True,
            name="pw_conv1_arm_convolve_1x1_s8",
        ),

        # arm_max_pool_s8() — output: (100, 50, 32)
        # CMSIS-NN/Source/PoolingFunctions/arm_max_pool_s8.c
        layers.MaxPool2D((2, 2), name="pool1_arm_max_pool_s8"),

        # --- Block 2 ---
        # arm_depthwise_conv_s8()
        # CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.c
        layers.DepthwiseConv2D(
            (3, 3), padding="same", activation="relu",
            use_bias=True, depth_multiplier=1,
            name="dw_conv2_arm_depthwise_conv_s8",
        ),

        # arm_convolve_1x1_s8() — pointwise 32→64 channels
        # CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8.c
        layers.Conv2D(
            64, (1, 1), padding="same", activation="relu",
            use_bias=True,
            name="pw_conv2_arm_convolve_1x1_s8",
        ),

        # arm_max_pool_s8() — output: (50, 25, 64)
        # CMSIS-NN/Source/PoolingFunctions/arm_max_pool_s8.c
        layers.MaxPool2D((2, 2), name="pool2_arm_max_pool_s8"),

        # arm_reshape_s8() — flatten: 50*25*64 = 80,000
        layers.Flatten(name="flatten_arm_reshape_s8"),

        # arm_fully_connected_s8() — 64 units (multiple of 4)
        # Smaller than Model A (128) — further reduces RAM and latency on M33
        # CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.c
        layers.Dense(64, activation="relu", use_bias=True,
                     name="dense1_arm_fully_connected_s8"),

        # arm_fully_connected_s8() + arm_softmax_s8()
        # CMSIS-NN/Source/SoftmaxFunctions/arm_softmax_s8.c
        layers.Dense(n_classes, activation="softmax", use_bias=True,
                     name="output_arm_softmax_s8"),
    ], name="model_f_depthwise_cnn")

    return model
