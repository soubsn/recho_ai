"""
CNN model architecture for Hopf reservoir computer classification.

Every layer is chosen to map directly to a named CMSIS-NN kernel function,
ensuring the entire inference graph runs on optimised Arm assembly — not
generic C fallbacks.

CMSIS-NN compatibility constraints:
  - Only ReLU activations (maps to arm_relu_s8)
  - Only Conv2D, MaxPool2D, Flatten, Dense
  - All tensor dimensions multiples of 4 (SIMD alignment)
  - Channels-last format: (200, 100, 1)
  - No batch normalisation
  - Bias enabled on all Conv2D and Dense layers (required by CMSIS-NN kernels)

Reference: CMSIS-NN/Source/ — arm_convolve_s8.c, arm_max_pool_s8.c,
           arm_fully_connected_s8.c, arm_softmax_s8.c, arm_relu_s8.c
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential


# Feature map dimensions from the reservoir — must be multiples of 4
INPUT_HEIGHT: int = 200   # time steps (200 = 8 * 25, multiple of 4)
INPUT_WIDTH: int = 100    # virtual nodes (100 = 4 * 25, multiple of 4)
INPUT_CHANNELS: int = 1   # single-channel grayscale


def build_model(n_classes: int = 5) -> Sequential:
    """
    Build the CMSIS-NN-compatible CNN.

    Each layer comment identifies the exact CMSIS-NN kernel it maps to
    at inference time on Arm Cortex-M targets.

    Args:
        n_classes: number of output classes for the softmax layer.

    Returns:
        Uncompiled Keras Sequential model.
    """
    model = Sequential([
        keras.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)),

        # Scale uint8 [0, 255] feature maps to [0, 1] for stable FP32
        # training. The TFLite converter folds this scale into the input
        # quantization params, so INT8 inference still accepts uint8
        # feature maps — no runtime preprocessing needed on the MCU.
        layers.Rescaling(1.0 / 255.0, name="rescale_input"),

        # --- Block 1 ---
        # arm_convolve_s8() — 2D convolution, INT8 in/out
        # CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c
        # On M33: uses DSP SIMD via SMLAD instruction
        # On M55: auto-detects Helium MVE, uses vmlaq_s8 intrinsics
        # On M85+Ethos-U55: offloaded to NPU via Vela compiler
        layers.Conv2D(
            32, (3, 3), padding="same", activation="relu",
            use_bias=True,  # required by cmsis_nn_conv_params
            name="conv1_arm_convolve_s8",
        ),
        # arm_relu_s8() — fused into convolution output by CMSIS-NN

        # arm_convolve_s8()
        # CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c
        layers.Conv2D(
            32, (3, 3), padding="same", activation="relu",
            use_bias=True,
            name="conv2_arm_convolve_s8",
        ),

        # arm_max_pool_s8() — 2D max pooling INT8
        # CMSIS-NN/Source/PoolingFunctions/arm_max_pool_s8.c
        # Output: (100, 50, 32) — all dims multiples of 2
        layers.MaxPool2D(
            (2, 2),
            name="pool1_arm_max_pool_s8",
        ),

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

        # arm_max_pool_s8() — 2D max pooling INT8
        # CMSIS-NN/Source/PoolingFunctions/arm_max_pool_s8.c
        # Output: (50, 25, 64) — all dims suitable for CMSIS-NN
        layers.MaxPool2D(
            (2, 2),
            name="pool2_arm_max_pool_s8",
        ),

        #OLD
        # arm_reshape_s8() — flatten to 1D vector
        # CMSIS-NN: reshape is a no-op (pointer reinterpret), no kernel needed
        # Flattened size: 50 * 25 * 64 = 80,000
        # layers.Flatten(name="flatten_arm_reshape_s8"),


        # arm_max_pool_s8() — aggressive spatial reduction before flatten.
        # (50, 25, 64) → (10, 5, 64) = 3,200 features. Keeps some spatial
        # structure (GAP threw too much away — model wouldn't learn) while
        # cutting dense1 25× vs. the raw 80,000-element flatten.
        layers.MaxPool2D((5, 5), name="pool3_arm_max_pool_s8"),

        # arm_reshape_s8() — flatten to 1D vector (pointer reinterpret).
        # Flattened size: 10 * 5 * 64 = 3,200.
        layers.Flatten(name="flatten_arm_reshape_s8"),

        # arm_fully_connected_s8() — dense layer INT8
        # CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.c
        # 128 output units (multiple of 4 for SIMD alignment)
        layers.Dense(
            128, activation="relu",
            use_bias=True,
            name="dense1_arm_fully_connected_s8",
        ),

        # arm_fully_connected_s8() + arm_softmax_s8()
        # CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.c
        # CMSIS-NN/Source/SoftmaxFunctions/arm_softmax_s8.c
        layers.Dense(
            n_classes, activation="softmax",
            use_bias=True,
            name="output_arm_softmax_s8",
        ),
    ])

    return model


def print_model_summary(model: Sequential) -> None:
    """Print model summary with per-layer CMSIS-NN kernel mapping."""
    model.summary()

    print("\n--- CMSIS-NN Kernel Map ---")
    for layer in model.layers:
        name = layer.name
        out_shape = layer.output.shape
        params = layer.count_params()

        if "convolve" in name or "conv" in name:
            kernel = "arm_convolve_s8()"
        elif "pool" in name:
            kernel = "arm_max_pool_s8()"
        elif "flatten" in name:
            kernel = "arm_reshape_s8() [no-op]"
        elif "fully_connected" in name or "dense" in name:
            kernel = "arm_fully_connected_s8()"
        elif "softmax" in name:
            kernel = "arm_fully_connected_s8() + arm_softmax_s8()"
        else:
            kernel = "UNKNOWN — check CMSIS-NN coverage!"

        print(f"  {name:40s} -> {str(out_shape):20s} [{params:>8,} params] => {kernel}")


def main() -> None:
    """Build and display the model."""
    model = build_model(n_classes=5)
    print_model_summary(model)


if __name__ == "__main__":
    main()
