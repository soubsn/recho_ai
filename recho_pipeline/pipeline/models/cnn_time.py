"""
Time-only CNN — variant (a) of recommendation #7.

Same trunk as pipeline.models.cnn_regularized but with (3, 1) conv kernels
instead of (3, 3). Convolutions never mix across the virtual-node axis;
each of the 100 virtual-node columns is processed as an independent 1D
signal through the same shared conv weights.

Rationale: the (200, 100) reshape in ingest.process_clip puts continuous
time along the virtual-node axis (adjacent columns = 1-sample apart) and
coarse time along the time-step axis (adjacent rows = 100 samples apart).
A (3, 3) kernel mixes the two scales. If that mixing is hurting more than
helping, a (3, 1) kernel removes it.

Pooling still uses (2, 2) so the fine-time axis eventually collapses —
pooling is a non-learned reduction, so it carries no cross-axis weight
learning.

CMSIS-NN compatibility is identical to cnn_regularized:
  - arm_convolve_s8 handles rectangular kernels natively
  - arm_max_pool_s8 handles (2, 2) as before
  - arm_avgpool_s8 handles the GAP head
  - All output channel counts (32, 64) remain multiples of 4
"""

from __future__ import annotations

from tensorflow import keras
from keras import layers, Sequential


INPUT_HEIGHT: int = 200
INPUT_WIDTH: int = 100
INPUT_CHANNELS: int = 1

DROPOUT_RATE: float = 0.5
DENSE_UNITS: int = 64


def build_model(n_classes: int = 5) -> Sequential:
    """Time-only CNN: (3, 1) kernels + GAP head + dropout."""
    model = Sequential([
        keras.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)),

        layers.Rescaling(1.0 / 255.0, name="rescale_input"),

        # --- Block 1 ---
        layers.Conv2D(
            32, (3, 1), padding="same", activation="relu",
            use_bias=True, name="conv1_arm_convolve_s8",
        ),
        layers.Conv2D(
            32, (3, 1), padding="same", activation="relu",
            use_bias=True, name="conv2_arm_convolve_s8",
        ),
        layers.MaxPool2D((2, 2), name="pool1_arm_max_pool_s8"),

        # --- Block 2 ---
        layers.Conv2D(
            64, (3, 1), padding="same", activation="relu",
            use_bias=True, name="conv3_arm_convolve_s8",
        ),
        layers.Conv2D(
            64, (3, 1), padding="same", activation="relu",
            use_bias=True, name="conv4_arm_convolve_s8",
        ),
        layers.MaxPool2D((2, 2), name="pool2_arm_max_pool_s8"),

        layers.GlobalAveragePooling2D(name="gap_arm_avgpool_s8"),

        layers.Dense(
            DENSE_UNITS, activation="relu",
            use_bias=True, name="dense1_arm_fully_connected_s8",
        ),
        layers.Dropout(DROPOUT_RATE, name="dropout_head"),

        layers.Dense(
            n_classes, activation="softmax",
            use_bias=True, name="output_arm_softmax_s8",
        ),
    ])

    return model
