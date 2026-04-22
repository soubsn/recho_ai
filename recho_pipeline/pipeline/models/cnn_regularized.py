"""
Regularized x(t)-only CNN for pretraining on the 50-class ESC-50 task.

Two changes vs pipeline.model.build_model:
  1. Flatten + Dense(128) replaced with GlobalAveragePooling2D + Dense(64).
     Kills ~405K of the 481K parameters — the giant FC layer was the
     overfit engine when training on 32 samples/class.
  2. Dropout(0.5) inserted before the final softmax. Train-only, no
     deploy cost.

Both changes remain CMSIS-NN compatible:
  - GlobalAveragePooling2D -> arm_avgpool_s8
  - Dropout -> removed at TFLite conversion time

Everything else mirrors pipeline.model — same conv stack, same input
shape, same channel counts, same naming scheme, same use_bias flags.
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
    """Regularized CNN: GAP head + dropout. Same conv trunk as pipeline.model."""
    model = Sequential([
        keras.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)),

        layers.Rescaling(1.0 / 255.0, name="rescale_input"),

        # --- Block 1 ---
        layers.Conv2D(
            32, (3, 3), padding="same", activation="relu",
            use_bias=True, name="conv1_arm_convolve_s8",
        ),
        layers.Conv2D(
            32, (3, 3), padding="same", activation="relu",
            use_bias=True, name="conv2_arm_convolve_s8",
        ),
        layers.MaxPool2D((2, 2), name="pool1_arm_max_pool_s8"),

        # --- Block 2 ---
        layers.Conv2D(
            64, (3, 3), padding="same", activation="relu",
            use_bias=True, name="conv3_arm_convolve_s8",
        ),
        layers.Conv2D(
            64, (3, 3), padding="same", activation="relu",
            use_bias=True, name="conv4_arm_convolve_s8",
        ),
        layers.MaxPool2D((2, 2), name="pool2_arm_max_pool_s8"),

        # GlobalAveragePooling2D collapses (50, 25, 64) -> (64,). Maps to
        # arm_avgpool_s8 at deploy time. Replaces the 405K-param Flatten +
        # Dense(128) with a parameter-free reduction + a 64x64 FC.
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
