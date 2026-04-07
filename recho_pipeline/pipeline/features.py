"""
Feature extraction for the Hopf reservoir computer pipeline.

Reshapes the processed x(t) stream into a 200 (time) x 100 (virtual nodes)
feature grid per clip, matching the reservoir state map format from:
  "Hopf physical reservoir computer for reconfigurable sound recognition"
  (Shougat et al., Scientific Reports 2023) — figure 2

Scales values to uint8 [0, 255] for INT8 CMSIS-NN input compatibility.

CMSIS-NN NOTE: The uint8 [0, 255] scaling here maps directly to the input
quantisation range expected by arm_convolve_s8(). During TFLite INT8
conversion, the converter will compute the zero-point and scale factor
to map this uint8 range to the int8 range [-128, 127] used by CMSIS-NN
kernels internally.
  — CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray


N_TIME_STEPS: int = 200
N_VIRTUAL_NODES: int = 100


def scale_to_uint8(feature_maps: NDArray[np.float64]) -> NDArray[np.uint8]:
    """
    Scale float feature maps to [0, 255] uint8.

    CMSIS-NN NOTE: arm_convolve_s8() expects int8 inputs. The TFLite
    converter handles the uint8-to-int8 offset (subtract 128) via
    the input zero-point parameter in cmsis_nn_conv_params.
    """
    fmin = feature_maps.min()
    fmax = feature_maps.max()
    if fmax - fmin < 1e-12:
        return np.zeros(feature_maps.shape, dtype=np.uint8)
    scaled = (feature_maps - fmin) / (fmax - fmin) * 255.0
    return np.round(scaled).astype(np.uint8)


def extract_features(
    processed_clips: NDArray[np.float64],
    labels: NDArray[np.int64],
    output_dir: Optional[str | Path] = None,
) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
    """
    Convert processed clips to uint8 feature maps and optionally save to disk.

    Args:
        processed_clips: shape (n_clips, 200, 100) from ingest.process_dataset()
        labels: shape (n_clips,) integer class labels
        output_dir: if provided, save .npy files here

    Returns:
        feature_maps: shape (n_clips, 200, 100) uint8
        labels: pass-through
    """
    assert processed_clips.shape[1:] == (N_TIME_STEPS, N_VIRTUAL_NODES), (
        f"Expected shape (n, {N_TIME_STEPS}, {N_VIRTUAL_NODES}), "
        f"got {processed_clips.shape}"
    )

    feature_maps = scale_to_uint8(processed_clips)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "feature_maps.npy", feature_maps)
        np.save(out / "labels.npy", labels)
        print(f"[features] Saved {feature_maps.shape[0]} maps to {out}")

    return feature_maps, labels


def visualise_features(
    feature_maps: NDArray[np.uint8],
    labels: NDArray[np.int64],
    class_names: list[str],
    output_dir: Optional[str | Path] = None,
) -> None:
    """
    Plot one representative feature map per class as a grayscale image.

    Each image shows the 200 (time) x 100 (virtual nodes) reservoir state map.
    """
    import matplotlib.pyplot as plt

    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]

    for cls in range(n_classes):
        mask = labels == cls
        if not np.any(mask):
            continue
        idx = np.where(mask)[0][0]
        ax = axes[cls]
        ax.imshow(feature_maps[idx], cmap="gray", aspect="auto", vmin=0, vmax=255)
        ax.set_title(f"Class {cls}: {class_names[cls]}")
        ax.set_xlabel("Virtual node index")
        ax.set_ylabel("Time step")

    plt.tight_layout()

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "feature_maps_overview.png", dpi=150)
        print(f"[features] Saved visualisation to {out / 'feature_maps_overview.png'}")
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    """Run feature extraction on synthetic data."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.sample_data import generate_dataset, CLASS_NAMES
    from pipeline.ingest import process_dataset

    print("[features] Generating small synthetic dataset ...")
    raw_x, labels = generate_dataset(n_clips_per_class=5, n_classes=5, cache=False)
    processed = process_dataset(raw_x)

    out_dir = Path(__file__).resolve().parent.parent / "output" / "features"
    feature_maps, labels = extract_features(processed, labels, output_dir=out_dir)
    print(f"  Feature maps: shape={feature_maps.shape}, dtype={feature_maps.dtype}")
    print(f"  Range: [{feature_maps.min()}, {feature_maps.max()}]")

    visualise_features(feature_maps, labels, CLASS_NAMES, output_dir=out_dir)


if __name__ == "__main__":
    main()
