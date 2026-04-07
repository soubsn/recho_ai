"""
Extended feature extraction for both x(t) and y(t) Hopf oscillator states.

The published papers (Shougat et al. 2021, 2023) use only x(t) as the reservoir
readout and discard y(t), noting it "likely stores information." This module
processes y(t) through the same pipeline as x(t) and computes derived
representations — orbit radius and phase angle — that expose information
encoded in the joint (x, y) dynamics.

Feature representations produced here:
  - y_features:   y(t) processed identically to x(t) — shape (n, 200, 100)
  - phase:        r(t) = sqrt(x^2 + y^2) — orbit radius, shape (n, 200, 100)
  - angle:        theta(t) = arctan2(y, x) — instantaneous phase, (n, 200, 100)
  - xy_dual:      x and y stacked as two channels — shape (n, 200, 100, 2)

All float outputs are in float64. Call pipeline.features.scale_to_uint8() to
convert single-channel representations to uint8 [0, 255] for CMSIS-NN input.

CMSIS-NN NOTE: arm_convolve_s8() supports multi-channel input natively.
The dual-channel input is passed directly to the first Conv2D layer, which
learns cross-channel patterns across x(t) and y(t) feature maps simultaneously.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray


# Feature map dimensions — must match ingest.py
N_TIME_STEPS: int = 200
N_VIRTUAL_NODES: int = 100


def extract_y_features(
    y_stream_raw: NDArray[np.float64],
    clip_length_s: float = 1.0,
    original_sr: int = 100_000,
    target_sr: int = 4_000,
    n_time: int = N_TIME_STEPS,
    n_nodes: int = N_VIRTUAL_NODES,
) -> NDArray[np.float64]:
    """
    Process raw y(t) clips through the same pipeline as x(t).

    y(t) receives no special treatment — identical steps to x(t):
      downsample → normalise [-1, +1] → atanh activation → reshape (n_time, n_nodes)

    Args:
        y_stream_raw: shape (n_clips, n_hw_samples) — raw y(t) at original_sr Hz
        clip_length_s: clip duration in seconds (used for tiling/truncation)
        original_sr: hardware sample rate in Hz (default 100 kHz)
        target_sr: target sample rate after downsampling (default 4 kHz)
        n_time: number of time steps in the output feature map (default 200)
        n_nodes: number of virtual nodes per time step (default 100)

    Returns:
        shape (n_clips, n_time, n_nodes) float64 — processed y(t) feature maps
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.ingest import (
        downsample, normalise, atanh_activation, _tile_to_length,
        DOWNSAMPLE_FACTOR, SAMPLES_PER_CLIP,
    )

    factor = original_sr // target_sr
    target_samples = n_time * n_nodes

    n_clips = y_stream_raw.shape[0]
    out = np.zeros((n_clips, n_time, n_nodes), dtype=np.float64)

    for i in range(n_clips):
        ds = y_stream_raw[i, ::factor].copy()
        ds = _tile_to_length(ds, target_samples)
        normed = normalise(ds)
        activated = atanh_activation(normed)
        out[i] = activated.reshape(n_time, n_nodes)

    return out


def compute_phase_features(
    x_features: NDArray[np.float64],
    y_features: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute orbit radius: r(t) = sqrt(x(t)^2 + y(t)^2).

    This is the distance of the spinning dot from the centre of the limit
    cycle at each moment in time. It encodes the amplitude of oscillation
    independently of rotation direction — a direct measure of how far the
    oscillator was perturbed from its natural orbit by the input signal.

    When the input signal is strong, the oscillator is pushed off its limit
    cycle and r(t) changes significantly. When there is no input, r(t) stays
    near the limit cycle radius sqrt(mu) ≈ sqrt(5) ≈ 2.24.

    Args:
        x_features: shape (n_clips, 200, 100) float64 — processed x(t)
        y_features: shape (n_clips, 200, 100) float64 — processed y(t)

    Returns:
        shape (n_clips, 200, 100) float64 — orbit radius at each sample
    """
    return np.sqrt(x_features ** 2 + y_features ** 2)


def compute_angle_features(
    x_features: NDArray[np.float64],
    y_features: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute instantaneous phase: theta(t) = arctan2(y(t), x(t)).

    This is the angular position of the spinning dot on the limit cycle.
    It encodes rotational information that neither x nor y captures alone —
    specifically the timing and frequency structure of perturbations,
    independently of their amplitude (complementary to orbit radius).

    Phase is unwrapped along the time axis to remove 2pi discontinuities,
    giving a monotonically increasing (or decreasing) phase signal whose
    rate of change encodes instantaneous frequency deviations caused by
    the input signal.

    Args:
        x_features: shape (n_clips, 200, 100) float64 — processed x(t)
        y_features: shape (n_clips, 200, 100) float64 — processed y(t)

    Returns:
        shape (n_clips, 200, 100) float64 — unwrapped instantaneous phase
    """
    raw_angle = np.arctan2(y_features, x_features)
    # Unwrap along axis=1 (time axis) per clip to remove 2pi jumps
    return np.unwrap(raw_angle, axis=1)


def build_dual_channel(
    x_features: NDArray[np.float64],
    y_features: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Stack x(t) and y(t) feature maps as two channels.

    The first Conv2D layer (arm_convolve_s8()) sees both feature maps
    simultaneously and learns cross-channel patterns — relationships between
    x(t) and y(t) dynamics at each spatial location.

    CMSIS-NN NOTE: arm_convolve_s8() handles multi-channel input natively
    with no additional overhead in the kernel. The input tensor channels
    are interleaved in memory per the NHWC layout expected by TFLite.

    Args:
        x_features: shape (n_clips, 200, 100) float64
        y_features: shape (n_clips, 200, 100) float64

    Returns:
        shape (n_clips, 200, 100, 2) float64 — [x, y] channel-last
    """
    return np.stack([x_features, y_features], axis=-1)


def scale_to_uint8(
    feature_maps: NDArray[np.float64],
) -> NDArray[np.uint8]:
    """
    Scale float feature maps globally to [0, 255] uint8.

    Used for single-channel representations (y_only, phase, angle) before
    passing to models that expect uint8 input matching the x(t) scaling.

    CMSIS-NN NOTE: uint8 [0, 255] maps to int8 [-128, 127] via zero-point
    offset during TFLite INT8 conversion (zero_point = 128).
    """
    fmin = feature_maps.min()
    fmax = feature_maps.max()
    if fmax - fmin < 1e-12:
        return np.zeros(feature_maps.shape, dtype=np.uint8)
    scaled = (feature_maps - fmin) / (fmax - fmin) * 255.0
    return np.round(scaled).astype(np.uint8)


def scale_dual_channel_to_uint8(
    xy_dual: NDArray[np.float64],
) -> NDArray[np.uint8]:
    """
    Scale the dual-channel feature map [n, 200, 100, 2] to uint8.

    Each channel is scaled independently so both x and y use their full
    [0, 255] dynamic range, preventing one channel from dominating.

    Returns:
        shape (n_clips, 200, 100, 2) uint8
    """
    out = np.zeros(xy_dual.shape, dtype=np.uint8)
    for c in range(xy_dual.shape[-1]):
        out[..., c] = scale_to_uint8(xy_dual[..., c])
    return out


def extract_all_representations(
    x_features_raw: NDArray[np.float64],
    y_features_raw: NDArray[np.float64],
) -> dict[str, NDArray]:
    """
    Compute all input representations from processed x(t) and y(t).

    Takes already-ingest-processed float64 feature maps (output of
    ingest.process_dataset()) and computes every representation needed
    for the multi-model training run.

    Args:
        x_features_raw: (n_clips, 200, 100) float64 — processed x(t)
        y_features_raw: (n_clips, 200, 100) float64 — processed y(t)

    Returns:
        dict with keys:
          "x_only"  — (n, 200, 100) uint8
          "y_only"  — (n, 200, 100) uint8
          "xy_dual" — (n, 200, 100, 2) uint8
          "phase"   — (n, 200, 100) uint8  (orbit radius)
          "angle"   — (n, 200, 100) uint8  (unwrapped phase angle)
    """
    phase_raw = compute_phase_features(x_features_raw, y_features_raw)
    angle_raw = compute_angle_features(x_features_raw, y_features_raw)
    xy_dual_raw = build_dual_channel(x_features_raw, y_features_raw)

    return {
        "x_only": scale_to_uint8(x_features_raw),
        "y_only": scale_to_uint8(y_features_raw),
        "xy_dual": scale_dual_channel_to_uint8(xy_dual_raw),
        "phase": scale_to_uint8(phase_raw),
        "angle": scale_to_uint8(angle_raw),
    }


def main() -> None:
    """Demonstrate feature extraction and visualise all representations."""
    import sys
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.sample_data import generate_dataset_xy, CLASS_NAMES
    from pipeline.ingest import process_dataset

    print("[features_xy] Generating small XY dataset ...")
    raw_x, raw_y, labels = generate_dataset_xy(n_clips_per_class=3, n_classes=5, cache=False)

    print("[features_xy] Processing through ingest pipeline ...")
    x_feat = process_dataset(raw_x)
    y_feat = extract_y_features(raw_y)

    print("[features_xy] Computing all representations ...")
    reps = extract_all_representations(x_feat, y_feat)

    for name, arr in reps.items():
        print(f"  {name:10s}: shape={arr.shape}, dtype={arr.dtype}, "
              f"range=[{arr.min()}, {arr.max()}]")

    # Visualise one clip per class, all representations
    rep_names = ["x_only", "y_only", "phase", "angle"]
    n_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(len(rep_names), n_classes,
                             figsize=(4 * n_classes, 4 * len(rep_names)))

    for row, rep_name in enumerate(rep_names):
        for cls in range(n_classes):
            mask = labels == cls
            if not np.any(mask):
                continue
            idx = np.where(mask)[0][0]
            ax = axes[row, cls]
            ax.imshow(reps[rep_name][idx], cmap="viridis", aspect="auto")
            ax.set_title(f"{rep_name}\nclass {cls}: {CLASS_NAMES[cls]}")
            ax.axis("off")

    plt.tight_layout()
    out_dir = Path(__file__).resolve().parent.parent / "pipeline" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "xy_feature_comparison.png", dpi=120)
    print(f"[features_xy] Saved visualisation to {out_dir / 'xy_feature_comparison.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
