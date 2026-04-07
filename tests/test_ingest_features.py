from __future__ import annotations

import numpy as np

from pipeline import features_xy, ingest


def test_process_clip_returns_expected_shape_and_finite_values() -> None:
    raw = np.sin(np.linspace(0.0, 12.0 * np.pi, 100_000, endpoint=False))

    out = ingest.process_clip(raw)

    assert out.shape == (ingest.N_TIME_STEPS, ingest.N_VIRTUAL_NODES)
    assert np.isfinite(out).all()


def test_extract_y_features_matches_ingest_pipeline_for_equivalent_input() -> None:
    raw = np.cos(np.linspace(0.0, 10.0 * np.pi, 100_000, endpoint=False))[None, :]

    x_processed = ingest.process_dataset(raw)
    y_processed = features_xy.extract_y_features(raw)

    np.testing.assert_allclose(x_processed, y_processed, rtol=1e-9, atol=1e-9)


def test_compute_phase_features_matches_radius_formula() -> None:
    x = np.array([[[3.0, 4.0], [5.0, 12.0]]], dtype=np.float64)
    y = np.array([[[4.0, 3.0], [12.0, 5.0]]], dtype=np.float64)

    phase = features_xy.compute_phase_features(x, y)

    expected = np.sqrt(x**2 + y**2)
    np.testing.assert_allclose(phase, expected)


def test_compute_angle_features_unwraps_time_axis() -> None:
    angles = np.array(
        [
            [-3.0, -2.0],
            [3.1, 2.8],
            [3.2, 2.9],
        ],
        dtype=np.float64,
    )
    x = np.cos(angles)[None, :, :]
    y = np.sin(angles)[None, :, :]

    unwrapped = features_xy.compute_angle_features(x, y)

    raw = np.arctan2(y, x)
    expected = np.unwrap(raw, axis=1)
    np.testing.assert_allclose(unwrapped, expected)
    assert np.abs(np.diff(unwrapped, axis=1)).max() < np.pi


def test_scale_to_uint8_constant_input_returns_zeros() -> None:
    arr = np.full((2, 3, 4), 7.5, dtype=np.float64)

    scaled = features_xy.scale_to_uint8(arr)

    assert scaled.dtype == np.uint8
    assert np.array_equal(scaled, np.zeros_like(scaled))


def test_extract_all_representations_returns_expected_contract() -> None:
    x = np.linspace(-1.0, 1.0, 2 * 200 * 100, dtype=np.float64).reshape(2, 200, 100)
    y = np.linspace(1.0, -1.0, 2 * 200 * 100, dtype=np.float64).reshape(2, 200, 100)

    reps = features_xy.extract_all_representations(x, y)

    assert set(reps) == {"x_only", "y_only", "xy_dual", "phase", "angle"}
    assert reps["x_only"].shape == (2, 200, 100)
    assert reps["y_only"].shape == (2, 200, 100)
    assert reps["phase"].shape == (2, 200, 100)
    assert reps["angle"].shape == (2, 200, 100)
    assert reps["xy_dual"].shape == (2, 200, 100, 2)
    assert all(arr.dtype == np.uint8 for arr in reps.values())
    assert np.array_equal(reps["xy_dual"][..., 0], features_xy.scale_to_uint8(x))
    assert np.array_equal(reps["xy_dual"][..., 1], features_xy.scale_to_uint8(y))
