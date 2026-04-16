"""
Tests for the ingest signal-processing core.

Covers the invariants that downstream models depend on:
  - skip-sample factor matches the documented 100 kHz -> 4 kHz ratio
  - _tile_to_length pads and truncates correctly in both directions
  - atanh_activation clamps to keep np.arctanh finite on pathological inputs
  - normalise handles all-zero and tiny-peak input without dividing by zero
  - process_clip always returns a (200, 100) float64 array for any input length
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.ingest import (
    DOWNSAMPLE_FACTOR,
    FS_HW,
    FS_TARGET,
    N_TIME_STEPS,
    N_VIRTUAL_NODES,
    SAMPLES_PER_CLIP,
    _tile_to_length,
    atanh_activation,
    downsample,
    normalise,
    process_clip,
)


def test_downsample_factor_matches_rates():
    assert DOWNSAMPLE_FACTOR == FS_HW // FS_TARGET
    assert DOWNSAMPLE_FACTOR == 25


def test_samples_per_clip_matches_feature_map():
    assert SAMPLES_PER_CLIP == N_TIME_STEPS * N_VIRTUAL_NODES


def test_downsample_skip_sampling():
    x = np.arange(100, dtype=np.float64)
    ds = downsample(x, factor=25)
    assert ds.shape == (4,)
    assert np.array_equal(ds, np.array([0.0, 25.0, 50.0, 75.0]))


def test_downsample_returns_independent_copy():
    x = np.arange(100, dtype=np.float64)
    ds = downsample(x, factor=25)
    ds[0] = -999.0
    assert x[0] == 0.0


def test_tile_to_length_pads_short_input():
    x = np.array([1.0, 2.0, 3.0])
    out = _tile_to_length(x, 8)
    assert out.shape == (8,)
    assert np.array_equal(out, np.array([1, 2, 3, 1, 2, 3, 1, 2], dtype=np.float64))


def test_tile_to_length_truncates_long_input():
    x = np.arange(20, dtype=np.float64)
    out = _tile_to_length(x, 5)
    assert np.array_equal(out, np.arange(5, dtype=np.float64))


def test_tile_to_length_exact_length_is_identity():
    x = np.arange(10, dtype=np.float64)
    out = _tile_to_length(x, 10)
    assert np.array_equal(out, x)


def test_normalise_scales_peak_to_unit():
    x = np.array([-4.0, 0.0, 2.0, 8.0])
    out = normalise(x)
    assert np.isclose(np.max(np.abs(out)), 1.0)
    assert np.isclose(out[3], 1.0)


def test_normalise_all_zero_input_returns_zeros():
    x = np.zeros(10, dtype=np.float64)
    out = normalise(x)
    assert np.array_equal(out, x)
    assert not np.any(np.isnan(out))


def test_normalise_tiny_peak_avoids_divide_by_zero():
    x = np.full(10, 1e-20, dtype=np.float64)
    out = normalise(x)
    assert np.all(out == 0.0)


def test_atanh_activation_clamps_extreme_zscores():
    # Pathological input: one huge outlier drives abs(z) above 1
    x = np.array([-1.0, -1.0, -1.0, 1000.0], dtype=np.float64)
    out = atanh_activation(x)
    assert np.all(np.isfinite(out)), "atanh must never produce +/-inf"


def test_atanh_activation_zero_variance_input():
    x = np.ones(10, dtype=np.float64)
    out = atanh_activation(x)
    assert np.all(out == 0.0)


def test_atanh_activation_matches_arctanh_within_clamp():
    # For well-behaved input (all |z| well below 1), output must equal arctanh(z) exactly
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    out = atanh_activation(x)
    z = (x - x.mean()) / x.std()
    # Most samples shouldn't hit the clamp
    assert np.isclose(out[np.abs(z) < 0.9], np.arctanh(z[np.abs(z) < 0.9])).all()


def test_process_clip_output_shape_and_dtype():
    rng = np.random.default_rng(42)
    raw = rng.standard_normal(FS_HW)  # 1 second at 100 kHz
    out = process_clip(raw)
    assert out.shape == (N_TIME_STEPS, N_VIRTUAL_NODES)
    assert out.dtype == np.float64
    assert np.all(np.isfinite(out))


def test_process_clip_handles_short_input_via_tiling():
    # 0.1 sec at 100 kHz -> after downsample only 400 samples; must tile to 20k
    raw = np.sin(np.linspace(0, 2 * np.pi, 10_000)).astype(np.float64)
    out = process_clip(raw)
    assert out.shape == (N_TIME_STEPS, N_VIRTUAL_NODES)
    assert np.all(np.isfinite(out))


def test_process_clip_handles_long_input_via_truncation():
    raw = np.zeros(10 * FS_HW, dtype=np.float64)  # 10 sec
    raw[::1000] = 1.0  # sparse spikes
    out = process_clip(raw)
    assert out.shape == (N_TIME_STEPS, N_VIRTUAL_NODES)
    assert np.all(np.isfinite(out))
