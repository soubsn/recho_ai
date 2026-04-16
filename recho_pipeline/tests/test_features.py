"""
Tests for feature-map scaling and the xy-derived representations.

scale_to_uint8 is the last float->int8 bridge before CMSIS-NN, so it needs
tight guarantees on range, monotonicity, and degenerate-input behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.features import scale_to_uint8
from pipeline.features_xy import (
    build_dual_channel,
    compute_angle_features,
    compute_phase_features,
    scale_dual_channel_to_uint8,
)


def test_scale_to_uint8_range():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 200, 100))
    out = scale_to_uint8(x)
    assert out.dtype == np.uint8
    assert out.min() == 0
    assert out.max() == 255


def test_scale_to_uint8_is_monotonic():
    # A sorted float input must map to a non-decreasing uint8 output
    x = np.linspace(-10.0, 10.0, 1000, dtype=np.float64)
    out = scale_to_uint8(x)
    assert np.all(np.diff(out.astype(np.int16)) >= 0)


def test_scale_to_uint8_constant_input_returns_zeros():
    x = np.full((5, 10, 10), 3.14, dtype=np.float64)
    out = scale_to_uint8(x)
    assert out.dtype == np.uint8
    assert np.all(out == 0)


def test_scale_to_uint8_near_constant_treated_as_constant():
    # Gap below 1e-12 must fall into the zero branch to avoid 1/0
    x = np.full((4, 4), 2.0, dtype=np.float64)
    x[0, 0] += 1e-15
    out = scale_to_uint8(x)
    assert np.all(out == 0)


def test_scale_to_uint8_preserves_shape():
    x = np.random.default_rng(1).standard_normal((2, 200, 100))
    assert scale_to_uint8(x).shape == x.shape


def test_compute_phase_radius_matches_formula():
    x = np.array([[3.0, 0.0], [1.0, -1.0]], dtype=np.float64)[None, ...]
    y = np.array([[4.0, 0.0], [1.0, 1.0]], dtype=np.float64)[None, ...]
    r = compute_phase_features(x, y)
    # Radius must always be non-negative
    assert np.all(r >= 0.0)
    assert np.isclose(r[0, 0, 0], 5.0)
    assert np.isclose(r[0, 0, 1], 0.0)
    assert np.isclose(r[0, 1, 0], np.sqrt(2.0))


def test_compute_angle_is_unwrapped_along_time_axis():
    # Build a phase ramp that crosses pi so raw arctan2 would jump by -2pi
    t = np.linspace(0, 4 * np.pi, 200)
    x = np.cos(t).reshape(1, 200, 1)
    y = np.sin(t).reshape(1, 200, 1)
    x = np.tile(x, (1, 1, 100))
    y = np.tile(y, (1, 1, 100))
    theta = compute_angle_features(x, y)
    # Unwrapped phase must be (approximately) monotonically increasing along time
    assert np.all(np.diff(theta, axis=1) > -0.1)


def test_build_dual_channel_layout_is_x_then_y():
    x = np.zeros((2, 200, 100), dtype=np.float64)
    y = np.ones((2, 200, 100), dtype=np.float64)
    xy = build_dual_channel(x, y)
    assert xy.shape == (2, 200, 100, 2)
    assert np.all(xy[..., 0] == 0.0)
    assert np.all(xy[..., 1] == 1.0)


def test_scale_dual_channel_per_channel_independent_range():
    # Channel 0 spans 0..1, channel 1 spans 0..1000 — after scaling both hit 255
    xy = np.zeros((1, 10, 10, 2), dtype=np.float64)
    xy[..., 0] = np.linspace(0, 1, 100).reshape(10, 10)
    xy[..., 1] = np.linspace(0, 1000, 100).reshape(10, 10)
    out = scale_dual_channel_to_uint8(xy)
    assert out.dtype == np.uint8
    assert out[..., 0].max() == 255
    assert out[..., 1].max() == 255
    assert out[..., 0].min() == 0
    assert out[..., 1].min() == 0
