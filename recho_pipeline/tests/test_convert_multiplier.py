"""
Tests for convert._compute_multiplier_shift.

This function ports TFLite's QuantizeMultiplier into Python and feeds the
CMSIS-NN requantise step. Getting the (multiplier, shift) pair wrong corrupts
every layer output on-device, so the edge cases need explicit coverage:
  - mantissa range [0.5, 1.0) after decomposition
  - integer multiplier fits into int32 (< 2**31)
  - scale == 0 short-circuits
  - round-trip reconstruction recovers the input scale to high precision
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline.convert import _compute_multiplier_shift


INT32_MAX = (1 << 31) - 1


def _reconstruct(multiplier: int, shift: int) -> float:
    """Undo the fixed-point encoding: scale ~= (multiplier / 2**31) * 2**shift."""
    return (multiplier / (1 << 31)) * (2.0 ** shift)


@pytest.mark.parametrize("scale", [
    1.0,
    0.5,
    0.25,
    1e-3,
    1e-6,
    1e-9,
    7.5,
    123.456,
    0.9999999,
])
def test_multiplier_shift_round_trip(scale: float):
    m, sh = _compute_multiplier_shift(scale)
    recovered = _reconstruct(m, sh)
    # 1 part in 2**30 is the best we can hope for from a 31-bit mantissa
    assert math.isclose(recovered, scale, rel_tol=1e-9)


def test_multiplier_fits_in_int32():
    for scale in [1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6]:
        m, _ = _compute_multiplier_shift(scale)
        assert 0 < m <= INT32_MAX, f"multiplier {m} overflows int32 for scale={scale}"


def test_multiplier_mantissa_in_canonical_range():
    # Re-deriving the mantissa from (multiplier / 2**31) must land in [0.5, 1.0)
    for scale in [1e-6, 1e-3, 0.5, 0.75, 1.0, 2.0, 10.0, 1234.0]:
        m, _ = _compute_multiplier_shift(scale)
        mantissa = m / (1 << 31)
        assert 0.5 <= mantissa < 1.0, (
            f"mantissa {mantissa} outside [0.5, 1.0) for scale={scale}"
        )


def test_scale_zero_short_circuits():
    m, sh = _compute_multiplier_shift(0.0)
    assert m == 0
    assert sh == 0


def test_scale_exactly_half_no_overflow():
    # 0.5 is a boundary case: mantissa exactly at 0.5 after 0 iterations
    m, sh = _compute_multiplier_shift(0.5)
    assert m <= INT32_MAX
    assert math.isclose(_reconstruct(m, sh), 0.5, rel_tol=1e-9)


def test_scale_exactly_one_no_overflow():
    # 1.0 triggers the `multiplier == 1 << 31` carry branch
    m, sh = _compute_multiplier_shift(1.0)
    assert m <= INT32_MAX
    assert math.isclose(_reconstruct(m, sh), 1.0, rel_tol=1e-9)


def test_shift_sign_matches_scale_magnitude():
    # scale < 1 -> shift <= 0 (right shift on device)
    # scale > 1 -> shift >= 0 (left shift, widening)
    m_small, sh_small = _compute_multiplier_shift(1e-6)
    m_large, sh_large = _compute_multiplier_shift(1e3)
    assert sh_small <= 0
    assert sh_large >= 0
