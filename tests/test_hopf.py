from __future__ import annotations

import numpy as np

from data import sample_data


def test_integrate_hopf_xy_is_deterministic() -> None:
    a_func = sample_data._make_sine(freq=300.0, amp=0.3)

    x1, y1 = sample_data.integrate_hopf_xy(a_func, duration=0.02, fs=2_000)
    x2, y2 = sample_data.integrate_hopf_xy(a_func, duration=0.02, fs=2_000)

    np.testing.assert_allclose(x1, x2, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(y1, y2, rtol=0.0, atol=0.0)


def test_integrate_hopf_matches_x_component_of_xy_solver() -> None:
    a_func = sample_data._make_two_sines(f1=180.0, f2=360.0, amp=0.15)

    x_only = sample_data.integrate_hopf(a_func, duration=0.02, fs=2_000)
    x_xy, _ = sample_data.integrate_hopf_xy(a_func, duration=0.02, fs=2_000)

    np.testing.assert_allclose(x_only, x_xy, rtol=1e-9, atol=1e-9)


def test_integrate_hopf_xy_zero_input_is_bounded_and_finite() -> None:
    x, y = sample_data.integrate_hopf_xy(lambda _t: 0.0, duration=0.05, fs=4_000)
    radius = np.sqrt(x**2 + y**2)

    assert x.shape == y.shape == (200,)
    assert np.isfinite(x).all()
    assert np.isfinite(y).all()
    assert np.isfinite(radius).all()
    assert float(radius.max()) < 10.0
    assert float(radius[-50:].std()) < 0.5


def test_class_factory_noise_is_seeded() -> None:
    noise_a = sample_data._class_factory(4, variation_seed=123)
    noise_b = sample_data._class_factory(4, variation_seed=123)
    ts = np.linspace(0.0, 0.01, 32, endpoint=False)

    vals_a = np.array([noise_a(float(t)) for t in ts])
    vals_b = np.array([noise_b(float(t)) for t in ts])

    np.testing.assert_allclose(vals_a, vals_b, rtol=0.0, atol=0.0)
