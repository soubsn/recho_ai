from __future__ import annotations

import os
import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_TF_SMOKE") != "1",
    reason="Set RUN_TF_SMOKE=1 to enable TensorFlow model smoke tests.",
)


@pytest.mark.parametrize(
    ("build_fn", "input_shape"),
    [
        ("build_x_only", (2, 200, 100, 1)),
        ("build_xy_dual", (2, 200, 100, 2)),
        ("build_depthwise", (2, 200, 100, 2)),
    ],
)
def test_single_input_models_produce_class_probabilities(build_fn, input_shape) -> None:
    from pipeline.models.cnn_x_only import build_model as build_x_only
    from pipeline.models.cnn_xy_dual import build_model as build_xy_dual
    from pipeline.models.depthwise_cnn import build_model as build_depthwise

    build_fn = {
        "build_x_only": build_x_only,
        "build_xy_dual": build_xy_dual,
        "build_depthwise": build_depthwise,
    }[build_fn]
    model = build_fn(n_classes=5)
    batch = np.zeros(input_shape, dtype=np.float32)

    out = model(batch, training=False).numpy()

    assert out.shape == (2, 5)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(2), rtol=1e-5, atol=1e-5)


def test_xy_fusion_model_accepts_two_inputs() -> None:
    from pipeline.models.cnn_xy_fusion import build_model as build_xy_fusion

    model = build_xy_fusion(n_classes=5)
    x_batch = np.zeros((2, 200, 100, 1), dtype=np.float32)
    y_batch = np.zeros((2, 200, 100, 1), dtype=np.float32)

    out = model([x_batch, y_batch], training=False).numpy()

    assert out.shape == (2, 5)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(2), rtol=1e-5, atol=1e-5)
