import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from pipeline.models.denoising.tcn_denoiser import TCNDenoiser, build_model


def test_build_tcn_denoiser_output_shape() -> None:
    model = build_model(input_timesteps=64)
    assert model.output_shape == (None, 64, 1)


def test_predict_streaming_matches_full_prediction() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 64, 2)).astype(np.float32)

    denoiser = TCNDenoiser()
    denoiser._model = build_model(input_timesteps=64)
    denoiser._is_fitted = True

    full = denoiser.predict(x)[0]

    state = None
    outputs = []
    for start in range(0, 64, 16):
        chunk = x[0, start:start + 16]
        out, state = denoiser.predict_streaming(chunk, state)
        outputs.append(out)
    streaming = np.concatenate(outputs, axis=0)

    np.testing.assert_allclose(streaming, full, atol=1e-5, rtol=1e-5)
