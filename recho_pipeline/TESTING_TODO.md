# Testing TODO

Companion to [REVIEW.md](REVIEW.md). Lists the test coverage that still needs
to be built out after the initial review pass, organised by risk.

## Current coverage (after review)

Added in this review:

- [tests/test_ingest.py](tests/test_ingest.py) — downsample factor,
  `_tile_to_length` padding/truncation, `atanh_activation` clamp + zero-var,
  `normalise` degenerate inputs, `process_clip` shape/dtype invariants on
  short / exact / long inputs.
- [tests/test_features.py](tests/test_features.py) — `scale_to_uint8` range,
  monotonicity, constant-input zero branch; `compute_phase_features`,
  `compute_angle_features` unwrap, `build_dual_channel` channel order,
  `scale_dual_channel_to_uint8` per-channel independence.
- [tests/test_convert_multiplier.py](tests/test_convert_multiplier.py) —
  `_compute_multiplier_shift` round-trip, int32 bounds, canonical mantissa
  range [0.5, 1.0), `scale == 0` short-circuit, `1.0` carry branch.

Pre-existing (denoising path only):

- `tests/test_denoise_data.py`, `tests/test_denoise_ingest.py`,
  `tests/test_denoiser_pipeline.py`, `tests/test_tcn_denoiser.py`.

## High priority — silent correctness risks

1. **End-to-end parity: TFLite Micro vs CMSIS-NN reference**
   Generate a canary input, run `.tflite` through the Python interpreter, run
   the same bytes through the exported `cmsis_params.h` path, assert that
   tensors match layer-by-layer within 1 LSB. Without this we can't prove the
   `_compute_multiplier_shift` output matches TFLite's kernel at runtime.

2. **`convert.py` layer-parameter extraction**
   `_extract_layer_params` identifies quantised ops by string-matching
   `tfl.pseudo_qconst` in the op name. Add a test that round-trips a tiny
   Keras model through QAT → `.tflite` → `_extract_layer_params` and asserts
   the expected layer list is recovered. Guards against MLIR op-name drift
   across TensorFlow versions.

3. **`knn_classifier.predict_numpy_mcu` vs `predict` agreement**
   Both should produce identical labels for the same inputs. A regression
   here means the "MCU-equivalent" path has drifted from the sklearn path.

4. **Ordered-labels footgun in `tcn.py` / any `validation_split=0.2` user**
   Assert training splits don't leak whole classes into validation when
   labels arrive ordered. Either add a shuffle test or a guard in the
   training helper.

## Medium priority — functional gaps

5. **`sample_data.generate_dataset_xy` cache key**
   The cache filename doesn't include `MU` / `OMEGA`. Add a test that changes
   those globals and asserts the cache is invalidated (or fix the cache key
   and test the fix).

6. **Hopf ODE integrator numerics**
   `denoise_ingest.simulate_hopf_reservoir` is a pure-Python Euler loop. Add
   a test that integrates a known-good frequency and checks the settled
   limit-cycle radius ≈ sqrt(MU) and peak frequency ≈ OMEGA / (2π).

7. **SPC Western-Electric rules**
   Unit tests for each of the four rules in isolation — craft a 30-point
   series that triggers only rule N and assert the others stay silent.

8. **Prototypical network few-shot path**
   PCA with `n_components=64` on 25 support samples is ill-posed. Test with
   the corrected `n_components=min(64, n_support - 1)` so the failure mode
   is caught at CI time, not training time.

9. **`features_xy.extract_all_representations` shapes and dtypes**
   Single integration test: feed a (3, 200, 100) float64 pair, assert every
   key is present and the shape/dtype matches the docstring.

## Low priority — nice to have

10. **`firmware/main.cpp` preprocess equivalence**
    Port `preprocess()` to a Python shim and assert it produces the same
    `int8` tensor as `ingest.process_clip() → scale_to_uint8() → -128`.
    Catches drift between host pipeline and MCU preprocessing.

11. **`convert_all.py` orchestration**
    Smoke test that iterates over a stub model registry and asserts every
    model successfully emits `.tflite` + `model_data.cc` + `cmsis_params.h`.

12. **`evaluate.py` metric computation**
    Golden-file test with a known prediction / label set to pin the exact
    accuracy / confusion-matrix output format.

13. **`data/denoise_data.py::mix_at_snr`**
    The core logic is verified, but there's no regression test. Add one
    that mixes at a known SNR and asserts the measured SNR is within
    0.1 dB of target.

## How to run

```bash
cd recho_pipeline
python -m pytest tests/ -q
```

New tests in this review (`test_ingest.py`, `test_features.py`,
`test_convert_multiplier.py`) run in under 10 s with no GPU / TF dependency.
