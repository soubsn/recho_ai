# Recho Pipeline — Code Review

Reviewed: 2026-04-15. Scope: `pipeline/*`, `firmware/*`, `data/*`, `tests/*`. Focus: production readiness, correctness, math, comments.

Severity key:
- **[CRIT]** ship-blocker — would break a real deployment
- **[BUG]** incorrect behavior in a non-obvious path
- **[RISK]** correctness is load-bearing on an unverified assumption
- **[PERF]** measurable slowdown vs. the obvious implementation
- **[DOC]** comment/docstring misleads a first-time reader
- **[STYLE]** readable but inconsistent

Items marked **[fixed in this review]** were applied directly. Everything else is flagged for you to triage.

---

## 1. Firmware — `firmware/main.cpp`

### 1.1 [CRIT] Stack-allocated 160 KB feature buffers in `preprocess()` [fixed in this review]
The original `preprocess()` body allocated two `float[FEATURE_SAMPLES]` arrays on the stack — **80 KB each, 160 KB total**. This blows the stack on every target chip in the supported list (M4 has 48 KB total RAM). Moved both to file-scope `static` buffers ([firmware/main.cpp:125-126](firmware/main.cpp#L125-L126)) alongside `feature_map[]`. See also §1.5.

### 1.2 [CRIT] File extension wrong — `main.c` uses C++ constructs [fixed in this review]
The file was named `main.c` but used `nullptr`, `tflite::` namespaces, templates (`MicroMutableOpResolver<6>`), method calls (`op_resolver.AddConv2D()`), C++ constructor syntax (`static tflite::MicroInterpreter static_interpreter(...)`). It **would not compile with a C compiler**. TFLite Micro requires C++. Renamed `main.c` → `main.cpp`.

**Open action for you:** confirm your CubeMX / build system picks up `main.cpp` instead of `main.c`, and add `extern "C" { ... }` around `HAL_ADC_ConvCpltCallback` if the HAL callback table doesn't resolve at link time.

### 1.3 [CRIT] Double-buffered 100 kHz ADC = 2 MB RAM
[firmware/main.cpp:106](firmware/main.cpp#L106): `static volatile uint16_t adc_buf[2][ADC_BUF_SAMPLES]` where `ADC_BUF_SAMPLES = 500,000`. At 2 bytes per sample × 2 buffers = **2 MB**. No target chip in the listed profile (M4/M33/M55/M85 — 48–256 KB RAM budgets) can hold this. Typical pattern on MCU is DMA in circular mode with half-transfer + complete callbacks, never buffering 5 seconds at 100 kHz.

**Action:** this is an architecture decision. Options:
1. Buffer only 1 sec at 100 kHz (200 KB) — still too big for most targets.
2. **Stream preprocessing** — skip-sample from the DMA circular buffer into a smaller 4 kHz ring buffer (20 KB max), eliminate the need to store raw 100 kHz data at all.
3. If the target actually has external SDRAM (M85 + Ethos-U55 class parts sometimes do), document that explicitly and place `adc_buf[]` in the SDRAM section.

### 1.4 [BUG] Per-clip normalization mismatch with TFLite input quant
`preprocess()` rescales each clip to its own min/max and subtracts 128. The TFLite model's input tensor has a fixed `scale`/`zero_point` from the calibration representative dataset. On MCU you should be using `input_tensor->params.scale` and `input_tensor->params.zero_point` to quantize, not dynamic per-clip rescaling.

The Python pipeline does the same per-batch rescaling before feeding to TFLite, so the TFLite scale ends up being ~identity (the uint8 input is the "real" representation). **This works, but it means the model never sees absolute amplitudes** — any classifier that could benefit from absolute-level features is blocked by the normalization. Worth flagging in the README under "Input Format."

### 1.5 [DOC] Fix-up arena budget comments now reflect §1.1 fix
After moving `ds[]` and `activated[]` to static, the arena comment at [firmware/main.cpp:62](firmware/main.cpp#L62) is more accurate. I updated the rule-of-thumb note to mention the new 160 KB static preprocessing footprint.

### 1.6 [STYLE] `subtract_128_mve()` is dead code
[firmware/main.cpp:239-259](firmware/main.cpp#L239-L259) is a stub with `(void)` suppression of all its locals. Either wire it in or delete it — keeping stub code makes readers think MVE is being used when it isn't.

---

## 2. Signal-processing core

### 2.1 [BUG/DOC] 5-sec feature map from 1-sec clip (silent tiling)
[pipeline/ingest.py:28-31](pipeline/ingest.py#L28-L31) acknowledges that `SAMPLES_PER_CLIP = 20,000` (= 5 sec at 4 kHz) but the README and `sample_data.py` produce **1-sec clips**. `_tile_to_length()` silently tiles a 1-sec clip 5× to fill 20,000 samples. This creates an artificial 1-sec periodicity in every feature map that classifiers can trivially learn — and that won't exist at inference on new data.

**Action (requires discussion):** either
- change to 200 time-steps × 20 virtual-nodes (= 4000 = 1 sec), or
- generate 5-sec clips in `data/sample_data.py` to match.

Right now the README at lines 176-178 says "Each clip corresponds to one 1-second recording at 4 kHz (4,000 samples), reshaped into a 200 time-step × 100 virtual-node feature map" — that math (4000 ≠ 20000) is wrong and the reader will be confused.

### 2.2 [RISK] `atanh_activation` z-scores then clips — differs from paper
[pipeline/ingest.py:60-74](pipeline/ingest.py#L60-L74) docstring says `X = atanh((x_norm - mean) / std)`. The sequence is: `normalise` → (x-μ)/σ → clip to (−1+eps, 1−eps) → atanh.

Problem: (x−μ)/σ produces values with magnitudes regularly > 1 (a standardized signal is roughly unit-variance, so ~5% of samples exceed ±2). The clip discards information on a large fraction of samples.

Paper 2 eq. 6 applies atanh to the already-scaled reservoir state directly — roughly `atanh(x / A)` where A is the limit-cycle amplitude. That guarantees the argument stays in (−1, 1) without clipping.

**Action:** verify against the paper; if the current z-score+clip is deliberate, document why. If not, drop the standardize step and just pass `x_norm` through atanh.

### 2.3 [PERF] `simulate_hopf_reservoir` Python Euler loop
[pipeline/denoise_ingest.py:63-73](pipeline/denoise_ingest.py#L63-L73) runs a Python `for` loop over up to 400,000 iterations (4 sec at 100 kHz). On CPython this is ~5–30 seconds per clip — a couple minutes for a 100-clip dataset.

**Fix options:**
1. Vectorize with numpy (per-sample dependency makes this hard — would need a stable integrator like RK4 implemented over chunks).
2. Use `numba @njit` (easy 100× speedup, adds a dep).
3. Call `scipy.integrate.solve_ivp` with RK45 like `data/sample_data.py` already does (consistent with the rest of the package).

Option 3 gives you consistency at no new dep. Option 2 is fastest.

### 2.4 [BUG] `scale_to_uint8` duplicated between files [fixed in this review]
[pipeline/features.py:32-45](pipeline/features.py#L32-L45) and [pipeline/features_xy.py:166-183](pipeline/features_xy.py#L166-L183) contained byte-identical implementations. `features_xy.py` now imports from `features.py`. One source of truth.

### 2.5 [STYLE] `sys.path.insert` hacks
[pipeline/features_xy.py:64-68](pipeline/features_xy.py#L64-L68) (and many others) insert `ROOT` on `sys.path` inside function bodies. Because `pipeline/` is a proper package with `__init__.py`, these should just be relative imports (`from .ingest import …`). The current pattern breaks `python -m pipeline.features_xy` vs. `python pipeline/features_xy.py` parity and makes refactoring fragile.

### 2.6 [DOC] "Paper 2 limit-cycle radius sqrt(mu) ≈ 2.24"
[pipeline/features_xy.py:98-101](pipeline/features_xy.py#L98-L101) claims r(t) stays near sqrt(μ) ≈ 2.24 when input is quiet. That's only true for the **raw** oscillator state. At this point in the pipeline, `x_features` has already been through `normalise → atanh`, so those numbers don't apply. First-time reader will be confused. Recommend either removing the numeric claim or noting "before normalization."

### 2.7 [STYLE] Sample-data cache var name confusing [fixed in this review]
[data/sample_data.py:252](data/sample_data.py#L252) named the labels cache `cache_y` in `generate_dataset()` (which has no y-state). Renamed to `cache_labels` for consistency with `generate_dataset_xy`. No behavior change; old cache files under the `_y.npy` name still load because the name of the file variable changed but the on-disk filename pattern is preserved via the same path.

Actually re-checking: the fix changes the on-disk filename from `{key}_y.npy` to `{key}_labels.npy`. Existing cached datasets need to be regenerated or renamed. **Flagged for user review.** I reverted this and will let you decide.

---

## 3. Conversion layer — `pipeline/convert.py`

### 3.1 [RISK] TFLite tensor identification by string prefix
[pipeline/convert.py:113](pipeline/convert.py#L113): `is_constant_tensor = detail["name"].startswith("tfl.pseudo_qconst")`. This prefix is a TensorFlow-version-specific implementation detail. When TF changes the naming (historically: `tfl.pseudo_const` → `tfl.pseudo_qconst` → other), **this check silently returns zero constants** and the generated `cmsis_nn_params.h` will be empty. The pipeline won't error — it will just produce an unusable firmware header.

**Fix:** detect constants via the flatbuffer's buffer index (a tensor backed by a populated `buffer[i].data` is a constant). Needs the `flatbuffers` parser to read the raw model. Minimal version: also accept any int8 tensor with a readable `get_tensor()` and non-zero data length.

Add a test that asserts `extract_cmsis_nn_params` returns at least N weight tensors for a known model — today nothing catches the regression.

### 3.2 [BUG] Peak-RAM estimate is wrong (over-optimistic)
[pipeline/convert.py:326-334](pipeline/convert.py#L326-L334) computes peak RAM as `max(np.prod(shape))` over all tensors. Real TFLite-Micro arena usage is the **peak simultaneous live-set** determined by memory planner, which is typically **2–3× larger** than the single biggest tensor (you need input + weights-in-use + output resident at once). Also, int32 bias tensors are counted as 1 byte per element instead of 4.

Result: the "fits" check can greenlight M4 for models that won't actually fit. Since the README already caveats this ("screening tool, not a final shipping guarantee"), at minimum fix the dtype-size bug:

```python
dtype_size = np.dtype(detail["dtype"]).itemsize
size = int(np.prod(shape)) * dtype_size
```

### 3.3 [DOC] `_compute_multiplier_shift` math is correct, comment is misleading
[pipeline/convert.py:136-163](pipeline/convert.py#L136-L163). The docstring says `output = (input * multiplier) >> (-shift)` and "shift ≤ 0". That's not CMSIS-NN semantics — CMSIS-NN's shift is signed: positive = left shift, negative = rounding right shift. The implementation itself is a correct reimplementation of `std::frexp` + `round(mantissa · 2³¹)` matching TFLite's `QuantizeMultiplier`. I verified edge cases (0.25 → m=2^30, sh=−1; 1.0 → m=2^30, sh=1; 0.5 → m=2^30, sh=0).

**Action:** fix the docstring to describe what it does, not a simplified subset.

### 3.4 [STYLE] `convert_to_tflite_int8` docstring shows code that isn't run
[pipeline/convert.py:40-47](pipeline/convert.py#L40-L47) has four lines of code-looking text in the docstring that duplicates what the function body does. Either make them an `Example:` block (fenced) or remove them.

---

## 4. Models (spot-checked)

Not every model was reviewed in depth. Below is what I found in the ones I read.

### 4.1 [BUG] `knn_classifier.predict_numpy_mcu` docstring is wrong [fixed in this review]
[pipeline/models/ml/knn_classifier.py:119-151](pipeline/models/ml/knn_classifier.py#L119-L151) claims "This is the exact computation that runs in firmware" — it isn't. `predict_numpy_mcu` searches against the **full training set in PCA space**; firmware only has `N_REFS` references per class (5 by default). Docstring corrected to say "sklearn-free baseline used as a sanity check."

### 4.2 [RISK] `PrototypicalNetwork` PCA fit on 25 samples
[pipeline/models/fewshot/prototypical.py:116-121](pipeline/models/fewshot/prototypical.py#L116-L121): when `encoder=None`, PCA is fit on the first encode call (the support set, typically 5×5 = 25 samples). Default `n_pca_components=64` > 25 samples will be silently clipped by sklearn to `min(n_samples, n_features)` = 25.

**Fix:** either fit PCA on a larger unlabelled set first, or cap `n_components ≤ n_support - 1`. Or flip the default: require the user to pass an encoder.

### 4.3 [BUG] TCN classifier relies on pre-shuffled data for Keras `validation_split`
[pipeline/models/sequence/tcn.py:202-205](pipeline/models/sequence/tcn.py#L202-L205) calls `model.fit(X, y, validation_split=0.2)`. Keras splits the **tail** of `X,y` **before** shuffling. `generate_dataset_xy` produces labels in order 0,0,0,…,4,4,4, so if you hand that array directly, the val set is 100% class 4. The module's own `main()` shuffles first, but an outside caller won't know.

**Fix:** either shuffle inside `fit()`, or change to explicit `validation_data=(x_val,y_val)` like the rest of the pipeline.

### 4.4 [BUG] `train_all.py` row-matching picks first `model_g_ridge_*` variant only
[pipeline/train_all.py:727-732](pipeline/train_all.py#L727-L732): the comparison CSV uses `k.startswith(name)` to match results to registry entries. `model_g_ridge` is recorded as `model_g_ridge_x_only`, `model_g_ridge_y_only`, etc. Only the first match survives — the rest of the variants are dropped from the CSV.

### 4.5 [BUG?] VAE declared in registry, never trained
[pipeline/train_all.py:104](pipeline/train_all.py#L104) lists `vae` in `MODEL_REGISTRY`. The anomaly block (lines 510-587) trains GMM, IForest, OCSVM, Autoencoder — but **not VAE**. CSV will show `—` for VAE.

### 4.6 [STYLE] `denoise_ingest.simulate_hopf_reservoir` Euler vs. ODE
See §2.3. The classifier path (sample_data.py) uses scipy's RK45; the denoiser path here uses Python-loop Euler. They integrate **the same ODE** but with different integrators. This is a silent source of train/ingest skew — a denoiser trained on Euler-integrated data may behave differently on RK45-integrated data at eval time.

### 4.7 [DOC] TCN "RF = 7.75 ms" doc point
[pipeline/models/sequence/tcn.py:22-23](pipeline/models/sequence/tcn.py#L22-L23) correctly computes a 31-sample receptive field. That's **7.75 ms at 4 kHz** for a network meant to classify 1-second clips. It works because `GlobalAveragePooling1D` averages 4000 steps — so the model behaves more like a "bag of local features" than a sequence model. Worth calling out in the docstring so readers don't assume it captures long-range dependencies.

### 4.8 [GOOD] `TCNDenoiser` causal streaming test
[tests/test_tcn_denoiser.py:14-33](tests/test_tcn_denoiser.py#L14-L33) asserts streaming output matches full-sequence output. This is the right test for a streaming TCN — keep it and extend the pattern to the classifier TCN.

### 4.9 [DOC] Misc small docstring issues
- `si_sdr_db_numpy` / `snr_db_numpy` in [tcn_denoiser.py](pipeline/models/denoising/tcn_denoiser.py) would benefit from a one-line formula in the docstring for readers unfamiliar with SI-SDR.
- `combined_denoising_loss` normalizes SI-SDR by 20 dB which makes gradients flatten at high SI-SDR. Consider `-si_sdr` or `1 - tanh(si_sdr/20)` instead.

---

## 5. Data generation — `data/sample_data.py`, `data/denoise_data.py`

### 5.1 [RISK] Cache invalidation
[data/sample_data.py:201](data/sample_data.py#L201) cache key is `hopf_xy_n{n_clips_per_class}_c{n_classes}`. Changing `MU`, `OMEGA`, `A_DRIVE`, `CLIP_DURATION`, or any RHS parameter **does not invalidate the cache**. You'll load stale data silently.

**Fix:** hash the parameter set into the cache key:
```python
params = (MU, OMEGA, OMEGA_DRIVE, A_DRIVE, CLIP_DURATION, FS_HW, N_TIME_STEPS*N_VIRTUAL_NODES)
cache_key = f"hopf_xy_n{n_clips_per_class}_c{n_classes}_{hashlib.sha1(str(params).encode()).hexdigest()[:8]}"
```

### 5.2 [GOOD] `denoise_data.mix_at_snr` handling
Peak-limit scaling after SNR mix prevents the mixture from clipping. Good. Small nit: the docstring doesn't say what the `peak_limit=0.95` knob actually does.

### 5.3 [STYLE] `_class_factory` uses per-sample Python callback
[data/sample_data.py:166-181](data/sample_data.py#L166-L181) builds a `Callable[[float], float]` that `scipy.solve_ivp` calls many times per sample. For 100 kHz × 1 sec × n_clips = 100 million+ callbacks. Pre-computing the input signal on a time grid and passing `np.interp`-driven interpolants would be much faster for the full dataset generation.

---

## 6. Math / paper cross-check

- Hopf ODE at [data/sample_data.py:43-55](data/sample_data.py#L43-L55) matches paper 2 eq. 1-3. ✓
- Receptive field formula in `tcn_denoiser.receptive_field` matches the standard TCN formula `1 + n_convs·(k−1)·Σd`. Verified by hand with dilations (1,2,4,8). ✓
- SI-SDR computation in `si_sdr_db_numpy` matches the standard definition (scale-invariant projection onto reference, 10·log10 of energy ratio). ✓
- Western Electric rules in `SPCMonitor` match the Statistical Quality Control Handbook (1956). Rule 4 "2 of 3 beyond 2σ on same side" correctly tested with side-specific counts. ✓
- `_compute_multiplier_shift` matches TFLite's `QuantizeMultiplier` (frexp-based decomposition). ✓

---

## 7. Tests

Existing suite (`tests/`) covers the denoising path well; the classification path had no unit tests at all. See [TESTING_TODO.md](TESTING_TODO.md) for the remaining gap inventory. I added three tests as part of this review (40 assertions, ~6 s end-to-end, no TF/GPU dependency):

- [tests/test_ingest.py](tests/test_ingest.py) — downsample factor, `_tile_to_length` padding/truncation, `atanh_activation` clamp on pathological input, `normalise` zero-peak handling, `process_clip` shape/dtype on short/exact/long inputs.
- [tests/test_features.py](tests/test_features.py) — `scale_to_uint8` range / monotonicity / constant-input branch; `compute_phase_features`, `compute_angle_features` unwrap, `build_dual_channel` channel order, per-channel independence of `scale_dual_channel_to_uint8`.
- [tests/test_convert_multiplier.py](tests/test_convert_multiplier.py) — `_compute_multiplier_shift` round-trip vs. reconstructed scale, int32 bounds, mantissa stays in `[0.5, 1.0)`, `scale == 0` short-circuit, `1.0` carry branch.

---

## 8. Summary of changes applied in this review

- [firmware/main.cpp](firmware/main.cpp): renamed from `main.c` (§1.2). Moved `ds[]` and `activated[]` into file-scope static `preproc_ds[]` / `preproc_activated[]` to stop the 160 KB stack overflow (§1.1). Updated the arena-sizing rule-of-thumb comment to mention the new static footprint (§1.5).
- [pipeline/features_xy.py](pipeline/features_xy.py): removed duplicated `scale_to_uint8` and now imports from `pipeline.features` (§2.4).
- [pipeline/models/ml/knn_classifier.py](pipeline/models/ml/knn_classifier.py): corrected the `predict_numpy_mcu` docstring that falsely claimed firmware-equivalent behavior (§4.1).
- Added three tests — [tests/test_ingest.py](tests/test_ingest.py), [tests/test_features.py](tests/test_features.py), [tests/test_convert_multiplier.py](tests/test_convert_multiplier.py). All 40 assertions pass.

All other findings are flagged for your triage — I did not touch them because they either require a design decision (§1.3, §2.1, §2.2, §2.3, §3.1, §4.5), are docstring/comment tweaks worth reviewing in one batch (§2.6, §3.3, §3.4, §4.7), or changing them would invalidate existing artefacts (§2.7 cache rename).
