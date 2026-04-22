# Prompt: build variant (b) — true 1D CNN over the raw time series

Paste this into a fresh Claude Code conversation when you're ready to build
variant (b) of recommendation #7 (the 1D-CNN sibling of `train_CNN_time.py`).
The prompt is self-contained: it explains the pipeline, the motivation, what
variant (a) already looks like, and what (b) needs to change.

---

## Context

I'm working on a Hopf reservoir computer pipeline that classifies audio clips
from a uint8 feature map produced by an analog Hopf oscillator. The final
model is deployed as INT8 on an Arm Cortex-M via CMSIS-NN. Repo layout
relevant to this task:

- Data cache: `/Users/nic-spect/data/recho_ai/Kaggle_Dog_vs_Cats/hopf_text/`
  (2019 clips, classes `cat` and `dog`).
- Ingest: [recho_pipeline/pipeline/ingest.py](recho_pipeline/pipeline/ingest.py)
  — loads raw x(t), downsamples 100 kHz → 4 kHz, normalises to [-1, 1], runs
  `atanh` activation, and reshapes the resulting 20000-sample sequence into a
  `(200 time_steps, 100 virtual_nodes)` 2D grid via `reshape(200, 100)`.
- Features: [recho_pipeline/pipeline/features.py](recho_pipeline/pipeline/features.py)
  — scales the 2D grid to `uint8` in `[0, 255]` for INT8 CMSIS-NN input.
- Baseline model: [recho_pipeline/pipeline/models/cnn_regularized.py](recho_pipeline/pipeline/models/cnn_regularized.py)
  — Conv2D (3, 3) stack + GAP + Dense. Input shape `(200, 100, 1)`.
- Baseline training: [recho_pipeline/pipeline/train_CNN.py](recho_pipeline/pipeline/train_CNN.py).
- Variant (a) already shipped:
  - Model: [recho_pipeline/pipeline/models/cnn_time.py](recho_pipeline/pipeline/models/cnn_time.py)
    — same trunk but (3, 1) kernels so convs never mix across the virtual-node
    axis.
  - Training: [recho_pipeline/pipeline/train_CNN_time.py](recho_pipeline/pipeline/train_CNN_time.py).

## The motivation

The `(200, 100)` reshape splits the original 20000-sample time series into a
2D grid where:
- adjacent columns = 1-sample apart (0.25 ms at 4 kHz)
- adjacent rows = 100 samples apart (25 ms at 4 kHz)

Variant (a) removes cross-axis mixing by using (3, 1) kernels but still
carries the 2D grid. **Variant (b) eliminates the grid entirely** and treats
the reservoir output as what it actually is: a single 1D time series of
20000 samples, processed with `Conv1D` + `MaxPool1D`.

## What I want you to build

Two new files, plus a prompt `.md` nothing to add:

### 1. `recho_pipeline/pipeline/models/cnn_1d.py`

A 1D CNN. Input shape `(20000, 1)`. CMSIS-NN compatible (every op must map to
`arm_convolve_s8`, `arm_max_pool_s8`, `arm_avgpool_s8`,
`arm_fully_connected_s8`, or `arm_softmax_s8`). Only ReLU activations.
`use_bias=True` everywhere. No batch norm. All output channel counts
multiples of 4.

Rough architecture — tune as needed to keep parameter count near the ~76K of
`cnn_regularized`:

```python
layers.Rescaling(1.0 / 255.0)
layers.Conv1D(32, kernel_size=21, strides=2, padding="same", activation="relu", use_bias=True)
layers.MaxPool1D(4)
layers.Conv1D(64, kernel_size=11, strides=2, padding="same", activation="relu", use_bias=True)
layers.MaxPool1D(4)
layers.Conv1D(64, kernel_size=7,  strides=1, padding="same", activation="relu", use_bias=True)
layers.GlobalAveragePooling1D()
layers.Dense(64, activation="relu", use_bias=True)
layers.Dropout(0.5)
layers.Dense(n_classes, activation="softmax", use_bias=True)
```

Name the layers following the `*_arm_convolve_s8`, `*_arm_max_pool_s8`,
`*_arm_fully_connected_s8`, `*_arm_softmax_s8` convention used in
[model.py](recho_pipeline/pipeline/model.py) so they're easy to grep through
the CMSIS-NN generation step.

### 2. `recho_pipeline/pipeline/train_CNN_1d.py`

A near-copy of [train_CNN_time.py](recho_pipeline/pipeline/train_CNN_time.py)
with these differences:

- **Feature shape**: the 1D model needs `(n, 20000, 1)` input, not
  `(n, 200, 100, 1)`. The simplest way is to call
  `feature_maps.reshape(n_clips, -1)` after `extract_features` returns the
  `(n, 200, 100)` uint8 array, then let `prepare_data` add the channel axis.
  Alternatively, skip the 2D reshape in `ingest` and work with the raw 20000.
  I'd prefer the first approach (flatten after extract) because it reuses the
  existing feature extraction path unchanged — easier to A/B.
- **`prepare_data`**: update the shape check. Currently it adds a channel
  axis only when `x.ndim == 3`; for the 1D path the input will be `(n, 20000)`
  so it also needs `(n, 20000, 1)`. A clean implementation: always
  `expand_dims(..., -1)` if the last axis isn't already the channel.
- **`save_feature_maps_preview`**: it expects 2D images. For the 1D case,
  either (a) plot `plt.plot(x_val[idx])` instead of `imshow`, or (b) reshape
  the 20000-sample array back to `(200, 100)` *just for the preview* so the
  visual sanity check still works. Option (b) is simpler and the preview is
  only cosmetic.
- **Checkpoint + preview paths**: suffix `_1d` so they don't collide with
  `_time` or the baseline. E.g. `pretrain_1d.h5`,
  `output/feature_maps_preview_1d/`, `output/checkpoints_1d/`.
- **`_resolve_build_model`**: import `build_model` from
  `pipeline.models.cnn_1d`; keep the `INPUT_REP != "x_only"` rejection.
- **`representative_data_gen`**: update the sample shape expansion. Currently
  it does `np.expand_dims(sample, axis=(0, -1))` to produce
  `(1, 200, 100, 1)`. The 1D path needs `(1, 20000, 1)`.

### Things NOT to change

- The ingest + feature extraction pipeline in `ingest.py` / `features.py`.
  The flatten from `(200, 100)` back to `(20000,)` should happen in the
  training script, so the same cache is reused.
- The overall `pretrain → finetune` flow and CLI-free `MODE` / `TUNE_LR`
  toggles at the top of the file.
- The CMSIS-NN compatibility discipline — it's the whole point.

## Deployment concerns to check before shipping

When the user runs `convert.py` on the trained 1D model:

1. TFLite will fold `Conv1D` into `Conv2D` with `kernel_shape=(k, 1)` and
   `input_shape=(1, 20000, 1)`. Verify by inspecting the generated
   `firmware/cmsis_nn_params.h` — every weight layer should still map to a
   real CMSIS-NN kernel.
2. Peak RAM will be dominated by the first intermediate activation —
   `(1, 10000, 32)` after the first conv with stride 2 = 320 KB at INT8.
   That blows the M4 (48 KB) and M33 (64 KB) RAM budgets from
   [convert.py `MCU_RAM_LIMITS`](recho_pipeline/pipeline/convert.py#L27-L32).
   Likely need a larger initial stride (4 or 8) or an initial pool before the
   first conv to keep the early activation small.
3. The generated input buffer in firmware becomes `int8_t input[20000]`
   rather than `int8_t input[200][100]`. Any hand-written firmware that
   assumes the 2D layout will need updating — flag this to the user.

## What to report back when done

- Summary of the two new files and what they do.
- Parameter count of the 1D model vs `cnn_regularized`.
- Expected peak RAM from the first conv output, and which MCU tiers fit.
- A one-line diff summary of `train_CNN_time.py` → `train_CNN_1d.py`.

Do not run training. I'll run it and paste logs back if it needs tuning.
