# Recho Pipeline

Post-processing pipeline for a **Hopf oscillator physical reservoir computer**,
targeting Arm Cortex-M microcontrollers with every inference layer mapped to a
named CMSIS-NN kernel function.

Supported targets: **M33**, **M55** (Helium MVE), **M85 + Ethos-U55 NPU**

---

## Installation

### Prerequisites

- [uv](https://docs.astral.sh/uv/) — if not installed, `setup.sh` will install it automatically
- Python 3.10 (via pyenv or system install)
- git

### Steps

```bash
git clone <repo-url>
cd recho_pipeline
bash setup.sh
source recho-dev/bin/activate
```

`setup.sh` will:
1. Install `uv` if it is not already on your `PATH`
2. Create a virtual environment named `recho-dev` using Python 3.10
3. Install all dependencies from `requirements.txt`
4. Optionally install `tensorflow-model-optimization` for quantisation-aware training (QAT)

To deactivate the environment:

```bash
deactivate
```

---

## Running the Pipeline

Each module is runnable standalone. Run them in order, or use the notebook for
an interactive walkthrough.

```bash
# 1. Generate synthetic Hopf oscillator data (cached to data/cache/ after first run)
python data/sample_data.py

# 2. Ingest: downsample → normalise → atanh activation
python pipeline/ingest.py

# 3. Feature extraction: reshape to 200×100, scale to uint8
python pipeline/features.py

# 4. Inspect the model architecture and CMSIS-NN kernel map
python pipeline/model.py

# 5. Quantisation-aware training
python pipeline/train.py

# 6. Convert to TFLite INT8 and generate firmware artefacts
python pipeline/convert.py
```

Or run the end-to-end notebook:

```bash
jupyter notebook notebooks/pipeline_demo.ipynb
```

---

## Package Overview

### Background

This pipeline implements the readout and deployment stage for a physical
reservoir computer based on a Hopf oscillator circuit. The analog circuit
produces two voltage states `x(t)` and `y(t)` at 100,000 Hz. Only `x(t)` is
used. The pipeline converts the raw signal into an INT8 CNN model whose every
operation maps to an optimised Arm assembly kernel, with zero generic C
fallbacks at inference time.

Based on:
- *"A Hopf physical reservoir computer"* — Shougat et al., Scientific Reports 2021
- *"Hopf physical reservoir computer for reconfigurable sound recognition"* — Shougat et al., Scientific Reports 2023

### Pipeline Stages

```
Hopf oscillator x(t) @ 100 kHz
        │
        ▼
┌─────────────────┐
│  1. ingest.py   │  Skip-sample 100 kHz → 4 kHz
│                 │  Normalise to [-1, +1]
│                 │  Apply atanh activation (eq. 6, paper 2)
└────────┬────────┘
         ▼
┌─────────────────┐
│  2. features.py │  Reshape to 200 (time) × 100 (virtual nodes) grid
│                 │  Scale to uint8 [0, 255] for CMSIS-NN input alignment
└────────┬────────┘
         ▼
┌─────────────────┐
│  3. model.py    │  CNN — every layer maps to a named CMSIS-NN kernel
└────────┬────────┘
         ▼
┌─────────────────┐
│  4. train.py    │  Quantisation-aware training (QAT)
│                 │  Fake-quantisation nodes simulate INT8 rounding
└────────┬────────┘
         ▼
┌─────────────────┐
│  5. convert.py  │  Full INT8 TFLite conversion
│                 │  Extract per-layer CMSIS-NN parameters
│                 │  Generate cmsis_nn_params.h + model_data.cc
│                 │  Print MCU fit report (M33 / M55 / M85)
└────────┬────────┘
         ▼
   firmware/cmsis_nn_params.h
   firmware/model_data.cc
   firmware/model.tflite
```

### CMSIS-NN Kernel Map

Every model layer targets a specific CMSIS-NN kernel. There are no generic C
fallbacks in the generated inference graph.

| Layer         | CMSIS-NN kernel                  | Source file                                          |
|---------------|----------------------------------|------------------------------------------------------|
| Conv2D + ReLU | `arm_convolve_s8()`              | `ConvolutionFunctions/arm_convolve_s8.c`             |
| MaxPool2D     | `arm_max_pool_s8()`              | `PoolingFunctions/arm_max_pool_s8.c`                 |
| Flatten       | `arm_reshape_s8()` (no-op)       | —                                                    |
| Dense + ReLU  | `arm_fully_connected_s8()`       | `FullyConnectedFunctions/arm_fully_connected_s8.c`   |
| Softmax       | `arm_softmax_s8()`               | `SoftmaxFunctions/arm_softmax_s8.c`                  |

### Generated Firmware Artefacts

`pipeline/convert.py` produces three files in `firmware/`:

| File                   | Contents                                                      |
|------------------------|---------------------------------------------------------------|
| `model.tflite`         | Fully INT8-quantised TFLite model                             |
| `cmsis_nn_params.h`    | `int8_t` weight arrays, `int32_t` bias arrays, per-channel quantisation parameters, `#define` constants for zero points and scales |
| `model_data.cc`        | TFLite Micro C array (`g_model_data[]`) for direct linking    |

`firmware/deployment_notes.md` contains per-MCU compiler flags, CMSIS-NN
linkage instructions, DMA configuration, and Vela/FVP commands.

### Model Architecture Constraints

The CNN is designed around CMSIS-NN's requirements:

- ReLU-only activations (`arm_relu_s8`) — no Swish, ELU, or custom activations
- Bias enabled on all Conv2D and Dense layers (required by CMSIS-NN kernel parameter structs)
- All tensor dimensions are multiples of 4 (SIMD alignment)
- Channels-last input: `(200, 100, 1)`
- No batch normalisation

---

## Project Structure

```
recho_pipeline/
├── data/
│   └── sample_data.py          # Hopf ODE integrator + 5 synthetic sound classes
├── pipeline/
│   ├── ingest.py               # Downsample, normalise, atanh activation
│   ├── features.py             # 200×100 reshape, uint8 scaling, visualisation
│   ├── model.py                # CNN definition with per-layer CMSIS-NN comments
│   ├── train.py                # QAT training loop + representative data generator
│   └── convert.py              # TFLite INT8 conversion + CMSIS-NN code generation
├── firmware/
│   ├── cmsis_nn_params.h       # Generated — do not edit
│   ├── model_data.cc           # Generated — do not edit
│   └── deployment_notes.md     # M33 / M55 / M85+Ethos-U55 deployment guide
├── notebooks/
│   └── pipeline_demo.ipynb     # End-to-end walkthrough with plots
├── setup.sh                    # uv venv setup script
├── requirements.txt
└── README.md
```

---

## Dependencies

| Package                          | Version    | Purpose                              |
|----------------------------------|------------|--------------------------------------|
| tensorflow                       | ≥2.13,<2.18 | Model training and TFLite conversion |
| numpy                            | ≥1.24      | Array operations                     |
| scipy                            | ≥1.11      | Hopf ODE integration (`solve_ivp`)   |
| matplotlib                       | ≥3.7       | Feature map visualisation            |
| flatbuffers                      | ≥23.5      | TFLite flatbuffer parsing            |
| tensorflow-model-optimization    | latest     | QAT (optional)                       |
