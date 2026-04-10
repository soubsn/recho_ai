# Recho Pipeline

Post-processing pipeline for a **Hopf oscillator physical reservoir computer**,
targeting Arm Cortex-M microcontrollers with every inference layer mapped to a
named CMSIS-NN kernel function.

The Hopf oscillator circuit produces two voltage states `x(t)` and `y(t)` at
100 kHz. This pipeline ingests those signals, extracts feature maps, trains a
zoo of 26 classifiers and anomaly detectors, and exports firmware-ready INT8
artefacts for deployment on **M4**, **M33**, **M55** (Helium MVE), or
**M85 + Ethos-U55**.

**Physical basis:**
- Shougat et al., *"A Hopf physical reservoir computer"*, Scientific Reports 2021 (paper 1) —
  establishes the Hopf oscillator as a reservoir computer, shows `x(t)` alone
  achieves competitive classification; introduces the 200 × 100 virtual-node
  feature map used throughout this pipeline.
- Shougat et al., *"Hopf physical reservoir computer for reconfigurable sound recognition"*,
  Scientific Reports 2023 (paper 2) — adds `y(t)`, phase/angle representations,
  and demonstrates *in-hours reconfigurability*: record 5 new examples, update the
  classifier immediately; no GPU or retraining required.

---

## Quick Start

```bash
git clone <repo-url>
cd recho_pipeline
bash setup.sh
source recho-dev/bin/activate

# Train all models (add --skip_keras / --skip_ml etc. to select subsets)
python -m pipeline.train_all

# Train the causal denoiser on paired noisy/clean data
python -m pipeline.train_denoiser

# Generate evaluation plots and comparison table
python -m pipeline.evaluate

# Evaluate denoising quality
python -m pipeline.evaluate_denoiser

# Convert the denoiser for firmware deployment
python -m pipeline.convert_denoiser

# Run tests
pytest -q tests
```

---

## Pipeline Overview

```
Hopf oscillator hardware
  x(t) @ 100 kHz          y(t) @ 100 kHz (optional)
       │                         │
       ▼                         ▼
┌──────────────────────────────────────┐
│  pipeline/ingest.py                  │
│  Skip-sample 100 kHz → 4 kHz (×25)  │
│  Normalise to [-1, +1]               │
│  atanh activation  (eq. 6, paper 2)  │
└──────────────┬───────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌────────────────┐  ┌────────────────────────────────────┐
│ features.py    │  │ features_xy.py                     │
│ x_only         │  │ y_only, xy_dual, phase r(t), θ(t) │
│ 200×100 uint8  │  │ 200×100 uint8 per representation   │
└───────┬────────┘  └───────────────┬────────────────────┘
        └──────────┬────────────────┘
                   ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  26 models  (pipeline/models/)                               │
    │                                                              │
    │  KERAS (A–G)    CNN, depthwise CNN, ridge regression         │
    │  CLASSICAL      SPC, Phase Portrait, RQA, Hilbert, Autocorr  │
    │  ML             SVM, Random Forest, GMM, KNN, Isolation Fst  │
    │  ANOMALY        Autoencoder, One-Class SVM, VAE, Contrastive │
    │  SEQUENCE       TCN, LSTM, Echo State Network                │
    │  FEW-SHOT       Prototypical Network                         │
    └───────────────────────────────┬──────────────────────────────┘
                                    ▼
                   ┌────────────────────────────┐
                   │  pipeline/train_all.py      │
                   │  results/model_comparison   │
                   │  .csv                       │
                   └───────────────┬─────────────┘
                                   ▼
                   ┌────────────────────────────┐
                   │  pipeline/evaluate.py       │
                   │  output/eval/*.png          │
                   │  grouped bar chart          │
                   │  MCU heatmap                │
                   │  use-case table             │
                   └───────────────┬─────────────┘
                                   ▼
                   ┌────────────────────────────┐
                   │  pipeline/convert.py        │
                   │  firmware/model.tflite      │
                   │  firmware/cmsis_nn_params.h │
                   │  firmware/model_data.cc     │
                   └────────────────────────────┘
```

---

## Classification / Anomaly Pipeline

The existing package remains centered on classification and anomaly detection
from Hopf reservoir outputs. That workflow is unchanged:

- `pipeline/ingest.py` prepares `x(t)` feature maps
- `pipeline/features.py` and `pipeline/features_xy.py` build classifier inputs
- `pipeline/train_all.py` and `pipeline/evaluate.py` manage the model zoo
- `pipeline/convert.py` exports the classification models for firmware

## Denoising Pipeline

The denoising workflow is now a separate, parallel path designed for paired
noisy/clean training rather than class labels:

```
clean waveform + noise waveform
        │
        ▼
data/denoise_data.py
  paired (clean, noise, mixture)
        │
        ▼
pipeline/denoise_ingest.py
  mixture → Hopf reservoir → downsampled x(t), y(t) sequence
        │
        ├──────────────► clean target waveform (aligned)
        ▼
pipeline/models/denoising/tcn_denoiser.py
  causal sequence-to-sequence TCN
        │
        ├──────────────► pipeline/train_denoiser.py
        ├──────────────► pipeline/evaluate_denoiser.py
        └──────────────► pipeline/convert_denoiser.py
```

This path keeps the denoiser separate from the 26-model classifier/anomaly zoo,
because the task shape is different: paired sequence regression instead of
classification or anomaly scoring.

Current deployment status:
- the causal TCN denoiser is officially supported on `M55` and `M85`
- `M4` and `M33` denoising are not part of the current supported product story
- supporting denoising on `M4/M33` would require a smaller model, not just a
  different converter setting

Conversion note:
- `pipeline.convert_denoiser` depends on a TFLite-compatible TensorFlow stack.
- Pin `protobuf>=3.20.3,<5` with TensorFlow 2.16.x. Newer protobuf releases can
  cause opaque MLIR conversion failures before export.

---

## Input Format

| Representation | Shape       | Dtype  | Description                                    |
|----------------|-------------|--------|------------------------------------------------|
| `x_only`       | (200, 100)  | uint8  | Published reservoir state — baseline input     |
| `y_only`       | (200, 100)  | uint8  | Second oscillator state (paper 2)              |
| `xy_dual`      | (200, 100, 2)| uint8 | x and y stacked as two channels               |
| `phase`        | (200, 100)  | uint8  | Orbit radius r(t) = √(x²+y²)                  |
| `angle`        | (200, 100)  | uint8  | Phase angle θ(t) = arctan2(y, x)              |
| raw signal     | (4000,)     | float32| Downsampled 4 kHz signal for sequence models  |

Each clip corresponds to one 1-second recording at 4 kHz (4,000 samples),
reshaped into a 200 time-step × 100 virtual-node feature map.

---

## Model Zoo

26 models across 6 categories:

Support labels used below:
- `officially supported`: part of the current deployment story for that core
- `conditional`: may work with board-specific tuning or a custom firmware path,
  but is not part of the default supported product profile
- `not recommended`: outside the current supported deployment profile

| # | Name | Category | Input | Deployment Support | Notes |
|---|------|----------|-------|--------------------|-------|
| A | CNN x-only | keras | x_only | Official: M55/M85 | Baseline CNN; not recommended on M4/M33 |
| B | CNN xy-dual | keras | xy_dual | Official: M55/M85 | Adds y(t); not recommended on M4/M33 |
| C | CNN phase | keras | phase | Official: M55/M85 | Radius input; not recommended on M4/M33 |
| D | CNN angle | keras | angle | Official: M55/M85 | Angle input; not recommended on M4/M33 |
| E | CNN late-fusion | keras | x+y | Official: M55/M85 | Premium two-branch model |
| F | Depthwise CNN | keras | xy_dual | Official: M33/M55/M85; Conditional: M4 | Lightweight learned model |
| G | Ridge readout | keras | x/y/xy | Official: M33/M55/M85; Conditional: M4 | Linear readout, minimal compute |
| H | SPC Monitor | classical | x+y stream | Official: M4/M33/M55/M85 | Per-sample alert, <1 ms |
| I | Phase Portrait | classical | x+y clips | Official: M4/M33/M55/M85 | Shoelace orbit features |
| J | Recurrence QA | classical | x clips | Official: M55/M85; Conditional: M33 | RQA features, O(N²) |
| K | Hilbert Transform | classical | x clips | Official: M4/M33/M55/M85 | Instantaneous freq/amp |
| L | Autocorrelation | classical | x clips | Official: M4/M33/M55/M85 | Periodicity features |
| M | SVM Classifier | ml | x/y/xy/phase/angle | Official: M85; Conditional: M33 | Useful analytically; not a default low-cost firmware target |
| N | Random Forest | ml | x+y clips | Official: M4/M33/M55/M85 | 28 handcrafted features |
| O | GMM Detector | ml | x clips | Official: M55/M85; Conditional: M33 | Normal-only unsupervised |
| P | KNN Classifier | ml | x clips | Official: M4/M33/M55/M85 | k=5, pure-numpy MCU impl |
| Q | Isolation Forest | ml | x+y clips | Official: M4/M33/M55/M85 | Fast anomaly, no labels |
| R | Autoencoder | anomaly | x clips | Official: M55/M85 | Recon error threshold |
| S | One-Class SVM | anomaly | x clips | Official: M55/M85; Conditional: M33 | Normal-only decision fn |
| T | VAE | anomaly | x clips | Official: M85 | ELBO score, smooth latent |
| U | Contrastive | anomaly | x clips | Official: M85 | SimCLR pretrain + proto |
| V | TCN | sequence | raw 4kHz | Official: M55/M85; Conditional: M33 | Causal dilated conv |
| W | LSTM | sequence | raw 4kHz | Official: M85 | TFLite Micro LSTM op |
| X | Echo State Net | sequence | x clips | Official: M4/M33/M55/M85 | Fixed reservoir, ridge out |
| Y | Prototypical Net | fewshot | x clips | Official: M4/M33/M55/M85 | 5-shot, no retraining |

---

## Use Case Recommendation

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| No labels available | Contrastive or VAE | Unsupervised pretraining; labels only needed for prototypes |
| Only normal data | VAE / Autoencoder / One-Class SVM | Trained on normal only; anomaly = high recon / ELBO score |
| <10 labelled examples | PrototypicalNetwork or Contrastive | Record 5 clips, update prototype; no GPU retraining |
| Must explain to customer | SPC Monitor or Phase Portrait | Physics-derived; thresholds in physical units |
| Real-time per-sample alert | SPC Monitor or Autocorrelation | O(1) per sample; no buffering; strong fit for M4/M33 |
| Switching tasks quickly | PrototypicalNetwork or KNN | Record 5 new clips, classification starts immediately |
| Best accuracy with premium hardware | CNN late-fusion or LSTM | Richer models; strongest story for M55/M85 products |
| M4/M33 tight budget | SPC or Random Forest or KNN | Clear low-cost deployment story; easier to defend commercially |

---

## Deployment Policy

`M4` and `M33` are not interchangeable in this package.

- `M4` is treated as the most cost-sensitive deployment tier. It is where the
  package leans on classical signal processing, tree-based models, nearest
  neighbour methods, and prototypes.
- `M33` adds headroom for a few lightweight learned models such as the
  depthwise CNN and ridge readout, but it is still not the default target for
  the heavier CNN and sequence families.
- `M55` is the point where the package's richer neural models become part of
  the official deployment story, including the current denoiser.
- `M85` is the premium tier for the heaviest sequence and representation
  learning models.

The converter's deployment summary is a screening tool, not a final shipping
guarantee. A successful `.tflite` export does not by itself prove board-level
support.

---

## Target Chips

| Chip | Effective Budget | Business Meaning | Officially Supported Families | Conditional / Notes |
|------|------------------|------------------|------------------------------|---------------------|
| Cortex-M4 | 48 KB | Lowest-cost always-on edge tier | SPC, Phase Portrait, Hilbert, Autocorrelation, Random Forest, KNN, Isolation Forest, ESN, Prototypical | Depthwise CNN and ridge readout are possible but not part of the default supported profile |
| Cortex-M33 | 64 KB | Main low-power embedded tier | M4 set plus Depthwise CNN and Ridge | Recurrence, SVM, GMM, One-Class SVM, and TCN are conditional rather than default |
| Cortex-M55 | 128 KB | Main neural inference tier | Small models plus CNN A-F, Recurrence, GMM, Autoencoder, One-Class SVM, TCN, current TCN denoiser | Best entry point for richer real-time ML on-device |
| Cortex-M85 + Ethos-U55 | 256 KB | Premium compute tier | Full model zoo, including LSTM, VAE, Contrastive, and current denoiser | Best fit when top-end temporal accuracy matters most |

---

## CMSIS-NN Kernel Map

| Layer | CMSIS-NN Kernel | Source |
|-------|-----------------|--------|
| Conv2D + ReLU | `arm_convolve_s8()` | `ConvolutionFunctions/arm_convolve_s8.c` |
| DepthwiseConv2D | `arm_depthwise_conv_s8()` | `ConvolutionFunctions/arm_depthwise_conv_s8.c` |
| Conv1D (TCN) | `arm_convolve_1_x_n_s8()` | `ConvolutionFunctions/arm_convolve_1_x_n_s8.c` |
| MaxPool2D | `arm_max_pool_s8()` | `PoolingFunctions/arm_max_pool_s8.c` |
| Dense + ReLU | `arm_fully_connected_s8()` | `FullyConnectedFunctions/arm_fully_connected_s8.c` |
| Softmax | `arm_softmax_s8()` | `SoftmaxFunctions/arm_softmax_s8.c` |

---

## Generated Firmware Artefacts

| File | Contents |
|------|----------|
| `firmware/model.tflite` | Fully INT8-quantised TFLite model |
| `firmware/cmsis_nn_params.h` | int8_t weights, int32_t biases, per-channel quant params |
| `firmware/model_data.cc` | TFLite Micro C array for direct linking |
| `firmware/prototypes.h` | int8_t prototype vectors (PrototypicalNetwork) |
| `firmware/knn_references.h` | int8_t reference embeddings (KNN) |
| `firmware/random_forest.h` | if-else decision tree (Random Forest) |
| `firmware/esn_output_weights.h` | float32 output weight matrix (Echo State Network) |
| `firmware/isolation_forest.h` | path-length counter (Isolation Forest) |

---

## Installation

```bash
git clone <repo-url>
cd recho_pipeline
bash setup.sh
source recho-dev/bin/activate
```

`setup.sh` creates a `uv` virtual environment named `recho-dev`, installs all
dependencies from `requirements.txt`, and optionally installs
`tensorflow-model-optimization` for QAT.

---

## Running the Pipeline

```bash
# Generate synthetic data
python data/sample_data.py

# Ingest: downsample → normalise → atanh
python pipeline/ingest.py

# Feature extraction
python pipeline/features.py

# Train a single canonical model
python pipeline/train.py

# Train all 26 models
python -m pipeline.train_all

# Train selected categories only
python -m pipeline.train_all --skip_keras --skip_sequence

# Evaluate and generate plots
python -m pipeline.evaluate

# Convert to TFLite INT8
python pipeline/convert.py
```

---

## Testing

```bash
# Fast suite
pytest -q tests

# With TensorFlow model smoke tests
RUN_TF_SMOKE=1 pytest -q tests/test_models_smoke.py

# Disable global plugin conflicts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests
```

---

## Project Structure

```
recho_pipeline/
├── data/
│   └── sample_data.py                  # Hopf ODE integrator + 5 synthetic classes
├── pipeline/
│   ├── ingest.py                       # Downsample, normalise, atanh activation
│   ├── features.py                     # 200×100 reshape, uint8 scaling
│   ├── features_xy.py                  # y(t), phase, angle, dual-channel features
│   ├── model.py                        # Canonical CNN (CMSIS-NN comment map)
│   ├── train.py                        # QAT training loop
│   ├── train_all.py                    # Train all 26 models; writes model_comparison.csv
│   ├── evaluate.py                     # Evaluation, comparison table, 7 plots
│   ├── convert.py                      # TFLite INT8 + firmware code generation
│   └── models/
│       ├── cnn_x_only.py               # Model A
│       ├── cnn_xy_dual.py              # Model B
│       ├── depthwise_cnn.py            # Model F
│       ├── ensemble.py                 # VoteEnsemble
│       ├── reservoir_readout.py        # ReservoirReadout (Model G)
│       ├── classical/
│       │   ├── spc.py                  # SPC Monitor (Western Electric rules)
│       │   ├── phase_portrait.py       # Phase Portrait Classifier
│       │   ├── recurrence.py           # Recurrence QA Classifier
│       │   ├── hilbert.py              # Hilbert Transform Classifier
│       │   └── autocorrelation.py      # Autocorrelation Classifier
│       ├── ml/
│       │   ├── svm_classifier.py       # SVM (PCA + GridSearchCV)
│       │   ├── random_forest.py        # Random Forest (28 features)
│       │   ├── gmm_anomaly.py          # GMM Anomaly Detector
│       │   ├── knn_classifier.py       # KNN Classifier
│       │   └── isolation_forest.py     # Isolation Forest
│       ├── sequence/
│       │   ├── tcn.py                  # TCN (causal dilated conv)
│       │   ├── lstm_classifier.py      # LSTM Classifier
│       │   └── esn_readout.py          # Echo State Network
│       ├── anomaly/
│       │   ├── autoencoder.py          # Autoencoder Anomaly Detector
│       │   ├── one_class_svm.py        # One-Class SVM Detector
│       │   ├── vae.py                  # VAE Anomaly Detector
│       │   └── contrastive.py          # Contrastive Classifier (SimCLR)
│       └── fewshot/
│           └── prototypical.py         # Prototypical Network (5-shot)
├── firmware/                           # Generated firmware artefacts
├── results/
│   └── model_comparison.csv            # Written by train_all.py
├── output/eval/                        # Written by evaluate.py
├── notebooks/
│   └── pipeline_demo.ipynb
├── tests/
│   ├── test_hopf.py
│   ├── test_ingest_features.py
│   ├── test_evaluate_contracts.py
│   └── test_models_smoke.py
├── setup.sh
├── requirements.txt
├── model_directory.md                  # Quick reference for all 26 models
└── README.md
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | ≥2.13,<2.18 | Keras models, TFLite conversion |
| numpy | ≥1.24 | Array operations |
| scipy | ≥1.11 | Hopf ODE integration, Hilbert transform |
| matplotlib | ≥3.7 | Feature map visualisation, evaluation plots |
| flatbuffers | ≥23.5 | TFLite flatbuffer parsing |
| scikit-learn | ≥1.3 | SVM, RF, GMM, KNN, Isolation Forest, PCA |
| joblib | ≥1.3 | sklearn model serialisation (.pkl) |
| tensorflow-model-optimization | latest | QAT (optional) |

---

## Further Reading

- [model_directory.md](model_directory.md) — per-model quick reference with training examples and deployment notes
- `firmware/deployment_notes.md` — per-MCU compiler flags, CMSIS-NN linkage, DMA, Vela/FVP
