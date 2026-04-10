# Recho Pipeline

Post-processing pipeline for a **Hopf oscillator physical reservoir computer**,
targeting Arm Cortex-M microcontrollers with every inference layer mapped to a
named CMSIS-NN kernel function.

The Hopf oscillator circuit produces two voltage states `x(t)` and `y(t)` at
100 kHz. This pipeline ingests those signals, extracts feature maps, trains a
zoo of 26 classifiers and anomaly detectors, and exports firmware-ready INT8
artefacts for deployment on **M33**, **M55** (Helium MVE), or **M85 + Ethos-U55**.

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

# Generate evaluation plots and comparison table
python -m pipeline.evaluate

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

| # | Name | Category | Input | MCU Target | Notes |
|---|------|----------|-------|-----------|-------|
| A | CNN x-only | keras | x_only | M33/M55 | Baseline; INT8 CMSIS-NN |
| B | CNN xy-dual | keras | xy_dual | M33/M55 | Adds y(t) channel |
| C | CNN phase | keras | phase | M33/M55 | Orbit radius input |
| D | CNN angle | keras | angle | M33/M55 | Phase angle input |
| E | CNN late-fusion | keras | x+y | M55 | Two-branch model |
| F | Depthwise CNN | keras | xy_dual | M33 | ~8× faster, depthwise-sep |
| G | Ridge readout | keras | x/y/xy | M33 | Linear classifier, no conv |
| H | SPC Monitor | classical | x+y stream | M33 | Per-sample alert, <1 ms |
| I | Phase Portrait | classical | x+y clips | M33 | Shoelace orbit features |
| J | Recurrence QA | classical | x clips | M55 | RQA features, O(N²) |
| K | Hilbert Transform | classical | x clips | M33 | Instantaneous freq/amp |
| L | Autocorrelation | classical | x clips | M33 | Periodicity features |
| M | SVM Classifier | ml | x/y/xy/phase/angle | M33 | PCA→SVC, GridSearchCV |
| N | Random Forest | ml | x+y clips | M33 | 28 handcrafted features |
| O | GMM Detector | ml | x clips | M55 | Normal-only unsupervised |
| P | KNN Classifier | ml | x clips | M33 | k=5, pure-numpy MCU impl |
| Q | Isolation Forest | ml | x+y clips | M33 | Fast anomaly, no labels |
| R | Autoencoder | anomaly | x clips | M55 | Recon error threshold |
| S | One-Class SVM | anomaly | x clips | M55 | Normal-only decision fn |
| T | VAE | anomaly | x clips | M85 | ELBO score, smooth latent |
| U | Contrastive | anomaly | x clips | M85 | SimCLR pretrain + proto |
| V | TCN | sequence | raw 4kHz | M55 | Causal dilated conv |
| W | LSTM | sequence | raw 4kHz | M85 | TFLite Micro LSTM op |
| X | Echo State Net | sequence | x clips | M33 | Fixed reservoir, ridge out |
| Y | Prototypical Net | fewshot | x clips | M33 | 5-shot, no retraining |

---

## Use Case Recommendation

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| No labels available | Contrastive or VAE | Unsupervised pretraining; labels only needed for prototypes |
| Only normal data | VAE / Autoencoder / One-Class SVM | Trained on normal only; anomaly = high recon / ELBO score |
| <10 labelled examples | PrototypicalNetwork or Contrastive | Record 5 clips, update prototype; no GPU retraining |
| Must explain to customer | SPC Monitor or Phase Portrait | Physics-derived; thresholds in physical units |
| Real-time per-sample alert | SPC Monitor or Autocorrelation | O(1) per sample; no buffering; <1 ms on M33 |
| Switching tasks quickly | PrototypicalNetwork or KNN | Record 5 new clips, classification starts immediately |
| Best accuracy (M85 only) | LSTM or TCN | Sequence models; capture full temporal structure |
| M33 tight budget (64 KB) | SVM or Random Forest or SPC | PCA-compressed; pkl << 64 KB; <5 ms inference |

---

## Target Chips

| Chip | RAM | Key Feature | Fits |
|------|-----|-------------|------|
| Cortex-M33 | 64 KB | CMSIS-NN INT8, no SIMD | Classical, ML, ESN, SPC, KNN, SVM, RF, Prototypical |
| Cortex-M55 | 128 KB | Helium MVE, 8× SIMD | + CNN A–F, TCN, Autoencoder, One-Class SVM, GMM |
| Cortex-M85 + Ethos-U55 | 256 KB | NPU offload | + LSTM, VAE, Contrastive |

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
