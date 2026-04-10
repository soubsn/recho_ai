# Model Directory

Quick reference for all 26 models in the Recho Pipeline.

---

## Quick Reference Table

| # | Model | Category | Class / Function | Input | MCU | Checkpoint |
|---|-------|----------|-----------------|-------|-----|-----------|
| A | CNN x-only | keras | `build_cnn_x_only()` | x_only (200×100) | M33/M55 | `checkpoints/model_a_cnn_x_only_best.keras` |
| B | CNN xy-dual | keras | `build_cnn_xy_dual()` | xy_dual (200×100×2) | M33/M55 | `checkpoints/model_b_cnn_xy_dual_best.keras` |
| C | CNN phase | keras | `build_model()` (cnn_x_only) | phase (200×100) | M33/M55 | `checkpoints/model_c_cnn_phase_best.keras` |
| D | CNN angle | keras | `build_model()` (cnn_x_only) | angle (200×100) | M33/M55 | `checkpoints/model_d_cnn_angle_best.keras` |
| E | CNN late-fusion | keras | `build_late_fusion()` | x+y (200×100×1 each) | M55 | `checkpoints/model_e_cnn_xy_fusion_best.keras` |
| F | Depthwise CNN | keras | `build_depthwise_cnn()` | xy_dual (200×100×2) | M33 | `checkpoints/model_f_depthwise_cnn_best.keras` |
| G | Ridge readout | keras/sklearn | `ReservoirReadout.fit_all_readouts()` | x/y/xy | M33 | `checkpoints/ridge_*.pkl` |
| H | SPC Monitor | classical | `SPCMonitor` | x+y stream | M33 | in-memory only |
| I | Phase Portrait | classical | `PhasePortraitClassifier` | x+y clips | M33 | `checkpoints/phase_portrait.pkl` |
| J | Recurrence QA | classical | `RecurrenceClassifier` | x clips | M55 | `checkpoints/recurrence.pkl` |
| K | Hilbert Transform | classical | `HilbertClassifier` | x clips | M33 | `checkpoints/hilbert.pkl` |
| L | Autocorrelation | classical | `AutocorrClassifier` | x clips | M33 | `checkpoints/autocorr.pkl` |
| M | SVM | ml | `SVMClassifier` | any 200×100 | M33 | `checkpoints/svm_*.pkl` |
| N | Random Forest | ml | `RandomForestModel` | x+y clips | M33 | `checkpoints/random_forest.pkl` |
| O | GMM Detector | ml | `GMMDetector` | x clips | M55 | `checkpoints/gmm_anomaly.pkl` |
| P | KNN | ml | `KNNClassifier` | x clips | M33 | `checkpoints/knn.pkl` |
| Q | Isolation Forest | ml | `IsolationForestModel` | x+y clips | M33 | `checkpoints/isolation_forest.pkl` |
| R | Autoencoder | anomaly | `AnomalyAutoencoder` | x clips | M55 | `checkpoints/autoencoder.keras` |
| S | One-Class SVM | anomaly | `OneClassSVMDetector` | x clips | M55 | `checkpoints/one_class_svm.pkl` |
| T | VAE | anomaly | `VAEDetector` | x clips | M85 | `checkpoints/vae_weights.h5` |
| U | Contrastive | anomaly | `ContrastiveClassifier` | x clips | M85 | `checkpoints/contrastive_encoder.keras` |
| V | TCN | sequence | `TCNClassifier` | raw 4kHz (4000,) | M55 | `checkpoints/tcn.keras` |
| W | LSTM | sequence | `LSTMClassifier` | raw 4kHz (T, 2) | M85 | `checkpoints/lstm.keras` |
| X | Echo State Net | sequence | `EchoStateReadout` | x clips | M33 | `checkpoints/esn.pkl` |
| Y | Prototypical Net | fewshot | `PrototypicalNetwork` | x clips | M33 | in-memory / `firmware/prototypes.h` |

---

## Recommendations

Ranked recommendations for common product scenarios.

### 1. Detecting a specific sound in a normal environment

**1. CNN x-only (A)**
Best default when the environment is relatively controlled and you want a
strong production classifier. It uses the published reservoir representation
directly and is the easiest model to position as the main baseline.

**2. Prototypical Network (Y)**
Best when the specific sound may vary by customer and you want fast onboarding
with only a few examples. This is especially useful if the workflow is
"record 5 examples of the target sound and start detecting immediately."

**3. SVM Classifier (M)**
Best classical ML backup when you want strong accuracy without relying on a
larger neural network. It is a good option for smaller datasets and gives a
credible lower-compute alternative to the CNN family.

### 2. Detecting a specific sound in a high-noise environment

**1. CNN late-fusion (E)**
Best overall choice in this package for difficult acoustic environments when
both `x(t)` and `y(t)` are available. It lets the model examine each channel
independently before combining evidence, which gives it the best chance of
separating target structure from noisy surroundings.

**2. TCN Classifier (V)**
Best sequence-oriented option when the timing pattern of the sound matters and
background noise is heavy. Because it works directly on the waveform over time,
it can capture temporal structure that image-style feature models may blur.

**3. Contrastive Classifier (U)**
Best if you expect domain shift and messy real-world conditions. Its main
strength is learning robust embeddings from large amounts of unlabelled data,
which is useful when office noise, room acoustics, and microphone conditions
vary from site to site.

### 3. Suppressing background noise

This package does **not** currently contain a dedicated noise-suppression,
source-separation, or speech-enhancement model. The models below are the
closest fits if the practical goal is to stay robust in noise or to model
"clean" behaviour, but they do not perform true waveform denoising as
implemented today.

**1. Autoencoder Anomaly Detector (R)**
Closest current fit if you want a model that learns the structure of normal
signals and reconstructs them. In principle this is the most natural starting
point for a future denoising model, because autoencoders can be adapted to map
noisy input to cleaner output. In the current package, however, it is used for
anomaly scoring rather than noise suppression.

**2. VAE Anomaly Detector (T)**
Second-best foundation for future suppression work. Like the autoencoder, it
learns a compact latent representation of normal behaviour, but with a smoother
latent space that can be more stable under variation. Again, this is a good
research direction, not a ready-made denoiser in the current code.

**3. TCN Classifier (V)**
Best practical fallback if your real need is not to remove noise from the
audio itself, but to keep detecting the target sound despite noise. Sequence
models are often the most sensible starting point when temporal structure must
survive cluttered backgrounds.

### 4. Lowest-cost embedded deployment

**1. Depthwise CNN (F)**
Best balance of modern pattern recognition and low compute cost. It is the
strongest candidate when the commercial goal is to fit useful intelligence onto
smaller, cheaper microcontrollers without giving up too much performance.

**2. Ridge Readout (G)**
Best ultra-lightweight classifier when memory, flash, and latency are all
tight. It is simple, fast, and easy to deploy, which makes it attractive for
very cost-sensitive products.

**3. SPC Monitor (H)**
Best if the requirement is continuous monitoring with almost no compute budget.
It is not a general classifier, but it is the cheapest way in the package to
provide immediate abnormality alerts on-device.

### 5. Fastest field reconfiguration

**1. Prototypical Network (Y)**
Best overall choice when new sound classes need to be added immediately in the
field. A few examples are enough to build a new class representation, with no
full retraining cycle.

**2. Contrastive Classifier (U)**
Best higher-end option when you want rapid adaptation on top of a strong
pretrained representation. It is especially attractive when the product can
learn from large pools of unlabelled data and then adapt quickly at deployment.

**3. KNN Classifier (P)**
Best simple operational fallback. New categories can be added by storing new
reference examples, making it easy to understand and easy to update.

### 6. Only normal data available

**1. GMM Detector (O)**
Best lightweight anomaly detector when you can gather lots of normal data but
very little fault data. It is commercially useful for monitoring and
preventive-maintenance style deployments.

**2. One-Class SVM Detector (S)**
Best when you want a stricter learned boundary around acceptable behaviour. It
is a strong option when the goal is to define what "normal" looks like and
reject everything else.

**3. Autoencoder Anomaly Detector (R)**
Best when you want a more intuitive anomaly story based on reconstruction
error. It is especially useful if you want to show examples of what the model
can and cannot reconstruct cleanly.

### 7. Most explainable to a customer

**1. SPC Monitor (H)**
Best pure explainability option. It is based on explicit statistical rules,
which makes it easy to justify in business, industrial, or regulated settings.

**2. Phase Portrait Classifier (I)**
Best when you want a physical, visual explanation of what differs between
classes. It gives a human-readable story about the geometry of the oscillator
orbit.

**3. Random Forest (N)**
Best when you still want a stronger learned model but need to show which
features mattered. Feature importance gives a more business-friendly narrative
than a black-box neural network.

### 8. Best raw accuracy if hardware budget is available

**1. CNN late-fusion (E)**
Best flagship classifier in the package when both oscillator channels are
available and compute budget is not the first constraint. It is the strongest
high-performance model to highlight in a premium product story.

**2. LSTM Classifier (W)**
Best memory-based sequence model for capturing complex temporal structure. It
is appropriate when the signal history matters and high-end hardware is
available.

**3. TCN Classifier (V)**
Best practical high-performance sequence alternative when you want strong
temporal modelling with lower deployment cost than the LSTM.

### 9. Always-on real-time alerting

**1. SPC Monitor (H)**
Best overall option for continuous live monitoring. It is the closest thing in
the package to an always-on guardrail that can react immediately.

**2. Autocorrelation Classifier (L)**
Best when the business problem depends on stable repetition and cycle quality.
It is useful for detecting when periodic structure weakens or drifts.

**3. Depthwise CNN (F)**
Best learned-model option when you still need embedded intelligence in a
low-latency setting. It offers a better chance of fitting on-device than the
heavier CNNs.

### 10. Best when only a small labelled dataset exists

**1. Prototypical Network (Y)**
Best fit for few-shot use cases by design. It is the clearest option when
labelled examples are scarce and deployment speed matters.

**2. SVM Classifier (M)**
Best traditional ML option for small datasets. It often generalises well
before deep models have enough data to shine.

**3. KNN Classifier (P)**
Best simple baseline when the data is limited and you want a model that stays
close to the examples themselves.

### 11. Best when both x(t) and y(t) are available

**1. CNN late-fusion (E)**
Best use of the full oscillator state when hardware budget allows. It gives
each channel its own path before combining evidence, making it the strongest
showcase of dual-channel sensing.

**2. CNN xy-dual (B)**
Best simpler dual-channel model when you want the benefit of both oscillator
states without the full cost of late fusion.

**3. Random Forest (N)**
Best explainable dual-channel alternative when you want to use information
from both states but keep the reasoning more transparent.

### 12. Future roadmap: true denoising

This package does **not** yet include a true denoising, source-separation, or
speech-enhancement model. If this becomes a product priority, the most natural
starting points in the current package are the `Autoencoder (R)`, `VAE (T)`,
and sequence models such as `TCN (V)`, but that would require a new training
objective and likely new paired noisy/clean data.

## Denoising Additions

These denoising components are implemented as a separate pipeline and are not
part of the A-Y classifier/anomaly ranking.

| Name | Type | Main Files | Input | Output | Target |
|------|------|------------|-------|--------|--------|
| TCN Denoiser | denoising | `data/denoise_data.py`, `pipeline/denoise_ingest.py`, `pipeline/models/denoising/tcn_denoiser.py` | Hopf reservoir sequence `(T, 2)` from noisy mixture | Clean waveform `(T, 1)` | M55 / M85 |

**What it does:** Generates paired noisy/clean waveforms, passes the noisy
mixture through the Hopf reservoir, and trains a causal TCN to reconstruct a
clean target waveform from the resulting `x(t), y(t)` sequence.

**Why it is separate:** Denoising is sequence-to-sequence regression, not
classification or anomaly detection. It uses different data, metrics,
conversion output, and deployment assumptions, so it lives in its own train,
evaluate, and convert entrypoints.

---

## Model Sections

### A — CNN x-only (Baseline)

**What it does:** Two-block CNN on the `x(t)` feature map. The published
baseline from Shougat et al. 2021 / 2023. Every layer maps to a CMSIS-NN
kernel.

**When to use:** Default choice when only `x(t)` is available and accuracy
matters more than speed.

**What it all means:** This is the standard "smart pattern recogniser" for the
main sensor signal. It looks at the signal as an image and learns the visual
signature of each class. If you want one strong default model that is easy to
position as the core product baseline, this is usually the place to start.

**Input requirements:** `x_only` feature map, shape `(200, 100)` uint8.

**Training example:**
```python
from pipeline.models.cnn_x_only import build_model
model = build_model(n_classes=5)
model.fit(X_train[..., np.newaxis], y_train, epochs=30)
```

**Deployment notes:** INT8 TFLite. Fits M33 (64 KB). Generate with
`pipeline/convert.py`. Checkpoint: `checkpoints/model_a_cnn_x_only_best.keras`.

**Known limitations:** Uses `x(t)` only — adding `y(t)` (model B or E) may
improve accuracy for frequency-modulated signals.

---

### B — CNN xy-dual

**What it does:** Same CNN backbone as A but takes `x(t)` and `y(t)` stacked
as a 2-channel input. Paper 2 shows that including `y(t)` improves accuracy
for classes that differ in phase rather than amplitude.

**When to use:** When both oscillator outputs are sampled and accuracy is
prioritised over flash size.

**What it all means:** This model listens with both "ears" instead of one.
That gives it a fuller picture of how the oscillator is moving, which can help
separate cases that look similar in one channel alone. It is a good choice
when you want to show investors that the platform can unlock more value simply
by reading more of the physics already present in the device.

**Input requirements:** `xy_dual` feature map, shape `(200, 100, 2)` uint8.

**Training example:**
```python
from pipeline.models.cnn_xy_dual import build_cnn_xy_dual
model = build_cnn_xy_dual(n_classes=5)
model.fit(X_xy_train, y_train, epochs=30)
```

**Deployment notes:** INT8 TFLite. Slightly larger than A (extra input
channel). Fits M33/M55.

**Known limitations:** Requires both ADC channels; adds hardware complexity.

---

### C — CNN phase (orbit radius)

**What it does:** Same CNN as A applied to the orbit radius representation
`r(t) = √(x²+y²)`. The radius is computed in `features_xy.py` before
training.

**When to use:** When the signal class differs in oscillation amplitude.

**What it all means:** Instead of watching the full motion, this model focuses
on how far the oscillator is being pushed from its normal orbit. In plain
language, it is measuring "how strong the disturbance is." This is useful when
the business problem is mainly about changes in intensity rather than timing.

**Input requirements:** `phase` feature map (orbit radius), shape `(200, 100)`.

**Training example:**
```python
reps = extract_all_representations(x_proc, y_proc)
model.fit(reps["phase"][..., np.newaxis], y_train, epochs=30)
```

**Deployment notes:** Identical architecture to A. `r(t)` must be computed
before sending data to MCU (one `arm_sqrt_f32()` per sample).

**Known limitations:** Loses phase-angle information — may underperform for
frequency-modulated signals.

---

### D — CNN angle (phase angle)

**What it does:** Same CNN as A applied to the phase angle
`θ(t) = arctan2(y, x)`. Captures rotational dynamics.

**When to use:** When signals differ in phase progression rather than amplitude.

**What it all means:** This model pays attention to where the oscillator is in
its cycle and how that cycle evolves over time. In non-technical terms, it is
better at seeing changes in rhythm than changes in strength. It is useful when
two events have similar energy but different timing patterns.

**Input requirements:** `angle` feature map, shape `(200, 100)`.

**Training example:**
```python
model.fit(reps["angle"][..., np.newaxis], y_train, epochs=30)
```

**Deployment notes:** `arctan2` computed offline; INT8 TFLite identical to A.

**Known limitations:** Loses amplitude information — complement with C or B.

---

### E — CNN late-fusion (x+y two-branch)

**What it does:** Two independent CNN branches (one for `x`, one for `y`)
whose outputs are concatenated and passed to a shared dense head. Captures
independent information in each oscillator state.

**When to use:** When both signals are available and maximum accuracy is
required, budget permitting.

**What it all means:** This is the "premium" version of the CNN family. It
lets the model study each oscillator channel separately before combining the
evidence, which can improve accuracy on harder problems. It is the right story
for high-value use cases where performance matters more than compute cost.

**Input requirements:** Separate `x_only` and `y_only` tensors, each
`(200, 100, 1)`.

**Training example:**
```python
model.fit([X_x_train, X_y_train], y_train, epochs=30)
```

**Deployment notes:** Two parallel inference branches — requires M55 or M85
(~180 KB activation RAM). Uses `model.predict([x, y])`.

**Known limitations:** ~2× inference time versus model A. Higher RAM.

---

### F — Depthwise CNN (M33 optimised)

**What it does:** Depthwise-separable convolutions on the `xy_dual` map.
`arm_depthwise_conv_s8()` + 1×1 `arm_convolve_s8()` pointwise step. ~8×
fewer MACs than A.

**When to use:** Tight M33 budget; latency-sensitive.

**What it all means:** This is the lightweight efficiency model. It tries to
keep most of the pattern-recognition power of a CNN while cutting the compute
bill enough to fit on smaller, cheaper microcontrollers. It is useful when the
commercial goal is low power, low cost, or edge deployment at scale.

**Input requirements:** `xy_dual`, shape `(200, 100, 2)`.

**Training example:**
```python
from pipeline.models.depthwise_cnn import build_depthwise_cnn
model = build_depthwise_cnn(n_classes=5)
```

**Deployment notes:** Dedicated CMSIS-NN kernel `arm_depthwise_conv_s8()`.
Fits comfortably in M33 64 KB.

**Known limitations:** Slight accuracy drop vs. standard conv (typical for
depthwise models). Validate A vs F on your dataset before deploying.

---

### G — Ridge Readout

**What it does:** Linear ridge regression on flattened feature maps. No
convolution — just a matrix multiply (`arm_fully_connected_s8()`). Three
variants: x_only, y_only, xy_concatenated.

**When to use:** Absolute minimum RAM/flash budget; explainability required.

**What it all means:** This is a very simple decision layer on top of the
reservoir features. It does not try to learn deep patterns; it mostly weighs
evidence and makes a fast linear decision. That makes it easy to deploy and
easy to explain, so it is a good fit for constrained hardware or early demos.

**Input requirements:** Flattened 200×100 = 20,000-element vector.

**Training example:**
```python
from pipeline.models.reservoir_readout import ReservoirReadout
rr = ReservoirReadout()
rr.fit_all_readouts(reps, labels)
```

**Deployment notes:** Saved as `checkpoints/ridge_*.pkl`. Load with joblib.
Extremely fast — sub-millisecond on M33.

**Known limitations:** Linear classifier; cannot capture non-linear class
boundaries. Use as a fast baseline, not a production model.

---

### H — SPC Monitor

**What it does:** Statistical Process Control monitor applying Western Electric
Rules 1–4 to the orbit radius `r = √(x²+y²)`. Anomaly detection with one
`arm_sqrt_f32()` per sample — no buffering.

**When to use:** Real-time per-sample streaming alert; hardware must respond
within microseconds.

**What it all means:** This is more like a guardrail than a classifier. It
continuously watches the signal and raises an alert when behaviour moves
outside the normal operating envelope. For investors, the message is that the
platform can support immediate monitoring and safety-style detection, not just
offline analytics.

**Input requirements:** Per-sample `(x, y)` floats from the ADC.

**Training example:**
```python
from pipeline.models.classical.spc import SPCMonitor
mon = SPCMonitor(sigma_n=3.0)
mon.fit(x_normal, y_normal)
result = mon.update(x_sample, y_sample)
# result: {"anomaly": bool, "radius": float, "sigma_level": float, "rule_violated": str}
```

**Deployment notes:** No checkpoint — fit parameters (mean, std) stored in
firmware as `float` constants. Rules implemented in ~50 lines of C.

**Known limitations:** Only monitors orbit radius — insensitive to changes
that preserve amplitude (e.g. frequency shift only).

---

### I — Phase Portrait Classifier

**What it does:** Extracts 4 geometric features from the Lissajous orbit
(area, eccentricity, centre drift, variance) and classifies with a
`RidgeClassifier`.

**When to use:** Explainable classification; when physical orbit geometry
distinguishes signal classes.

**What it all means:** This model turns the oscillator motion into a geometric
shape and then judges that shape using a few easy-to-understand measurements.
It is less about raw accuracy and more about having a story a human can follow:
"this class makes a wider orbit," or "that class shifts the centre." That is
valuable in customer presentations and regulated environments.

**Input requirements:** Paired `x_clip` and `y_clip`, shape `(200, 100)` each.

**Training example:**
```python
from pipeline.models.classical.phase_portrait import PhasePortraitClassifier
clf = PhasePortraitClassifier()
clf.fit(x_clips, y_clips, labels)
clf.save()
```

**Deployment notes:** Saved as `checkpoints/phase_portrait.pkl`. Feature
extraction uses the shoelace formula for orbit area — implementable on M33
with a small loop.

**Known limitations:** 4-feature linear model; may not separate all 5 classes.
Use alongside another model as an interpretability layer.

---

### J — Recurrence QA Classifier

**What it does:** Builds the recurrence matrix on a downsampled version of the
signal and extracts 5 RQA metrics (recurrence rate, determinism, laminarity,
entropy, trapping time). Classifies with `RandomForestClassifier`.

**When to use:** Rich characterisation of dynamical behaviour; research or
diagnostic use.

**What it all means:** This is a specialist dynamics model. It asks whether
the signal repeats, stalls, or evolves in structured ways over time. It is
useful when you want to study the behaviour deeply and understand the system,
but it is better positioned as an advanced analysis tool than as the cheapest
production option.

**Input requirements:** `x_clip`, shape `(200, 100)`. `max_samples=500` for
practical compute time (O(N²) truncated).

**Training example:**
```python
from pipeline.models.classical.recurrence import RecurrenceClassifier
clf = RecurrenceClassifier()
clf.fit(x_clips, labels)
```

**Deployment notes:** RQA computation is expensive — MCU deployment requires
pre-computation or a dedicated coprocessor. Checkpoint:
`checkpoints/recurrence.pkl`.

**Known limitations:** O(N²) complexity; not suitable for real-time use on M33
without significant approximation.

---

### K — Hilbert Transform Classifier

**What it does:** Applies `scipy.signal.hilbert` to extract instantaneous
amplitude and frequency, then classifies 5 features
(mean/std frequency, mean envelope, AM index, envelope_std) with `SVC(rbf)`.

**When to use:** Frequency-modulated or amplitude-modulated signals; when
frequency trajectory over time is the distinguishing feature.

**What it all means:** This model tries to answer two simple questions: "how
strong is the signal right now?" and "how fast is it oscillating right now?"
That makes it useful for problems where the meaning is carried by changing
pitch or envelope over time. It is a strong choice when customers already
think in terms of frequency, tone, or vibration speed.

**Input requirements:** `x_clip`, shape `(200, 100)` uint8, treated as 1D
time series internally.

**Training example:**
```python
from pipeline.models.classical.hilbert import HilbertClassifier
clf = HilbertClassifier()
clf.fit(x_clips, labels)
clf.plot_instantaneous(x_clips[0])
```

**Deployment notes:** Hilbert transform requires FFT on MCU (CMSIS-DSP
`arm_rfft_fast_f32()`). Checkpoint: `checkpoints/hilbert.pkl`.

**Known limitations:** Assumes narrowband signals; multi-frequency inputs
may produce unreliable instantaneous frequency estimates.

---

### L — Autocorrelation Classifier

**What it does:** Extracts 3 features (dominant period, periodicity strength,
decay rate) from the normalised autocorrelation function, then classifies with
`GradientBoostingClassifier`. Also provides a reference-based anomaly detector
via Pearson correlation.

**When to use:** Distinguishing periodic from aperiodic signals; period-
estimation tasks.

**What it all means:** This model compares the signal with delayed copies of
itself to see whether a stable rhythm is present. In plain language, it is a
"repeatability detector." It works well when the business problem depends on
regular cycles, heartbeat-like repetition, or loss of periodic structure.

**Input requirements:** `x_clip`, shape `(200, 100)`.

**Training example:**
```python
from pipeline.models.classical.autocorrelation import AutocorrClassifier
clf = AutocorrClassifier()
clf.fit(x_clips, labels)
clf.set_reference(normal_clips)
is_anom = clf.is_anomaly(test_clip)
```

**Deployment notes:** Autocorrelation via `arm_correlate_f32()` in CMSIS-DSP.
Checkpoint: `checkpoints/autocorr.pkl`.

**Known limitations:** Gradient boosting model may be large (~500 KB);
quantise to int8 decision thresholds for M33 deployment.

---

### M — SVM Classifier

**What it does:** PCA (50 components) followed by `SVC(kernel='rbf')` with
`GridSearchCV` over C and gamma. Five variants trained on x_only, y_only,
xy_concatenated, phase, and angle representations.

**When to use:** Strong non-linear classifier without deep learning; good
generalisation from small datasets.

**What it all means:** This is a classic machine-learning option that draws a
flexible boundary between classes without using a large neural network. It is
often a very strong baseline when you do not yet have massive datasets. For
investors, it shows that the platform does not depend on heavy AI to deliver
useful results.

**Input requirements:** Any 200×100 feature map (flattened internally to
20,000 dims → PCA → 50 dims).

**Training example:**
```python
from pipeline.models.ml.svm_classifier import SVMClassifier
clf = SVMClassifier()
clf.fit(reps["x_only"].reshape(n, -1), labels)
clf.save(name="x_only")
```

**Deployment notes:** `checkpoints/svm_x_only.pkl`. PCA + SVM inference on
MCU requires the 50 principal components (float32 matrix, ~4 KB) and the
support vectors. Feasible on M33 for small support vector counts.

**Known limitations:** SVM with RBF kernel scales quadratically in inference
with the number of support vectors.

---

### N — Random Forest (28 handcrafted features)

**What it does:** Extracts 28 signal-processing features (8 from x, 8 from y,
4 phase, 4 radius, 2 autocorr, 2 spectral shape) and classifies with
`RandomForestClassifier(n_estimators=200)`. Exports an if-else decision tree
to `firmware/random_forest.h`.

**When to use:** Interpretable classification; firmware deployment without
Keras; when feature importance plots are needed.

**What it all means:** This model turns the raw oscillator behaviour into a
set of named measurements and then makes decisions using many simple decision
rules. That means you can show which factors mattered most. It is a good fit
when stakeholders want a practical, explainable model that can still capture
non-linear behaviour.

**Input requirements:** Paired `x_clip` and `y_clip`, shape `(200, 100)` each.

**Training example:**
```python
from pipeline.models.ml.random_forest import RandomForestModel
rf = RandomForestModel()
rf.fit(x_clips, y_clips, labels)
rf.plot_feature_importance(top_n=20)
rf.export_firmware_header()
```

**Deployment notes:** `checkpoints/random_forest.pkl`. The exported header
contains the top-depth if-else tree; full forest requires ~50 KB.

**Known limitations:** 28-feature extraction requires full signal buffer.
Deep forests can exceed M33 flash if not depth-pruned.

---

### O — GMM Anomaly Detector

**What it does:** Fits a `GaussianMixture` model on PCA-compressed normal
clips. Anomaly score = negative log-likelihood. Threshold = 95th percentile
of training scores.

**When to use:** Unsupervised anomaly detection; when only normal data is
available at training time.

**What it all means:** This model learns what "normal" looks like and flags
anything that does not fit that pattern well. It is useful in real operating
environments where failures are rare and labelled fault data is scarce. That
makes it commercially attractive for monitoring and preventive maintenance.

**Input requirements:** `x_only` clips, shape `(200, 100)`.

**Training example:**
```python
from pipeline.models.ml.gmm_anomaly import GMMDetector
det = GMMDetector(n_components=8)
det.fit(normal_clips)
det.plot_likelihood_distribution(anomaly_clips)
```

**Deployment notes:** `checkpoints/gmm_anomaly.pkl`. GMM inference requires
covariance matrix operations — feasible on M55 with CMSIS-DSP matrix kernels.

**Known limitations:** Gaussian mixture assumption may not hold for all Hopf
oscillator classes. Tune `n_components` on your dataset.

---

### P — KNN Classifier

**What it does:** PCA (50 components) followed by `KNeighborsClassifier(k=5)`
with Euclidean distance. Also includes a pure-numpy MCU implementation
(`predict_numpy_mcu()`) that avoids sklearn at inference time. Exports int8
reference embeddings to `firmware/knn_references.h`.

**When to use:** Interpretable classification; few-shot extension (add a new
class by appending a reference vector).

**What it all means:** This model classifies a new signal by asking, "which
saved example does it look most like?" That is easy to understand and easy to
update. It is especially useful when new categories appear in the field and
you want to add them quickly without retraining a full model.

**Input requirements:** Any 200×100 feature map.

**Training example:**
```python
from pipeline.models.ml.knn_classifier import KNNClassifier
clf = KNNClassifier(k=5)
clf.fit(reps["x_only"].reshape(n, -1), labels)
clf.plot_k_comparison(X_test, y_test)
clf.export_firmware_header(n_references_per_class=5)
```

**Deployment notes:** `checkpoints/knn.pkl`. Firmware: load
`firmware/knn_references.h` int8 arrays; distance = `arm_fully_connected_s8()`
dot product.

**Known limitations:** O(n_references) per query — keep reference set small
(e.g. 5 per class) for MCU deployment.

---

### Q — Isolation Forest

**What it does:** `IsolationForest` on 28 handcrafted features (same as
Random Forest). Anomaly score = average path length in the isolation trees.
Exports a C path-length counter to `firmware/isolation_forest.h`.

**When to use:** Fast unsupervised anomaly detection; no need to tune a
threshold (automatic contamination estimate).

**What it all means:** This model looks for samples that seem unusually easy
to separate from the rest of the data. In plain language, it hunts for
"outliers" rather than trying to name every class. It is a good fit when you
care more about spotting unusual behaviour quickly than about assigning a
precise label.

**Input requirements:** Paired `x_clip` and `y_clip`.

**Training example:**
```python
from pipeline.models.ml.isolation_forest import IsolationForestModel
iso = IsolationForestModel(contamination=0.05)
iso.fit(x_clips, y_clips)
iso.export_firmware_header()
```

**Deployment notes:** `checkpoints/isolation_forest.pkl`. Firmware uses a
simplified path-length counter; full forest is too large for M33.

**Known limitations:** Path-length scores are not calibrated probabilities.
Use `score_anomaly()` to tune threshold on labelled data.

---

### R — Autoencoder Anomaly Detector

**What it does:** Conv2D encoder (stride-2) → 32-dimensional bottleneck →
Conv2DTranspose decoder. Trained on normal clips only. Anomaly score =
MSE reconstruction error. Threshold = 95th percentile of training scores.

**When to use:** Visual inspection of reconstructions; when reconstruction
error is a meaningful physical signal.

**What it all means:** This model compresses a normal signal and then tries to
rebuild it. If the rebuild is poor, the input is probably unusual. That gives
you an intuitive story for customers: the model has learned the shape of
healthy behaviour, and anything it cannot recreate cleanly may be a fault.

**Input requirements:** `x_only` clips, shape `(200, 100)`.

**Training example:**
```python
from pipeline.models.anomaly.autoencoder import AnomalyAutoencoder
ae = AnomalyAutoencoder(epochs=50)
ae.fit(normal_clips)
ae.visualise_reconstruction(test_clip)
```

**Deployment notes:** `checkpoints/autoencoder.keras`. Encoder alone runs
on M55; full encode-decode requires M55/M85.

**Known limitations:** MSE threshold is sensitive to input scaling.
Normalise inputs consistently between training and deployment.

---

### S — One-Class SVM Detector

**What it does:** `OneClassSVM(kernel='rbf', nu=0.05)` on PCA (50 components).
Signed decision function distance indicates how far a sample is from the
normal boundary.

**When to use:** Normal-only training; when a probabilistic boundary is
preferred over a reconstruction loss.

**What it all means:** This is another "learn normal, reject abnormal" model,
but instead of rebuilding the signal it draws a boundary around acceptable
behaviour. It is useful when you want a mathematically clean definition of the
safe operating region and do not need the extra complexity of a neural network.

**Input requirements:** `x_only` clips, shape `(200, 100)`.

**Training example:**
```python
from pipeline.models.anomaly.one_class_svm import OneClassSVMDetector
det = OneClassSVMDetector(nu=0.05)
det.fit(normal_clips)
det.plot_roc(anomaly_clips)
```

**Deployment notes:** `checkpoints/one_class_svm.pkl`. Requires support
vectors in flash — feasible on M55 for small `nu` (fewer support vectors).

**Known limitations:** Nu controls the fraction of training points treated
as outliers — set conservatively (0.01–0.1).

---

### T — VAE Anomaly Detector

**What it does:** Variational autoencoder with reparameterisation trick
(`z = mu + ε·exp(0.5·log_var)`). Loss = MSE reconstruction + KL divergence
(ELBO). Smooth latent space gives more reliable anomaly scores than the plain
autoencoder.

**When to use:** When reconstruction error alone is noisy; when a smooth
manifold of normal behaviour is desired.

**What it all means:** This is a more sophisticated version of the
autoencoder. It does not just memorise examples; it learns a smoother map of
what normal behaviour should look like. That can make anomaly scores more
stable and more robust when the real world is messy.

**Input requirements:** `x_only` clips, shape `(200, 100)`.

**Training example:**
```python
from pipeline.models.anomaly.vae import VAEDetector
det = VAEDetector(latent_dim=32, epochs=50)
det.fit(normal_clips)
score = det.anomaly_score(test_clip)
```

**Deployment notes:** `checkpoints/vae_weights.h5`. Encoder + sampling on
M85 recommended (256 KB). Encoder alone runs on M55 (128 KB).

**Known limitations:** KL weight balancing (beta-VAE) may be needed for
imbalanced feature distributions; requires tuning.

---

### U — Contrastive Classifier (SimCLR)

**What it does:** Unsupervised contrastive pretraining with NT-Xent loss
(SimCLR). Two augmented views of each clip are pushed together in a 128-dim
L2-normalised embedding space. After pretraining, new classes are added by
computing the mean embedding of 5 examples (prototype).

**When to use:** The flagship RECHO reconfigurability model. Pretrain once
overnight on unlabelled data; engineer labels 5 clips next morning.

**What it all means:** This is the strongest story for rapid adaptation. The
model first learns the general structure of the data without labels, then new
classes can be added from just a handful of examples. For investors, this is
the clearest example of a platform that can be reconfigured quickly in the
field instead of requiring a long retraining cycle.

**Input requirements:** `x_only` clips (no labels for pretraining).

**Training example:**
```python
from pipeline.models.anomaly.contrastive import ContrastiveClassifier
clf = ContrastiveClassifier(epochs=30)
clf.pretrain(all_clips)
clf.build_prototypes({"normal": normal[:5], "fault_a": fault[:5]})
label = clf.few_shot_classify(new_clip)
```

**Deployment notes:** `checkpoints/contrastive_encoder.keras`. Encoder runs
on M85+Ethos-U55; embedding is 128 int8 values; distance computed with
`arm_fully_connected_s8()`.

**Known limitations:** Augmentation hyperparameters (shift ±20, scale ±10%,
noise σ=0.01) tuned for Hopf oscillator — may need adjustment for other
sensors.

---

### V — TCN Classifier

**What it does:** Temporal Convolutional Network with causal zero-padding and
dilations [1, 2, 4, 8] using `arm_convolve_1_x_n_s8()`. Global average
pooling → Dense softmax.

**When to use:** Long-range temporal dependencies in the raw signal; when
LSTMs are too slow.

**What it all means:** This model reads the signal as a timeline rather than
as an image, so it is good at spotting patterns that unfold across time. It is
a practical option when timing matters a lot but you still need something fast
enough for embedded deployment.

**Input requirements:** Raw downsampled signal, shape `(4000,)` float32.

**Training example:**
```python
from pipeline.models.sequence.tcn import TCNClassifier
clf = TCNClassifier(n_classes=5, epochs=20)
clf.fit(raw_clips, labels)
```

**Deployment notes:** `checkpoints/tcn.keras`. Causal padding means no
future context is needed — real-time streaming possible on M55.

**Known limitations:** 4000-sample input requires 16 KB float32 buffer.
Use INT8 quantisation and `arm_convolve_1_x_n_s8()` for M55 deployment.

---

### W — LSTM Classifier

**What it does:** Two-layer LSTM (64 + 32 units) followed by dense layers.
Processes the raw signal as a `(T, 2)` sequence (x and y concatenated).
Maps to TFLite Micro `LSTMFull` op.

**When to use:** Maximum accuracy on sequence data; when M85/M55 RAM is
available.

**What it all means:** This is the heavyweight memory-based sequence model. It
is designed to remember what happened earlier in the signal and use that
history when making a decision. It is the right choice when you want to chase
top-end temporal accuracy and have enough hardware budget to support it.

**Input requirements:** Raw signal reshaped to `(T, 2)` float32 where
T = number of timesteps.

**Training example:**
```python
from pipeline.models.sequence.lstm_classifier import LSTMClassifier
clf = LSTMClassifier(n_classes=5, epochs=20)
clf.fit(x_clips, y_clips, labels)
```

**Deployment notes:** `checkpoints/lstm.keras`. Requires M55 (128 KB) or
M85 (256 KB). TFLite Micro LSTM op; enable `LSTM_FULL_KERNEL` in
`tensorflow/lite/micro/kernels/lstm_eval.h`.

**Known limitations:** Heaviest model — 64+32 LSTM units × 2 inputs × T
timesteps. Not suitable for M33 (64 KB).

---

### X — Echo State Network (ESN)

**What it does:** Fixed, sparse random recurrent reservoir (spectral radius
≤ 0.9, sparsity 10%); deterministic feature extraction; ridge regression
output weights trained in one step. Exports `float32` output weights to
`firmware/esn_output_weights.h`.

**When to use:** When training must be completed in < 1 second; research
into reservoir computing.

**What it all means:** This model keeps most of its internal dynamics fixed
and only learns the final readout, which makes training extremely fast. It is
appealing when you want quick turnaround, low training cost, or a direct link
to reservoir-computing ideas. It is also useful as a lightweight benchmark.

**Input requirements:** `x_only` clips, shape `(200, 100)`.

**Training example:**
```python
from pipeline.models.sequence.esn_readout import EchoStateReadout
esn = EchoStateReadout(reservoir_size=500, spectral_radius=0.9)
esn.fit(reps["x_only"].reshape(n, -1), labels)
esn.export_output_weights()
```

**Deployment notes:** `checkpoints/esn.pkl`. Reservoir matrix is fixed and
can be stored in flash as a sparse structure. Output weights = one
`arm_fully_connected_s8()` call.

**Known limitations:** Reservoir size (500 units) fixed at construction;
changing it requires retraining. Spectral radius must be < 1 for echo state
property.

---

### Y — Prototypical Network (5-shot)

**What it does:** Nearest-prototype classifier in the encoder's embedding
space. Each class prototype = mean embedding of ≤ 5 support examples.
Classification = Euclidean nearest neighbour. Supports online prototype
updates (running mean) without retraining. Exports int8 prototype arrays to
`firmware/prototypes.h`.

**When to use:** New signal class appears in the field; engineer records 5
examples; classification updates immediately.

**What it all means:** This is the fastest path to field reconfiguration. The
system stores a small prototype for each class, and new categories can be
added almost immediately from a few examples. For investors, this is easy to
understand and commercially compelling because it reduces data collection and
deployment friction.

**Input requirements:** `x_only` clips for both support set and queries.

**Training example:**
```python
from pipeline.models.fewshot.prototypical import PrototypicalNetwork
net = PrototypicalNetwork(encoder=None, n_pca_components=32)
net.build_prototypes({
    "normal": normal_clips[:5],
    "fault_a": fault_clips[:5],
})
label = net.classify(new_clip)
net.update_prototype("normal", another_normal_clip)
net.export_firmware_header()
```

**Deployment notes:** `firmware/prototypes.h` — int8_t arrays, one row per
class. Distance = `arm_fully_connected_s8()` dot product per prototype.
5 classes × 128-dim = 640 bytes flash overhead.

**Known limitations:** Accuracy limited by encoder quality. Using the
pretrained `ContrastiveClassifier` encoder (model U) improves performance
significantly over the default PCA encoder.
