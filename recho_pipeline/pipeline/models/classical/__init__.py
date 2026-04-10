"""
Classical signal processing methods for Hopf oscillator classification.

These methods require no neural network — all arithmetic, runnable on any MCU.
They provide baselines, anomaly detectors, and interpretable features for the
RECHO deployment scenario where labels are scarce or unavailable.

Models:
  SPCMonitor       — Statistical process control, per-sample real-time alerting
  PhasePortrait    — Geometric features from x(t) vs y(t) orbit
  RecurrenceQA     — Recurrence quantification analysis
  HilbertAnalysis  — Instantaneous amplitude/frequency via Hilbert transform
  AutocorrDetector — Periodicity detection via autocorrelation
"""

from pipeline.models.classical.spc import SPCMonitor
from pipeline.models.classical.phase_portrait import PhasePortraitClassifier
from pipeline.models.classical.recurrence import RecurrenceClassifier
from pipeline.models.classical.hilbert import HilbertClassifier
from pipeline.models.classical.autocorrelation import AutocorrClassifier

__all__ = [
    "SPCMonitor",
    "PhasePortraitClassifier",
    "RecurrenceClassifier",
    "HilbertClassifier",
    "AutocorrClassifier",
]
