"""
Sequence models for Hopf oscillator raw time series classification.

These models operate directly on raw or downsampled x(t)/y(t) signals,
bypassing the feature map reshape step used by the CNN models.

Models:
  TCNClassifier     — Temporal Convolutional Network on raw x(t)
  LSTMClassifier    — LSTM on x(t)+y(t) two-feature time series
  EchoStateReadout  — Echo State Network reservoir readout
"""

from pipeline.models.sequence.tcn import TCNClassifier
from pipeline.models.sequence.lstm_classifier import LSTMClassifier
from pipeline.models.sequence.esn_readout import EchoStateReadout

__all__ = [
    "TCNClassifier",
    "LSTMClassifier",
    "EchoStateReadout",
]
