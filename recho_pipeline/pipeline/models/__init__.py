"""
Multi-model architectures for Hopf oscillator x(t)/y(t) classification.

Each model is CMSIS-NN compatible and convertible to TFLite INT8 for
deployment on Arm Cortex-M33, M55, and M85+Ethos-U55 targets.

Models:
  A — cnn_x_only       baseline, x(t) only (replicates paper 2)
  B — cnn_xy_dual      two-channel x+y input
  C — cnn_phase        orbit radius sqrt(x^2+y^2)
  D — cnn_angle        instantaneous phase arctan2(y,x)
  E — cnn_xy_fusion    late fusion — separate x and y branches
  F — depthwise_cnn    depthwise separable CNN, M55/M33 optimised
  G — reservoir_readout ridge regression baseline (paper 1 method)
  H — ensemble         majority vote across multiple models
"""

from pipeline.models.cnn_x_only import build_model as build_cnn_x_only
from pipeline.models.cnn_xy_dual import build_model as build_cnn_xy_dual
from pipeline.models.cnn_phase import build_model as build_cnn_phase
from pipeline.models.cnn_angle import build_model as build_cnn_angle
from pipeline.models.cnn_xy_fusion import build_model as build_cnn_xy_fusion
from pipeline.models.depthwise_cnn import build_model as build_depthwise_cnn

__all__ = [
    "build_cnn_x_only",
    "build_cnn_xy_dual",
    "build_cnn_phase",
    "build_cnn_angle",
    "build_cnn_xy_fusion",
    "build_depthwise_cnn",
]
