"""
Machine learning classifiers and anomaly detectors for Hopf oscillator outputs.

These models operate on processed feature maps (200×100 grids) or handcrafted
feature vectors extracted from x(t)/y(t) time series.

Models:
  SVMClassifier       — SVM on PCA-reduced feature maps (5 input variants)
  RandomForestModel   — Random forest on 28 handcrafted features
  GMMDetector         — Gaussian mixture model, unsupervised anomaly detection
  KNNClassifier       — k-nearest neighbours on feature maps
  IsolationForestModel — Isolation forest anomaly detection
"""

from pipeline.models.ml.svm_classifier import SVMClassifier
from pipeline.models.ml.random_forest import RandomForestModel
from pipeline.models.ml.gmm_anomaly import GMMDetector
from pipeline.models.ml.knn_classifier import KNNClassifier
from pipeline.models.ml.isolation_forest import IsolationForestModel

__all__ = [
    "SVMClassifier",
    "RandomForestModel",
    "GMMDetector",
    "KNNClassifier",
    "IsolationForestModel",
]
