"""
Anomaly detection models for Hopf oscillator outputs.

All models in this module can be trained without fault labels.
The Hopf oscillator is sensitive to ANY input change — these models
exploit that sensitivity to detect anomalies without human annotation.

Models:
  AnomalyAutoencoder   — Convolutional autoencoder, reconstruction error
  OneClassSVMDetector  — One-class SVM on PCA-reduced feature maps
  VAEDetector          — Variational autoencoder, ELBO anomaly score
  ContrastiveClassifier — Contrastive learning for few-shot reconfiguration
"""

from pipeline.models.anomaly.autoencoder import AnomalyAutoencoder
from pipeline.models.anomaly.one_class_svm import OneClassSVMDetector
from pipeline.models.anomaly.vae import VAEDetector
from pipeline.models.anomaly.contrastive import ContrastiveClassifier

__all__ = [
    "AnomalyAutoencoder",
    "OneClassSVMDetector",
    "VAEDetector",
    "ContrastiveClassifier",
]
