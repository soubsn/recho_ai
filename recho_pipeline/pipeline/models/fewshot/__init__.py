"""
Few-shot learning models for Hopf oscillator classification.

These models can classify into new categories with just 1-5 examples,
enabling the RECHO reconfigurability use case: a customer records a
short fault signature and the classifier updates immediately.

Models:
  PrototypicalNetwork — prototype-based nearest-neighbour few-shot classifier
"""

from pipeline.models.fewshot.prototypical import PrototypicalNetwork

__all__ = [
    "PrototypicalNetwork",
]
