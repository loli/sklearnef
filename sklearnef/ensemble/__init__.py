"""
The :mod:`sklearnef.ensemble` module includes ensemble-based methods for
density learning and semi-supervised classification.
"""

from .forest import DensityForest
from .forest import SemiSupervisedRandomForestClassifier

__all__ = ["DensityForest",
           "SemiSupervisedRandomForestClassifier"]
