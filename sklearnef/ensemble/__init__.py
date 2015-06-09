"""
The :mod:`sklearnef.ensemble` module includes ensemble-based methods for
un-supervised and semi-supervised classification.
"""

#from .forest import SemiSupervisedRandomForestClassifier
from .forest import UnSupervisedRandomForestClassifier

__all__ = ["SemiSupervisedRandomForestClassifier",
           "UnSupervisedRandomForestClassifier"]
