"""
The :mod:`sklearnef.tree` module includes decision tree-based models for
un-supervised and semi-supervised classification.
"""

from .tree import SemiSupervisedDecisionTreeClassifier
from .tree import UnSupervisedDecisionTreeClassifier

__all__ = ["UnSupervisedDecisionTreeClassifier"
           "SemiSupervisedDecisionTreeClassifier"]
