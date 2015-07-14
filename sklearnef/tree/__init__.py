"""
=====================================================
Trees underlying the forests (:mod: `sklearnef.tree`)
=====================================================
.. currentmodule:: sklearnef.tree

This package contains the `DensityTree` and the
`SemiSupervisedDecisionTreeClassifier` ensemble methods.

Forest ensemble classes :mod:`sklearnef.tree.tree`
==================================================
Short description string.

.. module:: sklearnef.tree.tree
.. autosummary::
    :toctree: generated/
    
    DensityTree
    SemiSupervisedRandomForestClassifier
    GoodnessOfFit
    MECDF

"""

from .tree import SemiSupervisedDecisionTreeClassifier
from .tree import DensityTree
from .tree import GoodnessOfFit, MECDF

__all__ = ["DensityTree",
           "SemiSupervisedDecisionTreeClassifier",
           "GoodnessOfFit", "MECDF"]
