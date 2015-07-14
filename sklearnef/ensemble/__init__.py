"""
==========================================================
Forests for direct public use (:mod: `sklearnef.ensemble`)
==========================================================
.. currentmodule:: sklearnef.ensemble

This package contains the `DensityForest` and the
`SemiSupervisedRandomForestClassifier` ensemble methods.

Forest ensemble classes :mod:`sklearnef.ensemble.forest`
========================================================
Short description string.

.. module:: sklearnef.ensemble.forest
.. autosummary::
    :toctree: generated/
    
    DensityForest
    SemiSupervisedRandomForestClassifier

"""

#from .forest import SemiSupervisedRandomForestClassifier
from .forest import DensityForest

__all__ = ["SemiSupervisedRandomForestClassifier",
           "DensityForest"]
