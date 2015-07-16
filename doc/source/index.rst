=========
sklearnef
=========

:Release: |release|
:Date: |today|

**sklearnef** is an extension library for `sklearn <http://scikit-learn.org>`_,  providing density and semi-supervised forest written in Python.

Installation
============

.. toctree::
    :maxdepth: 1
    
    installation/asroot
    installation/asuser
    installation/developmentmode
    installation/uninstall   

..
    Information
    ===========

    .. toctree::
        :glob:
        :maxdepth: 1
        
        information/*

..
    Tutorials
    =========

    .. toctree::
        :glob:
        :maxdepth: 1
        
        tutorial/*

..
    Receipts
    ========

    .. toctree::
        :glob:
        :maxdepth: 1
        
        receipts/*

Reference
=========

:mod:`sklearn.ensemble`: Ensemble methods (decision forests)
------------------------------------------------------------

.. automodule:: sklearnef.ensemble

.. currentmodule:: sklearnef

.. autosummary::
   :toctree: generated/

   ensemble.DensityForest
   ensemble.SemiSupervisedRandomForestClassifier
   
:mod:`sklearn.tree`: Decision trees
-----------------------------------

.. automodule:: sklearnef.tree

.. currentmodule:: sklearnef

.. autosummary::
   :toctree: generated/

   tree.DensityTree
   tree.SemiSupervisedDecisionTreeClassifier   
   tree.GoodnessOfFit
   tree.MECDF




