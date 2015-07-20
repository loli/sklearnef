"""
This module gathers sklearnefs forest-based methods, including unsupervised (density)
and semi-supervised trees. Only single output problems are handled.
"""

# Authors: Oskar Maier <oskar.maier@googlemail.com>
#
# Licence: BSD 3 clause !TODO: Change

from __future__ import division

from sklearn.utils import check_array
from sklearn.ensemble.forest import ForestClassifier
from sklearn.tree._tree import DTYPE

from ..tree import (SemiSupervisedDecisionTreeClassifier,
                    DensityTree)

__all__ = ["DensityForest",
           "SemiSupervisedRandomForestClassifier"]

class DensityForest(ForestClassifier):
    """A forest based density estimator.

    A random forest is a meta estimator that fits a number of density trees
    on various sub-samples of the dataset and use averaging to improve the
    smoothness, predictive accuracy and to control over-fitting.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
        Note: this parameter is tree-specific.
    
    min_improvement : float (default=0)
        The minimum improvement a split must exhibit to be considered adequate.
        One of the strongest parameters for controlling over-fitting in density
        trees.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        Ignored if ``max_leaf_nodes`` is not None.
        Note: this parameter is tree-specific.

    min_samples_split : integer, optional (default=2)
        The minimum number of samples required to split an internal node.
        Note: this parameter is tree-specific.

    min_samples_leaf : integer, optional (default=1)
        The minimum number of samples in newly created leaves.  A split is
        discarded if after the split, one of the leaves would contain less then
        ``min_samples_leaf`` samples.
        Note: this parameter is tree-specific.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the input samples required to be at a
        leaf node.
        Note: this parameter is tree-specific.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
        If not None then ``max_depth`` will be ignored.
        Note: this parameter is tree-specific.

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    Attributes
    ----------
    estimators\_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    feature_importances\_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    References
    ----------

    .. [1] A. Criminisi, J. Shotton and E. Konukoglu, "Decision Forests: A 
           Unified Framework for Classification, Regression, Density
           Estimation, Manifold Learning and Semi-Supervised Learning",
           Foundations and Trends(r) in Computer Graphics and Vision, Vol. 7,
           No. 2-3, pp 81-227, 2012.

    See also
    --------
    DensityTree
    """    
    def __init__(self,
                 n_estimators=10,
                 criterion="unsupervised",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=None,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_improvement=0.,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(DensityForest, self).__init__(
            base_estimator=DensityTree(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes", "min_improvement",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_improvement = min_improvement

    def fit(self, X, y=None, sample_weight=None):
        """Fit estimator.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples whose density distribution to estimate.
            Internally, it will be converted to ``dtype=np.float32``.
            
        y : None
            Not used, kept only for interface conformity reasons.            
            
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
            Returns self.

        """
        X = check_array(X, accept_sdparse=False, order='C')
        return ForestClassifier.fit(self, X, y, sample_weight=sample_weight)

class SemiSupervisedRandomForestClassifier(ForestClassifier):
    """A forest based semi-supervised classifier.

    A random forest is a meta estimator that fits a number of semi-supervised
    trees on various sub-samples of the dataset and use averaging to improve
    the smoothness, predictive accuracy and to control over-fitting.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
        Note: this parameter is tree-specific.
        
    supervised_weight: float, optional (default=0.5)
        Factor balancing the supervised against the un-supervised measures of
        split quality. A value of `1.0` would mean to consider only the
        labelled samples, a value of `0.0` would equal a density tree.
        Note that a clean value of `1.0` is not allowed, at it would lead
        to non max-margin splits. Please use the original `sklearn`
        `RandomForestClassifier` for that effect.
        Note: this parameter is tree-specific.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        Ignored if ``max_leaf_nodes`` is not None.
        Note: this parameter is tree-specific.

    min_samples_split : integer, optional (default=2)
        The minimum number of samples required to split an internal node.
        Note: this parameter is tree-specific.

    min_samples_leaf : integer, optional (default=1)
        The minimum number of samples in newly created leaves.  A split is
        discarded if after the split, one of the leaves would contain less then
        ``min_samples_leaf`` samples.
        Note: this parameter is tree-specific.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the input samples required to be at a
        leaf node.
        Note: this parameter is tree-specific.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
        If not None then ``max_depth`` will be ignored.
        Note: this parameter is tree-specific.

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    Attributes
    ----------
    estimators\_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    feature_importances\_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    References
    ----------

    .. [1] A. Criminisi, J. Shotton and E. Konukoglu, "Decision Forests: A 
           Unified Framework for Classification, Regression, Density
           Estimation, Manifold Learning and Semi-Supervised Learning",
           Foundations and Trends(r) in Computer Graphics and Vision, Vol. 7,
           No. 2-3, pp 81-227, 2012.

    See also
    --------
    SemiSupervisedDecisionTreeClassifier
    """
    def __init__(self,
                 n_estimators=10,
                 criterion="semisupervised",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=None, #!TODO: Fix the default in the description
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 supervised_weight=.5,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 unsupervised_transformation='scale'): #"TODO: Put this in the description
        super(SemiSupervisedRandomForestClassifier, self).__init__(
            base_estimator=SemiSupervisedDecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "random_state", "supervised_weight",
                              "unsupervised_transformation"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)
 
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.supervised_weight = supervised_weight
        self.unsupervised_transformation = unsupervised_transformation
          
    def fit(self, X, y, sample_weight=None):
        """Fit estimator.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples whose density distribution to estimate.
            Internally, it will be converted to ``dtype=np.float32``.
            
        y : array-like, shape = [n_samples]
            The target values (class labels in classification).
            
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
            Returns self.

        """
        X = check_array(X, accept_sparse=False, order='C')
        return ForestClassifier.fit(self, X, y, sample_weight=sample_weight)          
          
#     def _validate_y_class_weight(self, y):
#         y = np.copy(y)
#         expanded_class_weight = None
#   
#         if self.class_weight is not None:
#             y_original = np.copy(y)
#   
#         self.classes_ = []
#         self.n_classes_ = []
#   
#         for k in range(self.n_outputs_):
#             classes_k, y[:, k] = np.unique(y[:, k], return_inverse=True)
#             # remove smallest label (assuming that always same (i.e. smallest) over all n_outputs and consistent)
#             self.classes_.append(classes_k[1:])
#             self.n_classes_.append(classes_k.shape[0] - 1)
#   
#         if self.class_weight is not None:
#             valid_presets = ('auto', 'subsample')
#             if isinstance(self.class_weight, six.string_types):
#                 if self.class_weight not in valid_presets:
#                     raise ValueError('Valid presets for class_weight include '
#                                      '"auto" and "subsample". Given "%s".'
#                                      % self.class_weight)
#                 if self.warm_start:
#                     warn('class_weight presets "auto" or "subsample" are '
#                          'not recommended for warm_start if the fitted data '
#                          'differs from the full dataset. In order to use '
#                          '"auto" weights, use compute_class_weight("auto", '
#                          'classes, y). In place of y you can use a large '
#                          'enough sample of the full training set target to '
#                          'properly estimate the class frequency '
#                          'distributions. Pass the resulting weights as the '
#                          'class_weight parameter.')
#   
#             if self.class_weight != 'subsample' or not self.bootstrap:
#                 if self.class_weight == 'subsample':
#                     class_weight = 'auto'
#                 else:
#                     class_weight = self.class_weight
#                 expanded_class_weight = compute_sample_weight(class_weight,
#                                                               y_original)
#   
#         return y, expanded_class_weight        

