"""Forest of tree-based ensemble methods

Those methods include un- and semi-supervised forests

The module structure is the following:

- The ``UnSupervisedRandomForestClassifier`` derived
  classes provide the user with concrete implementations of
  the density forest method.

- The ``SemiSupervisedRandomForestClassifier`` derived
  classes provide the user with concrete implementations of
  the semi-supervised classification forest method, which
  classfies the unlabelled data on-the-fly during training.
  
Only single output problems are handled.

"""

# Authors: Oskar Maier <oskar.maier@googlemail.com>
#
# Licence: BSD 3 clause !TODO: Change

from __future__ import division

from ..tree import (SemiSupervisedDecisionTreeClassifier,
                    UnSupervisedDecisionTreeClassifier)

from sklearn.ensemble.forest import ForestClassifier

__all__ = ["UnSupervisedRandomForestClassifier",
           #"SemiSupervisedRandomForestClassifier",
           ]

class UnSupervisedRandomForestClassifier(ForestClassifier):
    def __init__(self,
                 n_estimators=10,
                 criterion="unsupervised",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_improvement=0,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(UnSupervisedRandomForestClassifier, self).__init__(
            base_estimator=UnSupervisedDecisionTreeClassifier(),
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
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.

        Returns
        -------
        self : object
            Returns self.

        """
        ForestClassifier.fit(self, X, y, sample_weight=sample_weight)
        return self

# class SemiSupervisedRandomForestClassifier(ForestClassifier):
#     def __init__(self,
#                  n_estimators=10,
#                  criterion="semisupervised",
#                  max_depth=None,
#                  min_samples_split=2,
#                  min_samples_leaf=1,
#                  min_weight_fraction_leaf=0.,
#                  max_features="auto",
#                  max_leaf_nodes=None,
#                  bootstrap=True,
#                  oob_score=False,
#                  n_jobs=1,
#                  random_state=None,
#                  verbose=0,
#                  warm_start=False,
#                  class_weight=None):
#         super(SemiSupervisedRandomForestClassifier, self).__init__(
#             base_estimator=SemiSupervisedDecisionTreeClassifier(),
#             n_estimators=n_estimators,
#             estimator_params=("criterion", "max_depth", "min_samples_split",
#                               "min_samples_leaf", "min_weight_fraction_leaf",
#                               "max_features", "max_leaf_nodes",
#                               "random_state"),
#             bootstrap=bootstrap,
#             oob_score=oob_score,
#             n_jobs=n_jobs,
#             random_state=random_state,
#             verbose=verbose,
#             warm_start=warm_start,
#             class_weight=class_weight)
# 
#         self.criterion = criterion
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.min_samples_leaf = min_samples_leaf
#         self.min_weight_fraction_leaf = min_weight_fraction_leaf
#         self.max_features = max_features
#         self.max_leaf_nodes = max_leaf_nodes
#         
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

