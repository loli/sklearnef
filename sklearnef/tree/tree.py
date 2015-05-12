"""
This module gathers tree-based methods, including decision, regression and
randomized trees. Single and multi-output problems are both handled.
"""

# Authors: Oskar Maier <oskar.maier@googlemail.com>
#
# Licence: BSD 3 clause !TODO: Change

from __future__ import division

import numbers

import numpy as np
from scipy.sparse import issparse

from sklearn.utils import check_array, check_random_state, compute_sample_weight
from sklearn.utils.validation import NotFittedError, check_is_fitted

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import Criterion
from sklearn.tree._tree import Splitter
from sklearn.tree._tree import DepthFirstTreeBuilder, BestFirstTreeBuilder
from sklearn.tree._tree import Tree
from sklearn.tree import _tree

import _tree as _treeef

from scipy.stats import mvn
from scipy.stats._multivariate import multivariate_normal

__all__ = ["SemiSupervisedDecisionTreeClassifier",
           "UnSupervisedDecisionTreeClassifier"]


# =============================================================================
# Types and constants
# =============================================================================

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

CRITERIA_CLF = {"gini": _tree.Gini, "entropy": _tree.Entropy, "semisupervised": _treeef.LabeledOnlyEntropy}

DENSE_SPLITTERS = {"best": _tree.BestSplitter,
                   "presort-best": _tree.PresortBestSplitter,
                   "random": _tree.RandomSplitter,
                   "unsupervised": _treeef.UnSupervisedBestSplitter}

SPARSE_SPLITTERS = {"best": _tree.BestSparseSplitter,
                    "random": _tree.RandomSparseSplitter}

# =============================================================================
# Base decision tree
# =============================================================================

class UnSupervisedDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self,
                 criterion="unsupervised",
                 splitter="unsupervised",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 class_weight=None):
        super(UnSupervisedDecisionTreeClassifier, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state)
    
    def fit(self, X, y=None, sample_weight=None, check_input=True):
        random_state = check_random_state(self.random_state)
        if check_input:
            X = check_array(X, dtype=DTYPE)
            
        # Set min_weight_leaf from min_weight_fraction_leaf
        if self.min_weight_fraction_leaf != 0. and sample_weight is not None:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight))
        else:
            min_weight_leaf = 0.            
            
        # !TODO: replace this crude method to get the tree to provide sufficient memory for storing the leaf values
        # remove passed y (which we do not need and only keep for interface conformity reasons)
        y = np.zeros((X.shape[0], X.shape[1]**2 + X.shape[1] + 1))
        # initialise criterion here, since it requires another syntax than the default ones
        if 'unsupervised' == self.criterion:
            self.criterion =  _treeef.UnSupervisedClassificationCriterion(X.shape[0], X.shape[1])
        # initialize splitter here
        if 'unsupervised' == self.splitter:
            self.splitter = _treeef.UnSupervisedBestSplitter(self.criterion,
                                                             self.max_features_,
                                                             self.min_samples_leaf,
                                                             min_weight_leaf,
                                                             random_state)
        DecisionTreeClassifier.fit(self, X, y, sample_weight, check_input)
        return self

    def predict_proba(self, X):
        """Predict density distribution membership probabilities of the input samples X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``. No sparse matrixes allowed.

        Returns
        -------
        p : array of length n_samples
            The density distribution membership probability of the input
            samples.
        """
        check_is_fitted(self, 'n_outputs_')
        X = check_array(X, dtype=DTYPE, accept_sparse=None)

        n_samples, n_features = X.shape

        if self.tree_ is None:
            raise NotFittedError("Tree not initialized. Perform a fit first.")

        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             " match the input. Model n_features is %s and "
                             " input n_features is %s "
                             % (self.n_features_, n_features))

        # extract leaf values and leaf definition bounding boxes from the tree
        info = self.parse_tree_leaves()
        
        # compute the integral of the partition function
        # !TODO: This could be computed once for the first application, and
        # then re-used; but how to avoid re-training?
        pfi = self.compute_partition_function(info) # Modifies info objects range entries in-place!
        
        # returns the indices of the node (alle leaves) each sample dropped
        # into
        leaf_indices = self.tree_.apply(X)

        # construct the the associated multivariate Gaussian distribution for each unique
        # leaf and pass the associated samples through its pdf to obtain their density
        # values
        out = np.zeros(n_samples, np.float)
        for lidx in np.unique(leaf_indices):
            mnd = multivariate_normal(info[lidx]['mu'], info[lidx]['cov'])
            mask = lidx == leaf_indices
            out[mask] = info[lidx]['frac'] / pfi * mnd.pdf(X[mask])
        
        # return
        return out
        
    def compute_partition_function(self, info):
        r"""
        Computes the partition function of the tree. In the present case of
        axis-aligned weak learners, this is equivalent to the sum of the
        leaf-wise cumulative multivariate normal distributions.
        
        Warning
        -------
        Modifies info object in-place!
        
        Parameters
        ----------
        info : list of dicts
            As returned by `parse_tree_leaves()`.
        
        Returns
        -------
        z : float
            The partition function value for the tree.
        """
        info = self.info_inf_to_large(info)
        z = 0
        for leaf_info in info:
            if leaf_info is None:
                continue
            lower = [r[0] for r in leaf_info['range']]
            upper = [r[1] for r in leaf_info['range']]
            integral, _ = mvn.mvnun(lower, upper, leaf_info['mu'], leaf_info['cov'])
            z += leaf_info['frac'] * integral
        
        return z
    
    def info_inf_to_large(self, info, mult = 100):
        r"""
        Replaces the +/-inf objects in a tree leaves info object by the associated
        dimensions variance, multiplied by `mult`.
        
        Warning
        -------
        Modifies info object in-place!
        
        Parameters
        ----------
        info : list of dicts
            As returned by `parse_tree_leaves()`.
        mult : number
            The multiplicator, choose sufficiently high.
        
        Returns
        -------
        info : list of dicts
            The modified tree leaves info object.
        """
        for leaf_info in info:
            if leaf_info is None:
                continue
            rang = leaf_info['range']
            for i in range(len(rang)):
                l, u = rang[i]
                if np.isneginf(l):
                    l = -1 * mult * leaf_info['cov'][i, i]
                if np.isposinf(u):
                    u = mult * leaf_info['cov'][i, i]
                rang[i] = (l, u)
        return info
        
    def parse_tree_leaves(self):
        r"""
        Returns information about the leaves of this density tree.
            
        Returns
        -------
        info : list of dicts
            List containing a dict with the keys ['frac', 'cov', 'mu', 'range']
            for each leaf node and None for each internal node.
        """
        return self.__parse_tree_leaves_rec(self.tree_)
        
    def __parse_tree_leaves_rec(self, tree, pos = 0, rang = None):
        # init
        if rang is None:
            rang = [(-np.inf, np.inf)] * tree.n_features
        
        # get node info
        fid = tree.feature[pos]
        thr = tree.threshold[pos]
        
        # if not leave node...
        if not -2 == fid:
            
            info = [None]
    
            # ...update range of cell and ascend to left node
            lrang = rang[:]
            lrang[fid] = (lrang[fid][0], min(lrang[fid][1], thr))
            info.extend(self.__parse_tree_leaves_rec(tree, tree.children_left[pos], lrang))
            
            # ...update range of cell and ascend to right node
            rrang = rang[:]
            rrang[fid] = (max(rrang[fid][0], thr), rrang[fid][1])
            info.extend(self.__parse_tree_leaves_rec(tree, tree.children_right[pos], rrang))
            
            return info
            
        # if leave node, return information
        else:
            nval = tree.value[pos]
            frac = nval[0]
            cov = np.squeeze(nval[1:5]).reshape((2,2))
            mu = np.squeeze(nval[5:7])
            
            return [{'frac': frac,
                     'cov': cov,
                     'mu': mu,
                     'range': rang}]

        
    def predict_log_proba(self, X):
        """Predict density distribution membership log-probabilities of the input samples X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``. No sparse matrixes allowed.

        Returns
        -------
        p : array of length n_samples
            The density distribution membership log-probability of the
            input samples.
        """
        return np.log(self.predict_proba(X)) # self.n_outputs is set to > 1, even if only one outcome enforced
    
    def predict(self, X):
        """Not supported for density forest.
        
        Only kept for interface consistency reasons.
        """
        raise NotImplementedError("Density forest do not support the predict() method.")

class SemiSupervisedDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self,
                 criterion="semisupervised",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 class_weight=None):
        super(SemiSupervisedDecisionTreeClassifier, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state)

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Build a decision tree from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression). In the regression case, use ``dtype=np.float64`` and
            ``order='C'`` for maximum efficiency.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        self : object
            Returns self.
        """
        random_state = check_random_state(self.random_state)
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse="csc")
            if issparse(X):
                X.sort_indices()

                if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                    raise ValueError("No support for np.int64 index based "
                                     "sparse matrices")

        # Determine output settings
        n_samples, self.n_features_ = X.shape
        is_classification = isinstance(self, ClassifierMixin)

        y = np.atleast_1d(y)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        if is_classification:
            y = np.copy(y)

            self.classes_ = []
            self.n_classes_ = []

            if self.class_weight is not None:
                y_original = np.copy(y)

            for k in range(self.n_outputs_):
                classes_k, y[:, k] = np.unique(y[:, k], return_inverse=True)
                # remove smallest label (assuming that always same (i.e. smallest) over all n_outputs and consistent)
                self.classes_.append(classes_k[1:])
                self.n_classes_.append(classes_k.shape[0] - 1)

            if self.class_weight is not None:
                expanded_class_weight = compute_sample_weight(
                    self.class_weight, y_original)

        else:
            self.classes_ = [None] * self.n_outputs_
            self.n_classes_ = [1] * self.n_outputs_

        self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        # Check parameters
        max_depth = ((2 ** 31) - 1 if self.max_depth is None
                     else self.max_depth)
        max_leaf_nodes = (-1 if self.max_leaf_nodes is None
                          else self.max_leaf_nodes)

        if isinstance(self.max_features, six.string_types):
            if self.max_features == "auto":
                if is_classification:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1, int(self.max_features * self.n_features_))
            else:
                max_features = 0

        self.max_features_ = max_features

        if len(y) != n_samples:
            raise ValueError("Number of labels=%d does not match "
                             "number of samples=%d" % (len(y), n_samples))
        if self.min_samples_split <= 0:
            raise ValueError("min_samples_split must be greater than zero.")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be greater than zero.")
        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")
        if not isinstance(max_leaf_nodes, (numbers.Integral, np.integer)):
            raise ValueError("max_leaf_nodes must be integral number but was "
                             "%r" % max_leaf_nodes)
        if -1 < max_leaf_nodes < 2:
            raise ValueError(("max_leaf_nodes {0} must be either smaller than "
                              "0 or larger than 1").format(max_leaf_nodes))

        if sample_weight is not None:
            if (getattr(sample_weight, "dtype", None) != DOUBLE or
                    not sample_weight.flags.contiguous):
                sample_weight = np.ascontiguousarray(
                    sample_weight, dtype=DOUBLE)
            if len(sample_weight.shape) > 1:
                raise ValueError("Sample weights array has more "
                                 "than one dimension: %d" %
                                 len(sample_weight.shape))
            if len(sample_weight) != n_samples:
                raise ValueError("Number of weights=%d does not match "
                                 "number of samples=%d" %
                                 (len(sample_weight), n_samples))

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if self.min_weight_fraction_leaf != 0. and sample_weight is not None:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight))
        else:
            min_weight_leaf = 0.

        # Set min_samples_split sensibly
        min_samples_split = max(self.min_samples_split,
                                2 * self.min_samples_leaf)

        # Build tree
        criterion = self.criterion
        if not isinstance(criterion, Criterion):
            if is_classification:
                criterion = CRITERIA_CLF[self.criterion](self.n_outputs_,
                                                         self.n_classes_)
            else:
                criterion = CRITERIA_REG[self.criterion](self.n_outputs_)

        SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS

        splitter = self.splitter
        if not isinstance(self.splitter, Splitter):
            splitter = SPLITTERS[self.splitter](criterion,
                                                self.max_features_,
                                                self.min_samples_leaf,
                                                min_weight_leaf,
                                                random_state)

        self.tree_ = Tree(self.n_features_, self.n_classes_, self.n_outputs_)

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0:
            builder = DepthFirstTreeBuilder(splitter, min_samples_split,
                                            self.min_samples_leaf,
                                            min_weight_leaf,
                                            max_depth)
        else:
            builder = BestFirstTreeBuilder(splitter, min_samples_split,
                                           self.min_samples_leaf,
                                           min_weight_leaf,
                                           max_depth,
                                           max_leaf_nodes)

        builder.build(self.tree_, X, y, sample_weight)

        if self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self
