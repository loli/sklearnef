"""
This module gathers sklearnefs tree-based methods, including unsupervised (density)
and semi-supervised trees. Onyl single output problems are handled.
"""

# Authors: Oskar Maier <oskar.maier@googlemail.com>
#
# Licence: BSD 3 clause !TODO: Change

from __future__ import division

import numbers
import warnings

import numpy as np
from scipy.sparse import issparse

from sklearn.externals import six

from sklearn.utils import check_array, check_random_state, compute_sample_weight
from sklearn.utils.validation import NotFittedError, check_is_fitted

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.tree import DENSE_SPLITTERS
from sklearn.tree._tree import Criterion
from sklearn.tree._tree import Splitter
from sklearn.tree._tree import DepthFirstTreeBuilder, BestFirstTreeBuilder
from sklearn.tree._tree import Tree
from sklearn.tree import _tree

import _tree as _treeef

from scipy.stats import mvn # Fortran implementation for multivariate normal CDF estimation
from sklearn.base import ClassifierMixin
from numpy.linalg.linalg import LinAlgError

try:
    from scipy.stats import multivariate_normal
except ImportError:
    from scipy.stats._multivariate import multivariate_normal # older scipy versions


from sklearnef.tree import _diffentropy

__all__ = ["SemiSupervisedDecisionTreeClassifier",
           "UnSupervisedDecisionTreeClassifier"]

# =============================================================================
# Types and constants
# =============================================================================

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

CRITERIA_CLF = {"entropy": _tree.Entropy, "semisupervised": _treeef.LabeledOnlyEntropy, "labeledonly": _treeef.LabeledOnlyEntropy}
DENSE_SPLITTERS['unsupervised'] = _treeef.UnSupervisedBestSplitter

# =============================================================================
# Base decision tree
# =============================================================================

class UnSupervisedDecisionTreeClassifier(DecisionTreeClassifier):
    """A tree for density estimation.
    
    The forest attempts to learn the probability distribution which
    created the training data. For this purpose, bounded multivariate
    normal distributions are fitted to the data.
    
    The predict_proba() and predict_log_proba() methods should then
    be treated as probability density functions (PDF) of the learned
    distributions, while the integral over predict_proba() is exactly
    one.

    Parameters
    ----------
    !TODO: Implement also a random version of the best-splitter
    splitter : string, optional (default="unsupervised")
        The strategy used to choose the split at each node. Currently only
        "unsupervised" is supported, which is a variant of the "best" splitter
        strategy.

    max_features : int, float, string or None, optional (default=None)
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
        
    min_improvement : float (default=0)
        The minimum improvement a split must exhibit to be considered adequate.
        One of the strongest parameters for controlling over-fitting in density
        trees.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        Ignored if ``max_leaf_nodes`` is not None.

    min_samples_split : int, optional (default=2)
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int or None, optional (default=None)
        The minimum number of samples required to be at a leaf node. Must be
        at least as high as the number of features in the training set. If None,
        set to the number of features at trainign time.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the input samples required to be at a
        leaf node.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
        If not None then ``max_depth`` will be ignored.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    tree_ : Tree object
        The underlying Tree object.

    !TODO: This does not seem to get set. Why?
    max_features_ : int,
        The inferred value of max_features.

    feature_importances_ : array of shape = [n_features]
        The feature importances. The higher, the more important the
        feature. The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.

    See also
    --------
    SemiSupervisedDecisionTreeClassifier

    References
    ----------

    .. [1] A. Criminisi, J. Shotton and E. Konukoglu, "Decision Forests: A 
           Unified Framework for Classification, Regression, Density
           Estimation, Manifold Learning and Semi-Supervised Learning",
           Foundations and Trends(r) in Computer Graphics and Vision, Vol. 7,
           No. 2-3, pp 81-227, 2012.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearnef.tree import UnSupervisedDecisionTreeClassifier
    >>> clf = UnSupervisedDecisionTreeClassifier(random_state=0)
    >>> iris = load_iris()
    >>> clf.fit(iris.data)
    >>> clf.predict_proba(iris.data)
    array([  4.82956378e-01,   4.62110584e-01,   6.80794444e-01,
         4.96085162e-01,   3.06794415e-01,   1.87555200e-01,
         ...
    """    
    def __init__(self,
                 criterion="unsupervised",
                 splitter="unsupervised",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=None,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_improvement=0,
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
        
        self.min_improvement = min_improvement

        if not 'unsupervised' == criterion:
            raise ValueError("Currently only the \"unsupervised\" criterion "
                             "is supported for density estimation.")
        if not 'unsupervised' == splitter:
            raise ValueError("Currently only the \"unsupervised\" splitter "
                             "is supported for density estimation.")
    
    def fit(self, X, y=None, sample_weight=None, check_input=True):
        """Build a decision tree from the training set X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples whose density distribution to estimate.
            Internally, it will be converted to ``dtype=np.float32``.

        y : None
            Not used, kep only for interface conformity reasons.

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
        if check_input:
            X = check_array(X, dtype=DTYPE, order='C')
 
        if self.min_samples_leaf is None:
            self.min_samples_leaf = X.shape[1]
        elif not X.shape[1] <= self.min_samples_leaf:
            raise ValueError("The number of minimum samples per leaf of the "
                             "model must be at least as large as the number "
                             "of features. Model min_samples_leaf is %s and "
                             "input n_features is %s "
                             % (self.min_samples_leaf, X.shape[1]))
        # !TODO: replace this crude method to get the tree to provide sufficient memory for storing the leaf values
        # remove passed y (which we do not need and only keep for interface conformity reasons)
        #!TODO: Find a way to make an y conveying the required shape info without actually allocating any memory
        #y = np.zeros((X.shape[0], X.shape[1]**2 + X.shape[1] + 1), dtype=np.dtype('d'))
        #y = np.lib.stride_tricks.as_strided(np.asarray([0], dtype=np.bool), shape=(X.shape[0], X.shape[1]**2 + X.shape[1] + 1), strides=(0, 0))
        s = X.shape[1]**2 + X.shape[1] + 1
        if s > X.shape[0]:
            y = np.zeros((X.shape[0], s), dtype=np.dtype('d')) # uses n_output
        else:
            y = np.tile(range(s), X.shape[0]//s + 1)[:X.shape[0]] # uses n_max_classes
        # initialise criterion here, since it requires another syntax than the default ones
        if 'unsupervised' == self.criterion:
            self.criterion =  _treeef.UnSupervisedClassificationCriterion(X.shape[0], X.shape[1], self.min_improvement)
        DecisionTreeClassifier.fit(self, X, y, sample_weight, check_input)
        
        # parse the tree once and create the MVND objects associated with each leaf
        self.mvnds = self.parse_tree_leaves()
        
        return self

    def pdf(self, X, check_input=True):
        """Probability density function of the learned distribution.
        
        Notes
        -----
        Alias for predict_proba().
        
        See also
        --------
        predict_proba()
        """
        return self.predict_proba(X, check_input)
    
    def cdf(self, X, check_input=True):
        """Cummulative density function of the learned distribution.
        
        \f[
            F(x_1, x_2, ...) = P(X_1\leq x_1, X_2\lew x_2, ...)
        \f]
        
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``. No sparse matrixes allowed.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        p : array of length n_samples
            The responses of the CDF for all input samples.
        """
        check_is_fitted(self, 'n_outputs_')
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse=None)

        n_samples, n_features = X.shape

        if self.tree_ is None:
            raise NotFittedError("Tree not initialized. Perform a fit first.")

        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             " match the input. Model n_features is %s and "
                             " input n_features is %s "
                             % (self.n_features_, n_features))
            
        # returns the indices of the node (alle leaves) each sample dropped into
        #leaf_indices = self.tree_.apply(X)
        
        cdf = np.zeros(X.shape[0])
        
        for i, x in enumerate(X):
            for mvnd in self.mvnds:
                if mvnd is None:
                    continue
                if np.all(x > np.asarray(mvnd.upper)): # complete cell covered
                    cdf[i] += mvnd.cmnd #!TODO: multiply by frac?
                elif np.any(x > np.asarray(mvnd.lower)): # partially contained
                    _x = np.minimum(x, mvnd.upper)
                    cdf[i] += mvnd.cdf(_x)
        return cdf

    def predict_proba(self, X, check_input=True):
        """Cummulative density function of the learned distribution.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``. No sparse matrixes allowed.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        p : array of length n_samples
            The responses of the PDF for all input samples.
        """
        check_is_fitted(self, 'n_outputs_')
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse=None)

        n_samples, n_features = X.shape

        if self.tree_ is None:
            raise NotFittedError("Tree not initialized. Perform a fit first.")

        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             " match the input. Model n_features is %s and "
                             " input n_features is %s "
                             % (self.n_features_, n_features))
        
        # compute the distribution function integral value
        pfi = sum([mvnd.cmnd for mvnd in self.mvnds if mvnd is not None])
        
        # returns the indices of the node (alle leaves) each sample dropped into
        leaf_indices = self.tree_.apply(X)

        # construct the the associated multivariate Gaussian distribution for each unique
        # leaf and pass the associated samples through its pdf to obtain their density
        # values
        out = np.zeros(n_samples, np.float)
        in_singluar_samples = np.zeros(n_samples, np.bool)
        for lidx in np.unique(leaf_indices):
            mask = lidx == leaf_indices
            try:
                out[mask] = self.mvnds[lidx].frac / pfi * self.mvnds[lidx].pdf(X[mask])
            except LinAlgError:
                warnings.warn("Singular co-variance matrix(ces) detected. Associated samples will be set to global maximum.")
                in_singluar_samples |= mask
        
        # set samples that would have fallen in a singular matrix to the sample-wide maximum value
        out[in_singluar_samples] = out.max()
        
        # return
        return out
        
    def parse_tree_leaves(self):
        r"""
        Returns the pice-wise multivariate normal distributions of the
        leaves of this density tree. The returned list contains an entry
        for each node of the tree, where internal nodes are designated by
        None.
            
        Returns
        -------
        mvnds : list of MVND objects
            List containing MVND objects describing a bounded multivariate
            normal distribution each.
        """
        return self.__parse_tree_leaves_rec(self.tree_)
        
    def __parse_tree_leaves_rec(self, tree, pos = 0, rang = None):
        # init
        if rang is None:
            rang = [(-np.inf, np.inf)] * tree.n_features
        
        # get node info
        fid = tree.feature[pos]
        thr = tree.threshold[pos]
        
        # if not leaf node...
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
            
        # if leaf node, return information
        else:
            return [MVND(tree, rang, pos)]
        
    def predict_log_proba(self, X):
        r"""Log cummulative density function of the learned distribution.

        Notes
        -----
        Log version of predict_proba() and pdf().
        
        See also
        --------
        predict_proba(), pdf()
        """
        return np.log(self.predict_proba(X)) # self.n_outputs is set to > 1, even if only one outcome enforced
    
    def predict(self, X):
        r"""Not supported for density forest.
        
        Only kept for interface consistency reasons.
        """
        raise NotImplementedError("Density forests do not support the predict() method.")

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


class MVND():
    def __init__(self, tree, range, node_id):
        r"""Bounded multivariate normal distribution."""
        if not len(range) == tree.n_features:
            raise ValueError('The length of `range` must equal the \
                              number of features in the tree')
        
        self.__range = range
        self.__node_id = node_id
        self.__cmnd = None
        self.__n_features = tree.n_features
        self.__tree = tree
        
        self.inf_to_large()
        
    @property
    def __node_value(self):
        return self.__tree.value[self.node_id].flat
        
    @property
    def node_id(self):
        r"""The id of the represented node."""
        return self.__node_id
    
    @property
    def range(self):
        r"""The MVNDs bounding box."""
        return self.__range
    
    @property
    def frac(self):
        r"""The represented nodes weight in the tree."""
        return self.__node_value[0]
    
    @property
    def cov(self):
        r"""The MVNDs co-variance matrix."""
        return self.__node_value[1:self.__n_features * self.__n_features + 1].reshape((self.__n_features, self.__n_features))
    
    @property
    def mu(self):
        r"""The MVNDs mean."""
        return self.__node_value[self.__n_features * self.__n_features + 1:self.__n_features * self.__n_features + self.__n_features + 1]
    
    @property
    def lower(self):
        r"""The lower corner of the MVNDs bounding box."""
        return [l for l, _ in self.range]
    
    @property
    def upper(self):
        r"""The upper corner of the MVNDs bounding box."""
        return [u for _, u in self.range]
        
    def inf_to_large(self, mult = 100):
        r"""
        Replace possible +/-inf values in the MVNs bounding box by the
        associated dimensions variance, multiplied by `mult`.
        
        Parameters
        ----------
        mult : number
            The multiplicator, choose sufficiently high.
        """
        for d, (l, u) in enumerate(self.range):
            if np.isneginf(l):
                l = -1 * mult * self.cov[d, d]
            if np.isposinf(u):
                u = mult * self.cov[d, d]
            self.range[d] = (l, u)

    @property  
    def cmnd(self, resolution = 10000, abseps = 1e-20, releps = 1e-20):
        r"""
        Compute the bounded cummulative multivariate normal distribution
        i.e. the maximum of this bounded MVNs CDF.
        
        The computed value corresponds to the integral of the MVNs PDF
        over the bounded area. This is equivalent to the maximum of the
        CDF function.
        
        Notes
        -----
        For high dimensional MVNs, this approach can become quite slow. In
        that case lowering the resolution can help speed up the
        computation.
        
        Warning
        -------
        As the CDF of a MVN has no closed form solution, an iterative
        estimation approach is used, that can return varying results
        for the same input data due to the choice of sampling points.
        
        Parameters
        ----------
        resolution : int
            Resolution i.e. number of points to use in this iterative
            estimation approach to the CDF. The number is multiplied with
            the dimensionality.
        """
        # Lazy-computation when needed, but then fixed, as the
        # non-deterministic nature of the algorithm will underwise lead to
        # non-reproducible results
        if self.__cmnd is None:
            self.__cmnd = self.cdf(self.upper, resolution=resolution,
                                   abseps=abseps, releps=releps)
        return self.__cmnd

    @property
    def pdf(self):
        r"""Probability density function.
        
        Notes
        -----
        A small normalization value is added to the co-variance matrix to
        avoid most cases of singularity. 
        
        Warning
        -------
        Does not check whether the supplied values lie inside the MVNDs
        bounding box.
        """
        return multivariate_normal(self.mu, self.cov + _diffentropy._get_singularity_threshold()).pdf
    
    def cdf(self, x, resolution = 10000, abseps = 1e-20, releps = 1e-20):
        r"""Cumulative density function for a single point.
        
        Parameters
        ----------
        x : ndarray
            The point for which to compute the CDF.
        resolution : int
            Resolution i.e. number of points to use in this iterative
            estimation approach to the CDF. The number is multiplied with
            the dimensionality.
        
        Warning
        -------
        Does not check whether the supplied value lies inside the MVNDs
        bounding box.
        """
        return mvn.mvnun(self.lower, x, self.mu,
                         self.cov,
                         maxpts=resolution * len(self.lower),
                         abseps=abseps, releps=releps)[0]
    