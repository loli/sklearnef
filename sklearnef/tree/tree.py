"""
This module gathers sklearnefs tree-based methods, including unsupervised (density)
and semi-supervised trees. Only single output problems are handled.
"""

# Authors: Oskar Maier <oskar.maier@googlemail.com>
#
# Licence: BSD 3 clause !TODO: Change

from __future__ import division
import warnings

import numpy as np
from numpy.linalg.linalg import LinAlgError

from sklearn.utils import check_array
from sklearn.utils.validation import NotFittedError, check_is_fitted

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.tree import DENSE_SPLITTERS
from sklearn.tree import _tree

from sklearn.preprocessing.data import StandardScaler

from sklearnef.tree import _tree as _treeef
from sklearnef.tree import _diffentropy

from scipy.stats import mvn # Fortran implementation for multivariate normal CDF estimation
from scipy.spatial.distance import mahalanobis
from scipy.sparse.csgraph import shortest_path, csgraph_from_dense

try:
    from scipy.stats import multivariate_normal
except ImportError:
    from scipy.stats._multivariate import multivariate_normal # older scipy versions

__all__ = ["DensityTree",
           "SemiSupervisedDecisionTreeClassifier",
           "GoodnessOfFit", "MECDF"]

# =============================================================================
# Types and constants
# =============================================================================

SINGULARITY_REGULARIZATION_TERM = 1e-6

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

CRITERIA_CLF = {"entropy": _tree.Entropy, "labeledonly": _treeef.LabeledOnlyEntropy}
DENSE_SPLITTERS['unsupervised'] = _treeef.UnSupervisedBestSplitter
DENSE_SPLITTERS['semisupervised'] = _treeef.SemiSupervisedBestSplitter


# =============================================================================
# Base decision tree
# =============================================================================

class DensityTree(DecisionTreeClassifier):
    r"""A tree for density estimation.
    
    The tree attempts to learn the probability distribution which
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
        
    min_improvement : float (default=0.)
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
        set to the number of features at training time.

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

    max_features_ : int
        The inferred value of max_features.

    feature_importances_ : array of shape = [n_features]
        The feature importances. The higher, the more important the
        feature. The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.
        
    n_outputs_ : int
        For internal use only. Value does not convey any meaning for `DensityTree`s.
    
    n_classes_ : array of shape = [n_outputs]
        For internal use only. Value does not convey any meaning for `DensityTree`s.
        
    classes_ : array of shape = [n_outputs, n_classes]
        For internal use only. Value does not convey any meaning for `DensityTree`s.

    Notes
    -----
    A third party fortran library uses its own random number generator, hence the
    results of two consecutive training with the same data and same random seed
    can differ slightly.

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
    >>> from sklearnef.tree import DensityTree
    >>> clf = DensityTree(random_state=0)
    >>> iris = load_iris()
    >>> clf.fit(iris.data)
    >>> clf.predict_proba(iris.data)
    array([  5.17937424e+00,   4.95581747e+00,   7.30105111e+00,
             5.32017140e+00,   3.29015862e+00,   2.01140023e+00,
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
        super(DensityTree, self).__init__(
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
        r"""Build a density tree from the training set X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples whose density distribution to estimate.
            Internally, it will be converted to ``dtype=np.float32``.

        y : None
            Not used, kept only for interface conformity reasons.

        !TODO: Check if the density split algorithm real honors the sample_weight passed.
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.

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
            
        # Notes: I could get the tree to allocate more memory by sub-classign it and re-implementing a number of its
        # methods. The question is mainly, which approach would make more sense: This or the Tree-subclassing. Which
        # would, sadly, also require a re-writing of the fit() method, which I've avoided so far.
        # !TODO: replace this crude method to get the tree to provide sufficient memory for storing the leaf values
        # remove passed y (which we do not need and only keep for interface conformity reasons)
        s = X.shape[1]**2 + X.shape[1] + 1 # required memory (double)
        if s > X.shape[0]:
            y = np.zeros((X.shape[0], s), dtype=np.dtype('d')) # uses n_output
        else:
            y = np.tile(range(s), X.shape[0]//s + 1)[:X.shape[0]] # uses n_max_classes
            
        # initialize criterion here, since it requires another syntax than the default ones
        if 'unsupervised' == self.criterion:
            self.criterion =  _treeef.UnSupervisedClassificationCriterion(X.shape[0], X.shape[1], self.min_improvement)
            
        # class parent fit method
        DecisionTreeClassifier.fit(self, X, y, sample_weight, False)
        
        # parse the tree once and create the MVND objects associated with each leaf
        self.mvnds = self.parse_tree_leaves()

        return self

    def pdf(self, X, check_input=True):
        r"""Probability density function of the learned distribution.
        
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``. No sparse matrixes allowed.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        
        Notes
        -----
        Alias for predict_proba().
        
        See also
        --------
        predict_proba()
        """
        return self.predict_proba(X, check_input)
    
    def cdf(self, X, check_input=True):
        r"""Cumulative density function of the learned distribution.
        
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
        self.__is_fitted()
        X, n_samples, _ = self.__check_X(X, check_input)

        cdf = np.zeros(n_samples, dtype=np.float)
        
        for i, x in enumerate(X):
            for mvnd in self.mvnds:
                if mvnd is None:
                    continue
                if np.all(x > np.asarray(mvnd.upper)): # complete cell covered
                    cdf[i] += mvnd.cmnd * mvnd.frac
                elif np.any(x > np.asarray(mvnd.lower)): # partially contained
                    _x = np.minimum(x, mvnd.upper)
                    cdf[i] += mvnd.cdf(_x) * mvnd.frac
        return cdf

    def predict_proba(self, X, check_input=True):
        r"""Probability density function of the learned distribution.

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
        self.__is_fitted()
        X, n_samples, _ = self.__check_X(X, check_input)
        
        # compute the distribution function integral value
        pfi = sum([mvnd.frac * mvnd.cmnd for mvnd in self.mvnds if mvnd is not None])
        
        # returns the indices of the node (all leaves) each sample dropped into
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

    def predict_log_proba(self, X):
        r"""Log probability density function of the learned distribution.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``. No sparse matrixes allowed.

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
    
    def goodness_of_fit(self, X, eval_type = 'mean_squared_error', check_input = True):
        r"""Goodness of fit of the learned density distribution.
        
        Compares the learned distribution function with an empirical CDF constructed
        from the data-points in `X` i.e. roughly \f[error(CDF(X)-ECDF_X(X)\f].
        
        Provided measures
        -----------------
        `mean_squared_error`
            The mean squared error over all data-points of X.
        `mean_squared_error_weighted`
            The mean squared error over all data-points of X,
            weighted by the PDF.
        `maximum`
            Maximum error over all data-points of X.
        
        Notes
        -----
        The provided measures are better described as fit error, than as goodness of fit
        criteria in a statistical sense. Therefore, it is only suitable to compare
        different learned distributions against each other under the condition, that the
        same samples (`X`) are used. Higher values denote a stronger error. 
        
        Parameters
        ----------
        X : array_like
            Samples form the original distribution. Must be distinct from the ones used
            to train the tree to obtain meaningful results.
        eval_type : string
            The type of goodness measure. One of `mean_squared_error`,
            `mean_squared_error_weighted` and `kolmogorov_smirnov`.
        """
        self.__is_fitted()
        X, _, _ = self.__check_X(X, check_input)
        
        # initialize goodness of fit object
        gof = GoodnessOfFit(self.cdf, X)
        
        eval_types = ['mean_squared_error', 'mean_squared_error_weighted', 'maximum']
        if eval_type == 'mean_squared_error':
            return gof.mean_squared_error()
        elif eval_type == 'mean_squared_error_weighted':
            return gof.mean_squared_error_weighted(self.pdf)
        elif eval_type == 'maximum':
            return gof.maximum()
        else:
            raise ValueError("Invalid eval type {}. Expected one of: {}" .format(eval_type, eval_types))
        
    def parse_tree_leaves(self):
        r"""
        Returns the piecewise multivariate normal distributions of the
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
        r"""Extract cov, mean and frac of each leaves and creates a MVND
        object from them."""
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
        
    def __is_fitted(self):
        check_is_fitted(self, 'n_outputs_')
        if self.tree_ is None:
            raise NotFittedError("Tree not initialized. Perform a fit first.")
        
    def __check_X(self, X, check_input):
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse=None)
        n_samples, n_features = X.shape
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             " match the input. Model n_features is %s and "
                             " input n_features is %s "
                             % (self.n_features_, n_features))
        return X, n_samples, n_features        

class SemiSupervisedDecisionTreeClassifier(DecisionTreeClassifier):
    r"""A tree for semi-supervised classification.
    
    The tree takes a mix of labelled and unlabelled training samples,
    ultimately assigning class labels to the unlabelled data.
    
    Using induction to transduction, a classification tree is trained
    representing the findings of the whole training set, labelled as
    well as unlabelled.
    
    Parameters
    ----------
    splitter : string, optional (default="semisupervised")
        The strategy used to choose the split at each node. Currently only
        "semisupervised" is supported, which is a variant of the "best"
        splitter strategy.

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
        
    supervised_weight: float, optional (default=0.5)
        Factor balancing the supervised against the un-supervised measures of
        split quality. A value of `1.0` would mean to consider only the
        labelled samples, a value of `0.0` would equal a density tree.
        Note that a clean value of `1.0` is not allowed, at it would lead
        to non max-margin splits. Please use the original `sklearn`
        `DecisionTreeClassifier` for that effect.
        
    !TODO: Assert that this is only applied to the non-supervised part of the data.
           Maybe by initializing the Splitter later of something? Is this at all possible?
           
    unsupervised_transformation: string, object or None, optional (default='scale')
        Transformation method for the un-supervised samples (their split
        quality measure requires features of equal scale). Choices are:
            - 'scale', in which case the `StandardScaler` is employed.
            - Any object which implements the fit() and transform() methods.
            - None, in which the user is responsible for data normalization.

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

    max_features_ : int
        The inferred value of max_features.

    feature_importances_ : array of shape = [n_features]
        The feature importances. The higher, the more important the
        feature. The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.
        
    n_outputs_ : int
        For internal use only. Value does not convey any meaning for
        `SemiSupervisedDecisionTreeClassifier`s.
    
    n_classes_ : array of shape = [n_outputs]
        For internal use only. Value does not convey the same meaning for
        `SemiSupervisedDecisionTreeClassifier`s. First entry holds the count
        of unique class labels of the tree plus one (for the un-labelled class).
        All other entries are the same.
        
    classes_ : array of shape = [n_outputs, n_classes]
        For internal use only. Value does not convey the same meaning for
        `SemiSupervisedDecisionTreeClassifier`s: First entry holds the unique
        class labels of the tree, with the first being the un-labelled class
        label, which is not used for classification. All other entries are the
        same.

    Notes
    -----
    A third party fortran library uses its own random number generator, hence the
    results of two consecutive training with the same data and same random seed
    can differ slightly.

    See also
    --------
    DensityTree, sklearn.DecisionTreeClassifier

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
    >>> from sklearnef.tree import SemiSupervisedDecisionTreeClassifier
    >>> clf = SemiSupervisedDecisionTreeClassifier(random_state=0)
    >>> iris = load_iris()
    >>> clf.fit(iris.data, iris.target)
    >>> clf.predict_proba(iris.data)
    array([[ 0.99405824,  0.        ],
           [ 0.99405824,  0.        ],
         ...    
    """
    
    def __init__(self,
                 criterion="semisupervised",
                 splitter="semisupervised",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=None,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 #min_improvement=0,
                 supervised_weight=0.5,
                 unsupervised_transformation='scale',
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
        
        if 1.0 == supervised_weight:
            raise ValueError("A supervised_weight of 1.0 is not allowed, as it"\
                             "would results in non max-margin splits. Please "\
                             "use the sklearn.DecisionTreeClassifier for the same"\
                             "effect.")
        
        self.supervised_weight = supervised_weight
        if 'scale' == unsupervised_transformation:
            unsupervised_transformation = StandardScaler()
        self.unsupervised_transformation = unsupervised_transformation

        if not 'semisupervised' == criterion:
            raise ValueError("Currently only the \"semisupervised\" criterion "
                             "is supported for density estimation.")
        if not 'semisupervised' == splitter:
            raise ValueError("Currently only the \"semisupervised\" splitter "
                             "is supported for density estimation.")

    def fit(self, X, y, sample_weight=None, check_input=True):
        r"""Build a decision tree from the training set (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and can not be sparse..

        y : array-like, shape = [n_samples]
            The target values (class labels in classification). The lowest
            label is always taken as marker for the un-labelled samples
            (usually -1).

        !TODO: Check if the split algorithms (both) real honors the sample_weight passed.
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
 
        # apply transformation to data
        if self.unsupervised_transformation is not None:
            X = self.unsupervised_transformation.fit_transform(X)
 
        if self.min_samples_leaf is None:
            self.min_samples_leaf = X.shape[1]
        elif not X.shape[1] <= self.min_samples_leaf:
            raise ValueError("The number of minimum samples per leaf of the "
                             "model must be at least as large as the number "
                             "of features. Model min_samples_leaf is %s and "
                             "input n_features is %s "
                             % (self.min_samples_leaf, X.shape[1]))
        
        y = np.atleast_1d(y)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))
        
        if not 1 == y.shape[1]:
            raise ValueError("The semi-supervised classification allows only "
                             "for one output per sample. Provided class "
                             " ground truth holds %s outputs per sample."
                             % (y.shape[1]))

        # get mask denoting unlabelled samples
        mask_unlabelled = (np.min(y[:,0]) == y[:,0])

        # n_outputs_ and n_classes_ must be pre-computed to init the
        # classification criterion (note: always classification)
        # !TODO: This creates redundancy, as the same process will be
        # repeated int the parent classes fit() method
        n_classes_ = np.array([np.unique(y[:,0]).shape[0]], dtype=np.intp)
            
        # Expand y by tiling. This will cause the Tree to allocate sufficient
        # memory for storing the gaussian distributions per leaf.
        # !TODO: Can I find a better approach than this?
        s = X.shape[1]**2 + X.shape[1] + 1 # memory requirement (in double)
        y = np.tile(y, (1, s))
            
        # initialise criterion here, since it requires another syntax than the default ones
        if 'semisupervised' == self.criterion:
            self.criterion =  _treeef.SemiSupervisedClassificationCriterion(
                                        X.shape[0],
                                        X.shape[1],
                                        0, # disable min_improvement stop criteria
                                        self.supervised_weight,
                                        1, #self.n_outputs_ always 1
                                        n_classes_)
        DecisionTreeClassifier.fit(self, X, y, sample_weight, False)

        # parse the tree once and create the MVND objects associated with each leaf
        self.mvnds = self.parse_tree_leaves()

        # use transduction to classifiy the un-labelled training samples
        yu = self.transduction(X[mask_unlabelled], X[~mask_unlabelled],
                               y[~mask_unlabelled][:,0])
        
        Xa = np.concatenate((X[mask_unlabelled], X[~mask_unlabelled]), 0)
        ya = np.concatenate((yu, np.squeeze(y[~mask_unlabelled][:,0])), 0) # Note: y will not contain the unsupervised class
        
        # use induction/label counting to finalize decision tree
        self.__induction(Xa, ya)

        return self

    def predict(self, X, check_input=True):
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse=None)
        # apply transformation to data
        if self.unsupervised_transformation is not None:
            X = self.unsupervised_transformation.transform(X)
        return DecisionTreeClassifier.predict(self, X, False)[:,0] # only first of all outputs

    def predict_proba(self, X, check_input=True):
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse=None)
        # apply transformation to data
        if self.unsupervised_transformation is not None:
            X = self.unsupervised_transformation.transform(X)
        return DecisionTreeClassifier.predict_proba(self, X, False)[0][:,:-1] # only first of all outputs; remove last probability (is for unlabelled class)
    
    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))
        
    def transduction(self, Xu, Xl, yl):
        r"""
        Compute the class-memberships of the unlabelled `Xu` samples using
        geodisic distances on a surface formed by the trees piecewise
        Gaussians to the `yl` labelled set `Xl`. This is similar to label
        propagation.
        
        !TODO: Speed up this method.
        
        Parameters
        ----------
        Xu : array_like
            Unlabelled samples.
        Xl : array_like
            Labelled samples.
        yl : array_like
            Labels of the labelled samples.
            
        Returns
        -------
        xu : ndarray
            Labels of the unlabelled samples.
        
        Warning
        -------
        This method requires the Gaussian distribution to be saved in the
        tree leaves. This holds only true directly after fitting, while a
        call to __induction() will remove that information.
        !TODO: Keep the info and hence this transduction method valid! 
        """
        # prepare
        X = np.vstack((Xu, Xl))
        n = X.shape[0]
        nu = Xu.shape[0]

        # get leaves and other info
        leaf_indices = self.tree_.apply(X)
        
        # inline, memoization function to get icov
        @memoize
        def get_icov(nidx):
            """Get cov and compute its inverse with conditional regularization."""
            lidx = leaf_indices[nidx]
            try:
                return np.linalg.inv(self.mvnds[lidx].cov)
            except LinAlgError:
                v = [SINGULARITY_REGULARIZATION_TERM] * self.mvnds[lidx].cov.shape[0]
                return np.linalg.inv(self.mvnds[lidx].cov + np.diag(v))
        
        # compute pair-wise symmetric Mahalanobis distances
        pdists = np.zeros((n, n), np.float32)
        for i, si in enumerate(X):
            icov = get_icov(i)
            for j, sj in enumerate(X[i+1:]):
                j += i + 1
                #!TODO: Won't i need the mean here, too?
                pdists[i,j] = smahalanobis(si, sj, icov, get_icov(j))
                
        # search shortest path between all
        pdists_sparse = csgraph_from_dense(pdists, null_value=0)
        spaths = shortest_path(pdists_sparse, 'auto', directed=False)

        # find nearest labelled samples of each unlabelled sample
        argnearest = np.argmin(spaths[:nu,nu:], axis=1)
        
        # transfer labels
        yu = yl[argnearest]

        return yu
        
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
        """!TODO: Pack this and other shared methods in a common parent tree class?"""
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
       
    def __induction(self, X, y):
        r"""
        Re-writes the trees memory, changing the Gaussian distributin based
        nodes into the default class posteriors.
        
        Essentially, simply counts the class occurences per leaf and computes
        the class posteriori for each.
        """
        # prepare
        y = np.squeeze(y)

        # get leaf indices
        leaf_indices = self.tree_.apply(X)
        
        # convert to a zero-based class membership array
        class_k, y = np.unique(y, return_inverse=True)
        n_classes = len(class_k)
        
        # count class occurence per leaf and edit tree accordingly
        for lidx in np.unique(leaf_indices):
            mask = lidx == leaf_indices
            label_count = np.bincount(y[mask])
            label_count = np.pad(label_count, (0, n_classes - len(label_count)), 'constant')
            #!TODO: Is the unlabelled data really the last dimension?
            self.tree_.value[lidx][0][:-1] = label_count       
        
def smahalanobis(x, y, icovx, icovy):
    r"""The symmetric, cov-dependent Mahalanobis distance"""
    return 0.5 * (mahalanobis(x, y, icovx) + mahalanobis(y, x, icovy))

def memoize(f):
    r""" Memoization decorator for a function taking a single argument """
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret 
    return memodict().__getitem__

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
    def cmnd(self, resolution = 5000, abseps = 1e-20, releps = 1e-20):
        r"""
        Compute the bounded cummulative multivariate normal distribution
        i.e. the maximum of this bounded MVNs CDF.
        
        The computed value corresponds to the integral of the MVNs PDF
        over the bounded area. This is equivalent to the maximum of the
        CDF function.
        
        Notes
        -----
        For high dimensional MVNs, this approach can become quite slow. In
        that case, lowering the resolution can help speed up the
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
    
    def cdf(self, x, resolution = 5000, abseps = 1e-20, releps = 1e-20):
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
        d = len(self.lower)
        return mvn.mvnun(self.lower, x, self.mu,
                         self.cov,
                         maxpts=resolution * d * d,
                         abseps=abseps, releps=releps)[0]
                         
class MECDF:
    def __init__(self, X):
        r"""Multivariate estimated cumulative density function.
        
        Implemented as step-wise function.
        """
        self.X = np.asarray(X)
        if not 2 == self.X.ndim:
            raise ValueError('Dimensionality must be 2.')
        self.n, self.d = self.X.shape
        
    def cdf(self, X):
        r"""
        Compute the MECDF response for samples.
        
        Parameters
        ----------
        X : array_like
            One or more samples for which to compute the response.
        
        Returns
        -------
        cdf : ndarray
            MECDF response for the samples in `X`.
        """
        #!TODO: Is there a faster way to do this in bulk?
        X = np.atleast_2d(X)
        if not X.shape[1] == self.d:
            raise ValueError('Invalid dimensionality.')
        if not 2 == X.ndim:
            raise ValueError('Dimensionality must be 1 or 2.')
        cdf = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            cdf[i] = np.count_nonzero(np.all(self.X <= x, axis=1)) / float(self.n)
        return cdf
    
class GoodnessOfFit():
    
    def __init__(self, cdf, X):
        r"""
        Measures and statistics for goodness of fit criteria between a
        distribution defined by its `cdf` and a set of samples `X`.
        
        Frozen version to avoid expensive re-computations.
        
        Notes
        -----
        The samples in `X` can not be the same used to train the distribution
        behind the supplied `cdf`.
        
        Parameters
        ----------
        cdf : function
            a d-dimensional CDF function
        X : sequence
            a sequence of d-dimensional samples from which to compute the ECDF
        """
        X = np.atleast_2d(X)
        
        if not 2 == X.ndim:
            raise ValueError('X must be two dimensional.')
        if not hasattr(cdf, '__call__'):
            raise ValueError('cdf must be callable.')
        
        self.cdf = cdf
        self.ecdf = MECDF(X).cdf
        
        self.xmin = np.min(X, 0)
        self.xmax = np.max(X, 0)
        
        self.X = X
        
        self.__cdf_x = None
        self.__ecdf_x = None

    @property
    def cdf_x(self):
        if self.__cdf_x is None:
            self.__cdf_x = self.cdf(self.X)
        return self.__cdf_x
    
    @property
    def ecdf_x(self):
        if self.__ecdf_x is None:
            self.__ecdf_x = self.ecdf(self.X)
        return self.__ecdf_x
    
    def maximum(self):
        """
        Kolmogorov-Smirnov similar maximum-error test.
        Max error between CDF and ECDF at all points of X.
        
        \f[
            D_n = \sup_x\left|ECDF(x)-CDF(x)\right|
        \f]
        
        Notes
        -----
        While similar to the Kolmogorov-Smirnov test, an implementation of said criterion
        on higher dimensions is unpractical.
        """
        return np.abs(self.ecdf_x - self.cdf_x).max()
    
    def mean_squared_error(self):
        """
        Mean squared error between CDF and ECDF at all points of X.
        """
        return ((self.ecdf_x - self.cdf_x)**2).mean()
    
    def mean_squared_error_weighted(self, pdf):
        """
        Full error between CDF and ECDF at all points of X weighted by
        CDF'(x) (=PDF(x)).
        
        Maybe considered some type of Cramer-von Mises criterion.
        
        Parameters
        ----------
        pdf : function
            the d-dimensional PDF function correpsonding to the CDF
            function used to initialize the object
        """
        pdf_x = pdf(self.X)
        return (((self.ecdf_x - self.cdf_x)**2) * pdf_x).mean()
    
# def CvM(cdf, pdf, X):
#     """
#     Note: cdf can not be determined by X, i.e. X must differ from 
#     the X_cdf used to create CDF.
#     
#     \f[
#         \omega^2 = \integral_{-\inf}^{+\inf}\left[ECDF(x) - CDF(x)\right]^2 dCDF(x)
#     \f]
#     resolving the Stieltjesintegral, we get
#     \f[
#         \omega^2 = \integral_{-\inf}^{+\inf}\left[ECDF(x) - CDF(x)\right]^2 * CDF'(X) dx
#     \f]
#     now, taking into account that \f$CDF'(x) = PDF(x)\f$, we get
#     \f[
#         \omega^2 = \integral_{-\inf}^{+\inf}\left[ECDF(x) - CDF(x)\right]^2 * PDF(X) dx
#     \f]
#     """
#     n, d = X.shape
#     ecdf = MECDF(X)
#     omega_square = np.pow(ecdf(X) - cdf(X), 2) * pdf(X)
#     T = n * omega_square
#     # cant make multivariate rieman integral of fixed points that easily!
    
