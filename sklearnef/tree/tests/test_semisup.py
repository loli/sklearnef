"""
Testing for the tree module (sklearnef.tree).
"""

import numpy as np

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_in
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_greater_equal
from sklearn.utils.testing import assert_less
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import raises

from sklearn import datasets

from sklearn import tree
from sklearn.utils.validation import NotFittedError
from sklearnef.tree import UnSupervisedDecisionTreeClassifier
from sklearnef.tree import SemiSupervisedDecisionTreeClassifier
from nose.tools.nontrivial import with_setup
import pickle

# ---------- Datasets ----------
# load the iris dataset and randomly permute it
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

DATASETS = {
    "iris": {"X": iris.data, "y": iris.target},
}

# ---------- Definitions ----------
SEMISCLF_TREES = {
    "SemiSupervisedDecisionTreeClassifier": SemiSupervisedDecisionTreeClassifier
}

UNSCLF_TREES = {
    "UnSupervisedDecisionTreeClassifier": UnSupervisedDecisionTreeClassifier
}

ALL_TREES = dict()
#ALL_TREES.update(SEMISCLF_TREES)
ALL_TREES.update(UNSCLF_TREES)

# ---------- Test imports ----------
import sklearn.tree.tests.test_tree as sklearn_tests
SKLEARN_TESTS = {'ALL_TREES': None,
                 'CLF_TREES': None,
                 'REG_TREES': None,
                 'SPARSE_TREES': None}

# ---------- Set-ups --------
def setup_sklearn_tests():
    """Set-up the original sklearn tests."""
    SKLEARN_TESTS['ALL_TREES'] = sklearn_tests.ALL_TREES
    SKLEARN_TESTS['CLF_TREES'] = sklearn_tests.CLF_TREES
    SKLEARN_TESTS['REG_TREES'] = sklearn_tests.REG_TREES
    SKLEARN_TESTS['SPARSE_TREES'] = sklearn_tests.SPARSE_TREES
    sklearn_tests.ALL_TREES = ALL_TREES
    sklearn_tests.CLF_TREES = ALL_TREES
    sklearn_tests.REG_TREES = {}
    sklearn_tests.SPARSE_TREES = {}

def teardown_sklearn_tests():
    """Re-set the original sklearn tests."""
    sklearn_tests.ALL_TREES = SKLEARN_TESTS['ALL_TREES']
    sklearn_tests.CLF_TREES = SKLEARN_TESTS['CLF_TREES']
    sklearn_tests.REG_TREES = SKLEARN_TESTS['REG_TREES']
    sklearn_tests.SPARSE_TREES = SKLEARN_TESTS['SPARSE_TREES']

# ---------- Tests ----------
def test_semisupervised():
    """Test class working without checking results."""
    clf = SemiSupervisedDecisionTreeClassifier(random_state=2)
    clf.fit(iris.data, iris.target)
    clf.predict_proba(iris.data)
    
def test_semisupervised_as_supervised():
    """Test the semi-supervised tree as supervised tree."""
    ssy = iris.target.copy()
    ssy[-1] = -1 # un-labelled class, will be ignored by semi-supervised approach
    
    ssclf = SemiSupervisedDecisionTreeClassifier(random_state=0,
                                               min_samples_leaf=iris.data.shape[-1],
                                               supervised_weight=1.,
                                               unsupervised_transformation=None)
    ssclf.fit(iris.data, ssy)
    ssprob = ssclf.predict_proba(iris.data) # remove last (empty) class !TODO; Shouldn't that be the first? Anyway, add to predict method.
    sspredict = ssclf.predict(iris.data) + 1 # correct class indices # !TODO: should finally be added to rpredict method
    
    sclf = tree.DecisionTreeClassifier(random_state=0,
                                       min_samples_leaf=iris.data.shape[-1])
    #sclf.fit(iris.data[:-1], iris.target[:-1])
    sclf.fit(iris.data, iris.target) #!TODO: Should actually be the one above, at it is the only reliable
    sprob = sclf.predict_proba(iris.data)
    spredict = sclf.predict(iris.data)

    print np.abs(ssprob - sprob).max()
    
    #f1 = ssclf.tree_.feature == -2
    #f2 = sclf.tree_.feature == -2
    
    #print np.all(f1 == f2)
    
    #for i, t in enumerate(f1):
    #    if t:
    #        print ssclf.tree_.value[i][0][:-1], sclf.tree_.value[i][0]
    
#     #print np.all(ssclf.tree_.capacity == sclf.tree_.capacity)
#     #print np.all(ssclf.tree_.children_left == sclf.tree_.children_left)
#     #print np.all(ssclf.tree_.children_right == sclf.tree_.children_right)
#     #print np.all(ssclf.tree_.feature == sclf.tree_.feature)
#     print np.all(ssclf.tree_.impurity == sclf.tree_.impurity)
#     #print np.all(ssclf.tree_.max_depth == sclf.tree_.max_depth)
#     print np.all(ssclf.tree_.max_n_classes == sclf.tree_.max_n_classes)
#     print np.all(ssclf.tree_.n_classes == sclf.tree_.n_classes)
#     #print np.all(ssclf.tree_.n_features == sclf.tree_.n_features)
#     print np.all(ssclf.tree_.n_node_samples == sclf.tree_.n_node_samples)
#     print np.all(ssclf.tree_.n_outputs == sclf.tree_.n_outputs)
#     #print np.all(ssclf.tree_.node_count == sclf.tree_.node_count)
#     #print np.all(ssclf.tree_.threshold == sclf.tree_.threshold)
#     print np.all(ssclf.tree_.value == sclf.tree_.value)
#     print np.all(ssclf.tree_.weighted_n_node_samples == sclf.tree_.weighted_n_node_samples)
# 
#     print "impurity"
#     print ssclf.tree_.impurity
#     print sclf.tree_.impurity
#     
#     print "tree_.max_n_classes"
#     print ssclf.tree_.max_n_classes
#     print sclf.tree_.max_n_classes
#     
#     print "tree_.n_node_samples"
#     print ssclf.tree_.n_node_samples
#     print sclf.tree_.n_node_samples
#         
#     print "tree_.value"
#     print ssclf.tree_.value[0][:3]
#     print sclf.tree_.value[0][:3]
#     
#     print "weighted_n_node_samples"
#     print ssclf.tree_.weighted_n_node_samples
#     print sclf.tree_.weighted_n_node_samples
    
    print "clf.classes_"
    print ssclf.classes_
    
    #print "clf.criterion"
    #print ssclf.criterion # passed
    
    #print "clf.max_depth"
    #print ssclf.max_depth # passed
    
    print "clf.n_outputs_", ssclf.n_outputs_
    
    assert_array_equal(ssprob, sprob)
    assert_array_equal(sspredict, spredict)
    
    

