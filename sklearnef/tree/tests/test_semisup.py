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
from sklearn.utils.testing import assert_true, assert_false
from sklearn.utils.testing import raises

from sklearn import datasets

from sklearn import tree
from sklearn.utils.validation import NotFittedError
from sklearnef.tree import DensityTree
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
iris.target_semisup = np.copy(iris.target)
iris.target_semisup[np.random.randint(0, 2, iris.target_semisup.shape[0]) == 0] = -1

DATASETS = {
    "iris": {"X": iris.data, "y": iris.target},
    "iris_semisup": {"X": iris.data, "y": iris.target_semisup},
}

# ---------- Definitions ----------
SEMISCLF_TREES = {
    "SemiSupervisedDecisionTreeClassifier": SemiSupervisedDecisionTreeClassifier
}

UNSCLF_TREES = {
    "DensityTree": DensityTree
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
    
def test_semisupervised_probas():
    """Test probability results to sum to one."""
    clf = SemiSupervisedDecisionTreeClassifier(random_state=2)
    clf.fit(iris.data, iris.target)
    proba = clf.predict_proba(iris.data)
    assert_true(np.all(1 == proba.sum(1)))
    
def test_semisupervised_classes():
    """Test that lowest class never appears in results."""
    lowest_class_label = iris.target.min()
    clf = SemiSupervisedDecisionTreeClassifier(random_state=2)
    clf.fit(iris.data, iris.target)
    labels = clf.predict(iris.data)
    assert_false(lowest_class_label in labels)
    
def test_semisupervised_as_supervised():
    """Test the semi-supervised tree as supervised tree."""
    ssy = iris.target.copy()
    ssy[-1] = -1 # un-labelled class, will be ignored by semi-supervised approach
    
    ssclf = SemiSupervisedDecisionTreeClassifier(random_state=0,
                                               min_samples_leaf=iris.data.shape[-1],
                                               supervised_weight=.9999999999, # near 1, 1 is not allowed
                                               unsupervised_transformation=None)
    ssclf.fit(iris.data, ssy)
    ssprob = ssclf.predict_proba(iris.data)
    sspredict = ssclf.predict(iris.data)
    
    sclf = tree.DecisionTreeClassifier(random_state=0,
                                       min_samples_leaf=iris.data.shape[-1])
    sclf.fit(iris.data[:-1], iris.target[:-1])
    sprob = sclf.predict_proba(iris.data)
    spredict = sclf.predict(iris.data)
    
    assert_array_equal(ssprob, sprob)
    assert_array_equal(sspredict, spredict)
    
def test_class_labels():
    """Test if the correct class labels are returned."""
    X = DATASETS["iris_semisup"]["X"]
    y = np.copy(DATASETS["iris_semisup"]["y"])
    
    # default labels
    labels = np.unique(y)[1:]
    clf = SemiSupervisedDecisionTreeClassifier(random_state=2)
    clf.fit(X, y)
    
    _confirm_only_a_in_b(labels, clf.predict(X), "Predicted continuous labels are wrong.")
    _confirm_only_a_in_b(labels, clf.transduced_labels_, "Transduced continuous labels are wrong.")
    
    # far apart labels
    for l in labels:
        y[l == y] += l*100
    labels = np.unique(y)[1:]
    clf = SemiSupervisedDecisionTreeClassifier(random_state=2)
    clf.fit(X, y)

    _confirm_only_a_in_b(labels, clf.predict(X), "Predicted apart labels are wrong.")
    _confirm_only_a_in_b(labels, clf.transduced_labels_, "Transduced apart labels are wrong.")
     
# ---------- Helpers ----------
def _confirm_only_a_in_b(a, b, msg=None):
    print a, b
    occurences = np.zeros(b.shape[0], dtype=np.bool)
    for l in a:
        occurences += l == b
    assert_true(np.all(occurences), msg=msg)
 

