"""
Testing for the ensemble module (sklearnef.ensemble).
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

from sklearnef.ensemble import SemiSupervisedRandomForestClassifier, DensityForest

# ---------- Datasets ----------
# load the iris dataset, randomly permute it and mask some as unsupervised
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
SEMISCLF_FORESTS = {
    "SemiSupervisedRandomForestClassifier": SemiSupervisedRandomForestClassifier
}

UNSCLF_FORESTS = {
    "DensityForest": DensityForest
}

ALL_FORESTS = dict()
ALL_FORESTS.update(SEMISCLF_FORESTS)
ALL_FORESTS.update(UNSCLF_FORESTS)

# ---------- Test imports ----------

# ---------- Set-ups --------

# ---------- Tests ----------
def test_class_labels():
    """Test if the correct class labels are returned."""
    X = DATASETS["iris_semisup"]["X"]
    y = np.copy(DATASETS["iris_semisup"]["y"])
    
    # default labels
    labels = np.unique(y)[1:]
    clf = SemiSupervisedRandomForestClassifier(random_state=2)
    clf.fit(X, y)
    
    _confirm_only_a_in_b(labels, clf.predict(X), "Predicted continuous labels are wrong.")
    _confirm_only_a_in_b(labels, clf.transduced_labels_, "Transduced continuous labels are wrong.")
    
    # far apart labels
    for l in labels:
        y[l == y] += l*100
    labels = np.unique(y)[1:]
    clf = SemiSupervisedRandomForestClassifier(random_state=2)
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
