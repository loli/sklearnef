"""
Testing for the tree module (sklearnef.tree).
"""

import numpy as np

from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_less
from sklearn import datasets

from sklearnef.tree import UnSupervisedDecisionTreeClassifier
from sklearnef.tree import SemiSupervisedDecisionTreeClassifier

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
ALL_TREES.update(SEMISCLF_TREES)
ALL_TREES.update(UNSCLF_TREES)

# ---------- Tests ----------
def test_unsupervised_density():
    """Unsupervised learning of a datasets density."""
    for name, Tree in UNSCLF_TREES.items():
        for clid in np.unique(DATASETS['iris']['y']):
            mask = clid == DATASETS['iris']['y']
            clf = Tree(max_depth=1, max_features=1, random_state=0)
            clf.fit(DATASETS['iris']['X'][mask])
            prob_predict = clf.predict_proba(DATASETS['iris']['X'])
            assert_greater(prob_predict[mask].mean(),
                           prob_predict[~mask].mean(),
                           msg="Failed with {0}".format(name))
            prob_log_predict = clf.predict_log_proba(DATASETS['iris']['X'])
            assert_greater(prob_log_predict[mask].mean(),
                           prob_log_predict[~mask].mean(),
                           msg="Failed with {0}".format(name))
        

