"""
Testing for the tree module (sklearnef.tree).
"""
import pickle
from itertools import product
import platform

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

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
from sklearn.utils.validation import check_random_state

from sklearn.tree import DecisionTreeClassifier

from sklearnef.tree import UnSupervisedDecisionTreeClassifier
from sklearnef.tree import SemiSupervisedDecisionTreeClassifier

from sklearn import tree
from sklearn.tree._tree import TREE_LEAF
from sklearn import datasets

from sklearn.preprocessing._weights import _balance_weights


CLF_CRITERIONS = ("gini", "entropy")
REG_CRITERIONS = ("mse", )

SEMISCLF_TREES = {
    "SemiSupervisedDecisionTreeClassifier": SemiSupervisedDecisionTreeClassifier
}

UNSCLF_TREES = {
    "UnSupervisedDecisionTreeClassifier": UnSupervisedDecisionTreeClassifier
}

ALL_TREES = dict()
ALL_TREES.update(SEMISCLF_TREES)
ALL_TREES.update(UNSCLF_TREES)

# toy sample !TODO: Update to something usable for un- and semi-
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [-1, 1, 1]

# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

n_samples=1500
random_state = check_random_state(0)

X_multilabel, y_multilabel = datasets.make_multilabel_classification(random_state=0, return_indicator=True, n_samples=30, n_features=10)
blobs = datasets.make_blobs(n_samples=n_samples, n_features=2, centers=3, random_state=8, random_state=0)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
y_random = random_state.randint(0, 4, size=(20, ))

DATASETS = {
    "iris": {"X": iris.data, "y": iris.target},
    "blobs": {"X": blobs[0], "y": blobs[1]},
    "noisy_moons": {"X": noisy_moons[0], "y": noisy_moons[1]},
    "noisy_circles": {"X": noisy_circles[0], "y": noisy_circles[1]},
    "multilabel": {"X": X_multilabel, "y": y_multilabel},
    "zeros": {"X": np.zeros((20, 3)), "y": y_random}
}

from sklearn.tree.tests.test_tree import assert_tree_equal

def test_probability():
    """Predict probabilities using DecisionTreeClassifier."""

    for name, Tree in ALL_TREES.items():
        clf = Tree(max_depth=1, max_features=1, random_state=42)
        clf.fit(DATASETS['blobs']['X'], DATASETS['blobs']['y'])

        prob_predict = clf.predict_proba(DATASETS['blobs']['X'])
        assert_array_almost_equal(np.sum(prob_predict, 1),
                                  np.ones(DATASETS['blobs']['X'].shape[0]),
                                  err_msg="Failed with {0}".format(name))
        assert_array_equal(np.argmax(prob_predict, 1), #!TODO: Predict not applicable for un-supervised
                           clf.predict(DATASETS['blobs']['X']),
                           err_msg="Failed with {0}".format(name))
        assert_almost_equal(prob_predict, #!TODO: Applicable for me?
                            np.exp(clf.predict_log_proba(DATASETS['blobs']['X'])), 8,
                            err_msg="Failed with {0}".format(name))


def test_pure_set():
    """Check when y is pure."""
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
    y = [1, 1, 1, 1, 1, 1]

    for name, TreeClassifier in SEMISCLF_TREES.items():
        clf = TreeClassifier(random_state=0)
        clf.fit(X, y)
        assert_array_equal(clf.predict(X), y,
                           err_msg="Failed with {0}".format(name))


def test_importances():
    """Check variable importances."""
    X, y = datasets.make_classification(n_samples=2000,
                                        n_features=10,
                                        n_informative=3,
                                        n_redundant=0,
                                        n_repeated=0,
                                        shuffle=False,
                                        random_state=0)

    for name, Tree in ALL_TREES.items():
        clf = Tree(random_state=0)

        clf.fit(X, y)
        importances = clf.feature_importances_
        n_important = np.sum(importances > 0.1)

        assert_equal(importances.shape[0], 10, "Failed with {0}".format(name))
        assert_equal(n_important, 3, "Failed with {0}".format(name))

        X_new = clf.transform(X, threshold="mean")
        assert_less(0, X_new.shape[1], "Failed with {0}".format(name))
        assert_less(X_new.shape[1], X.shape[1], "Failed with {0}".format(name))

def test_max_features():
    """Check max_features."""

    for name, TreeEstimator in ALL_TREES.items():
        clf = TreeEstimator(max_features="auto")
        clf.fit(iris.data, iris.target)
        assert_equal(clf.max_features_, 2)
        
        est = TreeEstimator(max_features="sqrt")
        est.fit(iris.data, iris.target)
        assert_equal(est.max_features_,
                     int(np.sqrt(iris.data.shape[1])))

        est = TreeEstimator(max_features="log2")
        est.fit(iris.data, iris.target)
        assert_equal(est.max_features_,
                     int(np.log2(iris.data.shape[1])))

        est = TreeEstimator(max_features=1)
        est.fit(iris.data, iris.target)
        assert_equal(est.max_features_, 1)

        est = TreeEstimator(max_features=3)
        est.fit(iris.data, iris.target)
        assert_equal(est.max_features_, 3)

        est = TreeEstimator(max_features=0.01)
        est.fit(iris.data, iris.target)
        assert_equal(est.max_features_, 1)

        est = TreeEstimator(max_features=0.5)
        est.fit(iris.data, iris.target)
        assert_equal(est.max_features_,
                     int(0.5 * iris.data.shape[1]))

        est = TreeEstimator(max_features=1.0)
        est.fit(iris.data, iris.target)
        assert_equal(est.max_features_, iris.data.shape[1])

        est = TreeEstimator(max_features=None)
        est.fit(iris.data, iris.target)
        assert_equal(est.max_features_, iris.data.shape[1])

        # use values of max_features that are invalid
        est = TreeEstimator(max_features=10)
        assert_raises(ValueError, est.fit, X, y)

        est = TreeEstimator(max_features=-1)
        assert_raises(ValueError, est.fit, X, y)

        est = TreeEstimator(max_features=0.0)
        assert_raises(ValueError, est.fit, X, y)

        est = TreeEstimator(max_features=1.5)
        assert_raises(ValueError, est.fit, X, y)

        est = TreeEstimator(max_features="foobar")
        assert_raises(ValueError, est.fit, X, y)


def test_error():
    """Test that it gives proper exception on deficient input."""
    for name, TreeEstimator in ALL_TREES.items():
        # predict before fit
        est = TreeEstimator()
        assert_raises(Exception, est.predict_proba, X)

        est.fit(X, y)
        X2 = [-2, -1, 1]  # wrong feature shape for sample
        assert_raises(ValueError, est.predict_proba, X2)

        # Invalid values for parameters
        assert_raises(ValueError, TreeEstimator(min_samples_leaf=-1).fit, X, y)
        assert_raises(ValueError,
                      TreeEstimator(min_weight_fraction_leaf=-1).fit,
                      X, y)
        assert_raises(ValueError,
                      TreeEstimator(min_weight_fraction_leaf=0.51).fit,
                      X, y)
        assert_raises(ValueError, TreeEstimator(min_samples_split=-1).fit,
                      X, y)
        assert_raises(ValueError, TreeEstimator(max_depth=-1).fit, X, y)
        assert_raises(ValueError, TreeEstimator(max_features=42).fit, X, y)

        # Wrong dimensions
        est = TreeEstimator()
        y2 = y[:-1]
        assert_raises(ValueError, est.fit, X, y2)

        # Test with arrays that are non-contiguous.
        Xf = np.asfortranarray(X)
        est = TreeEstimator()
        est.fit(Xf, y)
        assert_almost_equal(est.predict(T), true_result)

        # predict before fitting
        est = TreeEstimator()
        assert_raises(Exception, est.predict, T)

        # predict on vector with different dims
        est.fit(X, y)
        t = np.asarray(T)
        assert_raises(ValueError, est.predict, t[:, 1:])

        # wrong sample shape
        Xt = np.array(X).T

        est = TreeEstimator()
        est.fit(np.dot(X, Xt), y)
        assert_raises(ValueError, est.predict, X)

        clf = TreeEstimator()
        clf.fit(X, y)
        assert_raises(ValueError, clf.predict, Xt)

def test_min_samples_leaf():
    """Test if leaves contain more than leaf_count training examples"""
    X = np.asfortranarray(iris.data.astype(tree._tree.DTYPE))
    y = iris.target

    # test both DepthFirstTreeBuilder and BestFirstTreeBuilder
    # by setting max_leaf_nodes
    for max_leaf_nodes in (None, 1000):
        for name, TreeEstimator in ALL_TREES.items():
            est = TreeEstimator(min_samples_leaf=5,
                                max_leaf_nodes=max_leaf_nodes,
                                random_state=0)
            est.fit(X, y)
            out = est.tree_.apply(X)
            node_counts = np.bincount(out)
            # drop inner nodes
            leaf_count = node_counts[node_counts != 0]
            assert_greater(np.min(leaf_count), 4,
                           "Failed with {0}".format(name))


def check_min_weight_fraction_leaf(name, datasets, sparse=False):
    """Test if leaves contain at least min_weight_fraction_leaf of the
    training set"""
    if sparse:
        X = DATASETS[datasets]["X_sparse"].astype(np.float32)
    else:
        X = DATASETS[datasets]["X"].astype(np.float32)
    y = DATASETS[datasets]["y"]

    weights = rng.rand(X.shape[0])
    total_weight = np.sum(weights)

    TreeEstimator = ALL_TREES[name]

    # test both DepthFirstTreeBuilder and BestFirstTreeBuilder
    # by setting max_leaf_nodes
    for max_leaf_nodes, frac in product((None, 1000), np.linspace(0, 0.5, 6)):
        est = TreeEstimator(min_weight_fraction_leaf=frac,
                            max_leaf_nodes=max_leaf_nodes,
                            random_state=0)
        est.fit(X, y, sample_weight=weights)

        if sparse:
            out = est.tree_.apply(X.tocsr())

        else:
            out = est.tree_.apply(X)

        node_weights = np.bincount(out, weights=weights)
        # drop inner nodes
        leaf_weights = node_weights[node_weights != 0]
        assert_greater_equal(
            np.min(leaf_weights),
            total_weight * est.min_weight_fraction_leaf,
            "Failed with {0} "
            "min_weight_fraction_leaf={1}".format(
                name, est.min_weight_fraction_leaf))


def test_min_weight_fraction_leaf():
    # Check on dense input
    for name in ALL_TREES:
        yield check_min_weight_fraction_leaf, name, "iris"


def test_pickle():
    """Check that tree estimator are pickable """
    for name, TreeClassifier in ALL_TREES.items():
        clf = TreeClassifier(random_state=0)
        clf.fit(iris.data, iris.target)
        score = clf.score(iris.data, iris.target)

        serialized_object = pickle.dumps(clf)
        clf2 = pickle.loads(serialized_object)
        assert_equal(type(clf2), clf.__class__)
        score2 = clf2.score(iris.data, iris.target)
        assert_equal(score, score2, "Failed to generate same score "
                                    "after pickling (classification) "
                                    "with {0}".format(name))

def test_multioutput():
    """Check estimators on multi-output problems."""
    X = [[-2, -1],
         [-1, -1],
         [-1, -2],
         [1, 1],
         [1, 2],
         [2, 1],
         [-2, 1],
         [-1, 1],
         [-1, 2],
         [2, -1],
         [1, -1],
         [1, -2]]

    y = [[-1, 0],
         [-1, 0],
         [-1, 0],
         [1, 1],
         [1, 1],
         [1, 1],
         [-1, 2],
         [-1, 2],
         [-1, 2],
         [1, 3],
         [1, 3],
         [1, 3]]

    T = [[-1, -1], [1, 1], [-1, 1], [1, -1]]
    y_true = [[-1, 0], [1, 1], [-1, 2], [1, 3]]

    # toy classification problem
    for name, TreeClassifier in ALL_TREES.items(): #!TODO: only prob allowed?
        clf = TreeClassifier(random_state=0)
        y_hat = clf.fit(X, y).predict(T)
        assert_array_equal(y_hat, y_true)
        assert_equal(y_hat.shape, (4, 2))

        proba = clf.predict_proba(T)
        assert_equal(len(proba), 2)
        assert_equal(proba[0].shape, (4, 2))
        assert_equal(proba[1].shape, (4, 4))

        log_proba = clf.predict_log_proba(T)
        assert_equal(len(log_proba), 2)
        assert_equal(log_proba[0].shape, (4, 2))
        assert_equal(log_proba[1].shape, (4, 4))


def test_classes_shape():
    """Test that n_classes_ and classes_ have proper shape."""
    for name, TreeClassifier in ALL_TREES.items():
        # Classification, single output
        clf = TreeClassifier(random_state=0)
        clf.fit(X, y)

        assert_equal(clf.n_classes_, 2)
        assert_array_equal(clf.classes_, [-1, 1])

        # Classification, multi-output
        _y = np.vstack((y, np.array(y) * 2)).T
        clf = TreeClassifier(random_state=0)
        clf.fit(X, _y)
        assert_equal(len(clf.n_classes_), 2)
        assert_equal(len(clf.classes_), 2)
        assert_array_equal(clf.n_classes_, [2, 2])
        assert_array_equal(clf.classes_, [[-1, 1], [-2, 2]])


def test_unbalanced_iris():
    """Check class rebalancing."""
    unbalanced_X = iris.data[:125]
    unbalanced_y = iris.target[:125]
    sample_weight = _balance_weights(unbalanced_y)

    for name, TreeClassifier in ALL_TREES.items():
        clf = TreeClassifier(random_state=0)
        clf.fit(unbalanced_X, unbalanced_y, sample_weight=sample_weight)
        assert_almost_equal(clf.predict(unbalanced_X), unbalanced_y)


def test_memory_layout():
    """Check that it works no matter the memory layout"""
    for (name, TreeEstimator), dtype in product(ALL_TREES.items(),
                                                [np.float64, np.float32]):
        est = TreeEstimator(random_state=0)

        # Nothing
        X = np.asarray(iris.data, dtype=dtype)
        y = iris.target
        assert_array_equal(est.fit(X, y).predict(X), y)

        # C-order
        X = np.asarray(iris.data, order="C", dtype=dtype)
        y = iris.target
        assert_array_equal(est.fit(X, y).predict(X), y)

        # F-order
        X = np.asarray(iris.data, order="F", dtype=dtype)
        y = iris.target
        assert_array_equal(est.fit(X, y).predict(X), y)

        # Contiguous
        X = np.ascontiguousarray(iris.data, dtype=dtype)
        y = iris.target
        assert_array_equal(est.fit(X, y).predict(X), y)

        # Strided
        X = np.asarray(iris.data[::3], dtype=dtype)
        y = iris.target[::3]
        assert_array_equal(est.fit(X, y).predict(X), y)


def _test_sample_weight(): #!TODO:Adapt this to the new forests? Sample-weight should still work! 
    """Check sample weighting."""
    # Test that zero-weighted samples are not taken into account
    X = np.arange(100)[:, np.newaxis]
    y = np.ones(100)
    y[:50] = 0.0

    sample_weight = np.ones(100)
    sample_weight[y == 0] = 0.0

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X, y, sample_weight=sample_weight)
    assert_array_equal(clf.predict(X), np.ones(100))

    # Test that low weighted samples are not taken into account at low depth
    X = np.arange(200)[:, np.newaxis]
    y = np.zeros(200)
    y[50:100] = 1
    y[100:200] = 2
    X[100:200, 0] = 200

    sample_weight = np.ones(200)

    sample_weight[y == 2] = .51  # Samples of class '2' are still weightier
    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    clf.fit(X, y, sample_weight=sample_weight)
    assert_equal(clf.tree_.threshold[0], 149.5)

    sample_weight[y == 2] = .5  # Samples of class '2' are no longer weightier
    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    clf.fit(X, y, sample_weight=sample_weight)
    assert_equal(clf.tree_.threshold[0], 49.5)  # Threshold should have moved

    # Test that sample weighting is the same as having duplicates
    X = iris.data
    y = iris.target

    duplicates = rng.randint(0, X.shape[0], 200)

    clf = DecisionTreeClassifier(random_state=1)
    clf.fit(X[duplicates], y[duplicates])

    sample_weight = np.bincount(duplicates, minlength=X.shape[0])
    clf2 = DecisionTreeClassifier(random_state=1)
    clf2.fit(X, y, sample_weight=sample_weight)

    internal = clf.tree_.children_left != tree._tree.TREE_LEAF
    assert_array_almost_equal(clf.tree_.threshold[internal],
                              clf2.tree_.threshold[internal])


def test_sample_weight_invalid():
    """Check sample weighting raises errors."""
    X = np.arange(100)[:, np.newaxis]
    y = np.ones(100)
    y[:50] = 0.0

    clf = DecisionTreeClassifier(random_state=0)

    sample_weight = np.random.rand(100, 1)
    assert_raises(ValueError, clf.fit, X, y, sample_weight=sample_weight)

    sample_weight = np.array(0)
    assert_raises(ValueError, clf.fit, X, y, sample_weight=sample_weight)

    sample_weight = np.ones(101)
    assert_raises(ValueError, clf.fit, X, y, sample_weight=sample_weight)

    sample_weight = np.ones(99)
    assert_raises(ValueError, clf.fit, X, y, sample_weight=sample_weight)


def check_class_weights(name):
    """Check class_weights resemble sample_weights behavior."""
    TreeClassifier = ALL_TREES[name]

    # Iris is balanced, so no effect expected for using 'auto' weights
    clf1 = TreeClassifier(random_state=0)
    clf1.fit(iris.data, iris.target)
    clf2 = TreeClassifier(class_weight='auto', random_state=0)
    clf2.fit(iris.data, iris.target)
    assert_almost_equal(clf1.feature_importances_, clf2.feature_importances_)

    # Make a multi-output problem with three copies of Iris
    iris_multi = np.vstack((iris.target, iris.target, iris.target)).T
    # Create user-defined weights that should balance over the outputs
    clf3 = TreeClassifier(class_weight=[{0: 2., 1: 2., 2: 1.},
                                        {0: 2., 1: 1., 2: 2.},
                                        {0: 1., 1: 2., 2: 2.}],
                          random_state=0)
    clf3.fit(iris.data, iris_multi)
    assert_almost_equal(clf2.feature_importances_, clf3.feature_importances_)
    # Check against multi-output "auto" which should also have no effect
    clf4 = TreeClassifier(class_weight='auto', random_state=0)
    clf4.fit(iris.data, iris_multi)
    assert_almost_equal(clf3.feature_importances_, clf4.feature_importances_)

    # Inflate importance of class 1, check against user-defined weights
    sample_weight = np.ones(iris.target.shape)
    sample_weight[iris.target == 1] *= 100
    class_weight = {0: 1., 1: 100., 2: 1.}
    clf1 = TreeClassifier(random_state=0)
    clf1.fit(iris.data, iris.target, sample_weight)
    clf2 = TreeClassifier(class_weight=class_weight, random_state=0)
    clf2.fit(iris.data, iris.target)
    assert_almost_equal(clf1.feature_importances_, clf2.feature_importances_)

    # Check that sample_weight and class_weight are multiplicative
    clf1 = TreeClassifier(random_state=0)
    clf1.fit(iris.data, iris.target, sample_weight**2)
    clf2 = TreeClassifier(class_weight=class_weight, random_state=0)
    clf2.fit(iris.data, iris.target, sample_weight)
    assert_almost_equal(clf1.feature_importances_, clf2.feature_importances_)


def test_class_weights():
    for name in ALL_TREES:
        yield check_class_weights, name


def check_class_weight_errors(name):
    """Test if class_weight raises errors and warnings when expected."""
    TreeClassifier = ALL_TREES[name]
    _y = np.vstack((y, np.array(y) * 2)).T

    # Invalid preset string
    clf = TreeClassifier(class_weight='the larch', random_state=0)
    assert_raises(ValueError, clf.fit, X, y)
    assert_raises(ValueError, clf.fit, X, _y)

    # Not a list or preset for multi-output
    clf = TreeClassifier(class_weight=1, random_state=0)
    assert_raises(ValueError, clf.fit, X, _y)

    # Incorrect length list for multi-output
    clf = TreeClassifier(class_weight=[{-1: 0.5, 1: 1.}], random_state=0)
    assert_raises(ValueError, clf.fit, X, _y)


def test_class_weight_errors():
    for name in ALL_TREES:
        yield check_class_weight_errors, name


def test_max_leaf_nodes():
    """Test greedy trees with max_depth + 1 leafs. """
    from sklearn.tree._tree import TREE_LEAF
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    k = 4
    for name, TreeEstimator in ALL_TREES.items():
        est = TreeEstimator(max_depth=None, max_leaf_nodes=k + 1).fit(X, y)
        tree = est.tree_
        assert_equal((tree.children_left == TREE_LEAF).sum(), k + 1)

        # max_leaf_nodes in (0, 1) should raise ValueError
        est = TreeEstimator(max_depth=None, max_leaf_nodes=0)
        assert_raises(ValueError, est.fit, X, y)
        est = TreeEstimator(max_depth=None, max_leaf_nodes=1)
        assert_raises(ValueError, est.fit, X, y)
        est = TreeEstimator(max_depth=None, max_leaf_nodes=0.1)
        assert_raises(ValueError, est.fit, X, y)


def test_max_leaf_nodes_max_depth():
    """Test preceedence of max_leaf_nodes over max_depth. """
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    k = 4
    for name, TreeEstimator in ALL_TREES.items():
        est = TreeEstimator(max_depth=1, max_leaf_nodes=k).fit(X, y)
        tree = est.tree_
        assert_greater(tree.max_depth, 1)


def test_arrays_persist():
    """Ensure property arrays' memory stays alive when tree disappears

    non-regression for #2726
    """
    for attr in ['n_classes', 'value', 'children_left', 'children_right',
                 'threshold', 'impurity', 'feature', 'n_node_samples']:
        value = getattr(DecisionTreeClassifier().fit([[0]], [0]).tree_, attr)
        # if pointing to freed memory, contents may be arbitrary
        assert_true(-2 <= value.flat[0] < 2,
                    'Array points to arbitrary memory')


def test_only_constant_features():
    random_state = check_random_state(0)
    X = np.zeros((10, 20))
    y = random_state.randint(0, 2, (10, ))
    for name, TreeEstimator in ALL_TREES.items():
        est = TreeEstimator(random_state=0)
        est.fit(X, y)
        assert_equal(est.tree_.max_depth, 0)


def test_with_only_one_non_constant_features():
    X = np.hstack([np.array([[1.], [1.], [0.], [0.]]),
                   np.zeros((4, 1000))])

    y = np.array([0., 1., 0., 1.0])
    for name, TreeEstimator in ALL_TREES.items():
        est = TreeEstimator(random_state=0, max_features=1)
        est.fit(X, y)
        assert_equal(est.tree_.max_depth, 1)
        assert_array_equal(est.predict_proba(X), 0.5 * np.ones((4, 2)))

def test_big_input():
    """Test if the warning for too large inputs is appropriate."""
    X = np.repeat(10 ** 40., 4).astype(np.float64).reshape(-1, 1)
    clf = DecisionTreeClassifier()
    try:
        clf.fit(X, [0, 1, 0, 1])
    except ValueError as e:
        assert_in("float32", str(e))


def test_realloc():
    from sklearn.tree._tree import _realloc_test
    assert_raises(MemoryError, _realloc_test)


def test_huge_allocations():
    n_bits = int(platform.architecture()[0].rstrip('bit'))

    X = np.random.randn(10, 2)
    y = np.random.randint(0, 2, 10)

    # Sanity check: we cannot request more memory than the size of the address
    # space. Currently raises OverflowError.
    huge = 2 ** (n_bits + 1)
    clf = DecisionTreeClassifier(splitter='best', max_leaf_nodes=huge)
    assert_raises(Exception, clf.fit, X, y)

    # Non-regression test: MemoryError used to be dropped by Cython
    # because of missing "except *".
    huge = 2 ** (n_bits - 1) - 1
    clf = DecisionTreeClassifier(splitter='best', max_leaf_nodes=huge)
    assert_raises(MemoryError, clf.fit, X, y)


def check_raise_error_on_1d_input(name):
    TreeEstimator = ALL_TREES[name]

    X = iris.data[:, 0].ravel()
    X_2d = iris.data[:, 0].reshape((-1, 1))
    y = iris.target

    assert_raises(ValueError, TreeEstimator(random_state=0).fit, X, y)

    est = TreeEstimator(random_state=0)
    est.fit(X_2d, y)
    assert_raises(ValueError, est.predict, X)


def test_1d_input():
    for name in ALL_TREES:
        yield check_raise_error_on_1d_input, name


def _check_min_weight_leaf_split_level(TreeEstimator, X, y, sample_weight):
    # Private function to keep pretty printing in nose yielded tests
    est = TreeEstimator(random_state=0)
    est.fit(X, y, sample_weight=sample_weight)
    assert_equal(est.tree_.max_depth, 1)

    est = TreeEstimator(random_state=0, min_weight_fraction_leaf=0.4)
    est.fit(X, y, sample_weight=sample_weight)
    assert_equal(est.tree_.max_depth, 0)


def check_min_weight_leaf_split_level(name):
    TreeEstimator = ALL_TREES[name]

    X = np.array([[0], [0], [0], [0], [1]])
    y = [0, 0, 0, 0, 1]
    sample_weight = [0.2, 0.2, 0.2, 0.2, 0.2]
    _check_min_weight_leaf_split_level(TreeEstimator, X, y, sample_weight)


def test_min_weight_leaf_split_level():
    for name in ALL_TREES:
        yield check_min_weight_leaf_split_level, name
