"""
Testing for the tree module (sklearnef.tree).
"""

import scipy.stats
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
#ALL_TREES.update(SEMISCLF_TREES)
ALL_TREES.update(UNSCLF_TREES)

# ---------- Test imports ----------
import sklearn.tree.tests.test_tree as sklearn_tests
sklearn_tests.ALL_TREES = ALL_TREES
sklearn_tests.CLF_TREES = ALL_TREES
sklearn_tests.REG_TREES = {}

# ---------- Tests ----------
def test_unsupervised_density():
    """Check learned class density of multiple, distributed multi-variate gaussians."""
    distance = 10
    sigma = 0.4
    n_samples = 500
    n_features = 2
    n_clusters = 4
    means = [[i * distance] * n_features for i in range(n_clusters)]
    cov = np.diag([sigma] * n_features)
    cov[np.triu_indices(n_features, 1)] = sigma/2
    cov[np.tril_indices(n_features, -1)] = sigma/2
    
    #!TODO: Should I fix the random state?
    X_train = np.concatenate([scipy.stats.multivariate_normal.rvs(mean, cov, n_samples) for mean in means])
    X_test = np.concatenate([scipy.stats.multivariate_normal.rvs(mean, cov, n_samples) for mean in means])
    y_test = np.sum([scipy.stats.multivariate_normal.pdf(X_test, mean, cov) for mean in means], 0)
    y_test_log = np.sum([scipy.stats.multivariate_normal.logpdf(X_test, mean, cov) for mean in means], 0)
    
    for name, Tree in UNSCLF_TREES.items():
        clf = Tree(random_state=0)
        clf.fit(X_train)
        prob_predict = clf.predict_proba(X_test)
        assert_array_almost_equal(prob_predict,
                                  y_test,
                                  err_msg="Failed with {0} using predict_proba() ".format(name))
        
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from matplotlib.mlab import griddata
    
    x = X_test[:,0]
    y = X_test[:,1]
    z = y_test
    
    z_min = min(prob_predict.min(), y_test.min())
    z_max = max(prob_predict.max(), y_test.max())
    
    xi = np.linspace(x.min(),x.max(),100)
    yi = np.linspace(y.min(),y.max(),100)
    zi = griddata(x,y,z,xi,yi, interp='linear')
    
    CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
    CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.PiYG)
    plt.colorbar() # draw colorbar
    
    plt.scatter(x,y,marker='o',c='b',s=5)
    plt.xlim(min(x),max(x))
    plt.ylim(min(y),max(y))
    plt.title('test')
    plt.show()
    
    #cmap = plt.get_cmap('PiYG')
    #norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    #plt.subplot(2, 1, 1)
    #im = plt.pcolormesh(x, y, y_test, cmap=cmap, norm=norm)
    #plt.colorbar()
    #plt.axis([x.min(), x.max(), y.min(), y.max()])
    #plt.title("gt")
    
    #plt.subplot(2, 1, 2)
    #im = plt.pcolormesh(x, y, prob_predict, cmap=cmap, norm=norm)
    #plt.colorbar()
    #plt.axis([x.min(), x.max(), y.min(), y.max()])
    #plt.title("results")
    
    #plt.show()
    

def test_unsupervised_density_iris():
    """Check learned class density on iris dataset."""
    for name, Tree in UNSCLF_TREES.items():
        for clid in np.unique(DATASETS['iris']['y']):
            mask = clid == DATASETS['iris']['y']
            clf = Tree(max_depth=1, max_features=1, random_state=0)
            clf.fit(DATASETS['iris']['X'][mask])
            prob_predict = clf.predict_proba(DATASETS['iris']['X'])
            assert_greater(prob_predict[mask].mean(),
                           prob_predict[~mask].mean(),
                           msg="Failed with {0} using predict_proba() ".format(name))
            prob_log_predict = clf.predict_log_proba(DATASETS['iris']['X'])
            assert_greater(prob_log_predict[mask].mean(),
                           prob_log_predict[~mask].mean(),
                           msg="Failed with {0} using predict_log_proba()".format(name))
            
def test_importances():
    #sklearn_tests.test_importances() # !TODO: pretty slow since now
    pass
    
def test_max_features():
    sklearn_tests.test_max_features()
        
def test_error():
    # !TODO: Wrong dimensions test won't raise an exception, as y is not used
    # sklearn_tests.test_error()
    pass
    
def test_min_samples_leaf():
    """Test if leaves contain more than leaf_count training examples"""
    X = iris.data
    y = iris.target

    # test both DepthFirstTreeBuilder and BestFirstTreeBuilder
    # by setting max_leaf_nodes
    for max_leaf_nodes in (None, 1000):
        for name, TreeEstimator in ALL_TREES.items():
            est = TreeEstimator(min_samples_leaf=5,
                                max_leaf_nodes=max_leaf_nodes,
                                random_state=0)
            est.fit(X, y)
            mask_leafs = -2 != est.tree_.feature
            nodes_in_leafs =  est.tree_.n_node_samples[mask_leafs]
            assert_greater(np.min(nodes_in_leafs), 4,
                           "Failed with {0}".format(name))    
