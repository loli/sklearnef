#!/usr/bin/env python

"""Learning density distribution from 2D gaussians and plot the results."""

# build-in modules
import argparse

# third-party modules
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

# path changes

# own modules
from sklearnef.ensemble import DensityForest

# information
__author__ = "Oskar Maier"
__version__ = "r0.1.0, 2015-06-08"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
Learning density distribution from 2D gaussians and plot the results.

The cluster centers will be distributed within a certain distance of
each other and the co-variance matrices skewed randomly according to
the supplied sigma multiplier.

No cut-lines will be plotted. To visualize these, please use the
plot_tree_gauss.py version of this script.
"""

# constants
N_FEATURES = 2

# code
def main():
    args = getArguments(getParser())

    # initialize the random seed
    np.random.seed(args.seed)
    
    # ----- Define gaussian distributions / clusters -----
    means = []
    for _ in range(args.n_clusters):
        means.append([np.random.randint(0, args.max_area) for _ in range(N_FEATURES)])
    covs = []
    for _ in range(args.n_clusters):
        cov = np.diag([(np.random.random() + .5) * args.sigma for _ in range(N_FEATURES)])
        n_tri_elements = (N_FEATURES * (N_FEATURES - 1)) / 2
        cov[np.triu_indices(N_FEATURES, 1)] = [(np.random.random() + .5) * args.sigma/2 for _ in range(n_tri_elements)]
        cov[np.tril_indices(N_FEATURES, -1)] = [(np.random.random() + .5) * args.sigma/2 for _ in range(n_tri_elements)]
        covs.append(cov)
    
    # ----- Sample train set -----
    X_train = np.concatenate([scipy.stats.multivariate_normal.rvs(mean, cov, args.n_samples) for mean, cov in zip(means, covs)])
    
    # ----- Grid -----
    x_lower = X_train[:,0].min() - 2 * args.sigma
    x_upper = X_train[:,0].max() + 2 * args.sigma
    y_lower = X_train[:,1].min() - 2 * args.sigma
    y_upper = X_train[:,1].max() + 2 * args.sigma
    grid = np.mgrid[x_lower:x_upper:args.resolution,y_lower:y_upper:args.resolution]
    
    
    # ----- Training -----
    clf = DensityForest(n_estimators=args.n_trees,
                        random_state=args.seed,
                        min_samples_leaf=N_FEATURES,
                        n_jobs=-1,
                        max_features='auto',
                        min_improvement=args.min_improvement)
    clf.fit(X_train)
    
    # ----- Prediction -----
    X_test_pred = np.rollaxis(grid, 0, 3).reshape((np.product(grid.shape[1:]), grid.shape[0]))
    prob_predict = clf.predict_proba(X_test_pred)
    
    # ----- Ground truth -----
    X_test_gt = np.rollaxis(grid, 0, 3)
    prob_gt = np.sum([scipy.stats.multivariate_normal.pdf(X_test_gt, mean, cov) for mean, cov in zip(means, covs)], 0)
    prob_gt /= args.n_clusters # normalize
    
    # ----- Plotting -----
    x, y = grid
    x = np.unique(x)
    y = np.unique(y)
    
    # first plot: gt
    plt.subplot(2, 1, 1)
    im = plt.imshow(prob_gt.T, extent=[min(x),max(x),min(y),max(y)], interpolation='none', cmap=plt.cm.afmhot, aspect='auto', origin='lower') #'auto'
    plt.colorbar()
    
    plt.xlim(min(x),max(x))
    plt.ylim(min(y),max(y))
    plt.title('GT')
    
    # second plot: prediction
    plt.subplot(2, 1, 2)
    im = plt.imshow(prob_predict.reshape((x.size,y.size)).T, extent=[min(x),max(x),min(y),max(y)], interpolation='none', cmap=plt.cm.afmhot, aspect='auto', origin='lower') #'auto'
    plt.colorbar()
    
    plt.xlim(min(x),max(x))
    plt.ylim(min(y),max(y))
    plt.title('Prediction')
    
    plt.show()
    
def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    return parser.parse_args()

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('--n-trees', default=10, type=int, help='The number of trees to train.')
    parser.add_argument('--n-clusters', default=4, type=int, help='The number of gaussian distributions to create.')
    parser.add_argument('--n-samples', default=1000, type=int, help='The number of training samples to draw from each gaussian.')
    parser.add_argument('--sigma', default=0.4, type=float, help='The sigma multiplier of the gaussian distributions.')
    parser.add_argument('--min-improvement', default=0, type=float, help='The minimum improvement require to consider a split valid.')
    parser.add_argument('--resolution', default=0.05, type=float, help='The plotting resolution.')
    parser.add_argument('--max-area', default=10, type=int, help='The maximum area over which the gaussians should be distributed.')
    parser.add_argument('--seed', default=None, type=int, help='The random seed to use. Fix to an integer to create reproducible results.')

    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    return parser

if __name__ == "__main__":
    main()
