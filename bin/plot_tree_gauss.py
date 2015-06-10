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
from sklearnef.ensemble import UnSupervisedRandomForestClassifier

# information
__author__ = "Oskar Maier"
__version__ = "r0.1.1, 2015-06-08"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
Learning density distribution from 2D gaussians and plot the results.

The cluster centers will be distributed within a certain distance of
each other and the co-variance matrices skewed randomly according to
the supplied sigma multiplier.

The classifier will be trained with a single tree to better observe the
trainign effects and max_features will be disabled.
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
    clf = UnSupervisedRandomForestClassifier(n_estimators=1,
                                             random_state=args.seed,
                                             min_samples_leaf=N_FEATURES,
                                             max_features=None,
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
    
    if not args.no_split_lines:
        info = clf.estimators_[0].parse_tree_leaves()
        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)
        hlines = ([], [], [])
        vlines = ([], [], [])
        for node in info:
            if node is not None: # leave node
                for pos in node['range'][0]: # xrange
                    if not np.isinf(pos):
                        xliml, xlimr = node['range'][1]
                        vlines[0].append(pos)
                        vlines[1].append(ymin if np.isinf(xliml) else xliml)
                        vlines[2].append(ymax if np.isinf(xlimr) else xlimr)
                for pos in node['range'][1]: # xrange
                    if not np.isinf(pos):
                        yliml, ylimr = node['range'][0]
                        hlines[0].append(pos)
                        hlines[1].append(xmin if np.isinf(yliml) else yliml)
                        hlines[2].append(xmax if np.isinf(ylimr) else ylimr)
        plt.hlines(hlines[0], hlines[1], hlines[2], colors='blue', linestyles='dotted')
        plt.vlines(vlines[0], vlines[1], vlines[2], colors='blue', linestyles='dotted')    
    
    # second plot: prediction
    plt.subplot(2, 1, 2)
    im = plt.imshow(prob_predict.reshape((x.size,y.size)).T, extent=[min(x),max(x),min(y),max(y)], interpolation='none', cmap=plt.cm.afmhot, aspect='auto', origin='lower') #'auto'
    plt.colorbar()
    
    plt.xlim(min(x),max(x))
    plt.ylim(min(y),max(y))
    plt.title('Prediction')
    
    # add split-lines
    if not args.no_split_lines:
        info = clf.estimators_[0].parse_tree_leaves()
        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)
        hlines = ([], [], [])
        vlines = ([], [], [])
        for node in info:
            if node is not None: # leave node
                for pos in node['range'][0]: # xrange
                    if not np.isinf(pos):
                        xliml, xlimr = node['range'][1]
                        vlines[0].append(pos)
                        vlines[1].append(ymin if np.isinf(xliml) else xliml)
                        vlines[2].append(ymax if np.isinf(xlimr) else xlimr)
                for pos in node['range'][1]: # xrange
                    if not np.isinf(pos):
                        yliml, ylimr = node['range'][0]
                        hlines[0].append(pos)
                        hlines[1].append(xmin if np.isinf(yliml) else yliml)
                        hlines[2].append(xmax if np.isinf(ylimr) else ylimr)
        plt.hlines(hlines[0], hlines[1], hlines[2], colors='blue', linestyles='dotted')
        plt.vlines(vlines[0], vlines[1], vlines[2], colors='blue', linestyles='dotted')
    
    plt.show()
    
def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    return parser.parse_args()

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('--n-clusters', default=4, type=int, help='The number of gaussian distributions to create.')
    parser.add_argument('--n-samples', default=1000, type=int, help='The number of training samples to draw from each gaussian.')
    parser.add_argument('--sigma', default=0.4, type=float, help='The sigma multiplier of the gaussian distributions.')
    parser.add_argument('--min-improvement', default=0, type=float, help='The minimum improvement require to consider a split valid.')
    parser.add_argument('--no-split-lines', action='store_true', help='Do not plot the split-lines.')
    parser.add_argument('--resolution', default=0.05, type=float, help='The plotting resolution.')
    parser.add_argument('--max-area', default=10, type=int, help='The maximum area over which the gaussians should be distributed.')
    parser.add_argument('--seed', default=None, type=int, help='The random seed to use. Fix to an integer to create reproducible results.')

    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    return parser

if __name__ == "__main__":
    main()
