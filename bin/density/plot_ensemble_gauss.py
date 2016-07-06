#!/usr/bin/env python

"""Learning density distribution from 2D gaussians and plot the results."""

# build-in modules
import os
import sys
import argparse

# third-party modules
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

# path changes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # enable import from parent directory

# own modules
from lib import generate_clusters, sample_data, scale_data, generate_grid, draw_split_lines
from sklearnef.ensemble import DensityForest

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

No cut-lines will be plotted. To visualize these, please use the
plot_tree_gauss.py version of this script.
"""

# code
def main():
    args = getArguments(getParser())

    # initialize the random seed
    np.random.seed(args.seed)
    
    # ----- Data generation ----
    means, covs = generate_clusters(args.max_area, args.n_clusters, args.sigma)
    
    (X_train, X_train_unlabelled, X_train_labelled),\
    (y_train, y_train_unlabelled, y_train_labelled),\
    y_train_gt = sample_data(means, covs, args.n_samples)
    
    # ----- Data scaling ----
    # Must be performed before to display final data in the right space
    if args.scaling:
        scale_data(X_train, (X_train, X_train_unlabelled, X_train_labelled, means))
    
    # ----- Grid -----
    grid = generate_grid(X_train, args.sigma, args.resolution)
    
    # ----- Training -----
    clf = DensityForest(n_estimators=args.n_trees,
                        random_state=args.seed,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        max_depth=args.max_depth,
                        max_features=args.max_features,
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
    args = parser.parse_args()
    if args.max_features is not None and  args.max_features not in ['auto', 'sqrt' 'log2']:
        args.max_features = int(args.max_features)
    return args

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('--n-trees', default=10, type=int, help='The number of trees to train.')
    parser.add_argument('--n-clusters', default=4, type=int, help='The number of gaussian distributions to create.')
    parser.add_argument('--n-samples', default=1000, type=int, help='The number of training samples to draw from each gaussian.')
    parser.add_argument('--sigma', default=0.4, type=float, help='The sigma multiplier of the gaussian distributions.')
    parser.add_argument('--max-depth', default=None, type=int, help='The maximum tree depth.')
    parser.add_argument('--max-features', default='auto', help='The number of features to consider at each split. Can be an integer or one of auto, sqrt and log2')    
    parser.add_argument('--min-improvement', default=-5.0, type=float, help='Minimum information gain required to consider another split. Note that the information gain can take on negative values in some situations.')
    parser.add_argument('--scaling', action='store_true', help='Enable data scaling.')
    parser.add_argument('--resolution', default=100, type=float, help='The plotting resolution i.e. dots per dimension.')
    parser.add_argument('--max-area', default=10, type=int, help='The maximum area over which the gaussians should be distributed.')
    parser.add_argument('--seed', default=None, type=int, help='The random seed to use. Fix to an integer to create reproducible results.')

    #parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    #parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    return parser

if __name__ == "__main__":
    main()
