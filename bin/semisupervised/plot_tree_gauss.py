#!/usr/bin/env python

"""Semi-supervised classification from 2D gaussians and plot the results."""

# build-in modules
import argparse

# third-party modules
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

# path changes

# own modules
from sklearnef.tree import SemiSupervisedDecisionTreeClassifier

# information
__author__ = "Oskar Maier"
__version__ = "r0.1.1, 2015-06-08"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
Semi-supervised classification from 2D gaussians and plot the results.

The cluster centers will be distributed within a certain distance of
each other and the co-variance matrices skewed randomly according to
the supplied sigma multiplier. The center of the clusters receive a
unique label, all other (unlabelled) samples are randomly drawn from
the gaussian distributions.

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
    X_train_unlabelled = np.concatenate([scipy.stats.multivariate_normal.rvs(mean, cov, args.n_samples) for mean, cov in zip(means, covs)]).astype(np.float32)
    y_train_unlabelled = np.full(X_train_unlabelled.shape[0], -1)
    y_train_gt = np.concatenate([[c] * args.n_samples for c in np.arange(len(means))], 0)
    X_train_labelled = np.asarray(means).astype(np.float32)
    y_train_labelled = np.arange(len(means))

    X_train = np.concatenate((X_train_unlabelled, X_train_labelled), 0)
    y_train = np.concatenate((y_train_unlabelled, y_train_labelled), 0)
    
    # ----- Grid -----
    x_lower = X_train[:,0].min() - 2 * args.sigma
    x_upper = X_train[:,0].max() + 2 * args.sigma
    y_lower = X_train[:,1].min() - 2 * args.sigma
    y_upper = X_train[:,1].max() + 2 * args.sigma
    grid = np.mgrid[x_lower:x_upper:args.resolution,y_lower:y_upper:args.resolution]
    
    
    # ----- Training -----
    clf = SemiSupervisedDecisionTreeClassifier(random_state=args.seed,
                                               max_depth=args.max_depth,
                                               max_features=None,
                                               supervised_weight=args.supervised_weight,
                                               unsupervised_transformation=None)
    clf.fit(X_train, y_train)
    
    # ----- Learned distribution -----
    X_test_pred = np.rollaxis(grid, 0, 3).reshape((np.product(grid.shape[1:]), grid.shape[0]))
    pdf = clf.pdf(X_test_pred)
    #cdf = clf.cdf(X_test_pred)
    
    # ----- Ground truth distribution -----
    X_test_gt = np.rollaxis(grid, 0, 3)
    prob_gt = np.sum([scipy.stats.multivariate_normal.pdf(X_test_gt, mean, cov) for mean, cov in zip(means, covs)], 0)
    prob_gt /= args.n_clusters # normalize
    
    # ----- Trasnduction -----
    y_train_result = clf.transduction(X_train_unlabelled, X_train_labelled, y_train_labelled)
    
    # ----- A-posteriori classification -----
    y_train_prediction = clf.predict(X_train_unlabelled)

    # ----- Goodness of fit measure -----
    #X_eval = np.concatenate([scipy.stats.multivariate_normal.rvs(mean, cov, args.n_samples) for mean, cov in zip(means, covs)])
    #gof = GoodnessOfFit(clf.cdf, X_eval, resolution=200)
    #print 'Goodness of fit evaluation over {}^{} grid-points:'.format(200, X_eval.shape[1])
    #print '\tkolmogorov_smirnov:', gof.kolmogorov_smirnov()
    #print '\tmean_squared_error:', gof.mean_squared_error()
    #print '\tmean_squared_error_weighted:', gof.mean_squared_error_weighted(clf.pdf)
    
    # ----- E(M)CDF -----
    #emcdf = gof.ecdf(X_test_pred)
    
    # ----- Plotting -----
    x, y = grid
    x = np.unique(x)
    y = np.unique(y) 
    
    # colour range: pdf
    pdf_vmin = min(prob_gt.min(), pdf.min())
    pdf_vmax = min(prob_gt.max(), pdf.max())
    
    # plot: gt - pdf
    plt.subplot(3, 1, 1)
    plt.imshow(prob_gt.T, extent=[min(x),max(x),min(y),max(y)], interpolation='none',
               cmap=plt.cm.afmhot, aspect='auto', origin='lower',
               vmin=pdf_vmin, vmax=pdf_vmax, alpha=.5) #'auto'
    plt.scatter(X_train_unlabelled[:,0], X_train_unlabelled[:,1], c=y_train_gt, s=20, alpha=.6)
    plt.scatter(X_train_labelled[:,0], X_train_labelled[:,1], c=y_train_labelled, s=100)
    plt.colorbar()
    
    plt.xlim(min(x),max(x))
    plt.ylim(min(y),max(y))
    plt.title('Ground-truth: PDF + samples')
    
    if not args.no_split_lines:
        draw_split_lines(clf, x, y)
    
    # plot: learned - pdf
    plt.subplot(3, 1, 2)
    plt.imshow(pdf.reshape((x.size,y.size)).T, extent=[min(x),max(x),min(y),max(y)],
               interpolation='none', cmap=plt.cm.afmhot, aspect='auto', origin='lower',
               vmin=pdf_vmin, vmax=pdf_vmax, alpha=.5) #'auto'
    plt.scatter(X_train_unlabelled[:,0], X_train_unlabelled[:,1], c=y_train_result, s=20, alpha=.6)
    plt.scatter(X_train_labelled[:,0], X_train_labelled[:,1], c=y_train_labelled, s=100)
    plt.colorbar()
    
    plt.xlim(min(x),max(x))
    plt.ylim(min(y),max(y))
    plt.title('Learned: PDF + samples')
    
    if not args.no_split_lines:
        draw_split_lines(clf, x, y)
    
    # plot: learned - pdf
    plt.subplot(3, 1, 3)
    plt.scatter(X_train_unlabelled[:,0], X_train_unlabelled[:,1], c=y_train_prediction + 1, s=20, alpha=.6)
    plt.scatter(X_train_labelled[:,0], X_train_labelled[:,1], c=y_train_labelled, s=100)
    plt.colorbar()
    
    plt.xlim(min(x),max(x))
    plt.ylim(min(y),max(y))
    plt.title('Learned: a-posteriori classification')
    
    # add split-lines
    if not args.no_split_lines:
        draw_split_lines(clf, x, y)
    
    plt.show()
    
def draw_split_lines(clf, x, y):
    """Draw the trees split lines into the current image."""
    info = clf.parse_tree_leaves()
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    hlines = ([], [], [])
    vlines = ([], [], [])
    for node in info:
        if node is not None: # leave node
            for pos in node.range[0]: # xrange
                if not np.isinf(pos):
                    xliml, xlimr = node.range[1]
                    vlines[0].append(pos)
                    vlines[1].append(ymin if np.isinf(xliml) else xliml)
                    vlines[2].append(ymax if np.isinf(xlimr) else xlimr)
            for pos in node.range[1]: # xrange
                if not np.isinf(pos):
                    yliml, ylimr = node.range[0]
                    hlines[0].append(pos)
                    hlines[1].append(xmin if np.isinf(yliml) else yliml)
                    hlines[2].append(xmax if np.isinf(ylimr) else ylimr)
    plt.hlines(hlines[0], hlines[1], hlines[2], colors='blue', linestyles='dotted')
    plt.vlines(vlines[0], vlines[1], vlines[2], colors='blue', linestyles='dotted')
    
def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    return parser.parse_args()

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('--n-clusters', default=4, type=int, help='The number of gaussian distributions to create.')
    parser.add_argument('--n-samples', default=1000, type=int, help='The number of training samples to draw from each gaussian.')
    parser.add_argument('--sigma', default=0.4, type=float, help='The sigma multiplier of the gaussian distributions.')
    parser.add_argument('--max-depth', default=None, type=int, help='The maximum tree depth.')
    parser.add_argument('--supervised-weight', default=0.5, type=float, help='The weight of the supervised metric against the un-supervised.')
    parser.add_argument('--no-split-lines', action='store_true', help='Do not plot the split-lines.')
    parser.add_argument('--resolution', default=0.05, type=float, help='The plotting resolution.')
    parser.add_argument('--max-area', default=10, type=int, help='The maximum area over which the gaussians should be distributed.')
    parser.add_argument('--seed', default=None, type=int, help='The random seed to use. Fix to an integer to create reproducible results.')

    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    return parser

import sklearnef.tree._tree as _treeef
from sklearn.tree import DecisionTreeClassifier


if __name__ == "__main__":
    main()
