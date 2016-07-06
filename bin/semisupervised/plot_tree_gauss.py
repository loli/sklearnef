#!/usr/bin/env python

"""Semi-supervised classification from 2D gaussians and plot the results."""

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
from sklearnef.tree import SemiSupervisedDecisionTreeClassifier
from sklearn.tree import export_graphviz

# information
__author__ = "Oskar Maier"
__version__ = "r0.1.2, 2015-06-08"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
Semi-supervised classification from 2D gaussians and plot the results.

The cluster centers will be distributed within a certain distance of
each other and the co-variance matrices skewed randomly according to
the supplied sigma multiplier. The center of the clusters receive a
unique label, all other (unlabelled) samples are randomly drawn from
the Gaussian distributions.

The classifier will be trained with a single tree to better observe the
training effects and max_features will be disabled.

Note: If scaling is enabled, the PDF in original space will reflect the unscaled version!
"""

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
    clf = SemiSupervisedDecisionTreeClassifier(random_state=args.seed,
                                               max_depth=args.max_depth,
                                               max_features=args.max_features,
                                               supervised_weight=args.supervised_weight,
                                               min_improvement=args.min_improvement,
                                               unsupervised_transformation='scale' if args.scaling else None)
    clf.fit(X_train, y_train)
    
    # ----- plot tree into file -----
    # Convert with: dot -Tps tree.dot -o tree.ps
    export_graphviz(clf)
    
    # ----- Learned distribution -----
    X_test_pred = np.rollaxis(grid, 0, 3).reshape((np.product(grid.shape[1:]), grid.shape[0]))
    pdf = clf.pdf(X_test_pred)
    
    # ----- Ground truth distribution -----
    X_test_gt = np.rollaxis(grid, 0, 3)
    prob_gt = np.sum([scipy.stats.multivariate_normal.pdf(X_test_gt, mean, cov) for mean, cov in zip(means, covs)], 0)
    prob_gt /= args.n_clusters # normalize
    
    # ----- Transduction -----
    y_train_result = clf.transduced_labels_
    
    # ----- A-posteriori classification / induction -----
    y_train_prediction = clf.predict(X_train_unlabelled)
    
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
    plt.title('Learned: PDF + samples labelled through transduction')
    
    if not args.no_split_lines:
        draw_split_lines(clf, x, y)
    
    # plot: learned - pdf
    plt.subplot(3, 1, 3)
    plt.scatter(X_train_unlabelled[:,0], X_train_unlabelled[:,1], c=y_train_prediction + 1, s=20, alpha=.6)
    plt.scatter(X_train_labelled[:,0], X_train_labelled[:,1], c=y_train_labelled, s=100)
    plt.colorbar()
    
    plt.xlim(min(x),max(x))
    plt.ylim(min(y),max(y))
    plt.title('Learned: a-posteriori classification / induction')
    
    # add split-lines
    if not args.no_split_lines:
        draw_split_lines(clf, x, y)
    
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
    parser.add_argument('--n-clusters', default=4, type=int, help='The number of gaussian distributions to create.')
    parser.add_argument('--n-samples', default=200, type=int, help='The number of training samples to draw from each gaussian.')
    parser.add_argument('--sigma', default=0.4, type=float, help='The sigma multiplier of the gaussian distributions.')
    parser.add_argument('--max-depth', default=None, type=int, help='The maximum tree depth.')
    parser.add_argument('--max-features', default=None, help='The number of features to consider at each split. Can be an integer or one of auto, sqrt and log2')
    parser.add_argument('--supervised-weight', default=0.5, type=float, help='The weight of the supervised metric against the un-supervised.')
    parser.add_argument('--min-improvement', default=-5.0, type=float, help='Minimum information gain required to consider another split. Note that the information gain can take on negative values in some situations.')
    parser.add_argument('--no-split-lines', action='store_true', help='Do not plot the split-lines.')
    parser.add_argument('--scaling', action='store_true', help='Enable data scaling.')
    parser.add_argument('--resolution', default=100, type=float, help='The plotting resolution i.e. dots per dimension.')
    parser.add_argument('--max-area', default=10, type=int, help='The maximum area over which the gaussians should be distributed.')
    parser.add_argument('--seed', default=None, type=int, help='The random seed to use. Fix to an integer to create reproducible results.')

    #parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    #parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    return parser

if __name__ == "__main__":
    main()
