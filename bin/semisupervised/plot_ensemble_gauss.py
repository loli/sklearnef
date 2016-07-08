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
from sklearn.metrics import accuracy_score

# path changes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # enable import from parent directory

# own modules
from lib import generate_clusters, sample_data, scale_data, generate_grid, draw_split_lines
from sklearnef.ensemble import SemiSupervisedRandomForestClassifier

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
the Gaussian distributions.
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
    
    y_train_unlabelled_gt = np.repeat(np.arange(len(means)), args.n_samples)
    
    # make custom map
    cmap = plt.get_cmap('jet', len(np.unique(y_train)))
    
    # ----- Data scaling ----
    # Must be performed before to display final data in the right space
    if args.scaling:
        scale_data(X_train, (X_train, X_train_unlabelled, X_train_labelled, means))
    
    # ----- Grid -----
    grid = generate_grid(X_train, args.sigma, args.resolution)
    
    # ----- Training -----
    clf = SemiSupervisedRandomForestClassifier(random_state=args.seed,
                                               n_estimators=args.n_trees,
                                               max_depth=args.max_depth,
                                               max_features=args.max_features,
                                               supervised_weight=args.supervised_weight,
                                               min_improvement=args.min_improvement,
                                               transduction_method=args.transduction_method,
                                               unsupervised_transformation='scale' if args.scaling else None)
    clf.fit(X_train, y_train)
    
    # ----- Learned distribution -----
    X_test_pred = np.rollaxis(grid, 0, 3).reshape((np.product(grid.shape[1:]), grid.shape[0]))
    pdf = clf.pdf(X_test_pred)
    pdf_tree = clf.estimators_[0].pdf(X_test_pred)
    
    # ----- Ground truth distribution -----
    X_test_gt = np.rollaxis(grid, 0, 3)
    prob_gt = np.sum([scipy.stats.multivariate_normal.pdf(X_test_gt, mean, cov) for mean, cov in zip(means, covs)], 0)
    prob_gt /= args.n_clusters # normalize
    
    # ----- Transduction -----
    y_train_result = clf.transduced_labels_
    
    # ----- A-posteriori classification -----
    y_train_prediction = clf.predict(X_train_unlabelled)
    
    # ----- Scoring -----
    print 'SCORES:'
    print '\t', accuracy_score(y_train_unlabelled_gt, y_train_result), 'Labeling through transduction'
    print '\t', accuracy_score(y_train_unlabelled_gt, y_train_prediction), 'Labeling through classification'
    
    # ----- Plotting -----
    x, y = grid
    x = np.unique(x)
    y = np.unique(y) 
    
    # colour range: pdf
    pdf_vmin = min(prob_gt.min(), pdf.min(), pdf_tree.min())
    pdf_vmax = min(prob_gt.max(), pdf.max(), pdf_tree.max())
    
    # plot: gt - pdf
    plt.subplot(3, 1, 1)
    img = plt.imshow(prob_gt.T, extent=[min(x),max(x),min(y),max(y)], interpolation='none',
                     cmap=plt.cm.afmhot, aspect='auto', origin='lower',
                     vmin=pdf_vmin, vmax=pdf_vmax, alpha=.5) #'auto'
    plt.scatter(X_train_unlabelled[:,0], X_train_unlabelled[:,1], c=cmap(y_train_gt.astype(np.uint8)), s=20, alpha=.6)
    plt.scatter(X_train_labelled[:,0], X_train_labelled[:,1], c=cmap(y_train_labelled.astype(np.uint8)), s=100)
    plt.colorbar(img)
    
    plt.xlim(min(x),max(x))
    plt.ylim(min(y),max(y))
    plt.title('Ground-truth: PDF + samples')
    
    if args.split_lines:
        draw_split_lines(clf.estimators_[0], x, y)
    
    # plot: learned - pdf
    plt.subplot(3, 1, 2)
    img = plt.imshow(pdf_tree.reshape((x.size,y.size)).T, extent=[min(x),max(x),min(y),max(y)],
                     interpolation='none', cmap=plt.cm.afmhot, aspect='auto', origin='lower',
                     vmin=pdf_vmin, vmax=pdf_vmax, alpha=.5) #'auto'
    plt.scatter(X_train_unlabelled[:,0], X_train_unlabelled[:,1], c=cmap(y_train_result.astype(np.uint8)), s=20, alpha=.6)
    plt.scatter(X_train_labelled[:,0], X_train_labelled[:,1], c=cmap(y_train_labelled.astype(np.uint8)), s=100)
    plt.colorbar(img)
    
    plt.xlim(min(x),max(x))
    plt.ylim(min(y),max(y))
    plt.title('Learned forest: PDF + samples labelled through transduction')
    
    if args.split_lines:
        draw_split_lines(clf.estimators_[0], x, y)
    
    # plot: learned - pdf
    plt.subplot(3, 1, 3)
    img = plt.imshow(pdf.reshape((x.size,y.size)).T, extent=[min(x),max(x),min(y),max(y)],
                     interpolation='none', cmap=plt.cm.afmhot, aspect='auto', origin='lower',
                     vmin=pdf_vmin, vmax=pdf_vmax, alpha=.5) #'auto'
    plt.scatter(X_train_unlabelled[:,0], X_train_unlabelled[:,1], c=cmap(y_train_prediction.astype(np.uint8)), s=20, alpha=.6)
    plt.scatter(X_train_labelled[:,0], X_train_labelled[:,1], c=cmap(y_train_labelled.astype(np.uint8)), s=100)
    plt.colorbar(img)
    
    plt.xlim(min(x),max(x))
    plt.ylim(min(y),max(y))
    plt.title('Learned forest: PDF + a-posteriori classification / induction')
    
    # add split-lines
    if args.split_lines:
        draw_split_lines(clf.estimators_[0], x, y)
    
    plt.show()
    
def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    args = parser.parse_args()
    if args.max_features not in ['auto', 'sqrt' 'log2']:
        args.max_features = int(args.max_features)
    return args


def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('--n-trees', default=10, type=int, help='The number of trees to train.')
    parser.add_argument('--n-clusters', default=4, type=int, help='The number of gaussian distributions to create.')
    parser.add_argument('--n-samples', default=200, type=int, help='The number of training samples to draw from each gaussian.')
    parser.add_argument('--sigma', default=0.4, type=float, help='The sigma multiplier of the gaussian distributions.')
    parser.add_argument('--max-depth', default=None, type=int, help='The maximum tree depth.')
    parser.add_argument('--transduction-method', default='diffusion', choices=['diffusion', 'approximate'], help='The transduction method to use.')
    parser.add_argument('--max-features', default='auto', help='The number of features to consider at each split. Can be an integer or one of auto, sqrt and log2')
    parser.add_argument('--supervised-weight', default=0.5, type=float, help='The weight of the supervised metric against the un-supervised.')
    parser.add_argument('--min-improvement', default=-5.0, type=float, help='Minimum information gain required to consider another split. Note that the information gain can take on negative values in some situations.')
    parser.add_argument('--split-lines', action='store_true', help='Plot the split-lines of the first tree in the forest.')
    parser.add_argument('--scaling', action='store_true', help='Enable data scaling.')
    parser.add_argument('--resolution', default=100, type=float, help='The plotting resolution i.e. dots per dimension.')
    parser.add_argument('--max-area', default=10, type=int, help='The maximum area over which the gaussians should be distributed.')
    parser.add_argument('--seed', default=None, type=int, help='The random seed to use. Fix to an integer to create reproducible results.')

    #parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    #parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    return parser

if __name__ == "__main__":
    main()
