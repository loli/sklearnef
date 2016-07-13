#!/usr/bin/env python

"""Semi-supervised classification from various distributions and plot the results."""

# build-in modules
import os
import sys
import argparse

# third-party modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.classification import accuracy_score

# path changes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # enable import from parent directory

# own modules
from lib import make_sklearn_dataset, generate_grid, draw_split_lines
from sklearnef.ensemble import SemiSupervisedRandomForestClassifier

# information
__author__ = "Oskar Maier"
__version__ = "r0.1.1, 2015-06-08"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
Semi-supervised classification from various distributions and plot the results."""

# constants
DATASETS = ['circles_distant', 'circles_near', 'moons', 'blobs', 's_curve', 'swiss_roll']

# code
def main():
    args = getArguments(getParser())

    # initialize the random seed
    np.random.seed(args.seed)
    
    # make dataset
    X, y = make_sklearn_dataset(args.dataset, args.n_samples)
        
    # normalize
    if args.scaling:
        X = StandardScaler().fit_transform(X).astype(np.float32)
    
    # ----- Create training and testing sets
    labelled_mask = np.zeros(y.shape, np.bool)
    for cid in np.unique(y):
        m = (cid == y)
        for _ in range(args.n_labelled):
            repeat = True
            while repeat:
                sel = np.random.randint(0, y.size)
                repeat = ~(m[sel] and ~labelled_mask[sel]) # belonging to target class AND not yet selected
            labelled_mask[sel] = True
    
    X_train = X
    X_train_unlabelled = X_train[~labelled_mask]
    y_train = y.copy()
    y_train[~labelled_mask] = -1
    y_train_unlabelled_gt = y[~labelled_mask]
    X_train_labelled = X_train[labelled_mask]
    y_train_labelled = y[labelled_mask]
    
    # make custom map
    cmap = plt.get_cmap('jet', len(np.unique(y_train)))
    
    # ----- Grid -----
    grid = generate_grid(X_train, X_train.std(), args.resolution)
    
    # ----- Training -----
    clf = SemiSupervisedRandomForestClassifier(random_state=args.seed,
                                               n_estimators=args.n_trees,
                                               max_depth=args.max_depth,
                                               max_features=args.max_features,
                                               supervised_weight=args.supervised_weight,
                                               min_improvement=args.min_improvement,
                                               transduction_method=args.transduction_method,
                                               unsupervised_transformation=None)
    clf.fit(X_train, y_train)
    
    # ----- Learned distribution -----
    X_test_pred = np.rollaxis(grid, 0, 3).reshape((np.product(grid.shape[1:]), grid.shape[0]))
    pdf = clf.pdf(X_test_pred)
    
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
    pdf_vmin = pdf.min()
    pdf_vmax = pdf.max()
    
    # plot: gt - pdf
    plt.subplot(3, 1, 1)
    plt.scatter(X_train_unlabelled[:,0], X_train_unlabelled[:,1], c=cmap(y_train_unlabelled_gt.astype(np.uint8)), s=20, alpha=.6)
    plt.scatter(X_train_labelled[:,0], X_train_labelled[:,1], c=cmap(y_train_labelled.astype(np.uint8)), s=100)

    
    plt.xlim(min(x),max(x))
    plt.ylim(min(y),max(y))
    plt.title('Ground-truth: PDF + samples')
    
    if args.split_lines:
        draw_split_lines(clf.estimators_[0], x, y)
    
    # plot: learned - pdf
    plt.subplot(3, 1, 2)
    img = plt.imshow(pdf.reshape((x.size,y.size)).T, extent=[min(x),max(x),min(y),max(y)],
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
    #plt.scatter(X_train_unlabelled[:,0], X_train_unlabelled[:,1], c=cmap(y_train_prediction.astype(np.uint8)), s=20, alpha=.6)
    #plt.scatter(X_train_labelled[:,0], X_train_labelled[:,1], c=cmap(y_train_labelled.astype(np.uint8)), s=100)
    plt.colorbar(img)
    
    plt.xlim(min(x),max(x))
    plt.ylim(min(y),max(y))
    plt.title('Learned forest: PDF + a-posteriori classification')
    
    # add split-lines
    if args.split_lines:
        draw_split_lines(clf.estimators_[0], x, y)
    
    if args.save:
       plt.savefig(args.save)
    else:
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
    parser.add_argument('dataset', choices=DATASETS, help='The dataset to use.')
    parser.add_argument('--n-trees', default=10, type=int, help='The number of trees to train.')
    parser.add_argument('--n-labelled', default=1, type=int, help='The number labelled samples per class.')
    parser.add_argument('--n-samples', default=200, type=int, help='The number of training samples to draw from each dataset.')
    parser.add_argument('--max-depth', default=None, type=int, help='The maximum tree depth.')
    parser.add_argument('--transduction-method', default='diffusion', choices=['diffusion', 'approximate'], help='The transduction method to use.')
    parser.add_argument('--max-features', default='auto', help='The number of features to consider at each split. Can be an integer or one of auto, sqrt and log2')
    parser.add_argument('--supervised-weight', default=0.5, type=float, help='The weight of the supervised metric against the un-supervised.')
    parser.add_argument('--min-improvement', default=-5.0, type=float, help='Minimum information gain required to consider another split. Note that the information gain can take on negative values in some situations.')
    parser.add_argument('--split-lines', action='store_true', help='Plot the split-lines of the first tree in the forest.')
    parser.add_argument('--scaling', action='store_true', help='Enable data scaling.')
    parser.add_argument('--resolution', default=100, type=float, help='The plotting resolution i.e. dots per dimension.')
    parser.add_argument('--seed', default=None, type=int, help='The random seed to use. Fix to an integer to create reproducible results.')
    parser.add_argument('--save', default=None, help='Save the plot to this file instead of displaying it.')

    #parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    #parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    return parser

if __name__ == "__main__":
    main()
