#!/usr/bin/env python

"""Semi-supervised classification from various distributions and plot the results."""

# build-in modules
import argparse

# third-party modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# path changes

# own modules
from sklearnef.ensemble import SemiSupervisedRandomForestClassifier
from sklearn.metrics.classification import accuracy_score

# information
__author__ = "Oskar Maier"
__version__ = "r0.1.0, 2015-06-08"
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
    
    # create dataset
    if 'circles_distant' == args.dataset: # labels=3, seed=1, n-samples=1000, max-depth=4 OR labels=4, seed=1, n-samples=1000, max-depth=4
        dataset = datasets.make_circles(n_samples=args.n_samples, factor=.5, noise=.05)
    elif 'moons' == args.dataset: # labels=2, seed=13, n-samples=500, max-depth=4 OR labels=1, seed=27, n-samples=500, max-depth=4
        dataset = datasets.make_moons(n_samples=args.n_samples, noise=.05)
    elif 'blobs' == args.dataset: # labels=1, seed=0, n-samples=100, max-depth=3 
        dataset = datasets.make_blobs(n_samples=args.n_samples, random_state=8)
    elif 'circles_near' == args.dataset: # labels = 20, seed=0, n-samples=2000, max-depth=5
        dataset = datasets.make_circles(n_samples=args.n_samples, noise=.05)
    elif 's_curve' == args.dataset: # labels=10, seed=35, n-samples=2500, max-depth=7
        scurve1 = datasets.make_s_curve(n_samples=args.n_samples // 2, noise=.05)
        scurve1 = np.vstack((scurve1[0][:, 0], scurve1[0][:, 2])).T
        scurve2 = datasets.make_s_curve(n_samples=args.n_samples // 2, noise=.05)
        scurve2 = np.vstack((scurve2[0][:, 0], scurve2[0][:, 2])).T + [.5, .5] # offset
        dataset = np.concatenate((scurve1, scurve2), 0), \
                  np.concatenate((np.asarray([0] * scurve1.shape[0]),
                                  np.asarray([1] * scurve2.shape[0])), 0)
    elif 'swiss_roll' == args.dataset: # labels = 10, seed = 35, n-samples=2500, max-depth=5
        sroll1 = datasets.make_swiss_roll(n_samples=args.n_samples // 2, noise=.05)
        sroll1 = np.vstack((sroll1[0][:,0], sroll1[0][:,2])).T
        sroll2 = datasets.make_swiss_roll(n_samples=args.n_samples // 2, noise=.05)
        sroll2 = np.vstack((sroll2[0][:,0], sroll2[0][:,2])).T * 0.75 # shrink
        dataset = np.concatenate((sroll1, sroll2), 0), \
                  np.concatenate((np.asarray([0] * sroll1.shape[0]),
                                  np.asarray([1] * sroll2.shape[0])), 0)
        
    # split and normalize
    X, y = dataset
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
    
    # ----- Grid -----
    x_lower = X[:,0].min() - X[:,0].std()
    x_upper = X[:,0].max() + X[:,0].std()
    y_lower = X[:,1].min() - X[:,1].std()
    y_upper = X[:,1].max() + X[:,1].std()
    grid = np.mgrid[x_lower:x_upper:args.resolution,y_lower:y_upper:args.resolution]
    
    # ----- Training -----
    clf = SemiSupervisedRandomForestClassifier(random_state=args.seed,
                                               n_estimators=args.n_trees,
                                               max_depth=args.max_depth,
                                               max_features=args.max_features,
                                               supervised_weight=args.supervised_weight,
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
    print '\t', accuracy_score(y_train_unlabelled_gt, y_train_result), 'Labeling throught transduction'
    print '\t', accuracy_score(y_train_unlabelled_gt, y_train_prediction), 'Labeling throught classification'
    
    # ----- Plotting -----
    x, y = grid
    x = np.unique(x)
    y = np.unique(y) 
    
    # colour range: pdf
    pdf_vmin = pdf.min()
    pdf_vmax = pdf.max()
    
    # plot: gt - pdf
    plt.subplot(3, 1, 1)
    #plt.imshow(prob_gt.T, extent=[min(x),max(x),min(y),max(y)], interpolation='none',
    #           cmap=plt.cm.afmhot, aspect='auto', origin='lower',
    #           vmin=pdf_vmin, vmax=pdf_vmax, alpha=.5) #'auto'
    plt.scatter(X_train_unlabelled[:,0], X_train_unlabelled[:,1], c=y_train_unlabelled_gt, s=20, alpha=.6)
    plt.scatter(X_train_labelled[:,0], X_train_labelled[:,1], c=y_train_labelled, s=100)
    plt.colorbar()
    
    plt.xlim(min(x),max(x))
    plt.ylim(min(y),max(y))
    plt.title('Ground-truth: PDF + samples')
    
    if args.split_lines:
        draw_split_lines(clf.estimators_[0], x, y)
    
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
    plt.title('Learned forest: PDF + samples labelled through transduction')
    
    if args.split_lines:
        draw_split_lines(clf.estimators_[0], x, y)
    
    # plot: learned - pdf
    plt.subplot(3, 1, 3)
    plt.imshow(pdf.reshape((x.size,y.size)).T, extent=[min(x),max(x),min(y),max(y)],
               interpolation='none', cmap=plt.cm.afmhot, aspect='auto', origin='lower',
               vmin=pdf_vmin, vmax=pdf_vmax, alpha=.5) #'auto'
    plt.scatter(X_train_unlabelled[:,0], X_train_unlabelled[:,1], c=y_train_prediction + 1, s=20, alpha=.6)
    plt.scatter(X_train_labelled[:,0], X_train_labelled[:,1], c=y_train_labelled, s=100)
    plt.colorbar()
    
    plt.xlim(min(x),max(x))
    plt.ylim(min(y),max(y))
    plt.title('Learned forest: PDF + a-posteriori classification')
    
    # add split-lines
    if args.split_lines:
        draw_split_lines(clf.estimators_[0], x, y)
    
    #plt.show()
    
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
    parser.add_argument('dataset', choices=DATASETS, help='The dataset to use.')
    parser.add_argument('--n-trees', default=10, type=int, help='The number of trees to train.')
    parser.add_argument('--n-labelled', default=1, type=int, help='The number labelled samples per class.')
    parser.add_argument('--n-samples', default=200, type=int, help='The number of training samples to draw from each gaussian.')
    parser.add_argument('--max-depth', default=None, type=int, help='The maximum tree depth.')
    parser.add_argument('--max-features', default='auto', help='The number of features to consider at each split.')
    parser.add_argument('--supervised-weight', default=0.5, type=float, help='The weight of the supervised metric against the un-supervised.')
    parser.add_argument('--split-lines', action='store_true', help='Plot the split-lines of the first tree in the forest.')
    parser.add_argument('--resolution', default=0.05, type=float, help='The plotting resolution.')
    parser.add_argument('--seed', default=None, type=int, help='The random seed to use. Fix to an integer to create reproducible results.')

    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    return parser

if __name__ == "__main__":
    main()
