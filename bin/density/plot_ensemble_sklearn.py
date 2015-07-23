#!/usr/bin/env python

"""Learning density distribution from sklearn created datasets and plot the results."""

# build-in modules
import argparse

# third-party modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# path changes

# own modules
from sklearnef.ensemble import DensityForest

# information
__author__ = "Oskar Maier"
__version__ = "r0.1.1, 2015-06-08"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
Learning density distribution from sklearn created datasets and plot
the results.
"""

# constants
DATASETS = ['circles_distant', 'circles_near', 'moons', 'blobs', 's_curve', 'swiss_roll']

# code
def main():
    args = getArguments(getParser())

    # initialize the random seed
    np.random.seed(args.seed)
    
    # create dataset
    if 'circles_distant' == args.dataset:
        dataset = datasets.make_circles(n_samples=args.n_samples, factor=.5, noise=.05)
    elif 'moons' == args.dataset:
        dataset = datasets.make_moons(n_samples=args.n_samples, noise=.05)
    elif 'blobs' == args.dataset:
        dataset = datasets.make_blobs(n_samples=args.n_samples, random_state=8)
    elif 'circles_near' == args.dataset:
        dataset = datasets.make_circles(n_samples=args.n_samples, noise=.05)
    elif 's_curve' == args.dataset:
        dataset = datasets.make_s_curve(n_samples=args.n_samples, noise=.05)
        dataset = np.vstack((dataset[0][:, 0], dataset[0][:, 2])).T, None
    elif 'swiss_roll' == args.dataset:
        dataset = datasets.make_swiss_roll(n_samples=args.n_samples, noise=.05)
        dataset = np.vstack((dataset[0][:,0], dataset[0][:,2])).T, None
        
    # split and normalize
    X, _ = dataset
    X = StandardScaler().fit_transform(X)
    
    # ----- Grid -----
    x_lower = X[:,0].min() - X[:,0].std()
    x_upper = X[:,0].max() + X[:,0].std()
    y_lower = X[:,1].min() - X[:,1].std()
    y_upper = X[:,1].max() + X[:,1].std()
    grid = np.mgrid[x_lower:x_upper:args.resolution,y_lower:y_upper:args.resolution]
    
    # ----- Training -----
    clf = DensityForest(n_estimators=args.n_trees,
                        random_state=args.seed,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        max_features='auto',
                        min_improvement=args.min_improvement)
    clf.fit(X)
    
    # ----- Prediction -----
    X_test_pred = np.rollaxis(grid, 0, 3).reshape((np.product(grid.shape[1:]), grid.shape[0]))
    prob_predict = clf.predict_proba(X_test_pred)
    
    # ----- Plotting -----
    x, y = grid
    x = np.unique(x)
    y = np.unique(y)
    
    # first plot: gt
    plt.subplot(2, 1, 1, axisbg='k')
    plt.scatter(X[:, 0], X[:, 1], c='w', alpha=.3, edgecolors='none')
    
    plt.xlim(min(x),max(x))
    plt.ylim(min(y),max(y))
    plt.title('GT: {}'.format(args.dataset))
    
    # second plot: prediction
    plt.subplot(2, 1, 2)
    im = plt.imshow(prob_predict.reshape((x.size,y.size)).T, extent=[min(x),max(x),min(y),max(y)], interpolation='none', cmap=plt.cm.afmhot, aspect='auto', origin='lower') #'auto'
    plt.colorbar()
    
    plt.xlim(min(x),max(x))
    plt.ylim(min(y),max(y))
    plt.title('Learned density: {}'.format(args.dataset))
    
    plt.show()
    
def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    return parser.parse_args()

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('dataset', choices=DATASETS, help='The dataset to use.')
    parser.add_argument('--n-trees', default=10, type=int, help='The number of trees to train.')
    parser.add_argument('--n-samples', default=2500, type=int, help='The number of samples to draw.')
    parser.add_argument('--min-improvement', default=0, type=float, help='The minimum improvement require to consider a split valid.')
    parser.add_argument('--resolution', default=0.01, type=float, help='The plotting resolution.')
    parser.add_argument('--seed', default=None, type=int, help='The random seed to use. Fix to an integer to create reproducible results.')

    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    return parser

if __name__ == "__main__":
    main()
