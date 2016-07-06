#!/usr/bin/env python

"""Learning density distribution from sklearn created datasets and plot the results."""

# build-in modules
import os
import sys
import argparse

# third-party modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# path changes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # enable import from parent directory

# own modules
from lib import make_sklearn_dataset, generate_grid, draw_split_lines, plot_density, plot_gt
from sklearnef.ensemble import DensityForest

# information
__author__ = "Oskar Maier"
__version__ = "r0.1.3, 2015-06-08"
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
    if args.scaling:
        X = StandardScaler().fit_transform(X).astype(np.float32)
    
    # ----- Grid -----
    grid = generate_grid(X, X.std(), args.resolution)
    
    # ----- Training -----
    clf = DensityForest(n_estimators=args.n_trees,
                        random_state=args.seed,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        max_depth=args.max_depth,
                        max_features=args.max_features,
                        min_improvement=args.min_improvement)
    clf.fit(X)
    
    # ----- Prediction -----
    X_test_pred = np.rollaxis(grid, 0, 3).reshape((np.product(grid.shape[1:]), grid.shape[0]))
    prob_predict = clf.predict_proba(X_test_pred)
    
    # ----- Plotting -----
    x, y = grid
    x = np.unique(x)
    y = np.unique(y)
    
    if not args.skipgt:
        if not args.skipdensity:
            plt.subplot(2, 1, 1, axisbg='k')
            plot_gt(X, x, y, args)
            plt.subplot(2, 1, 2)
            plot_density(prob_predict, x, y, args)
        else:
            plt.subplot(1, 1, 1, axisbg='k')
            plot_gt(X, x, y, args)
    else:
        plt.subplot(1, 1, 2)
        plot_density(prob_predict, x, y, args)
    
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
    parser.add_argument('--n-samples', default=2500, type=int, help='The number of samples to draw.')
    parser.add_argument('--max-depth', default=None, type=int, help='The maximum tree depth.')
    parser.add_argument('--max-features', default='auto', help='The number of features to consider at each split. Can be an integer or one of auto, sqrt and log2')    
    parser.add_argument('--min-improvement', default=-5.0, type=float, help='Minimum information gain required to consider another split. Note that the information gain can take on negative values in some situations.')
    parser.add_argument('--scaling', action='store_true', help='Enable data scaling.')
    parser.add_argument('--resolution', default=100, type=float, help='The plotting resolution i.e. dots per dimension.')
    parser.add_argument('--seed', default=None, type=int, help='The random seed to use. Fix to an integer to create reproducible results.')
    parser.add_argument('--save', help='Save the plot into an image file instead of plotting it.')
    parser.add_argument('--skipgt', action='store_true', help='Do not plot the ground truth image.')
    parser.add_argument('--skipdensity', action='store_true', help='Do not plot the density image.')
    
    #parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    #parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    return parser

if __name__ == "__main__":
    main()
