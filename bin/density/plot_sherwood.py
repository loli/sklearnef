#!/usr/bin/env python

"""Plot an example of Microsoft sherwood library."""

# build-in modules
import argparse

# third-party modules
import numpy
import matplotlib.pyplot as plt

# path changes

# own modules
from sklearnef.ensemble import DensityForest

# information
__author__ = "Oskar Maier"
__version__ = "r0.1.2, 2015-05-18"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
Plot an example of Microsofts sherwood library.
"""

# code
def main():
    args = getArguments(getParser())
    
    numpy.random.seed(args.seed)
    
    # parse dataset
    data = numpy.genfromtxt(args.dataset)
    if not 2 == data.ndim:
        raise Exception("Can only plot 2D datasets.")
    
    # train forest
    clf = DensityForest(n_estimators=args.n_trees,
                        random_state=args.seed,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        max_depth=args.max_depth,
                        max_features=args.max_features,
                        min_improvement=args.min_improvement)
    clf.fit(data)
    
    # generate plot grid
    xrange = (data[:,0].min() - data[:,0].std(), data[:,0].max() + data[:,0].std())
    yrange = (data[:,1].min() - data[:,1].std(), data[:,1].max() + data[:,1].std())
    xresolution = (xrange[1] - xrange[0]) / float(args.resolution)
    yresolution = (yrange[1] - yrange[0]) / float(args.resolution) 
    
    grid = numpy.mgrid[xrange[0]:xrange[1]:xresolution,yrange[0]:yrange[1]:yresolution]
    x, y = grid
    x = numpy.unique(x)
    y = numpy.unique(y)
    
    # apply forest
    X = numpy.swapaxes(grid.reshape(2, numpy.product(grid.shape[1:])), 0, 1)
    z = clf.predict_proba(X)

    # first plot: gt
    plt.subplot(2, 1, 1, axisbg='k')
    plt.scatter(data[:, 0], data[:, 1], c='w', alpha=.3, edgecolors='none')
    
    plt.xlim(min(x),max(x))
    plt.ylim(min(y),max(y))
    plt.title('GT: {}'.format(args.dataset))

    # second plot: prediction
    plt.subplot(2, 1, 2)
    im = plt.imshow(z.reshape((x.size,y.size)).T, extent=[min(x),max(x),min(y),max(y)], interpolation='none', cmap=plt.cm.afmhot, aspect='auto', origin='lower') #'auto'
    plt.colorbar()
    
    plt.xlim(min(x),max(x))
    plt.ylim(min(y),max(y))
    plt.title('Learned density: {}'.format(args.dataset))
    
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
    parser.add_argument('dataset', help='One of the sharewood density example datasets (a text file containing a table).')
    parser.add_argument('--n-trees', default=10, type=int, help='The number of trees to train.')
    parser.add_argument('--max-depth', default=None, type=int, help='The maximum tree depth.')
    parser.add_argument('--max-features', default='auto', help='The number of features to consider at each split. Can be an integer or one of auto, sqrt and log2')    
    parser.add_argument('--min-improvement', default=-5.0, type=float, help='Minimum information gain required to consider another split. Note that the information gain can take on negative values in some situations.')
    parser.add_argument('--seed', default=None, type=int, help='The random seed to use. Fix to an integer to create reproducible results.')
    parser.add_argument('-r', '--resolution', dest='resolution', type=int, default=200, help='Plot resolution (points-per-dimension).')
    return parser

if __name__ == "__main__":
    main()
