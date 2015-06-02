#!/usr/bin/env python

"""Plot an example of Microsoft sherwood library."""

# build-in modules
import argparse

# third-party modules
import numpy
import matplotlib.pyplot as plt

# path changes

# own modules
from tree import UnSupervisedDecisionTreeClassifier

# information
__author__ = "Oskar Maier"
__version__ = "r0.1.0, 2015-05-18"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
Plot an example of Microsofts sherwood library.
"""

# code
def main():
    args = getArguments(getParser())
    
    # parse dataset
    data = numpy.genfromtxt(args.dataset)
    if not 2 == data.ndim:
        raise Exception("Can only plot 2D datasets.")
    
    # train forest
    clf = UnSupervisedDecisionTreeClassifier(random_state=0, min_samples_leaf=20)
    clf.fit(data)
    
    # generate plot grid
    xrange = (data[:,0].min(), data[:,0].max())
    yrange = (data[:,1].min(), data[:,1].max())
    xresolution = (xrange[1] - xrange[0]) / float(args.resolution)
    yresolution = (yrange[1] - yrange[0]) / float(args.resolution) 
    
    grid = numpy.mgrid[xrange[0]:xrange[1]:xresolution,yrange[0]:yrange[1]:yresolution]
    x, y = grid
    x = numpy.unique(x)
    y = numpy.unique(y)
    
    # apply forest
    X = numpy.swapaxes(grid.reshape(2, numpy.product(grid.shape[1:])), 0, 1)
    z = clf.predict_proba(X)
    z = z.reshape(grid.shape[1:])

    # plot
    plt.subplot(2, 1, 1)
    CS = plt.contour(x,y,z.T,15,linewidths=0.5,colors='k')
    CS = plt.contourf(x,y,z.T,100,cmap=plt.cm.PiYG)
    plt.colorbar()
    
    plt.scatter(data[:,0], data[:,1], alpha=0.2)
    
    plt.xlim(min(x), max(x))
    plt.ylim(min(y), max(y))
    plt.title('Learned density distribution of: {}'.format(args.dataset))
    
    # plot split-lines
    info = clf.parse_tree_leaves()
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    hlines = ([], [], [])
    vlines = ([], [], [])
    for node in info:
        if node is not None: # leave node
            for pos in node['range'][0]: # xrange
                if not numpy.isinf(pos):
                    xliml, xlimr = node['range'][1]
                    vlines[0].append(pos)
                    vlines[1].append(ymin if numpy.isinf(xliml) else xliml)
                    vlines[2].append(ymax if numpy.isinf(xlimr) else xlimr)
            for pos in node['range'][1]: # xrange
                if not numpy.isinf(pos):
                    yliml, ylimr = node['range'][0]
                    hlines[0].append(pos)
                    hlines[1].append(xmin if numpy.isinf(yliml) else yliml)
                    hlines[2].append(xmax if numpy.isinf(ylimr) else ylimr)
    plt.hlines(hlines[0], hlines[1], hlines[2], colors='blue', linestyles='dotted')
    plt.vlines(vlines[0], vlines[1], vlines[2], colors='blue', linestyles='dotted')
    
    plt.show()

def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    return parser.parse_args()

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('dataset', help='One of the sharewood density example datasets (a text file containing a table).')
    #parser.add_argument('output', help='Target volume.')
    parser.add_argument('-r', '--resolution', dest='resolution', type=int, default=100, help='Plot resolution (points-per-dimension).')
    #parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    #parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    #parser.add_argument('-f', dest='force', action='store_true', help='Silently override existing output images.')
    return parser

if __name__ == "__main__":
    main()
